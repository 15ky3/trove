"""Paths, settings persistence and the on-disk repo layout.

Settings live in CONFIG_DIR/settings.json (a volume) so the token and the queue
options survive a container rebuild. Environment variables only seed the
defaults: whatever is set in the web interface wins.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/config"))

SETTINGS_FILE = CONFIG_DIR / "settings.json"
JOBS_FILE = CONFIG_DIR / "jobs.json"
SECRET_FILE = CONFIG_DIR / "session.key"

#: Marker file written into every downloaded repo. It makes the library listing
#: exact (repo id, revision, timestamp) instead of guessing from folder names.
MARKER_NAME = ".trove.json"
#: Former name, still read so downloads made under it do not vanish from the
#: library.
LEGACY_MARKER_NAMES = (".hf-docker.json",)

REPO_TYPES = ("model", "dataset", "space")
REPO_SUBDIR = {"model": "models", "dataset": "datasets", "space": "spaces"}

DEFAULT_SETTINGS: dict[str, Any] = {
    "hf_token": "",
    "endpoint": "",
    # How many transfers may run at the same time.
    "max_concurrent": 2,
    # How many files of one repo are fetched at the same time (hf_hub
    # max_workers). Lower than huggingface_hub's own default of 8: every file in
    # flight holds Xet transfer buffers, and this multiplies with
    # max_concurrent. Four keeps a NAS out of trouble and still saturates a
    # normal line; the interface allows up to 32.
    "max_workers": 4,
    # Drop finished jobs from the queue list automatically.
    "auto_clear_done": False,
    # Download speed ceiling in Mbit/s, 0 for none. Enforced by a local proxy
    # in the worker (see app/throttle.py), because the kernel this ships to
    # cannot shape a container's incoming traffic.
    "max_download_mbit": 0.0,
}

#: Beyond this a limit stops meaning anything on the hardware this runs on.
MAX_DOWNLOAD_MBIT = 10_000.0

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class Settings:
    """Thread-safe JSON settings with an environment fallback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self) -> None:
        data = dict(DEFAULT_SETTINGS)
        try:
            raw = json.loads(SETTINGS_FILE.read_text())
            if isinstance(raw, dict):
                data.update({k: v for k, v in raw.items() if k in DEFAULT_SETTINGS})
        except (OSError, ValueError):
            pass

        # The environment only seeds values that were never saved.
        if not data["hf_token"]:
            data["hf_token"] = os.getenv("HF_TOKEN", "").strip()
        if not data["endpoint"]:
            data["endpoint"] = os.getenv("HF_ENDPOINT", "").strip()

        with self._lock:
            self._data = data

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        with self._lock:
            payload = json.dumps(self._data, indent=2)
        tmp.write_text(payload)
        os.chmod(tmp, 0o600)
        tmp.replace(SETTINGS_FILE)

    def get(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key, DEFAULT_SETTINGS.get(key))

    def all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in values.items():
            if key not in DEFAULT_SETTINGS:
                continue
            if key in ("max_concurrent", "max_workers"):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                # Same bounds as the input fields in the web interface.
                value = max(1, min(16 if key == "max_concurrent" else 32, value))
            elif key == "max_download_mbit":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                # json.loads accepts a bare NaN, and clamping one yields the
                # maximum rather than nothing: min(10_000, nan) keeps 10_000
                # because the comparison is false. Refuse it like any other
                # value that is not a number.
                if not math.isfinite(value):
                    continue
                # A negative ceiling would mean "stop entirely"; it means "off".
                value = max(0.0, min(MAX_DOWNLOAD_MBIT, round(value, 1)))
            elif key == "auto_clear_done":
                value = bool(value)
            else:
                value = str(value or "").strip()
            clean[key] = value

        with self._lock:
            self._data.update(clean)
        self.save()
        return self.all()

    def public(self) -> dict[str, Any]:
        """Settings for the interface — the token only leaves here masked."""
        data = self.all()
        token = data.pop("hf_token", "")
        data["token_set"] = bool(token)
        data["token_hint"] = f"{token[:3]}…{token[-4:]}" if len(token) > 10 else ("set" if token else "")
        return data


settings = Settings()


def session_secret() -> bytes:
    """Persistent session key, so sign-ins survive a restart."""
    try:
        return SECRET_FILE.read_bytes()
    except OSError:
        key = secrets.token_bytes(32)
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            SECRET_FILE.write_bytes(key)
            os.chmod(SECRET_FILE, 0o600)
        except OSError:
            pass
        return key


def valid_repo_id(repo_id: str) -> bool:
    return bool(_REPO_ID_RE.match(repo_id or "")) and ".." not in repo_id


def type_root(repo_type: str) -> Path:
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Unknown repo type: {repo_type}")
    return DATA_DIR / REPO_SUBDIR[repo_type]


def local_dir_for(repo_type: str, repo_id: str) -> Path:
    """Target folder: <DATA_DIR>/<models|datasets|spaces>/<org>/<name>."""
    if not valid_repo_id(repo_id):
        raise ValueError(f"Invalid repo ID: {repo_id!r}")
    root = type_root(repo_type)
    path = (root / repo_id).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError("Path escapes the data directory")
    return path


def ensure_dirs() -> None:
    for repo_type in REPO_TYPES:
        type_root(repo_type).mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def inside_data_dir(path: Path) -> bool:
    """Guard against path traversal for upload/delete paths from the UI."""
    try:
        resolved = path.resolve()
        root = DATA_DIR.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents

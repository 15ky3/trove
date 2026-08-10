"""The local library: listing repos, measuring their size, deleting them."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

from .config import LEGACY_MARKER_NAMES, MARKER_NAME, REPO_TYPES, local_dir_for, type_root

# Size cache: walking a 500 GB directory takes a while, and the interface asks
# for the listing on every view switch.
_SIZE_CACHE: dict[str, tuple[float, int, int]] = {}
_CACHE_TTL = 60.0

# Internal folders that do not count as repo content.
_INTERNAL = {".cache", ".git", ".locks"}


def _dir_stats(path: Path) -> tuple[int, int]:
    """(bytes, file_count), served from a short-lived cache."""
    key = str(path)
    now = time.time()
    cached = _SIZE_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1], cached[2]

    total = 0
    files = 0
    for root, dirnames, filenames in os.walk(path, onerror=lambda _e: None):
        dirnames[:] = [d for d in dirnames if d not in _INTERNAL]
        for name in filenames:
            try:
                stat = os.stat(os.path.join(root, name), follow_symlinks=False)
            except OSError:
                continue
            total += stat.st_size
            files += 1

    _SIZE_CACHE[key] = (now, total, files)
    return total, files


def invalidate(path: Path | None = None) -> None:
    if path is None:
        _SIZE_CACHE.clear()
    else:
        _SIZE_CACHE.pop(str(path), None)


def _has_files(path: Path) -> bool:
    try:
        return any(entry.is_file() for entry in os.scandir(path))
    except OSError:
        return False


def _read_marker(path: Path) -> dict[str, Any]:
    for name in (MARKER_NAME, *LEGACY_MARKER_NAMES):
        try:
            data = json.loads((path / name).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def _iter_repo_dirs(root: Path) -> Iterator[Path]:
    """Repos live at <org>/<name>; ones without an org sit directly below."""
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name.lower())
    except OSError:
        return
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False) or entry.name.startswith("."):
            continue
        path = Path(entry.path)
        # A folder holding files is itself a repo (e.g. "gpt2"); a folder
        # holding only other folders is an org namespace.
        if _has_files(path) or (path / MARKER_NAME).exists():
            yield path
            continue
        children = [Path(c.path) for c in os.scandir(path) if c.is_dir(follow_symlinks=False)]
        if children:
            yield from sorted(children, key=lambda p: p.name.lower())
        else:
            yield path


def list_repos(repo_type: str | None = None, refresh: bool = False) -> list[dict[str, Any]]:
    if refresh:
        invalidate()

    types = [repo_type] if repo_type in REPO_TYPES else list(REPO_TYPES)
    out: list[dict[str, Any]] = []

    for rtype in types:
        root = type_root(rtype)
        if not root.exists():
            continue
        for path in _iter_repo_dirs(root):
            marker = _read_marker(path)
            size, files = _dir_stats(path)
            rel = path.relative_to(root).as_posix()
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            out.append(
                {
                    "repo_id": marker.get("repo_id") or rel,
                    "repo_type": marker.get("repo_type") or rtype,
                    "path": str(path),
                    "size": size,
                    "files": files,
                    "revision": marker.get("revision") or "",
                    # Full sha — the UI shortens it, the update check compares it.
                    "commit": marker.get("commit") or "",
                    "downloaded_at": marker.get("downloaded_at") or mtime,
                    "complete": bool(marker),
                    "files_selected": marker.get("files") or [],
                    "allow_patterns": marker.get("allow_patterns") or [],
                    "ignore_patterns": marker.get("ignore_patterns") or [],
                    "partial": bool(
                        marker.get("files")
                        or marker.get("allow_patterns")
                        or marker.get("ignore_patterns")
                    ),
                }
            )

    out.sort(key=lambda r: r["downloaded_at"], reverse=True)
    return out


def repo_files(repo_type: str, repo_id: str, limit: int = 2000) -> list[dict[str, Any]]:
    path = local_dir_for(repo_type, repo_id)
    if not path.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for root, dirnames, filenames in os.walk(path, onerror=lambda _e: None):
        dirnames[:] = [d for d in dirnames if d not in _INTERNAL]
        for name in filenames:
            if name == MARKER_NAME or name in LEGACY_MARKER_NAMES:
                continue
            full = Path(root) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            out.append(
                {
                    "name": full.relative_to(path).as_posix(),
                    "size": stat.st_size,
                }
            )
            if len(out) >= limit:
                out.sort(key=lambda f: f["name"])
                return out
    out.sort(key=lambda f: f["name"])
    return out


def delete_repo(repo_type: str, repo_id: str) -> dict[str, Any]:
    path = local_dir_for(repo_type, repo_id)
    if not path.is_dir():
        raise FileNotFoundError(f"{repo_id} is not stored locally")

    size, files = _dir_stats(path)
    shutil.rmtree(path)
    invalidate(path)

    # Clean up an org folder left empty, so the library stays tidy.
    parent = path.parent
    root = type_root(repo_type)
    if parent != root and parent.is_dir() and not any(parent.iterdir()):
        try:
            parent.rmdir()
        except OSError:
            pass

    return {"deleted": str(path), "freed": size, "files": files}


def disk_usage() -> dict[str, int]:
    from .config import DATA_DIR

    try:
        usage = shutil.disk_usage(DATA_DIR)
        return {"total": usage.total, "free": usage.free}
    except OSError:
        return {"total": 0, "free": 0}

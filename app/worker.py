"""The transfer worker: one process per job.

Usage:  python -m app.worker '<json-payload>'

Why a process rather than a thread:
  * snapshot_download/upload_folder cannot be interrupted — a process can
    (SIGTERM), and the next attempt resumes cleanly.
  * Blocking network calls can never stall the API server.
  * The monkeypatch for upload progress stays contained to this one job.

Talking to the parent: one JSON line per event on stdout.
  meta      total size and file count are known
  progress  progress, throttled to about once a second
  log       a line for the job log
  done      finished successfully
  error     gave up with an error
"""

from __future__ import annotations

import errno
import fnmatch
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from . import storage
from .config import LEGACY_MARKER_NAMES, MARKER_NAME

# Progress comes out of huggingface_hub's own tqdm objects; we do not want
# their output on stderr.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

EMIT_INTERVAL = 0.9

# snapshot_download gives up on the whole repo as soon as one file errors: the
# files still open run to the end, but everything not yet started is dropped and
# the call raises. Without a retry a single dropped connection ends the transfer
# and the user has to notice and restart it by hand. Files that finished are on
# disk and get skipped, so another attempt only fetches what is genuinely left.
MAX_ATTEMPTS = 5
#: Seconds to wait before attempt 2, 3, 4, 5 …; the last value repeats.
RETRY_BACKOFF = (10, 30, 60, 120)


# --------------------------------------------------------------------------- IPC


_emit_lock = threading.Lock()


def emit(event: str, **payload: Any) -> None:
    payload["e"] = event
    line = json.dumps(payload, default=str)
    with _emit_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log(message: str, level: str = "info") -> None:
    emit("log", msg=message, level=level)


# ------------------------------------------------------------------- Aggregator


class Progress:
    """Collects bytes from several threads and reports upwards, throttled."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.base = 0  # bytes already on disk before this run started
        self.written = 0  # bytes written to disk (reconstruct)
        self.transfer = 0  # bytes actually pulled over the network
        self.total = 0
        self.total_files = 0
        self.done_files = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_bytes = 0
        self._last_time = time.monotonic()
        self._speed = 0.0

    def add(self, kind: str, n: int) -> None:
        with self.lock:
            if kind == "transfer":
                self.transfer = max(0, self.transfer + n)
            else:
                self.written = max(0, self.written + n)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            # Xet writes to disk in bursts while the network is already ahead
            # (or the other way round when dedup kicks in). Whichever counter is
            # further along is the honest live estimate.
            reference = max(self.base + self.written, self.base + self.transfer)
            now = time.monotonic()
            elapsed = now - self._last_time
            if elapsed >= 0.25:
                rate = (reference - self._last_bytes) / elapsed
                # Exponentially smoothed, or the readout jumps on every chunk.
                self._speed = rate if self._speed == 0 else 0.35 * rate + 0.65 * self._speed
                self._last_bytes = reference
                self._last_time = now
            # Capped, so a retry cannot push the bar past 100%.
            reported = min(reference, self.total) if self.total else reference

            return {
                "done_bytes": reported,
                "total_bytes": self.total,
                "total_files": self.total_files,
                "done_files": self.done_files,
                "speed": max(0.0, self._speed),
            }

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._loop, name="progress", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def flush(self) -> None:
        emit("progress", **self.snapshot())

    def _loop(self) -> None:
        while not self._stop.wait(EMIT_INTERVAL):
            self.flush()


progress = Progress()


# --------------------------------------------------------------------- Download


class _NullSink:
    """Swallows tqdm output; we render progress in the interface instead."""

    def write(self, *_args: Any) -> None:
        return None

    def flush(self) -> None:
        return None


def _reporter_tqdm_class():
    """A tqdm subclass that mirrors every update() into the aggregator.

    snapshot_download builds two aggregate bars from it: 'Downloading bytes'
    (network traffic) and 'Reconstructing' (written to disk). We count them
    separately — on Xet repos the network figure is lower thanks to dedup.
    """
    from huggingface_hub.utils import tqdm as hf_tqdm

    class ReporterTqdm(hf_tqdm):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            desc = str(kwargs.get("desc") or "")
            # Only byte bars count; the "Fetching n files" bar counts files.
            if kwargs.get("unit") != "B":
                self._kind = None
            else:
                self._kind = "transfer" if "Downloading bytes" in desc else "written"
            # disable=False plus a null sink: tqdm keeps its counters running
            # (huggingface_hub reads bar.n) but writes nowhere.
            kwargs["disable"] = False
            kwargs["file"] = _NullSink()
            kwargs.setdefault("mininterval", 0.5)
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1) -> Any:
            if n and self._kind:
                progress.add(self._kind, int(n))
            return super().update(n)

    return ReporterTqdm


def _as_pattern(filename: str) -> str:
    """Turn an exact filename into an fnmatch pattern.

    huggingface_hub filters by pattern only. A name like `model_[0].gguf` would
    be read as a character class, so each special character is wrapped in a
    one-element class, which fnmatch takes literally.
    """
    return "".join(f"[{c}]" if c in "*?[" else c for c in filename)


def _is_fatal(exc: BaseException) -> bool:
    """Would another attempt fail the same way?

    The cause chain is walked, not just the exception itself: a failed listing
    arrives as a `DryRunError` wrapping the real reason, and that reason decides.
    A repo that does not exist is hopeless, a Hub that timed out is not.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_fatal_cause(current):
            return True
        current = current.__cause__
    return False


def _is_fatal_cause(exc: BaseException) -> bool:
    """Everything not named here — dropped connections, timeouts, a Hub that
    returns 5xx, a Xet transfer that gives up — is worth another attempt."""
    from huggingface_hub.errors import (
        BadRequestError,
        DisabledRepoError,
        HFValidationError,
        HfHubHTTPError,
        RemoteEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    # Missing, gated, renamed, or simply not a valid request: nothing changes by
    # asking again. GatedRepoError subclasses RepositoryNotFoundError.
    if isinstance(
        exc,
        (
            RepositoryNotFoundError,
            DisabledRepoError,
            RevisionNotFoundError,
            RemoteEntryNotFoundError,
            HFValidationError,
            BadRequestError,
        ),
    ):
        return True
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", 0)
        return status in (400, 401, 403, 404)
    # A full disk or a folder we may not write to. HfHubHTTPError is an OSError
    # too, so this check has to come after it.
    if isinstance(exc, OSError) and exc.errno in (
        errno.ENOSPC,
        errno.EDQUOT,
        errno.EACCES,
        errno.EPERM,
        errno.EROFS,
    ):
        return True
    return False


def _describe(exc: BaseException) -> str:
    """The message worth showing.

    huggingface_hub wraps a failed listing in a `DryRunError` that says to check
    the connection or the token, whatever actually went wrong. The cause names
    the real problem, so it goes first.
    """
    cause = exc.__cause__
    if cause is not None and type(cause) is not type(exc):
        return f"{type(cause).__name__}: {cause}"
    return f"{type(exc).__name__}: {exc}"


def _backoff(attempt: int) -> int:
    return RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]


def _drop_leftovers(dest: Path) -> None:
    """Clear half-written files from a transfer that was killed.

    They cannot be resumed — huggingface_hub names each temporary file after the
    attempt that created it — and nothing else ever removes them, so a repo that
    was interrupted a few times can sit on hundreds of gigabytes of dead weight.
    This is the one moment where deleting them is safe: no transfer is writing
    into this folder yet, and the API refuses a second job for the same repo.
    """
    freed, count = storage.drop_leftover_parts(dest)
    if count:
        # A transfer killed early leaves files that are still empty; saying
        # "0 B reclaimed" for those reads like a bug.
        reclaimed = f" — {_fmt(freed)} reclaimed" if freed else ""
        log(f"Cleared {count} unusable part file(s) from an earlier attempt{reclaimed}.")


def run_download(payload: dict[str, Any]) -> None:
    from huggingface_hub import snapshot_download

    repo_id: str = payload["repo_id"]
    repo_type: str = payload.get("repo_type", "model")
    revision: str | None = payload.get("revision") or None
    dest = Path(payload["dest"])
    picked = payload.get("files") or []
    allow = list(payload.get("allow_patterns") or [])
    # Picking files is just a friendlier way of writing allow_patterns; giving
    # both yields the union.
    allow += [_as_pattern(f) for f in picked]
    allow = allow or None
    ignore = payload.get("ignore_patterns") or None
    token = os.getenv("HF_TOKEN") or None
    endpoint = os.getenv("HF_ENDPOINT") or None

    common = dict(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(dest),
        token=token,
        endpoint=endpoint,
        allow_patterns=allow,
        ignore_patterns=ignore,
    )

    dest.mkdir(parents=True, exist_ok=True)
    _drop_leftovers(dest)

    commit = ""
    attempt = 0
    progress.start()
    try:
        while True:
            attempt += 1
            try:
                log(f"Listing files in {repo_id} …")

                plan = snapshot_download(**common, dry_run=True, tqdm_class=_reporter_tqdm_class())
                total = sum(int(f.file_size or 0) for f in plan)
                pending = [f for f in plan if f.will_download]
                already = total - sum(int(f.file_size or 0) for f in pending)
                commit = plan[0].commit_hash if plan else ""

                # Re-read on every attempt: the failed one may well have
                # completed a few files before it died, and those now count.
                with progress.lock:
                    progress.total = total
                    progress.total_files = len(plan)
                    progress.done_files = len(plan) - len(pending)
                    progress.base = already
                    progress.written = 0
                    progress.transfer = 0

                emit(
                    "meta",
                    total_bytes=total,
                    done_bytes=already,
                    total_files=len(plan),
                    done_files=len(plan) - len(pending),
                    commit=commit,
                )

                if not pending:
                    log("Every file is already here — nothing to do.")
                else:
                    log(f"{len(pending)} file(s), {_fmt(total - already)} to fetch.")

                path = snapshot_download(
                    **common,
                    max_workers=int(payload.get("max_workers") or 4),
                    tqdm_class=_reporter_tqdm_class(),
                )
                break
            except Exception as exc:  # noqa: BLE001 - decided by _is_fatal
                if _is_fatal(exc) or attempt >= MAX_ATTEMPTS:
                    raise
                delay = _backoff(attempt)
                log(_describe(exc), "error")
                log(
                    f"Attempt {attempt} of {MAX_ATTEMPTS} stopped. Trying again in "
                    f"{delay}s — every file that finished stays on disk.",
                    "warn",
                )
                # The file that errored deletes its own temporary data, but a
                # Xet transfer that died mid-write may not have; either way
                # nothing here can be resumed.
                _drop_leftovers(dest)
                time.sleep(delay)
    finally:
        progress.stop()

    with progress.lock:
        progress.done_files = progress.total_files
        # Once it succeeded, the bar is at 100% by definition.
        progress.base = max(progress.total, progress.base + progress.written)
        progress.written = 0
    progress.flush()

    _write_marker(
        dest,
        {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "revision": revision or "main",
            "commit": commit,
            "downloaded_at": time.time(),
            "size": total,
            "files": picked,
            "allow_patterns": list(payload.get("allow_patterns") or []),
            "ignore_patterns": ignore or [],
        },
    )

    emit("done", path=str(path))


# ----------------------------------------------------------------------- Upload


def _iter_upload_files(src: Path, allow: list[str] | None, ignore: list[str] | None) -> Iterable[Path]:
    for root, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in (".git",)]
        for name in filenames:
            full = Path(root) / name
            rel = full.relative_to(src).as_posix()
            if allow and not any(fnmatch.fnmatch(rel, p) for p in allow):
                continue
            if ignore and any(fnmatch.fnmatch(rel, p) for p in ignore):
                continue
            yield full


def _patch_upload_display() -> bool:
    """Hook into the live display of upload_folder.

    huggingface_hub renders upload progress through an internal _LiveDisplay
    class; there is no public callback API. We swap in a variant that reports
    the same counters upwards instead of drawing them on stderr. Should a
    huggingface_hub update break the patch, the upload still runs — it just
    loses the byte readout.
    """
    try:
        from huggingface_hub import _upload_pipeline as pipeline

        base = pipeline._LiveDisplay
    except Exception:  # noqa: BLE001
        return False

    class ReportingDisplay(base):  # type: ignore[misc, valid-type]
        def __init__(self, total_files: int, enabled: bool = True) -> None:
            # `enabled` is part of the signature huggingface_hub calls with; we
            # ignore its value because the base renderer must stay switched off.
            super().__init__(total_files=total_files, enabled=False)
            # _active decides whether Xet callbacks are created at all.
            self._active = True
            self._tty = False
            with progress.lock:
                progress.total_files = total_files

        def _render_loop(self) -> None:
            while not self._stop_event.wait(EMIT_INTERVAL):
                self._report()

        def _report(self) -> None:
            with self._lock:
                committed = self._committed
                uploaded = self._xet_bytes
                total_files = self._total
            with progress.lock:
                progress.done_files = committed
                progress.total_files = total_files
                progress.transfer = uploaded
                progress.written = uploaded
                progress.base = 0
            progress.flush()

        def close(self) -> None:
            super().close()
            self._report()

    pipeline._LiveDisplay = ReportingDisplay
    return True


def run_upload(payload: dict[str, Any]) -> None:
    hooked = _patch_upload_display()

    from huggingface_hub import HfApi

    repo_id: str = payload["repo_id"]
    repo_type: str = payload.get("repo_type", "model")
    src = Path(payload["src"])
    allow = payload.get("allow_patterns") or None
    ignore = list(payload.get("ignore_patterns") or [])
    revision = payload.get("revision") or None
    token = os.getenv("HF_TOKEN") or None
    endpoint = os.getenv("HF_ENDPOINT") or None

    if not src.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {src}")
    if not token:
        raise RuntimeError("Uploading needs a token with write access (see Settings).")

    # Never upload our own bookkeeping.
    for pattern in (".cache/*", MARKER_NAME, *LEGACY_MARKER_NAMES, ".git/*", "**/.DS_Store"):
        if pattern not in ignore:
            ignore.append(pattern)

    files = list(_iter_upload_files(src, allow, ignore))
    total = sum(f.stat().st_size for f in files if f.exists())
    with progress.lock:
        progress.total = total
        progress.total_files = len(files)
    emit("meta", total_bytes=total, done_bytes=0, total_files=len(files), done_files=0, dest=repo_id)

    if not files:
        raise RuntimeError(f"No files left in {src} after filtering — nothing to upload.")

    api = HfApi(token=token, endpoint=endpoint, library_name="trove")

    if payload.get("create_repo", True):
        url = api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=bool(payload.get("private", False)),
            exist_ok=True,
        )
        log(f"Repo ready: {url}")

    log(f"Uploading {len(files)} file(s), {_fmt(total)}, to {repo_id} …")
    if not hooked:
        log("Byte-level upload progress unavailable — the huggingface_hub API changed.", "warn")

    progress.start()
    try:
        info = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(src),
            repo_type=repo_type,
            revision=revision,
            commit_message=payload.get("commit_message") or "Upload via Trove",
            allow_patterns=allow,
            ignore_patterns=ignore or None,
        )
    finally:
        progress.stop()

    with progress.lock:
        progress.done_files = progress.total_files
        progress.base = progress.total
        progress.written = 0
        progress.transfer = 0
    progress.flush()

    log(f"Committed as {getattr(info, 'oid', '') or 'unknown'}.")
    emit("done", path=str(src))


# ------------------------------------------------------------------------ Utils


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    try:
        (path / MARKER_NAME).write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        log(f"Could not write the marker file: {exc}", "warn")


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def _on_sigterm(_signum: int, _frame: Any) -> None:
    raise SystemExit(143)


def main(argv: list[str]) -> int:
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    try:
        payload = json.loads(argv[1])
    except (IndexError, ValueError) as exc:
        emit("error", msg=f"Invalid payload: {exc}")
        return 2

    try:
        if payload.get("kind") == "upload":
            run_upload(payload)
        else:
            run_download(payload)
    except SystemExit:
        progress.stop()
        log("Cancelled.", "warn")
        return 143
    except KeyboardInterrupt:
        progress.stop()
        return 143
    except Exception as exc:  # noqa: BLE001 - everything goes to the interface
        progress.stop()
        emit("error", msg=_describe(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

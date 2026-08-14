"""The local library: listing repos, measuring their size, deleting them."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

from .config import (
    DATA_DIR,
    LEGACY_MARKER_NAMES,
    MARKER_NAME,
    REPO_TYPES,
    local_dir_for,
    type_root,
)

# Size cache: walking a 500 GB directory takes a while, and the interface asks
# for the listing on every view switch.
_SIZE_CACHE: dict[str, tuple[float, int, int, int]] = {}
_CACHE_TTL = 60.0

# Internal folders that do not count as repo content.
_INTERNAL = {".cache", ".git", ".locks"}

#: Where huggingface_hub keeps its per-file bookkeeping inside a local dir.
_DOWNLOAD_CACHE = (".cache", "huggingface", "download")


def _download_cache(path: Path) -> Path:
    return path.joinpath(*_DOWNLOAD_CACHE)


def _dir_stats(path: Path) -> tuple[int, int, int]:
    """(bytes, file_count, leftover_bytes), served from a short-lived cache."""
    key = str(path)
    now = time.time()
    cached = _SIZE_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1], cached[2], cached[3]

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

    leftover = leftover_size(path)
    _SIZE_CACHE[key] = (now, total, files, leftover)
    return total, files, leftover


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
            size, files, leftover = _dir_stats(path)
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
                    # Space held by half-written files from a killed transfer;
                    # the next download of this repo clears it.
                    "leftover": leftover,
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


def _iter_content_files(path: Path) -> Iterator[Path]:
    """Every file that belongs to the repo itself — no bookkeeping, no marker."""
    for root, dirnames, filenames in os.walk(path, onerror=lambda _e: None):
        dirnames[:] = [d for d in dirnames if d not in _INTERNAL]
        for name in filenames:
            if name == MARKER_NAME or name in LEGACY_MARKER_NAMES:
                continue
            yield Path(root) / name


def repo_files(repo_type: str, repo_id: str, limit: int = 2000) -> list[dict[str, Any]]:
    path = local_dir_for(repo_type, repo_id)
    if not path.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for full in _iter_content_files(path):
        try:
            stat = full.stat()
        except OSError:
            continue
        out.append({"name": full.relative_to(path).as_posix(), "size": stat.st_size})
        if len(out) >= limit:
            break
    out.sort(key=lambda f: f["name"])
    return out


def local_etags(repo_type: str, repo_id: str) -> dict[str, str]:
    """The content hash huggingface_hub recorded for each downloaded file.

    It keeps a `<file>.metadata` next to its download cache holding three lines:
    the commit, the etag, and a timestamp. The etag is the git blob id for plain
    files and the LFS sha256 for large ones — the same values the Hub reports,
    so comparing them says exactly which files moved.
    """
    root = _download_cache(local_dir_for(repo_type, repo_id))
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in root.rglob("*.metadata"):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        if len(lines) < 2 or not lines[1].strip():
            continue
        name = path.relative_to(root).as_posix().removesuffix(".metadata")
        out[name] = lines[1].strip()
    return out


def _iter_leftover_parts(path: Path) -> Iterator[Path]:
    """Half-written files a killed transfer left behind.

    huggingface_hub downloads every file to `<name>.<etag>.<uuid>.incomplete`
    and picks a fresh uuid on the next attempt, so a leftover can never be
    resumed. It normally deletes its own file, but a process that dies without
    running its cleanup — SIGKILL, an out-of-memory kill, a container that goes
    down — leaves it behind for good. Nothing ever collects it, and `.cache` is
    excluded from the reported repo size, so the space disappears silently.
    """
    root = _download_cache(path)
    if root.is_dir():
        yield from root.rglob("*.incomplete")


def leftover_size(path: Path) -> int:
    """How much space those leftovers hold."""
    total = 0
    for part in _iter_leftover_parts(path):
        try:
            total += part.stat().st_size
        except OSError:
            continue
    return total


def drop_leftover_parts(path: Path) -> tuple[int, int]:
    """Delete them and report what was reclaimed — (bytes, count).

    Only safe while no transfer is writing into `path`: a running download owns
    an `.incomplete` file of its own.
    """
    total = 0
    count = 0
    for part in _iter_leftover_parts(path):
        try:
            size = part.stat().st_size
            part.unlink()
        except OSError:
            continue
        total += size
        count += 1
    if count:
        invalidate(path)
    return total, count


def _safe_member(path: Path, name: str) -> Path:
    """Resolve a repo-relative file name, refusing anything that leaves the repo.

    The names come from the interface, so they are treated as hostile: a `..`
    segment, an absolute path or a symlink pointing outside would otherwise
    delete files anywhere on the host.
    """
    clean = (name or "").strip().replace("\\", "/")
    if not clean or clean.startswith("/"):
        raise ValueError(f"Invalid file name: {name!r}")

    # "." and "./" normalise away to no parts at all, so the checks below have
    # nothing to look at — and an empty name is not a file either way.
    parts = PurePosixPath(clean).parts
    if not parts or ".." in parts:
        raise ValueError(f"Invalid file name: {name!r}")
    # Case-folded: on a case-insensitive volume ".Cache/…" and ".TROVE.json"
    # reach the very files these two lines exist to protect.
    if parts[0].casefold() in _INTERNAL:
        raise ValueError(f"{parts[0]} holds bookkeeping, not repo content")
    if parts[-1].casefold() in (MARKER_NAME, *LEGACY_MARKER_NAMES):
        raise ValueError("The download record cannot be deleted on its own")

    target = path / clean
    try:
        resolved = target.resolve()
        root = path.resolve()
    except OSError as exc:
        raise ValueError(f"Invalid file name: {name!r}") from exc
    if root not in resolved.parents:
        raise ValueError(f"{name!r} points outside the repo")
    return target


def _iter_bookkeeping(cache_path: Path) -> Iterator[Path]:
    """The `.metadata` record and the `.incomplete` parts belonging to a file.

    The part match is by prefix, so it also catches parts of files whose name
    merely extends this one — deleting `model.safetensors` clears a leftover of
    `model.safetensors.index.json` too. That costs nothing: a part file can
    never be resumed, so every one of them is dead weight whoever wrote it.
    """
    meta = cache_path.with_name(f"{cache_path.name}.metadata")
    if meta.is_file():
        yield meta
    prefix = f"{cache_path.name}."
    try:
        with os.scandir(cache_path.parent) as entries:
            for entry in entries:
                if entry.name.startswith(prefix) and entry.name.endswith(".incomplete"):
                    yield Path(entry.path)
    except OSError:
        return


def _drop_bookkeeping(path: Path, name: str) -> int:
    """Forget a file was ever downloaded, and report the bytes that frees.

    The `.metadata` next to the download cache is what the update check reads:
    left behind, a deleted file still counts as tracked and the next update
    fetches it again.
    """
    cache_root = _download_cache(path)
    cache_path = cache_root / name
    freed = 0
    for extra in list(_iter_bookkeeping(cache_path)):
        try:
            size = extra.stat().st_size
            extra.unlink()
        except OSError:
            continue
        freed += size
    _prune_empty_dirs(cache_path.parent, cache_root)
    return freed


def _prune_empty_dirs(start: Path, stop: Path) -> None:
    """Remove folders a deletion left behind, up to but excluding `stop`."""
    current = start
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _narrow_marker(path: Path, remaining: list[str]) -> bool:
    """Record what is left as the selection for this copy; False if there is none.

    An update re-fetches exactly what the marker names, so a file deleted here
    has to leave the selection as well — otherwise the next update pulls it
    straight back. A pattern used at download time goes: it would widen the
    selection to everything it once matched, deleted files included.

    A folder without a marker keeps not having one. Writing the first one here
    would dress an unknown folder up as a tracked download.
    """
    marker = _read_marker(path)
    if not marker:
        return False
    marker["files"] = remaining
    marker["allow_patterns"] = []
    (path / MARKER_NAME).write_text(json.dumps(marker, indent=2))
    return True


def delete_files(repo_type: str, repo_id: str, names: Iterable[str]) -> dict[str, Any]:
    """Delete picked files from a stored repo and narrow it to what is left."""
    path = local_dir_for(repo_type, repo_id)
    if not path.is_dir():
        raise FileNotFoundError(f"{repo_id} is not stored locally")

    # Validate every name before deleting anything: a bad one in the middle
    # would otherwise leave the repo half edited.
    targets = [(name, _safe_member(path, name)) for name in names]

    freed = 0
    deleted: list[str] = []
    missing: list[str] = []
    failed: list[str] = []
    for name, target in targets:
        if not target.is_file():
            # Someone else got there first, or the interface is showing a
            # listing that has since moved on.
            missing.append(name)
            continue
        try:
            size = target.stat().st_size
            target.unlink()
        except OSError:
            # A file we may not touch — an ACL, a read-only mount. The ones
            # already unlinked stay gone, so the run has to carry on and narrow
            # the selection anyway: stopping here would leave the marker naming
            # files that no longer exist, and the next update would fetch them.
            failed.append(name)
            continue
        deleted.append(name)
        freed += size + _drop_bookkeeping(path, target.relative_to(path).as_posix())
        _prune_empty_dirs(target.parent, path)

    invalidate(path)
    remaining = sorted(p.relative_to(path).as_posix() for p in _iter_content_files(path))
    warnings: list[str] = []
    if failed:
        warnings.append(
            f"{len(failed)} file(s) could not be removed — check the permissions on the folder."
        )

    result: dict[str, Any] = {
        "deleted": deleted,
        "missing": missing,
        "failed": failed,
        "freed": freed,
        "remaining": len(remaining),
        "removed_repo": False,
        "warning": "",
    }

    if not remaining:
        # Nothing worth keeping the folder for — and an empty selection reads as
        # "the whole repo" on the next update, which is the opposite of what
        # emptying it out asked for.
        result["freed"] += delete_repo(repo_type, repo_id)["freed"]
        result["removed_repo"] = True
        result["warning"] = " ".join(warnings)
        return result

    try:
        if not _narrow_marker(path, remaining):
            # No record to narrow, so the promise the interface makes for this
            # button does not hold here. Say it rather than let an update
            # quietly restore what was just deleted.
            warnings.append(
                "This copy has no download record, so an update fetches the whole repo again "
                "— including what you just deleted."
            )
    except OSError as exc:
        # The files are gone either way; say so rather than failing the call,
        # but do not let the stale selection pass unmentioned.
        warnings.append(
            f"Deleted, but the download record could not be updated ({exc}). An update may fetch them again."
        )

    result["warning"] = " ".join(warnings)
    return result


def delete_repo(repo_type: str, repo_id: str) -> dict[str, Any]:
    path = local_dir_for(repo_type, repo_id)
    if not path.is_dir():
        raise FileNotFoundError(f"{repo_id} is not stored locally")

    size, files, leftover = _dir_stats(path)
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

    return {"deleted": str(path), "freed": size + leftover, "files": files}


def disk_usage() -> dict[str, int]:
    try:
        usage = shutil.disk_usage(DATA_DIR)
        return {"total": usage.total, "free": usage.free}
    except OSError:
        return {"total": 0, "free": 0}

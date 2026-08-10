"""Shared fixtures.

The app reads DATA_DIR and CONFIG_DIR into module constants at import time, so
they have to point somewhere disposable *before* anything under `app` is
imported — hence the environment setup at the top of this file, which pytest
loads first.

Nothing in here talks to the Hub. Everything that would is faked: the worker
process is replaced by a stub that emits whatever events a test asks for, and
`hub` is monkeypatched per test.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(tempfile.mkdtemp(prefix="trove-tests-"))
os.environ["DATA_DIR"] = str(ROOT / "data")
os.environ["CONFIG_DIR"] = str(ROOT / "config")
os.environ["UI_PASSWORD"] = "test-password"
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HF_ENDPOINT", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, jobs, storage  # noqa: E402

PASSWORD = "test-password"


# --------------------------------------------------------------- Clean slate


@pytest.fixture(autouse=True)
def fresh_state():
    """Every test starts with empty folders, default settings, an empty queue."""
    shutil.rmtree(ROOT, ignore_errors=True)
    config.ensure_dirs()
    config.settings.update(dict(config.DEFAULT_SETTINGS))
    storage.invalidate()

    manager = jobs.manager
    manager.jobs.clear()
    manager.order.clear()
    manager._procs.clear()
    manager._tasks.clear()
    manager._cancelling.clear()
    manager._progress_at.clear()
    manager._stopping = False
    manager._on_event = None
    yield
    shutil.rmtree(ROOT, ignore_errors=True)


# ------------------------------------------------------------- Repo fixtures


def make_repo(
    repo_id: str,
    repo_type: str = "model",
    files: dict[str, int] | None = None,
    marker: dict[str, Any] | None = None,
    etags: dict[str, str] | None = None,
    leftovers: dict[str, int] | None = None,
) -> Path:
    """Build a repo folder on disk the way a real download leaves it.

    `files` maps name to byte size, `etags` writes the per-file `.metadata`
    huggingface_hub keeps, `leftovers` writes `.incomplete` part files.
    """
    path = config.local_dir_for(repo_type, repo_id)
    path.mkdir(parents=True, exist_ok=True)

    for name, size in (files or {"config.json": 10}).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * size)

    if marker is not None:
        (path / config.MARKER_NAME).write_text(json.dumps(marker))

    cache = path / ".cache" / "huggingface" / "download"
    for name, etag in (etags or {}).items():
        meta = cache / f"{name}.metadata"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(f"deadbeef\n{etag}\n1700000000.0\n")
    for name, size in (leftovers or {}).items():
        part = cache / f"{name}.incomplete"
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"y" * size)

    storage.invalidate(path)
    return path


@pytest.fixture
def repo_factory():
    return make_repo


# ------------------------------------------------------------- Worker stubs


#: Stands in for `app.worker`. Reads a spec and behaves as told, so a test can
#: produce a clean run, a reported error, a hang or a hard crash on demand.
STUB_WORKER = r"""
import json, os, signal, sys, time
spec = json.loads(sys.argv[1])
for event in spec.get("emit", []):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()
if spec.get("stderr"):
    sys.stderr.write(spec["stderr"] + "\n")
    sys.stderr.flush()
if spec.get("suicide"):
    os.kill(os.getpid(), spec["suicide"])
if spec.get("hang"):
    # Ignore SIGTERM when asked, so a test can exercise the kill escalation.
    if spec.get("stubborn"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(spec["hang"])
sys.exit(spec.get("code", 0))
"""


def use_stub_worker(monkeypatch, spec: dict[str, Any]) -> None:
    """Replace the worker command with the stub, for every job."""
    command = [sys.executable, "-c", STUB_WORKER, json.dumps(spec)]
    monkeypatch.setattr(jobs.manager, "_build_command", lambda job: list(command))


@pytest.fixture
def stub_worker(monkeypatch):
    def apply(spec: dict[str, Any]) -> None:
        use_stub_worker(monkeypatch, spec)

    return apply


# -------------------------------------------------------------------- Async


async def _drain_queue() -> None:
    """Stop every transfer and wait for its task, the way a shutdown does.

    Tests routinely leave a stub worker running. Closing an event loop while a
    subprocess is still being spawned on it hangs, so the queue is wound down
    first rather than cancelled from underneath.
    """
    manager = jobs.manager
    for job_id in list(manager.jobs):
        job = manager.jobs.get(job_id)
        if job is not None and job.status in jobs.ACTIVE:
            try:
                await manager.cancel(job_id)
            except KeyError:
                pass
    pending = list(manager._tasks.values())
    if pending:
        await asyncio.wait(pending, timeout=15)
    # Collect the subprocess transports while their loop is still open. Left to
    # the garbage collector they are finalised after the loop closes, and every
    # one of them raises "Event loop is closed" from its destructor.
    gc.collect()


def run(coro):
    """Run one coroutine to completion on a fresh loop, then drain the queue."""

    async def main():
        try:
            return await coro
        finally:
            await _drain_queue()

    return asyncio.run(main())


async def wait_for(check, timeout: float = 10.0, interval: float = 0.02):
    """Poll until `check()` is truthy, or fail the test."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = check()
        if value:
            return value
        await asyncio.sleep(interval)
    raise AssertionError("condition was never met")

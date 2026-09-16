"""The transfer queue: download/upload jobs and their worker processes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import throttle
from .config import DEFAULT_SETTINGS, JOBS_FILE, local_dir_for, settings
from . import storage

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"

ACTIVE = (QUEUED, RUNNING)
LOG_LIMIT = 200

# A running transfer that has not moved a single byte for this long is reported
# as stalled, so the queue stops looking busy while nothing happens.
STALL_AFTER = 300.0
STALL_INTERVAL = 30.0

# How often a download whose worker died without a word may be started again.
# The worker handles network trouble by itself, so reaching this point means the
# process was killed — out of memory is by far the most common reason. Every
# file that finished is on disk, so a fresh attempt only fetches the remainder.
MAX_CRASH_RESTARTS = 3


def _exit_reason(code: int) -> str:
    """Explain a worker that ended without reporting an error of its own.

    A process killed by a signal is reported as the negated signal number; 143
    is the worker's own exit code when it caught a SIGTERM and shut down.
    """
    if code == -9:
        return (
            "The worker was killed (SIGKILL) — on a NAS this is almost always the "
            "kernel running out of memory. Lower 'Parallel transfers' or "
            "'Files at once' in Settings."
        )
    if code in (-15, 143):
        return "The worker was stopped from outside (SIGTERM)."
    if code < 0:
        return f"The worker was killed by signal {-code}."
    return f"Worker exited with code {code}"


@dataclass
class Job:
    id: str
    kind: str  # download | upload
    repo_id: str
    repo_type: str = "model"
    revision: str = ""
    status: str = QUEUED
    dest: str = ""
    src: str = ""
    total_bytes: int = 0
    done_bytes: int = 0
    total_files: int = 0
    done_files: int = 0
    speed: float = 0.0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    files: list[str] = field(default_factory=list)
    allow_patterns: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=list)
    private: bool = False
    create_repo: bool = True
    commit_message: str = ""
    stalled: bool = False
    #: How often this job was restarted after its worker died unexpectedly.
    restarts: int = 0
    logs: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=LOG_LIMIT))

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if k != "logs"}
        data["logs"] = list(self.logs)[-60:]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        logs = data.pop("logs", [])
        known = {k: v for k, v in data.items() if k in cls.__annotations__}
        job = cls(**known)
        job.logs.extend(logs[-30:])
        return job


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._cancelling: set[str] = set()
        self._tasks: dict[str, asyncio.Task] = {}
        self._on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._lock = asyncio.Lock()
        self._progress_at: dict[str, float] = {}
        self._watchdog: asyncio.Task | None = None
        self._stopping = False

    # ------------------------------------------------------------- Lifecycle

    def bind(self, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._on_event = callback

    async def start(self) -> None:
        self._load()
        self._watchdog = asyncio.create_task(self._watch_for_stalls())
        await self._pump()

    async def shutdown(self) -> None:
        # Transfers stopped by a shutdown stay marked as running, so the next
        # start picks them up again instead of writing them off as cancelled.
        self._stopping = True
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        for job_id in list(self._procs):
            await self._terminate(job_id)
        # Let every transfer's task run out before the queue is written, or the
        # file can miss the last state change and the pipes outlive the loop.
        pending = list(self._tasks.values())
        if pending:
            await asyncio.wait(pending, timeout=15)
        self._save()

    # ----------------------------------------------------------- Persistence

    def _load(self) -> None:
        try:
            raw = json.loads(JOBS_FILE.read_text())
        except (OSError, ValueError):
            return
        for entry in raw.get("jobs", []):
            try:
                job = Job.from_dict(dict(entry))
            except (TypeError, ValueError):
                continue
            # Re-queue transfers a restart interrupted; hf_hub resumes files
            # it had already started.
            if job.status == RUNNING:
                job.status = QUEUED
                job.speed = 0.0
            job.stalled = False
            # A fresh start gets a fresh restart budget.
            job.restarts = 0
            self.jobs[job.id] = job
            self.order.append(job.id)

    def _save(self) -> None:
        try:
            JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {"jobs": [self.jobs[i].to_dict() for i in self.order if i in self.jobs]}
            tmp = JOBS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=1, default=str))
            tmp.replace(JOBS_FILE)
        except OSError:
            pass

    # --------------------------------------------------------------- Events

    async def _emit(self, event: str, **payload: Any) -> None:
        if self._on_event:
            await self._on_event({"type": event, **payload})

    async def _push(self, job: Job) -> None:
        await self._emit("job", job=job.to_dict())

    def snapshot(self) -> list[dict[str, Any]]:
        return [self.jobs[i].to_dict() for i in self.order if i in self.jobs]

    def stats(self) -> dict[str, int]:
        counts = {QUEUED: 0, RUNNING: 0, DONE: 0, ERROR: 0, CANCELLED: 0}
        for job in self.jobs.values():
            counts[job.status] = counts.get(job.status, 0) + 1
        return counts

    # ---------------------------------------------------------- Public API

    async def add_download(
        self,
        repo_id: str,
        repo_type: str = "model",
        revision: str = "",
        files: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> Job:
        dest = local_dir_for(repo_type, repo_id)
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind="download",
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision or "",
            dest=str(dest),
            files=files or [],
            allow_patterns=allow_patterns or [],
            ignore_patterns=ignore_patterns or [],
        )
        return await self._enqueue(job)

    async def add_upload(
        self,
        repo_id: str,
        src: str,
        repo_type: str = "model",
        private: bool = False,
        create_repo: bool = True,
        commit_message: str = "",
        revision: str = "",
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind="upload",
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision or "",
            src=src,
            private=private,
            create_repo=create_repo,
            commit_message=commit_message,
            allow_patterns=allow_patterns or [],
            ignore_patterns=ignore_patterns or [],
        )
        return await self._enqueue(job)

    async def _enqueue(self, job: Job) -> Job:
        self.jobs[job.id] = job
        self.order.append(job.id)
        self._log(job, "Added to the queue.")
        self._save()
        await self._push(job)
        await self._pump()
        return job

    async def cancel(self, job_id: str) -> Job:
        job = self._require(job_id)
        if job.status == RUNNING:
            self._cancelling.add(job_id)
            await self._terminate(job_id)
        elif job.status == QUEUED:
            job.status = CANCELLED
            job.finished_at = time.time()
            self._log(job, "Removed from the queue.", "warn")
            self._save()
            await self._push(job)
        return job

    async def retry(self, job_id: str) -> Job:
        job = self._require(job_id)
        if job.status in ACTIVE:
            return job
        job.status = QUEUED
        job.error = ""
        job.speed = 0.0
        job.done_bytes = 0
        job.done_files = 0
        job.started_at = 0.0
        job.finished_at = 0.0
        job.restarts = 0
        self._log(job, "Trying again.")
        self._save()
        await self._push(job)
        await self._pump()
        return job

    async def remove(self, job_id: str) -> None:
        job = self._require(job_id)
        if job.status in ACTIVE:
            await self.cancel(job_id)
        self.jobs.pop(job_id, None)
        if job_id in self.order:
            self.order.remove(job_id)
        self._save()
        await self._emit("removed", id=job_id)

    async def clear_finished(self) -> int:
        removed = [i for i in list(self.order) if self.jobs[i].status not in ACTIVE]
        for job_id in removed:
            self.jobs.pop(job_id, None)
            self.order.remove(job_id)
        self._save()
        await self._emit("jobs", jobs=self.snapshot())
        return len(removed)

    async def reschedule(self) -> None:
        """Start whatever the current concurrency limit now allows.

        Called after the limit changes in Settings, so a raised value takes
        effect on jobs that are already waiting.
        """
        await self._pump()

    def _require(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    # ------------------------------------------------------------ Scheduler

    async def _pump(self) -> None:
        async with self._lock:
            limit = int(settings.get("max_concurrent") or 1)
            running = sum(1 for j in self.jobs.values() if j.status == RUNNING)
            for job_id in self.order:
                if running >= limit:
                    break
                job = self.jobs.get(job_id)
                if job is None or job.status != QUEUED:
                    continue
                job.status = RUNNING
                job.started_at = time.time()
                running += 1
                self._tasks[job_id] = asyncio.create_task(self._run(job))
                await self._push(job)

    def _build_command(self, job: Job) -> list[str]:
        if job.kind == "upload":
            payload = {
                "kind": "upload",
                "repo_id": job.repo_id,
                "repo_type": job.repo_type,
                "src": job.src,
                "revision": job.revision,
                "private": job.private,
                "create_repo": job.create_repo,
                "commit_message": job.commit_message,
                "allow_patterns": job.allow_patterns,
                "ignore_patterns": job.ignore_patterns,
            }
        else:
            payload = {
                "kind": "download",
                "repo_id": job.repo_id,
                "repo_type": job.repo_type,
                "revision": job.revision,
                "dest": job.dest,
                "files": job.files,
                "allow_patterns": job.allow_patterns,
                "ignore_patterns": job.ignore_patterns,
                "max_workers": int(settings.get("max_workers") or DEFAULT_SETTINGS["max_workers"]),
            }
        return [sys.executable, "-u", "-m", "app.worker", json.dumps(payload)]

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        # The token goes through the environment on purpose: argv shows up in `ps`.
        token = settings.get("hf_token") or ""
        if token:
            env["HF_TOKEN"] = token
        else:
            env.pop("HF_TOKEN", None)
        endpoint = settings.get("endpoint") or ""
        if endpoint:
            env["HF_ENDPOINT"] = endpoint
        else:
            env.pop("HF_ENDPOINT", None)
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
        # A speed limit is a proxy on localhost that every worker shares, read
        # at the moment the worker starts. Nothing is removed when no limit is
        # set: a proxy the operator configured for the container stays the
        # worker's proxy.
        env.update(throttle.shared.env())
        return env

    async def _run(self, job: Job) -> None:
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._build_command(job),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            self._procs[job.id] = proc
            # A job counts as running from the moment it is scheduled, but the
            # process only exists a few milliseconds later. A cancel — or a
            # shutdown — that landed in that window found nothing to signal and
            # quietly did nothing, leaving the transfer running. Catch it up.
            if job.id in self._cancelling or self._stopping:
                await self._terminate(job.id)
            stderr_task = asyncio.create_task(self._drain_stderr(job, proc))

            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    self._log(job, line)
                    continue
                await self._handle_event(job, event)

            code = await proc.wait()
            await stderr_task
        except FileNotFoundError as exc:
            job.error = f"Could not start the worker process: {exc}"
            code = 1
        except Exception as exc:  # noqa: BLE001
            job.error = f"{type(exc).__name__}: {exc}"
            code = 1
        finally:
            self._procs.pop(job.id, None)
            self._tasks.pop(job.id, None)

        cancelled = job.id in self._cancelling
        self._cancelling.discard(job.id)
        self._progress_at.pop(job.id, None)
        job.speed = 0.0
        job.stalled = False

        # The container is going down. Leaving the job as it is means the next
        # start re-queues it and carries on where the files on disk end.
        if self._stopping and job.status == RUNNING:
            self._save()
            return

        job.finished_at = time.time()

        if cancelled:
            job.status = CANCELLED
            self._log(job, "Transfer cancelled.", "warn")
        elif code == 0 and job.status == RUNNING:
            job.status = DONE
            job.restarts = 0
            if job.total_bytes:
                job.done_bytes = job.total_bytes
            if job.total_files:
                job.done_files = job.total_files
            self._log(job, "Finished.", "ok")
            storage.invalidate()
        elif job.status == RUNNING:
            # An empty job.error means the process died without saying anything:
            # it was killed. The worker reports everything it can handle itself,
            # so this is worth another go — with a limit, in case it is the kind
            # of crash that repeats.
            killed = not job.error
            if killed and job.kind == "download" and job.restarts < MAX_CRASH_RESTARTS:
                job.restarts += 1
                job.status = QUEUED
                job.finished_at = 0.0
                self._log(
                    job,
                    f"{_exit_reason(code)} Starting over ({job.restarts}/{MAX_CRASH_RESTARTS}) — "
                    "finished files are kept.",
                    "warn",
                )
            else:
                job.status = ERROR
                if not job.error:
                    job.error = _exit_reason(code)
                self._log(job, job.error, "error")

        # The job may have been removed while its process was still shutting
        # down; broadcasting it now would resurrect it in every open tab.
        if job.id not in self.jobs:
            return
        self._save()
        await self._push(job)

        if settings.get("auto_clear_done") and job.status == DONE:
            await asyncio.sleep(3)
            await self.remove(job.id)

        await self._pump()

    async def _drain_stderr(self, job: Job, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        async for raw in proc.stderr:
            text = raw.decode("utf-8", "replace").strip()
            if not text or text.startswith("\r"):
                continue
            # Keep tqdm and hf chatter out of the job log.
            if "it/s" in text or "B/s" in text or text.startswith("Found "):
                continue
            self._log(job, text, "warn" if "warn" in text.lower() else "info")

    async def _handle_event(self, job: Job, event: dict[str, Any]) -> None:
        kind = event.get("e")
        if kind == "meta":
            job.total_bytes = int(event.get("total_bytes") or 0)
            job.done_bytes = int(event.get("done_bytes") or 0)
            job.total_files = int(event.get("total_files") or 0)
            job.done_files = int(event.get("done_files") or 0)
            self._save()
        elif kind == "progress":
            if int(event.get("done_bytes") or 0) > job.done_bytes:
                self._progress_at[job.id] = time.time()
                job.stalled = False
            job.done_bytes = int(event.get("done_bytes") or 0)
            job.total_bytes = int(event.get("total_bytes") or job.total_bytes)
            job.total_files = int(event.get("total_files") or job.total_files)
            job.done_files = int(event.get("done_files") or job.done_files)
            job.speed = float(event.get("speed") or 0.0)
        elif kind == "log":
            self._log(job, str(event.get("msg", "")), str(event.get("level", "info")))
        elif kind == "done":
            if event.get("path"):
                job.dest = job.dest or str(event["path"])
        elif kind == "error":
            job.error = str(event.get("msg", "Unknown error"))
            self._log(job, job.error, "error")

        await self._push(job)

    def _log(self, job: Job, message: str, level: str = "info") -> None:
        text = message[:2000]
        # A blocked transfer can repeat one line hundreds of times. Counting the
        # repeats instead of appending them keeps the earlier — and far more
        # useful — history inside the log limit.
        if job.logs:
            last = job.logs[-1]
            if last["msg"] == text and last["level"] == level:
                last["n"] = last.get("n", 1) + 1
                last["t"] = time.time()
                return
        job.logs.append({"t": time.time(), "level": level, "msg": text})

    async def _watch_for_stalls(self) -> None:
        """Mark running jobs that stopped moving, so the UI can say so."""
        while True:
            await asyncio.sleep(STALL_INTERVAL)
            now = time.time()
            for job in list(self.jobs.values()):
                if job.status != RUNNING or job.stalled:
                    continue
                since = self._progress_at.get(job.id) or job.started_at or now
                if now - since < STALL_AFTER:
                    continue
                job.stalled = True
                self._log(
                    job,
                    "No data received for several minutes. Check the connection "
                    "to the Hub; on NAS hardware a depleted system entropy pool "
                    "can block transfers as well.",
                    "warn",
                )
                await self._push(job)

    async def _terminate(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=8)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


manager = JobManager()

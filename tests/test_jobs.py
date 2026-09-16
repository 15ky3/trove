"""The transfer queue: scheduling, cancellation, crashes, restarts, persistence."""

from __future__ import annotations

import asyncio
import json
import signal

import pytest

from app import config, jobs
from app.jobs import CANCELLED, DONE, ERROR, QUEUED, RUNNING, Job, manager
from conftest import run, use_stub_worker, wait_for


def settled(job_id: str) -> bool:
    job = manager.jobs.get(job_id)
    return bool(job and job.status not in (QUEUED, RUNNING))


def messages(job: Job) -> str:
    return " | ".join(entry["msg"] for entry in job.logs)


class TestExitReason:
    def test_sigkill_points_at_memory(self):
        reason = jobs._exit_reason(-9)
        assert "SIGKILL" in reason
        assert "memory" in reason

    def test_sigterm_is_named(self):
        assert "SIGTERM" in jobs._exit_reason(-15)
        assert "SIGTERM" in jobs._exit_reason(143)

    def test_other_signals_are_numbered(self):
        assert "signal 6" in jobs._exit_reason(-6)

    def test_plain_exit_codes_pass_through(self):
        assert "code 2" in jobs._exit_reason(2)


class TestJobSerialisation:
    def test_round_trip(self):
        job = Job(id="abc", kind="download", repo_id="org/name", restarts=2)
        job.logs.append({"t": 1.0, "level": "info", "msg": "hello"})
        clone = Job.from_dict(json.loads(json.dumps(job.to_dict())))
        assert clone.id == "abc"
        assert clone.restarts == 2
        assert list(clone.logs)[-1]["msg"] == "hello"

    def test_unknown_fields_are_ignored(self):
        job = Job.from_dict({"id": "a", "kind": "download", "repo_id": "org/name", "from_the_future": 1})
        assert job.id == "a"

    def test_log_history_is_bounded(self):
        job = Job(id="a", kind="download", repo_id="org/name")
        for i in range(jobs.LOG_LIMIT + 50):
            job.logs.append({"t": 0.0, "level": "info", "msg": str(i)})
        assert len(job.logs) == jobs.LOG_LIMIT
        assert len(job.to_dict()["logs"]) <= 60


class TestLogCollapsing:
    def test_repeats_are_counted_not_appended(self):
        job = Job(id="a", kind="download", repo_id="org/name")
        for _ in range(5):
            manager._log(job, "entropy pool exhausted", "warn")
        assert len(job.logs) == 1
        assert job.logs[-1]["n"] == 5

    def test_a_different_line_starts_a_new_entry(self):
        job = Job(id="a", kind="download", repo_id="org/name")
        manager._log(job, "one")
        manager._log(job, "two")
        assert len(job.logs) == 2

    def test_the_same_text_at_a_different_level_is_separate(self):
        job = Job(id="a", kind="download", repo_id="org/name")
        manager._log(job, "same", "info")
        manager._log(job, "same", "warn")
        assert len(job.logs) == 2

    def test_very_long_lines_are_trimmed(self):
        job = Job(id="a", kind="download", repo_id="org/name")
        manager._log(job, "x" * 5000)
        assert len(job.logs[-1]["msg"]) == 2000


class TestLifecycle:
    def test_a_clean_run_finishes(self, monkeypatch):
        async def scenario():
            use_stub_worker(
                monkeypatch,
                {"emit": [{"e": "meta", "total_bytes": 100, "total_files": 2}, {"e": "done", "path": "/data/x"}]},
            )
            job = await manager.add_download("org/name")
            await wait_for(lambda: settled(job.id))
            return manager.jobs[job.id]

        job = run(scenario())
        assert job.status == DONE
        assert job.done_bytes == 100
        assert job.done_files == 2
        assert job.restarts == 0

    def test_a_reported_error_fails_without_restarting(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"emit": [{"e": "error", "msg": "Repo not found"}], "code": 1})
            job = await manager.add_download("org/name")
            await wait_for(lambda: settled(job.id))
            return manager.jobs[job.id]

        job = run(scenario())
        assert job.status == ERROR
        assert job.error == "Repo not found"
        assert job.restarts == 0, "the worker said why — repeating it would not help"

    def test_progress_events_move_the_job(self, monkeypatch):
        async def scenario():
            use_stub_worker(
                monkeypatch,
                {
                    "emit": [
                        {"e": "meta", "total_bytes": 1000, "total_files": 4},
                        {"e": "progress", "done_bytes": 500, "done_files": 2, "speed": 12.5},
                        {"e": "done"},
                    ]
                },
            )
            job = await manager.add_download("org/name")
            await wait_for(lambda: settled(job.id))
            return manager.jobs[job.id]

        job = run(scenario())
        assert job.status == DONE
        assert job.speed == 0.0, "a finished job is not moving"
        assert job.done_bytes == 1000

    def test_worker_stdout_that_is_not_json_becomes_a_log_line(self, monkeypatch):
        async def scenario():
            command = ["/bin/sh", "-c", "echo plain text line; exit 0"]
            monkeypatch.setattr(manager, "_build_command", lambda job: list(command))
            job = await manager.add_download("org/name")
            await wait_for(lambda: settled(job.id))
            return manager.jobs[job.id]

        assert "plain text line" in messages(run(scenario()))

    def test_a_missing_worker_binary_is_an_error(self, monkeypatch):
        async def scenario():
            monkeypatch.setattr(manager, "_build_command", lambda job: ["/nonexistent/binary"])
            job = await manager.add_download("org/name")
            await wait_for(lambda: settled(job.id))
            return manager.jobs[job.id]

        job = run(scenario())
        assert job.status == ERROR
        assert "Could not start" in job.error


class TestCrashHandling:
    def test_a_killed_worker_is_not_called_cancelled(self, monkeypatch):
        # Read the state while the job is still mid-flight: the queue is wound
        # down once the scenario returns, which would cancel it for real.
        async def scenario():
            use_stub_worker(monkeypatch, {"suicide": signal.SIGKILL})
            job = await manager.add_download("org/name")
            await wait_for(lambda: manager.jobs[job.id].restarts >= 1)
            live = manager.jobs[job.id]
            return live.status, messages(live)

        status, log = run(scenario())
        assert status != CANCELLED
        assert "SIGKILL" in log

    def test_restarts_are_bounded_then_it_fails(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"suicide": signal.SIGKILL})
            job = await manager.add_download("org/name")
            await wait_for(lambda: settled(job.id), timeout=30)
            return manager.jobs[job.id]

        job = run(scenario())
        assert job.status == ERROR
        assert job.restarts == jobs.MAX_CRASH_RESTARTS
        assert "SIGKILL" in job.error

    def test_uploads_are_not_restarted(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"suicide": signal.SIGKILL})
            job = await manager.add_upload("org/name", src=str(config.DATA_DIR))
            await wait_for(lambda: settled(job.id))
            return manager.jobs[job.id]

        job = run(scenario())
        assert job.status == ERROR
        assert job.restarts == 0

    def test_a_manual_retry_clears_the_budget(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"suicide": signal.SIGKILL})
            job = await manager.add_download("org/name")
            await wait_for(lambda: settled(job.id), timeout=30)
            use_stub_worker(monkeypatch, {"emit": [{"e": "done"}]})
            await manager.retry(job.id)
            await wait_for(lambda: settled(job.id))
            return manager.jobs[job.id]

        job = run(scenario())
        assert job.status == DONE
        assert job.restarts == 0
        assert job.error == ""


class TestCancellation:
    def test_a_running_transfer_can_be_stopped(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"hang": 30})
            job = await manager.add_download("org/name")
            await wait_for(lambda: manager.jobs[job.id].status == RUNNING)
            await manager.cancel(job.id)
            await wait_for(lambda: settled(job.id))
            return manager.jobs[job.id]

        job = run(scenario())
        assert job.status == CANCELLED
        assert job.restarts == 0

    def test_a_waiting_transfer_is_simply_dropped(self, monkeypatch):
        async def scenario():
            config.settings.update({"max_concurrent": 1})
            use_stub_worker(monkeypatch, {"hang": 30})
            first = await manager.add_download("org/one")
            await wait_for(lambda: manager.jobs[first.id].status == RUNNING)
            second = await manager.add_download("org/two")
            assert manager.jobs[second.id].status == QUEUED
            await manager.cancel(second.id)
            await manager.cancel(first.id)
            await wait_for(lambda: settled(first.id))
            return manager.jobs[second.id]

        assert run(scenario()).status == CANCELLED

    def test_a_stubborn_process_is_killed(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"hang": 60, "stubborn": True})
            job = await manager.add_download("org/name")
            await wait_for(lambda: manager.jobs[job.id].status == RUNNING)
            await manager.cancel(job.id)
            await wait_for(lambda: settled(job.id), timeout=30)
            return manager.jobs[job.id]

        assert run(scenario()).status == CANCELLED

    def test_cancelling_an_unknown_job_raises(self):
        with pytest.raises(KeyError):
            run(manager.cancel("nope"))

    def test_a_cancel_during_startup_still_stops_it(self, monkeypatch):
        """A job is 'running' from the moment it is scheduled, but its process
        appears a few milliseconds later. Cancelling in that window used to
        signal nothing at all and leave the transfer running to the end."""

        async def scenario():
            use_stub_worker(monkeypatch, {"hang": 30})
            job = await manager.add_download("org/name")
            # No await in between: the worker task has not run yet.
            assert manager._procs.get(job.id) is None
            await manager.cancel(job.id)
            await wait_for(lambda: settled(job.id))
            return manager.jobs[job.id].status

        assert run(scenario()) == CANCELLED

    def test_a_shutdown_during_startup_stops_the_process(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"hang": 30})
            job = await manager.add_download("org/name")
            assert manager._procs.get(job.id) is None
            await manager.shutdown()
            return manager._procs, manager._tasks

        procs, tasks = run(scenario())
        assert procs == {}, "shutdown must not return while a process is still up"
        assert tasks == {}


class TestScheduling:
    def test_the_concurrency_limit_holds(self, monkeypatch):
        async def scenario():
            config.settings.update({"max_concurrent": 2})
            use_stub_worker(monkeypatch, {"hang": 30})
            ids = [(await manager.add_download(f"org/n{i}")).id for i in range(5)]
            await wait_for(lambda: sum(1 for i in ids if manager.jobs[i].status == RUNNING) == 2)
            await asyncio.sleep(0.2)
            running = sum(1 for i in ids if manager.jobs[i].status == RUNNING)
            for job_id in ids:
                await manager.cancel(job_id)
            return running

        assert run(scenario()) == 2

    def test_raising_the_limit_starts_more(self, monkeypatch):
        async def scenario():
            config.settings.update({"max_concurrent": 1})
            use_stub_worker(monkeypatch, {"hang": 30})
            ids = [(await manager.add_download(f"org/n{i}")).id for i in range(3)]
            await wait_for(lambda: sum(1 for i in ids if manager.jobs[i].status == RUNNING) == 1)
            config.settings.update({"max_concurrent": 3})
            await manager.reschedule()
            await wait_for(lambda: sum(1 for i in ids if manager.jobs[i].status == RUNNING) == 3)
            for job_id in ids:
                await manager.cancel(job_id)
            return True

        assert run(scenario())

    def test_finished_jobs_can_be_cleared(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"emit": [{"e": "done"}]})
            done = await manager.add_download("org/done")
            await wait_for(lambda: settled(done.id))
            use_stub_worker(monkeypatch, {"hang": 30})
            busy = await manager.add_download("org/busy")
            await wait_for(lambda: manager.jobs[busy.id].status == RUNNING)
            removed = await manager.clear_finished()
            still_there = set(manager.jobs)
            await manager.cancel(busy.id)
            return removed, still_there, busy.id

        removed, still_there, busy_id = run(scenario())
        assert removed == 1
        assert still_there == {busy_id}

    def test_removing_a_running_job_cancels_it_first(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"hang": 30})
            job = await manager.add_download("org/name")
            await wait_for(lambda: manager.jobs[job.id].status == RUNNING)
            await manager.remove(job.id)
            return job.id in manager.jobs

        assert run(scenario()) is False


class TestPersistence:
    def test_the_queue_is_written_to_disk(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"emit": [{"e": "done"}]})
            job = await manager.add_download("org/name")
            await wait_for(lambda: settled(job.id))
            return job.id

        job_id = run(scenario())
        stored = json.loads(config.JOBS_FILE.read_text())
        assert [j["id"] for j in stored["jobs"]] == [job_id]

    def test_a_shutdown_leaves_transfers_running_so_they_resume(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"hang": 30})
            job = await manager.add_download("org/name")
            await wait_for(lambda: manager.jobs[job.id].status == RUNNING)
            await manager.shutdown()
            await asyncio.sleep(0.3)
            return job.id

        job_id = run(scenario())
        stored = json.loads(config.JOBS_FILE.read_text())
        entry = next(j for j in stored["jobs"] if j["id"] == job_id)
        assert entry["status"] == RUNNING, "a shutdown is not a failure"

    def test_loading_re_queues_interrupted_transfers(self):
        config.JOBS_FILE.write_text(
            json.dumps(
                {
                    "jobs": [
                        {"id": "a", "kind": "download", "repo_id": "org/name", "status": RUNNING, "restarts": 3},
                        {"id": "b", "kind": "download", "repo_id": "org/other", "status": DONE},
                    ]
                }
            )
        )
        manager._load()
        assert manager.jobs["a"].status == QUEUED
        assert manager.jobs["a"].restarts == 0, "a fresh start gets a fresh budget"
        assert manager.jobs["b"].status == DONE

    def test_a_corrupt_queue_file_is_survivable(self):
        config.JOBS_FILE.write_text("{not json")
        manager._load()
        assert manager.jobs == {}

    def test_unreadable_entries_are_skipped(self):
        config.JOBS_FILE.write_text(json.dumps({"jobs": [{"nonsense": True}, {"id": "ok", "kind": "download", "repo_id": "org/n"}]}))
        manager._load()
        assert list(manager.jobs) == ["ok"]


class TestStallWatchdog:
    def test_a_transfer_that_stops_moving_is_flagged(self, monkeypatch):
        monkeypatch.setattr(jobs, "STALL_AFTER", 0.05)
        monkeypatch.setattr(jobs, "STALL_INTERVAL", 0.02)

        async def scenario():
            use_stub_worker(monkeypatch, {"hang": 30})
            job = await manager.add_download("org/name")
            await wait_for(lambda: manager.jobs[job.id].status == RUNNING)
            watchdog = asyncio.create_task(manager._watch_for_stalls())
            await wait_for(lambda: manager.jobs[job.id].stalled)
            watchdog.cancel()
            await manager.cancel(job.id)
            return messages(manager.jobs[job.id])

        assert "No data received" in run(scenario())

    def test_progress_clears_the_flag(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"emit": [{"e": "progress", "done_bytes": 10}], "hang": 5})
            job = await manager.add_download("org/name")
            await wait_for(lambda: manager.jobs[job.id].done_bytes == 10)
            manager.jobs[job.id].stalled = True
            await manager._handle_event(manager.jobs[job.id], {"e": "progress", "done_bytes": 20})
            stalled = manager.jobs[job.id].stalled
            await manager.cancel(job.id)
            return stalled

        assert run(scenario()) is False


class TestWorkerEnvironment:
    def test_the_token_travels_in_the_environment_not_in_argv(self):
        config.settings.update({"hf_token": "hf_secret_value"})
        job = Job(id="a", kind="download", repo_id="org/name", dest="/data/x")
        command = manager._build_command(job)
        assert "hf_secret_value" not in " ".join(command)
        assert manager._build_env()["HF_TOKEN"] == "hf_secret_value"

    def test_an_empty_token_is_removed_rather_than_passed_blank(self):
        config.settings.update({"hf_token": "", "endpoint": ""})
        env = manager._build_env()
        assert "HF_TOKEN" not in env
        assert "HF_ENDPOINT" not in env

    def test_files_at_once_reaches_the_worker(self):
        config.settings.update({"max_workers": 3})
        job = Job(id="a", kind="download", repo_id="org/name", dest="/data/x")
        payload = json.loads(manager._build_command(job)[-1])
        assert payload["max_workers"] == 3

    def test_the_speed_limit_reaches_the_worker(self):
        config.settings.update({"max_download_mbit": 25})
        job = Job(id="a", kind="download", repo_id="org/name", dest="/data/x")
        payload = json.loads(manager._build_command(job)[-1])
        assert payload["limit_mbit"] == 25.0

    def test_no_speed_limit_is_passed_as_zero(self):
        config.settings.update({"max_download_mbit": 0})
        job = Job(id="a", kind="download", repo_id="org/name", dest="/data/x")
        payload = json.loads(manager._build_command(job)[-1])
        assert payload["limit_mbit"] == 0

    def test_upload_payload_carries_its_own_fields(self):
        job = Job(id="a", kind="upload", repo_id="org/name", src="/data/src", private=True)
        payload = json.loads(manager._build_command(job)[-1])
        assert payload["kind"] == "upload"
        assert payload["src"] == "/data/src"
        assert payload["private"] is True
        # Uploads are deliberately not throttled.
        assert "limit_mbit" not in payload


class TestStats:
    def test_counts_by_status(self, monkeypatch):
        async def scenario():
            use_stub_worker(monkeypatch, {"emit": [{"e": "done"}]})
            job = await manager.add_download("org/name")
            await wait_for(lambda: settled(job.id))
            return manager.stats()

        stats = run(scenario())
        assert stats[DONE] == 1
        assert stats[ERROR] == 0

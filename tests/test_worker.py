"""The transfer worker: retry policy, progress maths, marker and IPC helpers."""

from __future__ import annotations

import errno
import json

import httpx
import pytest
from huggingface_hub.errors import (
    DryRunError,
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from app import config, storage, worker


def http_error(status: int, cls=HfHubHTTPError):
    request = httpx.Request("GET", "https://huggingface.co/x")
    return cls("boom", response=httpx.Response(status, request=request))


def emitted(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


class TestPatternEscaping:
    def test_plain_name_is_unchanged(self):
        assert worker._as_pattern("model.safetensors") == "model.safetensors"

    @pytest.mark.parametrize("char", ["*", "?", "["])
    def test_glob_characters_are_neutralised(self, char):
        # huggingface_hub filters by pattern only, so a literal bracket in a
        # filename must not turn into a character class.
        assert worker._as_pattern(f"a{char}b") == f"a[{char}]b"

    def test_matches_only_itself(self):
        import fnmatch

        pattern = worker._as_pattern("model_[0].gguf")
        assert fnmatch.fnmatch("model_[0].gguf", pattern)
        assert not fnmatch.fnmatch("model_0.gguf", pattern)


class TestIsFatal:
    @pytest.mark.parametrize(
        "exc",
        [
            RepositoryNotFoundError("gone", response=httpx.Response(404, request=httpx.Request("GET", "https://x"))),
            GatedRepoError("gated", response=httpx.Response(403, request=httpx.Request("GET", "https://x"))),
            RevisionNotFoundError("rev", response=httpx.Response(404, request=httpx.Request("GET", "https://x"))),
            http_error(401),
            http_error(403),
            http_error(404),
            http_error(400),
            OSError(errno.ENOSPC, "No space left on device"),
            OSError(errno.EACCES, "Permission denied"),
            OSError(errno.EROFS, "Read-only file system"),
        ],
    )
    def test_hopeless_errors_stop_at_once(self, exc):
        assert worker._is_fatal(exc)

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("connection refused"),
            httpx.ReadTimeout("timed out"),
            http_error(500),
            http_error(502),
            http_error(429),
            OSError(errno.EIO, "I/O error"),
            LocalEntryNotFoundError("hub unreachable"),
            RuntimeError("xet gave up"),
        ],
    )
    def test_transient_errors_are_retried(self, exc):
        assert not worker._is_fatal(exc)

    def test_looks_through_the_wrapper(self):
        # A failed listing arrives as a DryRunError that blames the connection
        # whatever actually went wrong; the cause has to decide.
        cause = RepositoryNotFoundError(
            "gone", response=httpx.Response(404, request=httpx.Request("GET", "https://x"))
        )
        wrapped = DryRunError("Dry run cannot be performed")
        wrapped.__cause__ = cause
        assert worker._is_fatal(wrapped)

    def test_wrapper_around_a_transient_cause_still_retries(self):
        wrapped = DryRunError("Dry run cannot be performed")
        wrapped.__cause__ = httpx.ConnectError("connection refused")
        assert not worker._is_fatal(wrapped)

    def test_deep_chain(self):
        inner = OSError(errno.ENOSPC, "No space left on device")
        middle = RuntimeError("write failed")
        middle.__cause__ = inner
        outer = DryRunError("nope")
        outer.__cause__ = middle
        assert worker._is_fatal(outer)

    def test_survives_a_cycle(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert worker._is_fatal(a) is False

    def test_a_bare_entry_not_found_is_not_fatal(self):
        # LocalEntryNotFoundError means the Hub could not be reached at all.
        assert not worker._is_fatal(EntryNotFoundError("missing"))


class TestDescribe:
    def test_plain_exception(self):
        assert worker._describe(ValueError("bad")) == "ValueError: bad"

    def test_prefers_the_cause(self):
        wrapped = DryRunError("check your internet connection or authentication token")
        wrapped.__cause__ = RepositoryNotFoundError(
            "Repository Not Found", response=httpx.Response(404, request=httpx.Request("GET", "https://x"))
        )
        assert worker._describe(wrapped).startswith("RepositoryNotFoundError:")

    def test_ignores_a_cause_of_the_same_type(self):
        outer = ValueError("outer")
        outer.__cause__ = ValueError("inner")
        assert worker._describe(outer) == "ValueError: outer"


class TestBackoff:
    def test_follows_the_table(self):
        assert [worker._backoff(n) for n in (1, 2, 3, 4)] == list(worker.RETRY_BACKOFF)

    def test_last_value_repeats(self):
        assert worker._backoff(9) == worker.RETRY_BACKOFF[-1]

    def test_total_wait_stays_reasonable(self):
        total = sum(worker._backoff(n) for n in range(1, worker.MAX_ATTEMPTS))
        assert total <= 300, "a doomed transfer should not sit in backoff for many minutes"


class TestDropLeftovers:
    def test_clears_and_reports(self, repo_factory, capsys):
        path = repo_factory("org/name", leftovers={"a": 2048, "b": 1024})
        worker._drop_leftovers(path)
        assert storage.leftover_size(path) == 0
        message = emitted(capsys)[0]["msg"]
        assert "2 unusable part file(s)" in message
        assert "reclaimed" in message

    def test_says_nothing_when_there_is_nothing(self, repo_factory, capsys):
        worker._drop_leftovers(repo_factory("org/name"))
        assert emitted(capsys) == []

    def test_empty_parts_do_not_claim_zero_bytes(self, repo_factory, capsys):
        path = repo_factory("org/name", leftovers={"a": 0})
        worker._drop_leftovers(path)
        assert "0 B" not in emitted(capsys)[0]["msg"]


class TestProgress:
    def test_starts_at_zero(self):
        progress = worker.Progress()
        snapshot = progress.snapshot()
        assert snapshot["done_bytes"] == 0
        assert snapshot["speed"] >= 0

    def test_counts_from_the_baseline(self):
        progress = worker.Progress()
        progress.base = 500
        progress.total = 1000
        progress.add("written", 200)
        assert progress.snapshot()["done_bytes"] == 700

    def test_takes_whichever_counter_leads(self):
        # Xet writes to disk in bursts while the network is ahead, or the other
        # way round when dedup kicks in.
        progress = worker.Progress()
        progress.total = 1000
        progress.add("written", 100)
        progress.add("transfer", 400)
        assert progress.snapshot()["done_bytes"] == 400

    def test_never_reports_past_the_total(self):
        progress = worker.Progress()
        progress.total = 100
        progress.add("written", 250)
        assert progress.snapshot()["done_bytes"] == 100

    def test_does_not_go_negative(self):
        progress = worker.Progress()
        progress.add("written", -50)
        assert progress.snapshot()["done_bytes"] == 0

    def test_flush_emits_a_progress_event(self, capsys):
        progress = worker.Progress()
        progress.total = 10
        progress.flush()
        event = emitted(capsys)[0]
        assert event["e"] == "progress"
        assert event["total_bytes"] == 10


class TestMarker:
    def test_written_as_json(self, tmp_path):
        worker._write_marker(tmp_path, {"repo_id": "org/name", "commit": "abc"})
        stored = json.loads((tmp_path / config.MARKER_NAME).read_text())
        assert stored["repo_id"] == "org/name"

    def test_unwritable_folder_only_warns(self, tmp_path, capsys):
        worker._write_marker(tmp_path / "absent", {"a": 1})
        assert emitted(capsys)[0]["level"] == "warn"


class TestUploadFileSelection:
    def test_walks_everything_by_default(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.bin").write_text("a")
        (tmp_path / "sub" / "b.bin").write_text("b")
        found = {p.relative_to(tmp_path).as_posix() for p in worker._iter_upload_files(tmp_path, None, None)}
        assert found == {"a.bin", "sub/b.bin"}

    def test_allow_patterns_restrict(self, tmp_path):
        (tmp_path / "a.bin").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        found = {p.name for p in worker._iter_upload_files(tmp_path, ["*.bin"], None)}
        assert found == {"a.bin"}

    def test_ignore_patterns_exclude(self, tmp_path):
        (tmp_path / "a.bin").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        found = {p.name for p in worker._iter_upload_files(tmp_path, None, ["*.txt"])}
        assert found == {"a.bin"}

    def test_git_folder_is_skipped(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        (tmp_path / "a.bin").write_text("a")
        found = {p.name for p in worker._iter_upload_files(tmp_path, None, None)}
        assert found == {"a.bin"}


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "0 B"), (512, "512 B"), (1536, "1.5 KB"), (1024 * 1024 * 3, "3.0 MB")],
    )
    def test_human_sizes(self, value, expected):
        assert worker._fmt(value) == expected


class TestEmit:
    def test_one_json_line_per_event(self, capsys):
        worker.emit("meta", total_bytes=5)
        worker.log("hello", "warn")
        events = emitted(capsys)
        assert events[0] == {"e": "meta", "total_bytes": 5}
        assert events[1] == {"e": "log", "msg": "hello", "level": "warn"}


class TestEntryPoint:
    def test_invalid_payload_is_reported(self, capsys):
        assert worker.main(["app.worker", "not json"]) == 2
        assert emitted(capsys)[0]["e"] == "error"

    def test_missing_payload_is_reported(self, capsys):
        assert worker.main(["app.worker"]) == 2
        assert emitted(capsys)[0]["e"] == "error"

    def test_failure_is_emitted_as_an_error_event(self, capsys, monkeypatch):
        def boom(_payload):
            raise RuntimeError("nope")

        monkeypatch.setattr(worker, "run_download", boom)
        assert worker.main(["app.worker", json.dumps({"kind": "download"})]) == 1
        assert emitted(capsys)[-1]["msg"] == "RuntimeError: nope"

    def test_a_stop_signal_ends_as_cancelled(self, capsys, monkeypatch):
        def stop(_payload):
            raise SystemExit(143)

        monkeypatch.setattr(worker, "run_download", stop)
        assert worker.main(["app.worker", json.dumps({"kind": "download"})]) == 143
        assert emitted(capsys)[-1]["msg"] == "Cancelled."


class TestSpeedLimitWiring:
    """The worker starts the limiter before the transfer and stops it after."""

    class FakeLimiter:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    def arm(self, monkeypatch):
        seen: dict = {}
        limiter = self.FakeLimiter()

        def fake_start(value, log=None):
            seen["value"] = value
            seen["log"] = log
            return limiter

        monkeypatch.setattr(worker.throttle, "start_limit", fake_start)
        return seen, limiter

    def test_the_limit_from_the_payload_is_applied_and_released(self, monkeypatch):
        seen, limiter = self.arm(monkeypatch)
        monkeypatch.setattr(worker, "run_download", lambda payload: None)

        assert worker.main(["app.worker", json.dumps({"kind": "download", "limit_mbit": 25})]) == 0
        assert seen["value"] == 25
        assert seen["log"] is worker.log
        assert limiter.stopped

    def test_a_payload_without_a_limit_asks_for_none(self, monkeypatch):
        seen, _ = self.arm(monkeypatch)
        monkeypatch.setattr(worker, "run_upload", lambda payload: None)

        assert worker.main(["app.worker", json.dumps({"kind": "upload"})]) == 0
        assert seen["value"] is None

    def test_the_limiter_is_released_when_the_transfer_fails(self, monkeypatch, capsys):
        _, limiter = self.arm(monkeypatch)

        def boom(_payload):
            raise RuntimeError("nope")

        monkeypatch.setattr(worker, "run_download", boom)
        assert worker.main(["app.worker", json.dumps({"kind": "download", "limit_mbit": 5})]) == 1
        assert limiter.stopped
        capsys.readouterr()

    def test_the_limiter_is_released_on_cancellation(self, monkeypatch, capsys):
        _, limiter = self.arm(monkeypatch)

        def stop(_payload):
            raise SystemExit(143)

        monkeypatch.setattr(worker, "run_download", stop)
        assert worker.main(["app.worker", json.dumps({"kind": "download", "limit_mbit": 5})]) == 143
        assert limiter.stopped
        capsys.readouterr()

    def test_without_a_limiter_nothing_is_stopped(self, monkeypatch):
        monkeypatch.setattr(worker.throttle, "start_limit", lambda value, log=None: None)
        monkeypatch.setattr(worker, "run_download", lambda payload: None)
        assert worker.main(["app.worker", json.dumps({"kind": "download"})]) == 0

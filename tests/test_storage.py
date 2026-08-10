"""The local library: listing, sizing, leftover accounting, deletion."""

from __future__ import annotations

import time

import pytest

from app import config, storage


def find(repos, repo_id):
    return next(r for r in repos if r["repo_id"] == repo_id)


class TestListRepos:
    def test_empty_library(self):
        assert storage.list_repos() == []

    def test_marker_wins_over_the_folder_name(self, repo_factory):
        repo_factory(
            "org/name",
            marker={"repo_id": "canonical/name", "repo_type": "model", "commit": "abc123", "revision": "main"},
        )
        entry = storage.list_repos()[0]
        assert entry["repo_id"] == "canonical/name"
        assert entry["commit"] == "abc123"
        assert entry["complete"] is True

    def test_folder_without_a_marker_is_still_listed(self, repo_factory):
        repo_factory("org/name")
        entry = storage.list_repos()[0]
        assert entry["repo_id"] == "org/name"
        assert entry["complete"] is False
        assert entry["commit"] == ""

    def test_repo_without_an_org(self, repo_factory):
        repo_factory("gpt2")
        assert storage.list_repos()[0]["repo_id"] == "gpt2"

    def test_org_folder_is_not_a_repo(self, repo_factory):
        repo_factory("org/one")
        repo_factory("org/two")
        ids = {r["repo_id"] for r in storage.list_repos()}
        assert ids == {"org/one", "org/two"}

    def test_filters_by_type(self, repo_factory):
        repo_factory("org/model")
        repo_factory("org/data", repo_type="dataset")
        assert [r["repo_id"] for r in storage.list_repos("dataset")] == ["org/data"]
        assert [r["repo_id"] for r in storage.list_repos("model")] == ["org/model"]
        assert len(storage.list_repos()) == 2

    def test_full_commit_is_kept(self, repo_factory):
        sha = "a" * 40
        repo_factory("org/name", marker={"commit": sha})
        # The update check compares commits, so it must not be shortened here.
        assert storage.list_repos()[0]["commit"] == sha

    @pytest.mark.parametrize(
        "marker",
        [
            {"files": ["model.gguf"]},
            {"allow_patterns": ["*.gguf"]},
            {"ignore_patterns": ["*.bin"]},
        ],
    )
    def test_partial_is_detected(self, repo_factory, marker):
        repo_factory("org/name", marker=marker)
        assert storage.list_repos()[0]["partial"] is True

    def test_whole_copy_is_not_partial(self, repo_factory):
        repo_factory("org/name", marker={"commit": "abc"})
        assert storage.list_repos()[0]["partial"] is False

    def test_newest_first(self, repo_factory):
        repo_factory("org/old", marker={"downloaded_at": 1000})
        repo_factory("org/new", marker={"downloaded_at": 2000})
        assert [r["repo_id"] for r in storage.list_repos()] == ["org/new", "org/old"]


class TestSizes:
    def test_counts_bytes_and_files(self, repo_factory):
        repo_factory("org/name", files={"a.bin": 100, "b.bin": 50, "sub/c.bin": 25})
        entry = storage.list_repos()[0]
        assert entry["size"] == 175
        assert entry["files"] == 3

    def test_internal_folders_do_not_count(self, repo_factory):
        path = repo_factory("org/name", files={"a.bin": 100}, etags={"a.bin": "etag"})
        (path / ".git").mkdir()
        (path / ".git" / "big").write_bytes(b"x" * 500)
        storage.invalidate()
        entry = storage.list_repos()[0]
        assert entry["size"] == 100
        assert entry["files"] == 1

    def test_result_is_cached(self, repo_factory):
        path = repo_factory("org/name", files={"a.bin": 100})
        assert storage.list_repos()[0]["size"] == 100
        (path / "b.bin").write_bytes(b"x" * 900)
        assert storage.list_repos()[0]["size"] == 100, "should still be served from cache"
        assert storage.list_repos(refresh=True)[0]["size"] == 1000

    def test_cache_expires(self, repo_factory, monkeypatch):
        path = repo_factory("org/name", files={"a.bin": 100})
        assert storage.list_repos()[0]["size"] == 100
        (path / "b.bin").write_bytes(b"x" * 900)
        # Take the real clock before replacing it, or the stand-in calls itself.
        later = time.time() + storage._CACHE_TTL + 1
        monkeypatch.setattr(storage.time, "time", lambda: later)
        assert storage.list_repos()[0]["size"] == 1000


class TestLeftoverParts:
    def test_none_by_default(self, repo_factory):
        path = repo_factory("org/name")
        assert storage.leftover_size(path) == 0

    def test_counted_but_kept_out_of_the_repo_size(self, repo_factory):
        repo_factory("org/name", files={"a.bin": 100}, leftovers={"big": 4000, "small": 25})
        entry = storage.list_repos()[0]
        assert entry["size"] == 100, "part files must not inflate the repo size"
        assert entry["leftover"] == 4025

    def test_dropping_reclaims_and_reports(self, repo_factory):
        path = repo_factory("org/name", leftovers={"one": 300, "two": 700})
        assert storage.drop_leftover_parts(path) == (1000, 2)
        assert storage.leftover_size(path) == 0

    def test_dropping_leaves_everything_else_alone(self, repo_factory):
        path = repo_factory("org/name", files={"a.bin": 10}, etags={"a.bin": "etag"}, leftovers={"p": 40})
        storage.drop_leftover_parts(path)
        assert (path / "a.bin").exists()
        assert storage.local_etags("model", "org/name") == {"a.bin": "etag"}

    def test_dropping_nothing_is_harmless(self, repo_factory):
        path = repo_factory("org/name")
        assert storage.drop_leftover_parts(path) == (0, 0)

    def test_missing_folder_is_harmless(self):
        missing = config.local_dir_for("model", "org/absent")
        assert storage.leftover_size(missing) == 0
        assert storage.drop_leftover_parts(missing) == (0, 0)


class TestLocalEtags:
    def test_reads_the_second_line(self, repo_factory):
        repo_factory("org/name", etags={"a.bin": "sha-a", "sub/b.bin": "sha-b"})
        assert storage.local_etags("model", "org/name") == {"a.bin": "sha-a", "sub/b.bin": "sha-b"}

    def test_empty_when_nothing_was_recorded(self, repo_factory):
        repo_factory("org/name")
        assert storage.local_etags("model", "org/name") == {}

    def test_malformed_records_are_skipped(self, repo_factory):
        path = repo_factory("org/name", etags={"good.bin": "sha"})
        cache = path / ".cache" / "huggingface" / "download"
        (cache / "truncated.bin.metadata").write_text("only-a-commit\n")
        (cache / "blank.bin.metadata").write_text("commit\n\n123\n")
        assert storage.local_etags("model", "org/name") == {"good.bin": "sha"}


class TestRepoFiles:
    def test_lists_sorted_without_bookkeeping(self, repo_factory):
        repo_factory(
            "org/name",
            files={"z.bin": 3, "a.bin": 1},
            marker={"commit": "abc"},
            etags={"a.bin": "sha"},
        )
        names = [f["name"] for f in storage.repo_files("model", "org/name")]
        assert names == ["a.bin", "z.bin"]

    def test_reports_sizes(self, repo_factory):
        repo_factory("org/name", files={"a.bin": 42})
        assert storage.repo_files("model", "org/name")[0]["size"] == 42

    def test_missing_repo_is_empty(self):
        assert storage.repo_files("model", "org/absent") == []

    def test_limit_is_honoured(self, repo_factory):
        repo_factory("org/name", files={f"f{i}.bin": 1 for i in range(20)})
        assert len(storage.repo_files("model", "org/name", limit=5)) == 5


class TestDelete:
    def test_removes_the_folder(self, repo_factory):
        path = repo_factory("org/name", files={"a.bin": 100})
        result = storage.delete_repo("model", "org/name")
        assert not path.exists()
        assert result["freed"] == 100
        assert result["files"] == 1

    def test_freed_includes_leftovers(self, repo_factory):
        repo_factory("org/name", files={"a.bin": 100}, leftovers={"p": 900})
        assert storage.delete_repo("model", "org/name")["freed"] == 1000

    def test_empty_org_folder_goes_too(self, repo_factory):
        path = repo_factory("org/only")
        storage.delete_repo("model", "org/only")
        assert not path.parent.exists()

    def test_org_folder_with_siblings_stays(self, repo_factory):
        repo_factory("org/one")
        path = repo_factory("org/two")
        storage.delete_repo("model", "org/one")
        assert path.parent.is_dir()

    def test_unknown_repo_raises(self):
        with pytest.raises(FileNotFoundError):
            storage.delete_repo("model", "org/absent")


class TestDiskUsage:
    def test_reports_totals(self):
        usage = storage.disk_usage()
        assert usage["total"] > 0
        assert usage["free"] >= 0

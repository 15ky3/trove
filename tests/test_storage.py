"""The local library: listing, sizing, leftover accounting, deletion."""

from __future__ import annotations

import json
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


class TestDeleteFiles:
    def test_removes_the_picked_file_only(self, repo_factory):
        path = repo_factory("org/name", files={"a.bin": 100, "b.bin": 50}, marker={"commit": "abc"})
        result = storage.delete_files("model", "org/name", ["a.bin"])
        assert not (path / "a.bin").exists()
        assert (path / "b.bin").exists()
        assert result["deleted"] == ["a.bin"]
        assert result["freed"] == 100
        assert result["remaining"] == 1

    def test_the_download_record_goes_too(self, repo_factory):
        # Left behind, the update check would still count the file as tracked
        # and pull it straight back.
        repo_factory(
            "org/name",
            files={"a.bin": 10, "b.bin": 10},
            marker={"commit": "abc"},
            etags={"a.bin": "sha-a", "b.bin": "sha-b"},
        )
        storage.delete_files("model", "org/name", ["a.bin"])
        assert storage.local_etags("model", "org/name") == {"b.bin": "sha-b"}

    @pytest.mark.parametrize("part", ["a.bin.incomplete", "a.bin.deadbeef.0123-4567.incomplete"])
    def test_half_written_parts_of_that_file_go_too(self, repo_factory, part):
        path = repo_factory("org/name", files={"a.bin": 10, "b.bin": 10}, marker={"commit": "abc"})
        cache = path / ".cache" / "huggingface" / "download"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / part).write_bytes(b"y" * 400)
        result = storage.delete_files("model", "org/name", ["a.bin"])
        assert not (cache / part).exists()
        assert result["freed"] == 410

    def test_parts_of_another_file_stay(self, repo_factory):
        path = repo_factory(
            "org/name",
            files={"a.bin": 10, "b.bin": 10},
            marker={"commit": "abc"},
            leftovers={"b.bin": 400},
        )
        storage.delete_files("model", "org/name", ["a.bin"])
        assert storage.leftover_size(path) == 400

    def test_an_emptied_folder_is_removed(self, repo_factory):
        path = repo_factory("org/name", files={"onnx/model.onnx": 10, "a.bin": 10}, marker={"commit": "abc"})
        storage.delete_files("model", "org/name", ["onnx/model.onnx"])
        assert not (path / "onnx").exists()
        assert path.is_dir()

    def test_the_selection_narrows_to_what_is_left(self, repo_factory):
        path = repo_factory(
            "org/name",
            files={"a.bin": 10, "sub/b.bin": 10, "c.bin": 10},
            marker={"commit": "abc"},
        )
        storage.delete_files("model", "org/name", ["a.bin"])
        marker = json.loads((path / config.MARKER_NAME).read_text())
        assert marker["files"] == ["c.bin", "sub/b.bin"]
        # The repo now counts as a partial copy, so the update goes file by file.
        assert storage.list_repos()[0]["partial"] is True

    def test_a_download_pattern_is_dropped(self, repo_factory):
        # "*.gguf" would match the deleted file again on the next update.
        path = repo_factory(
            "org/name",
            files={"q4.gguf": 10, "q8.gguf": 10},
            marker={"commit": "abc", "allow_patterns": ["*.gguf"]},
        )
        storage.delete_files("model", "org/name", ["q8.gguf"])
        marker = json.loads((path / config.MARKER_NAME).read_text())
        assert marker["allow_patterns"] == []
        assert marker["files"] == ["q4.gguf"]

    def test_a_folder_without_a_record_does_not_get_one(self, repo_factory):
        # Inventing a marker here would dress an unknown folder up as a
        # tracked download.
        path = repo_factory("org/name", files={"a.bin": 10, "b.bin": 10})
        storage.delete_files("model", "org/name", ["a.bin"])
        assert not (path / config.MARKER_NAME).exists()
        assert storage.list_repos()[0]["complete"] is False

    def test_deleting_the_last_file_removes_the_repo(self, repo_factory):
        path = repo_factory("org/name", files={"a.bin": 100}, marker={"commit": "abc"})
        result = storage.delete_files("model", "org/name", ["a.bin"])
        assert result["removed_repo"] is True
        assert not path.exists()
        assert storage.list_repos() == []

    def test_a_file_that_is_already_gone_is_reported_not_raised(self, repo_factory):
        repo_factory("org/name", files={"a.bin": 10, "b.bin": 10}, marker={"commit": "abc"})
        result = storage.delete_files("model", "org/name", ["absent.bin"])
        assert result == {
            "deleted": [], "missing": ["absent.bin"], "failed": [], "freed": 0,
            "remaining": 2, "removed_repo": False, "warning": "",
        }

    def test_a_file_that_cannot_be_removed_does_not_abort_the_rest(self, repo_factory, monkeypatch):
        # A Synology ACL or a read-only mount hits one file in the middle. The
        # ones already gone stay gone, so the selection has to be narrowed
        # anyway — otherwise the next update fetches them back.
        path = repo_factory("org/name", files={"a.bin": 10, "b.bin": 10, "c.bin": 10}, marker={"commit": "abc"})
        real = storage.Path.unlink

        def refuse(self, *args, **kwargs):
            if self.name == "b.bin":
                raise PermissionError("operation not permitted")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(storage.Path, "unlink", refuse)
        result = storage.delete_files("model", "org/name", ["a.bin", "b.bin", "c.bin"])

        assert result["deleted"] == ["a.bin", "c.bin"]
        assert result["failed"] == ["b.bin"]
        assert (path / "b.bin").exists()
        assert "permissions" in result["warning"]
        marker = json.loads((path / config.MARKER_NAME).read_text())
        assert marker["files"] == ["b.bin"]

    def test_a_copy_without_a_record_says_the_promise_does_not_hold(self, repo_factory):
        # Nothing to narrow, so an update still fetches the whole repo. The
        # interface promises the opposite, so this has to be said out loud.
        repo_factory("org/name", files={"a.bin": 10, "b.bin": 10})
        result = storage.delete_files("model", "org/name", ["a.bin"])
        assert "no download record" in result["warning"]

    def test_a_copy_with_a_record_warns_about_nothing(self, repo_factory):
        repo_factory("org/name", files={"a.bin": 10, "b.bin": 10}, marker={"commit": "abc"})
        assert storage.delete_files("model", "org/name", ["a.bin"])["warning"] == ""

    def test_several_at_once(self, repo_factory):
        path = repo_factory("org/name", files={"a.bin": 10, "b.bin": 20, "c.bin": 30})
        result = storage.delete_files("model", "org/name", ["a.bin", "c.bin"])
        assert result["freed"] == 40
        assert [f["name"] for f in storage.repo_files("model", "org/name")] == ["b.bin"]
        assert path.is_dir()

    @pytest.mark.parametrize(
        "name",
        [
            "../../../etc/passwd",
            "sub/../../escape.bin",
            "/etc/passwd",
            "..",
            "",
            "   ",
            # These normalise away to no path at all — the guards below have
            # nothing left to inspect.
            ".",
            "./",
            ". ",
            ".cache/huggingface/download/a.bin.metadata",
            ".git/config",
            config.MARKER_NAME,
            "..\\..\\escape.bin",
            # A case-insensitive volume would resolve these to the real thing.
            ".Cache/huggingface/download/a.bin.metadata",
            ".TROVE.json",
            "sub/.TrOvE.json",
        ],
    )
    def test_traversal_and_bookkeeping_are_refused(self, repo_factory, name):
        repo_factory("org/name", files={"a.bin": 10})
        with pytest.raises(ValueError):
            storage.delete_files("model", "org/name", [name])

    def test_a_symlink_out_of_the_repo_is_refused(self, repo_factory, tmp_path):
        path = repo_factory("org/name", files={"a.bin": 10})
        outside = tmp_path / "secret.txt"
        outside.write_text("keep me")
        (path / "link.bin").symlink_to(outside)
        with pytest.raises(ValueError):
            storage.delete_files("model", "org/name", ["link.bin"])
        assert outside.exists()

    def test_one_bad_name_deletes_nothing(self, repo_factory):
        path = repo_factory("org/name", files={"a.bin": 10, "b.bin": 10})
        with pytest.raises(ValueError):
            storage.delete_files("model", "org/name", ["a.bin", "../escape"])
        assert (path / "a.bin").exists()

    def test_an_unknown_repo_raises(self):
        with pytest.raises(FileNotFoundError):
            storage.delete_files("model", "org/absent", ["a.bin"])

    def test_an_unwritable_marker_is_a_warning_not_a_failure(self, repo_factory, monkeypatch):
        path = repo_factory("org/name", files={"a.bin": 10, "b.bin": 10}, marker={"commit": "abc"})

        def refuse(*_args, **_kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(storage.Path, "write_text", refuse)
        result = storage.delete_files("model", "org/name", ["a.bin"])
        assert not (path / "a.bin").exists()
        assert "could not be updated" in result["warning"]

    def test_sizes_are_recounted_afterwards(self, repo_factory):
        repo_factory("org/name", files={"a.bin": 100, "b.bin": 50})
        assert storage.list_repos()[0]["size"] == 150
        storage.delete_files("model", "org/name", ["a.bin"])
        assert storage.list_repos()[0]["size"] == 50


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

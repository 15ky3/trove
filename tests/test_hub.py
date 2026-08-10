"""The huggingface_hub wrapper: error translation and field mapping.

No test in here reaches the network — `hub.api` is replaced by a fake client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from app import hub


def response(status: int = 404) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/x"))


class FakeApi:
    """Returns what it was given, or raises what it was given."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def _answer(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    repo_info = _answer
    whoami = _answer
    list_models = _answer
    list_datasets = _answer
    list_spaces = _answer


@pytest.fixture
def fake_api(monkeypatch):
    def install(result=None, error=None):
        client = FakeApi(result, error)
        monkeypatch.setattr(hub, "api", lambda token=None: client)
        return client

    return install


def sibling(name: str, size: int = 0, blob_id: str = "", sha256: str | None = None):
    lfs = SimpleNamespace(sha256=sha256, size=size) if sha256 else None
    return SimpleNamespace(rfilename=name, size=size, blob_id=blob_id, lfs=lfs)


class TestErrorTranslation:
    def test_gated_repos_say_what_to_do(self, fake_api):
        fake_api(error=GatedRepoError("gated", response=response(403)))
        with pytest.raises(hub.HubError, match="gated"):
            hub.resolve("org/name")

    def test_missing_repos_mention_the_name(self, fake_api):
        fake_api(error=RepositoryNotFoundError("gone", response=response(404)))
        with pytest.raises(hub.HubError, match="org/name"):
            hub.resolve("org/name")

    def test_unknown_revision_mentions_it(self, fake_api):
        fake_api(error=RevisionNotFoundError("rev", response=response(404)))
        with pytest.raises(hub.HubError, match="v9"):
            hub.resolve("org/name", revision="v9")

    def test_anything_else_still_becomes_a_hub_error(self, fake_api):
        fake_api(error=httpx.ConnectError("connection refused"))
        with pytest.raises(hub.HubError):
            hub.resolve("org/name")

    def test_all_three_readers_share_the_translation(self, fake_api):
        fake_api(error=RepositoryNotFoundError("gone", response=response(404)))
        for call in (hub.resolve, hub.file_state, hub.repo_info):
            with pytest.raises(hub.HubError):
                call("org/name")


class TestResolve:
    def test_returns_the_canonical_id(self, fake_api):
        fake_api(result=SimpleNamespace(id="google-bert/bert-base-uncased", sha="abc123"))
        assert hub.resolve("bert-base-uncased") == {
            "repo_id": "google-bert/bert-base-uncased",
            "sha": "abc123",
        }

    def test_falls_back_to_what_was_asked_for(self, fake_api):
        fake_api(result=SimpleNamespace(id=None, sha=None))
        assert hub.resolve("org/name")["repo_id"] == "org/name"


class TestFileState:
    def test_lfs_hash_wins_for_large_files(self, fake_api):
        fake_api(
            result=SimpleNamespace(
                id="org/name",
                sha="commit",
                siblings=[sibling("model.bin", 100, blob_id="blob", sha256="lfs-sha")],
            )
        )
        assert hub.file_state("org/name")["files"]["model.bin"] == "lfs-sha"

    def test_blob_id_is_used_for_plain_files(self, fake_api):
        fake_api(result=SimpleNamespace(id="org/name", sha="c", siblings=[sibling("README.md", 10, blob_id="blob")]))
        assert hub.file_state("org/name")["files"]["README.md"] == "blob"

    def test_files_without_a_hash_come_back_empty(self, fake_api):
        fake_api(result=SimpleNamespace(id="org/name", sha="c", siblings=[sibling("x")]))
        assert hub.file_state("org/name")["files"]["x"] == ""

    def test_asks_for_file_metadata(self, fake_api):
        client = fake_api(result=SimpleNamespace(id="o/n", sha="c", siblings=[]))
        hub.file_state("o/n")
        assert client.calls[0]["files_metadata"] is True

    def test_no_siblings_is_fine(self, fake_api):
        fake_api(result=SimpleNamespace(id="o/n", sha="c", siblings=None))
        assert hub.file_state("o/n")["files"] == {}


class TestRepoInfo:
    def test_sums_sizes_and_sorts_files(self, fake_api):
        fake_api(
            result=SimpleNamespace(
                id="org/name",
                sha="abc",
                siblings=[sibling("z.bin", 30), sibling("a.bin", 12)],
                private=False,
                gated=False,
                downloads=5,
                likes=2,
                pipeline_tag="text-generation",
                tags=["gguf"],
                last_modified=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        )
        info = hub.repo_info("org/name")
        assert [f["name"] for f in info["files"]] == ["a.bin", "z.bin"]
        assert info["total_size"] == 42
        assert info["updated_at"] > 0

    def test_lfs_size_is_used_when_the_plain_one_is_missing(self, fake_api):
        entry = SimpleNamespace(rfilename="m.bin", size=0, blob_id="", lfs=SimpleNamespace(sha256="s", size=500))
        fake_api(
            result=SimpleNamespace(
                id="org/name", sha="", siblings=[entry], private=False, gated=None,
                downloads=0, likes=0, pipeline_tag=None, tags=None, last_modified=None,
            )
        )
        assert hub.repo_info("org/name")["total_size"] == 500

    def test_missing_timestamp_is_zero(self, fake_api):
        fake_api(
            result=SimpleNamespace(
                id="o/n", sha="", siblings=[], private=False, gated=None,
                downloads=0, likes=0, pipeline_tag=None, tags=None, last_modified=None,
            )
        )
        assert hub.repo_info("o/n")["updated_at"] == 0.0


class TestSearch:
    def test_maps_the_fields(self, fake_api):
        fake_api(
            result=[
                SimpleNamespace(
                    id="org/name", author="org", downloads=10, likes=3,
                    pipeline_tag="text-generation", library_name="transformers",
                    private=False, gated=False, last_modified=None,
                    tags=["gguf", "license:mit", "text-generation"],
                )
            ]
        )
        result = hub.search("name")[0]
        assert result["repo_id"] == "org/name"
        assert result["downloads"] == 10
        assert "license:mit" not in result["tags"], "namespaced tags are noise in the list"

    def test_author_falls_back_to_the_id(self, fake_api):
        fake_api(result=[SimpleNamespace(id="org/name", last_modified=None)])
        assert hub.search("x")[0]["author"] == "org"

    def test_failures_become_hub_errors(self, fake_api):
        fake_api(error=RuntimeError("hub down"))
        with pytest.raises(hub.HubError, match="Search failed"):
            hub.search("x")

    @pytest.mark.parametrize("repo_type", ["model", "dataset", "space"])
    def test_every_type_is_searchable(self, fake_api, repo_type):
        fake_api(result=[])
        assert hub.search("x", repo_type) == []


class TestWhoami:
    def test_returns_name_and_orgs(self, fake_api):
        fake_api(result={"name": "someone", "orgs": [{"name": "org-a"}, {"nope": 1}]})
        assert hub.whoami("hf_x") == {"name": "someone", "orgs": ["org-a"]}

    def test_a_rejected_token_is_explained(self, fake_api):
        fake_api(error=HfHubHTTPError("401", response=response(401)))
        with pytest.raises(hub.HubError, match="Token rejected"):
            hub.whoami("hf_bad")

    def test_network_failure_is_wrapped(self, fake_api):
        fake_api(error=httpx.ConnectError("no route"))
        with pytest.raises(hub.HubError):
            hub.whoami("hf_x")

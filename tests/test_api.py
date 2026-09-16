"""The HTTP surface: auth, validation, and every endpoint's contract.

The Hub is faked throughout, and the worker is replaced by a stub that simply
hangs — enough for a job to reach `running` without a byte of traffic.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app import __version__, config, hub, storage, throttle
from app.jobs import RUNNING, manager
from conftest import PASSWORD, use_stub_worker

PROTECTED = [
    ("GET", "/api/settings"),
    ("GET", "/api/disk"),
    ("GET", "/api/search"),
    ("GET", "/api/repo?repo_id=org/name"),
    ("GET", "/api/library"),
    ("GET", "/api/library/files?repo_id=org/name"),
    ("GET", "/api/library/updates"),
    ("GET", "/api/jobs"),
    ("PUT", "/api/settings"),
    ("POST", "/api/settings/test-token"),
    ("POST", "/api/library/update"),
    ("POST", "/api/library/delete"),
    ("POST", "/api/library/files/delete"),
    ("POST", "/api/jobs/download"),
    ("POST", "/api/jobs/upload"),
    ("POST", "/api/jobs/abc/cancel"),
    ("POST", "/api/jobs/abc/retry"),
    ("POST", "/api/jobs/clear"),
    ("DELETE", "/api/jobs/abc"),
]


@pytest.fixture
def anon(monkeypatch):
    """A client that never signed in, with the worker stubbed out."""
    use_stub_worker(monkeypatch, {"hang": 30})
    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def client(anon):
    assert anon.post("/api/login", json={"password": PASSWORD}).status_code == 200
    return anon


@pytest.fixture
def fake_hub(monkeypatch):
    """Point every Hub call at canned answers."""

    def install(**overrides):
        defaults = {
            "resolve": lambda repo_id, repo_type="model", revision=None: {"repo_id": repo_id, "sha": "remote-sha"},
            "search": lambda q, repo_type="model", limit=30: [{"repo_id": "org/found", "repo_type": repo_type}],
            "repo_info": lambda repo_id, repo_type="model", revision=None: {
                "repo_id": repo_id, "repo_type": repo_type, "sha": "s", "files": [], "total_size": 0
            },
            "file_state": lambda repo_id, repo_type="model", revision=None: {
                "repo_id": repo_id, "sha": "remote-sha", "files": {}
            },
            "whoami": lambda token=None: {"name": "tester", "orgs": []},
        }
        defaults.update(overrides)
        for name, fn in defaults.items():
            monkeypatch.setattr(hub, name, fn)

    return install


# ------------------------------------------------------------------------ Auth


class TestAuth:
    @pytest.mark.parametrize(("method", "path"), PROTECTED)
    def test_everything_needs_a_session(self, anon, method, path):
        assert anon.request(method, path, json={}).status_code == 401

    def test_wrong_password_is_rejected(self, anon):
        assert anon.post("/api/login", json={"password": "nope"}).status_code == 401

    def test_right_password_opens_a_session(self, anon):
        assert anon.post("/api/login", json={"password": PASSWORD}).json() == {"authenticated": True}
        assert anon.get("/api/settings").status_code == 200

    def test_logout_closes_it(self, client):
        client.post("/api/logout")
        assert client.get("/api/settings").status_code == 401

    def test_a_forged_cookie_does_not_work(self, anon):
        anon.cookies.set("hfd_session", "made.up.token")
        assert anon.get("/api/settings").status_code == 401

    def test_session_reports_the_state(self, anon):
        before = anon.get("/api/session").json()
        assert before["auth_required"] is True
        assert before["authenticated"] is False
        assert before["version"] == __version__
        anon.post("/api/login", json={"password": PASSWORD})
        assert anon.get("/api/session").json()["authenticated"] is True

    def test_without_a_password_the_app_is_open(self, monkeypatch, anon):
        monkeypatch.setenv("UI_PASSWORD", "")
        assert anon.get("/api/session").json()["auth_required"] is False
        assert anon.get("/api/settings").status_code == 200

    def test_login_without_a_password_configured_just_succeeds(self, monkeypatch, anon):
        monkeypatch.setenv("UI_PASSWORD", "")
        assert anon.post("/api/login", json={"password": ""}).json() == {"authenticated": True}

    def test_healthz_stays_open(self, anon):
        assert anon.get("/healthz").json()["ok"] is True


# -------------------------------------------------------------------- Settings


class TestSettingsEndpoint:
    def test_read_never_leaks_the_token(self, client):
        config.settings.update({"hf_token": "hf_supersecret"})
        body = client.get("/api/settings").json()
        assert "hf_token" not in body
        assert "supersecret" not in body["token_hint"]

    def test_write_round_trip(self, client):
        body = client.put("/api/settings", json={"max_workers": 6, "max_concurrent": 3}).json()
        assert body["max_workers"] == 6
        assert body["max_concurrent"] == 3

    def test_out_of_range_values_are_clamped(self, client):
        assert client.put("/api/settings", json={"max_workers": 999}).json()["max_workers"] == 32

    def test_the_speed_limit_round_trips(self, client):
        assert client.put("/api/settings", json={"max_download_mbit": 30}).json()["max_download_mbit"] == 30.0
        assert client.get("/api/settings").json()["max_download_mbit"] == 30.0

    def test_a_speed_limit_of_nan_is_refused(self, client):
        # Pydantic lets NaN through to the validator, where it used to clamp to
        # the maximum instead of being dropped.
        client.put("/api/settings", json={"max_download_mbit": 20})
        # A JSON encoder refuses to write NaN, so it arrives as a raw body —
        # which json.loads on the server side happily accepts.
        body = client.put(
            "/api/settings",
            content='{"max_download_mbit": NaN}',
            headers={"content-type": "application/json"},
        ).json()
        assert body["max_download_mbit"] == 20.0

    def test_a_negative_speed_limit_means_off(self, client):
        assert client.put("/api/settings", json={"max_download_mbit": -1}).json()["max_download_mbit"] == 0.0

    def test_saving_a_speed_limit_arms_the_limiter(self, client):
        try:
            client.put("/api/settings", json={"max_download_mbit": 16})
            assert throttle.shared.mbit == 16
            assert throttle.shared.url
        finally:
            throttle.shared.stop()

    def test_raising_the_limit_retunes_without_a_restart(self, client):
        # Transfers already running point at this port.
        try:
            client.put("/api/settings", json={"max_download_mbit": 16})
            url = throttle.shared.url
            client.put("/api/settings", json={"max_download_mbit": 64})
            assert throttle.shared.url == url
            assert throttle.shared.mbit == 64
        finally:
            throttle.shared.stop()

    def test_clearing_the_limit_takes_the_limiter_down(self, client):
        client.put("/api/settings", json={"max_download_mbit": 16})
        client.put("/api/settings", json={"max_download_mbit": 0})
        assert throttle.shared.url == ""

    def test_omitted_fields_are_left_alone(self, client):
        client.put("/api/settings", json={"endpoint": "https://mirror"})
        assert client.put("/api/settings", json={"max_workers": 5}).json()["endpoint"] == "https://mirror"

    def test_token_check_without_a_token(self, client):
        assert client.post("/api/settings/test-token", json={}).status_code == 400

    def test_token_check_passes_through(self, client, fake_hub):
        fake_hub()
        assert client.post("/api/settings/test-token", json={"token": "hf_x"}).json()["name"] == "tester"

    def test_a_rejected_token_is_a_400(self, client, fake_hub):
        def reject(token=None):
            raise hub.HubError("Token rejected")

        fake_hub(whoami=reject)
        assert client.post("/api/settings/test-token", json={"token": "hf_x"}).status_code == 400


# ----------------------------------------------------------------------- Hub


class TestHubEndpoints:
    def test_search(self, client, fake_hub):
        fake_hub()
        assert client.get("/api/search?q=llama").json()["results"][0]["repo_id"] == "org/found"

    def test_search_failure_is_a_502(self, client, fake_hub):
        def boom(*_a, **_k):
            raise hub.HubError("Hub down")

        fake_hub(search=boom)
        assert client.get("/api/search?q=x").status_code == 502

    def test_search_rejects_a_bad_type(self, client, fake_hub):
        fake_hub()
        assert client.get("/api/search?q=x&repo_type=nope").status_code == 422

    def test_repo_info_adds_the_local_path(self, client, fake_hub):
        fake_hub()
        body = client.get("/api/repo?repo_id=org/name").json()
        assert body["local_dir"].endswith("models/org/name")
        assert body["local"] is False

    def test_repo_info_uses_the_canonical_id_for_the_path(self, client, fake_hub):
        fake_hub(
            repo_info=lambda repo_id, repo_type="model", revision=None: {
                "repo_id": "google-bert/bert-base-uncased", "repo_type": "model", "sha": "s", "files": [], "total_size": 0
            }
        )
        body = client.get("/api/repo?repo_id=bert-base-uncased").json()
        assert body["local_dir"].endswith("google-bert/bert-base-uncased")

    def test_repo_info_rejects_a_bad_id(self, client):
        assert client.get("/api/repo?repo_id=../etc").status_code == 400

    def test_unknown_repo_is_a_404(self, client, fake_hub):
        def missing(*_a, **_k):
            raise hub.HubError("not found")

        fake_hub(repo_info=missing)
        assert client.get("/api/repo?repo_id=org/name").status_code == 404


# ------------------------------------------------------------------- Library


class TestLibraryEndpoints:
    def test_lists_what_is_on_disk(self, client, repo_factory):
        repo_factory("org/name", files={"a.bin": 100})
        body = client.get("/api/library").json()
        assert body["repos"][0]["repo_id"] == "org/name"
        assert body["data_dir"] == str(config.DATA_DIR)

    def test_reports_leftover_space(self, client, repo_factory):
        repo_factory("org/name", files={"a.bin": 10}, leftovers={"p": 5000})
        assert client.get("/api/library").json()["repos"][0]["leftover"] == 5000

    def test_files_of_a_repo(self, client, repo_factory):
        repo_factory("org/name", files={"a.bin": 1, "b.bin": 2})
        body = client.get("/api/library/files?repo_id=org/name").json()
        assert [f["name"] for f in body["files"]] == ["a.bin", "b.bin"]

    def test_files_rejects_a_bad_id(self, client):
        assert client.get("/api/library/files?repo_id=../etc").status_code == 400

    def test_files_says_when_the_listing_was_cut(self, client, repo_factory, monkeypatch):
        monkeypatch.setattr("app.main.FILE_LIST_LIMIT", 2)
        repo_factory("org/name", files={f"f{i}.bin": 1 for i in range(5)})
        body = client.get("/api/library/files?repo_id=org/name").json()
        assert body["truncated"] is True
        assert len(body["files"]) == 2

    def test_files_is_not_marked_truncated_when_it_fits(self, client, repo_factory):
        repo_factory("org/name", files={"a.bin": 1})
        assert client.get("/api/library/files?repo_id=org/name").json()["truncated"] is False

    def test_delete_removes_it(self, client, repo_factory):
        repo_factory("org/name", files={"a.bin": 100})
        body = client.post("/api/library/delete", json={"repo_id": "org/name"}).json()
        assert body["freed"] == 100
        assert storage.list_repos() == []

    def test_delete_of_something_absent_is_a_404(self, client):
        assert client.post("/api/library/delete", json={"repo_id": "org/absent"}).status_code == 404

    def test_delete_rejects_a_bad_id(self, client):
        assert client.post("/api/library/delete", json={"repo_id": "../etc"}).status_code == 400

    def test_delete_rejects_a_bad_type(self, client):
        assert client.post("/api/library/delete", json={"repo_id": "a/b", "repo_type": "nope"}).status_code == 400


class TestFileDeletion:
    def test_deletes_the_picked_files(self, client, repo_factory):
        repo_factory("org/name", files={"a.bin": 100, "b.bin": 50}, marker={"commit": "abc"})
        body = client.post(
            "/api/library/files/delete", json={"repo_id": "org/name", "files": ["a.bin"]}
        ).json()
        assert body["deleted"] == ["a.bin"]
        assert body["freed"] == 100
        assert [f["name"] for f in storage.repo_files("model", "org/name")] == ["b.bin"]

    def test_the_next_update_leaves_them_out(self, client, repo_factory, fake_hub):
        repo_factory(
            "org/name",
            files={"a.bin": 10, "b.bin": 10},
            marker={"commit": "abc"},
            etags={"a.bin": "sha-a", "b.bin": "sha-b"},
        )
        fake_hub()
        client.post("/api/library/files/delete", json={"repo_id": "org/name", "files": ["a.bin"]})
        client.post("/api/library/update", json={"repo_id": "org/name"})
        job = next(j for j in manager.jobs.values())
        assert job.files == ["b.bin"]

    def test_emptying_a_repo_removes_it(self, client, repo_factory):
        repo_factory("org/name", files={"a.bin": 10}, marker={"commit": "abc"})
        body = client.post(
            "/api/library/files/delete", json={"repo_id": "org/name", "files": ["a.bin"]}
        ).json()
        assert body["removed_repo"] is True
        assert storage.list_repos() == []

    def test_an_empty_list_is_a_400(self, client, repo_factory):
        repo_factory("org/name")
        assert client.post(
            "/api/library/files/delete", json={"repo_id": "org/name", "files": [" "]}
        ).status_code == 400

    @pytest.mark.parametrize(
        "name",
        ["../../etc/passwd", "/etc/passwd", ".cache/x", config.MARKER_NAME, ".", "./", ".TROVE.json"],
    )
    def test_traversal_is_a_400(self, client, repo_factory, name):
        repo_factory("org/name", files={"a.bin": 10})
        assert client.post(
            "/api/library/files/delete", json={"repo_id": "org/name", "files": [name]}
        ).status_code == 400

    def test_a_bad_id_is_a_400(self, client):
        assert client.post(
            "/api/library/files/delete", json={"repo_id": "../etc", "files": ["a.bin"]}
        ).status_code == 400

    def test_a_bad_type_is_a_400(self, client, repo_factory):
        repo_factory("org/name")
        assert client.post(
            "/api/library/files/delete", json={"repo_id": "org/name", "repo_type": "nope", "files": ["a.bin"]}
        ).status_code == 400

    def test_an_unknown_repo_is_a_404(self, client):
        assert client.post(
            "/api/library/files/delete", json={"repo_id": "org/absent", "files": ["a.bin"]}
        ).status_code == 404

    def test_a_repo_being_transferred_is_a_409(self, client, repo_factory, fake_hub):
        # The transfer writes into that folder; pulling files out mid-run would
        # either be undone straight away or break it.
        repo_factory("org/name", files={"a.bin": 10}, marker={"commit": "old"})
        fake_hub()
        client.post("/api/library/update", json={"repo_id": "org/name"})
        assert client.post(
            "/api/library/files/delete", json={"repo_id": "org/name", "files": ["a.bin"]}
        ).status_code == 409

    def test_a_file_already_gone_is_reported(self, client, repo_factory):
        repo_factory("org/name", files={"a.bin": 10})
        body = client.post(
            "/api/library/files/delete", json={"repo_id": "org/name", "files": ["absent.bin"]}
        ).json()
        assert body["missing"] == ["absent.bin"]
        assert body["deleted"] == []


# ------------------------------------------------------------- Update checking


class TestUpdateCheck:
    def test_a_whole_copy_is_compared_by_commit(self, client, repo_factory, fake_hub):
        repo_factory("org/name", marker={"commit": "old-sha"})
        fake_hub()
        entry = client.get("/api/library/updates").json()["repos"][0]
        assert entry["outdated"] is True
        assert entry["remote_commit"] == "remote-sha"

    def test_a_matching_commit_is_up_to_date(self, client, repo_factory, fake_hub):
        repo_factory("org/name", marker={"commit": "remote-sha"})
        fake_hub()
        assert client.get("/api/library/updates").json()["outdated"] == 0

    def test_a_copy_without_a_record_is_skipped(self, client, repo_factory, fake_hub):
        repo_factory("org/name")
        fake_hub()
        entry = client.get("/api/library/updates").json()["repos"][0]
        assert entry["outdated"] is False
        assert "No download record" in entry["skipped"]

    def test_a_partial_copy_is_compared_file_by_file(self, client, repo_factory, fake_hub):
        repo_factory(
            "org/name",
            marker={"commit": "old", "files": ["a.bin", "b.bin"]},
            etags={"a.bin": "sha-a", "b.bin": "sha-b"},
        )
        fake_hub(
            file_state=lambda *_a, **_k: {
                "repo_id": "org/name", "sha": "new", "files": {"a.bin": "sha-a", "b.bin": "CHANGED"}
            }
        )
        entry = client.get("/api/library/updates").json()["repos"][0]
        assert entry["outdated"] is True
        assert entry["changed_files"] == ["b.bin"]
        assert entry["tracked_files"] == 2

    def test_a_partial_copy_ignores_files_it_never_took(self, client, repo_factory, fake_hub):
        repo_factory("org/name", marker={"commit": "old", "files": ["a.bin"]}, etags={"a.bin": "sha-a"})
        fake_hub(
            file_state=lambda *_a, **_k: {
                "repo_id": "org/name", "sha": "new", "files": {"a.bin": "sha-a", "other.bin": "whatever"}
            }
        )
        assert client.get("/api/library/updates").json()["repos"][0]["outdated"] is False

    def test_a_partial_copy_without_per_file_records_is_skipped(self, client, repo_factory, fake_hub):
        repo_factory("org/name", marker={"commit": "old", "files": ["a.bin"]})
        fake_hub()
        entry = client.get("/api/library/updates").json()["repos"][0]
        assert "predates" in entry["skipped"]

    def test_hub_failures_are_reported_per_repo(self, client, repo_factory, fake_hub):
        repo_factory("org/name", marker={"commit": "old"})

        def boom(*_a, **_k):
            raise hub.HubError("gated")

        fake_hub(resolve=boom)
        entry = client.get("/api/library/updates").json()["repos"][0]
        assert entry["error"] == "gated"
        assert entry["outdated"] is False


class TestUpdateQueueing:
    def test_queues_a_named_repo(self, client, repo_factory, fake_hub):
        repo_factory("org/name", marker={"commit": "old"})
        fake_hub()
        assert client.post("/api/library/update", json={"repo_id": "org/name"}).json()["queued"] == ["org/name"]

    def test_a_partial_copy_stays_partial(self, client, repo_factory, fake_hub):
        repo_factory("org/name", marker={"commit": "old", "files": ["a.bin"], "ignore_patterns": ["*.md"]})
        fake_hub()
        client.post("/api/library/update", json={"repo_id": "org/name"})
        job = next(j for j in manager.jobs.values())
        assert job.files == ["a.bin"]
        assert job.ignore_patterns == ["*.md"]

    def test_a_repo_not_in_the_library_is_a_404(self, client, fake_hub):
        fake_hub()
        assert client.post("/api/library/update", json={"repo_id": "org/absent"}).status_code == 404

    def test_a_bad_id_is_a_400(self, client):
        assert client.post("/api/library/update", json={"repo_id": "../etc"}).status_code == 400

    def test_queueing_twice_is_a_409(self, client, repo_factory, fake_hub):
        repo_factory("org/name", marker={"commit": "old"})
        fake_hub()
        client.post("/api/library/update", json={"repo_id": "org/name"})
        assert client.post("/api/library/update", json={"repo_id": "org/name"}).status_code == 409

    def test_all_outdated_queues_only_those(self, client, repo_factory, fake_hub):
        repo_factory("org/stale", marker={"commit": "old"})
        repo_factory("org/current", marker={"commit": "remote-sha"})
        fake_hub()
        assert client.post("/api/library/update", json={"all_outdated": True}).json()["queued"] == ["org/stale"]


# ---------------------------------------------------------------------- Jobs


class TestDownloadEndpoint:
    def test_queues_a_job(self, client, fake_hub):
        fake_hub()
        body = client.post("/api/jobs/download", json={"repo_id": "org/name"}).json()
        assert body["repo_id"] == "org/name"
        assert body["kind"] == "download"

    def test_uses_the_canonical_id_and_says_so(self, client, fake_hub):
        fake_hub(resolve=lambda *_a, **_k: {"repo_id": "google-bert/bert-base-uncased", "sha": "s"})
        body = client.post("/api/jobs/download", json={"repo_id": "bert-base-uncased"}).json()
        assert body["repo_id"] == "google-bert/bert-base-uncased"
        assert any("redirects to" in entry["msg"] for entry in body["logs"])

    def test_rejects_a_bad_id(self, client):
        assert client.post("/api/jobs/download", json={"repo_id": "../etc"}).status_code == 400

    def test_rejects_a_bad_type(self, client):
        assert client.post("/api/jobs/download", json={"repo_id": "a/b", "repo_type": "nope"}).status_code == 400

    def test_an_unknown_repo_is_a_404(self, client, fake_hub):
        def missing(*_a, **_k):
            raise hub.HubError("not found")

        fake_hub(resolve=missing)
        assert client.post("/api/jobs/download", json={"repo_id": "org/name"}).status_code == 404

    def test_the_same_repo_twice_is_a_409(self, client, fake_hub):
        fake_hub()
        client.post("/api/jobs/download", json={"repo_id": "org/name"})
        assert client.post("/api/jobs/download", json={"repo_id": "org/name"}).status_code == 409

    def test_blank_selections_are_dropped(self, client, fake_hub):
        fake_hub()
        body = client.post(
            "/api/jobs/download",
            json={"repo_id": "org/name", "files": ["a.bin", "  "], "allow_patterns": [""]},
        ).json()
        assert body["files"] == ["a.bin"]
        assert body["allow_patterns"] == []


class TestUploadEndpoint:
    def test_rejects_a_path_outside_the_data_dir(self, client):
        config.settings.update({"hf_token": "hf_x"})
        body = {"repo_id": "org/name", "path": "/etc"}
        assert client.post("/api/jobs/upload", json=body).status_code == 400

    def test_missing_folder_is_a_404(self, client):
        config.settings.update({"hf_token": "hf_x"})
        body = {"repo_id": "org/name", "path": str(config.DATA_DIR / "models" / "absent")}
        assert client.post("/api/jobs/upload", json=body).status_code == 404

    def test_without_a_token_it_refuses(self, client, repo_factory):
        path = repo_factory("org/name")
        assert client.post("/api/jobs/upload", json={"repo_id": "org/name", "path": str(path)}).status_code == 400

    def test_queues_with_a_token(self, client, repo_factory):
        path = repo_factory("org/name")
        config.settings.update({"hf_token": "hf_x"})
        body = client.post("/api/jobs/upload", json={"repo_id": "org/name", "path": str(path)}).json()
        assert body["kind"] == "upload"
        assert body["src"] == str(path)


class TestJobControl:
    def test_listing(self, client, fake_hub):
        fake_hub()
        client.post("/api/jobs/download", json={"repo_id": "org/name"})
        assert len(client.get("/api/jobs").json()["jobs"]) == 1

    def test_cancel(self, client, fake_hub):
        fake_hub()
        job_id = client.post("/api/jobs/download", json={"repo_id": "org/name"}).json()["id"]
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200

    def test_cancel_of_an_unknown_job_is_a_404(self, client):
        assert client.post("/api/jobs/nope/cancel").status_code == 404

    def test_retry_of_an_unknown_job_is_a_404(self, client):
        assert client.post("/api/jobs/nope/retry").status_code == 404

    def test_delete_of_an_unknown_job_is_a_404(self, client):
        assert client.delete("/api/jobs/nope").status_code == 404

    def test_delete_removes_it(self, client, fake_hub):
        fake_hub()
        job_id = client.post("/api/jobs/download", json={"repo_id": "org/name"}).json()["id"]
        assert client.delete(f"/api/jobs/{job_id}").json() == {"ok": True}
        assert client.get("/api/jobs").json()["jobs"] == []

    def test_clear_reports_how_many_went(self, client):
        assert client.post("/api/jobs/clear").json() == {"removed": 0}


# ----------------------------------------------------------------- WebSocket


class TestWebSocket:
    def test_refuses_without_a_session(self, anon):
        from starlette.websockets import WebSocketDisconnect

        # The socket is closed rather than refused, so the rejection shows up on
        # the first read — nothing from the queue is ever delivered.
        with pytest.raises(WebSocketDisconnect) as refused:
            with anon.websocket_connect("/ws") as ws:
                ws.receive()
        assert refused.value.code == 4401

    def test_sends_the_queue_on_connect(self, client, fake_hub):
        fake_hub()
        client.post("/api/jobs/download", json={"repo_id": "org/name"})
        with client.websocket_connect("/ws") as ws:
            message = ws.receive_json()
        assert message["type"] == "jobs"
        assert message["jobs"][0]["repo_id"] == "org/name"


# ------------------------------------------------------------------ Frontend


class TestFrontend:
    def test_index_is_served_with_versioned_assets(self, anon):
        html = anon.get("/").text
        assert f"/css/style.css?v={__version__}" in html
        assert f"/js/app.js?v={__version__}" in html

    def test_the_page_itself_is_not_cached(self, anon):
        assert anon.get("/").headers["cache-control"] == "no-store"

    def test_index_html_goes_through_the_rewrite_too(self, anon):
        assert f"?v={__version__}" in anon.get("/index.html").text

    def test_unknown_pages_fall_back_to_the_app(self, anon):
        assert anon.get("/some/deep/link").status_code == 200

    def test_unknown_api_paths_stay_json(self, anon):
        response = anon.get("/api/nothing-here")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")

    def test_static_assets_are_served(self, anon):
        assert anon.get("/css/style.css").status_code == 200
        assert anon.get("/js/app.js").status_code == 200


class TestJobsAreReallyStubbed:
    def test_the_stub_worker_runs_instead_of_the_real_one(self, client, fake_hub):
        """Guards the fixture itself: a leak here would hit the network."""
        fake_hub()
        client.post("/api/jobs/download", json={"repo_id": "org/name"})
        job = next(iter(manager.jobs.values()))
        command = manager._build_command(job)
        assert "app.worker" not in " ".join(command)
        assert job.status in (RUNNING, "queued")

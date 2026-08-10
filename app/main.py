"""The FastAPI app: REST API, WebSocket live updates, and serving the UI."""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import BaseModel, Field

from . import __version__, hub, storage
from .config import (
    DATA_DIR,
    REPO_TYPES,
    ensure_dirs,
    inside_data_dir,
    local_dir_for,
    session_secret,
    settings,
    valid_repo_id,
)
from .jobs import ACTIVE, manager

STATIC_DIR = Path(__file__).parent / "static"
COOKIE_NAME = "hfd_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 14

_serializer = URLSafeTimedSerializer(session_secret(), salt="trove-session")


def auth_required() -> bool:
    return bool(os.getenv("UI_PASSWORD", "").strip())


def _valid_session(token: str | None) -> bool:
    if not auth_required():
        return True
    if not token:
        return False
    try:
        _serializer.loads(token, max_age=SESSION_MAX_AGE)
    except BadSignature:
        return False
    except Exception:  # noqa: BLE001 - abgelaufene Signatur o.ae.
        return False
    return True


async def require_auth(request: Request) -> None:
    if not _valid_session(request.cookies.get(COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="Not signed in")


# ------------------------------------------------------------------ WebSocket


class EventHub:
    """Fans job events out to every open browser tab."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_hub = EventHub()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_dirs()
    manager.bind(ws_hub.broadcast)
    await manager.start()
    yield
    await manager.shutdown()


app = FastAPI(title="Trove", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------- Bodies


class LoginBody(BaseModel):
    password: str = ""


class SettingsBody(BaseModel):
    hf_token: str | None = None
    endpoint: str | None = None
    max_concurrent: int | None = None
    max_workers: int | None = None
    auto_clear_done: bool | None = None


class TokenTestBody(BaseModel):
    token: str | None = None


class DownloadBody(BaseModel):
    repo_id: str
    repo_type: str = "model"
    revision: str = ""
    # Individually picked files (exact names from the repo listing).
    files: list[str] = Field(default_factory=list)
    allow_patterns: list[str] = Field(default_factory=list)
    ignore_patterns: list[str] = Field(default_factory=list)


class UploadBody(BaseModel):
    repo_id: str
    path: str
    repo_type: str = "model"
    private: bool = False
    create_repo: bool = True
    commit_message: str = ""
    revision: str = ""
    allow_patterns: list[str] = Field(default_factory=list)
    ignore_patterns: list[str] = Field(default_factory=list)


class DeleteBody(BaseModel):
    repo_id: str
    repo_type: str = "model"


class UpdateBody(BaseModel):
    repo_id: str = ""
    repo_type: str = "model"
    # Queue every repo the check found outdated, instead of a named one.
    all_outdated: bool = False


# ------------------------------------------------------------------ Basics/auth


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "jobs": manager.stats()}


@app.get("/api/session")
async def session_state(request: Request) -> dict[str, Any]:
    return {
        "auth_required": auth_required(),
        "authenticated": _valid_session(request.cookies.get(COOKIE_NAME)),
        "version": __version__,
    }


@app.post("/api/login")
async def login(body: LoginBody, response: Response) -> dict[str, Any]:
    expected = os.getenv("UI_PASSWORD", "").strip()
    if not expected:
        return {"authenticated": True}
    # Constant-time compare, so the password cannot be probed by timing.
    if not secrets.compare_digest(body.password, expected):
        await asyncio.sleep(0.6)
        raise HTTPException(status_code=401, detail="Wrong password")

    token = _serializer.dumps({"ts": time.time()})
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return {"authenticated": True}


@app.post("/api/logout")
async def logout(response: Response) -> dict[str, Any]:
    response.delete_cookie(COOKIE_NAME)
    return {"authenticated": False}


# --------------------------------------------------------------------- Settings


@app.get("/api/settings", dependencies=[Depends(require_auth)])
async def get_settings() -> dict[str, Any]:
    return settings.public()


@app.put("/api/settings", dependencies=[Depends(require_auth)])
async def put_settings(body: SettingsBody) -> dict[str, Any]:
    values = {k: v for k, v in body.model_dump().items() if v is not None}
    settings.update(values)
    # Apply a changed concurrency limit to waiting jobs right away.
    await manager._pump()  # noqa: SLF001
    return settings.public()


@app.post("/api/settings/test-token", dependencies=[Depends(require_auth)])
async def test_token(body: TokenTestBody) -> dict[str, Any]:
    token = (body.token or "").strip() or (settings.get("hf_token") or "")
    if not token:
        raise HTTPException(status_code=400, detail="No token stored")
    try:
        return await asyncio.to_thread(hub.whoami, token)
    except hub.HubError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/disk", dependencies=[Depends(require_auth)])
async def disk() -> dict[str, Any]:
    usage = await asyncio.to_thread(storage.disk_usage)
    return {**usage, "path": str(DATA_DIR)}


# ------------------------------------------------------------------------ Hub


@app.get("/api/search", dependencies=[Depends(require_auth)])
async def search(
    q: str = "",
    repo_type: str = Query("model", pattern="^(model|dataset|space)$"),
    limit: int = Query(30, ge=1, le=100),
) -> dict[str, Any]:
    try:
        results = await asyncio.to_thread(hub.search, q, repo_type, limit)
    except hub.HubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"results": results}


@app.get("/api/repo", dependencies=[Depends(require_auth)])
async def repo(
    repo_id: str,
    repo_type: str = Query("model", pattern="^(model|dataset|space)$"),
    revision: str = "",
) -> dict[str, Any]:
    if not valid_repo_id(repo_id):
        raise HTTPException(status_code=400, detail="Invalid repo ID")
    try:
        info = await asyncio.to_thread(hub.repo_info, repo_id, repo_type, revision or None)
    except hub.HubError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Use the canonical id the Hub resolved to, not what the caller typed:
    # otherwise a legacy alias shows a target folder the download never uses.
    info["local_dir"] = str(local_dir_for(repo_type, info["repo_id"]))
    info["local"] = Path(info["local_dir"]).is_dir()
    return info


# --------------------------------------------------------------------- Library


@app.get("/api/library", dependencies=[Depends(require_auth)])
async def library(
    repo_type: str = Query("", pattern="^(model|dataset|space)?$"),
    refresh: bool = False,
) -> dict[str, Any]:
    repos = await asyncio.to_thread(storage.list_repos, repo_type or None, refresh)
    return {"repos": repos, "data_dir": str(DATA_DIR)}


@app.get("/api/library/files", dependencies=[Depends(require_auth)])
async def library_files(
    repo_id: str,
    repo_type: str = Query("model", pattern="^(model|dataset|space)$"),
) -> dict[str, Any]:
    if not valid_repo_id(repo_id):
        raise HTTPException(status_code=400, detail="Invalid repo ID")
    files = await asyncio.to_thread(storage.repo_files, repo_type, repo_id)
    return {"files": files, "path": str(local_dir_for(repo_type, repo_id))}


# How many repos to ask the Hub about at once. The check is one API call per
# repo, so a large library would otherwise crawl — or look like a scraper.
UPDATE_CHECK_CONCURRENCY = 8


async def _check_updates() -> list[dict[str, Any]]:
    """Compare the commit each repo was downloaded at against the Hub."""
    repos = await asyncio.to_thread(storage.list_repos, None, False)
    limit = asyncio.Semaphore(UPDATE_CHECK_CONCURRENCY)

    async def check(repo: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "repo_id": repo["repo_id"],
            "repo_type": repo["repo_type"],
            "revision": repo["revision"],
            "commit": repo["commit"],
            "remote_commit": "",
            "outdated": False,
            "skipped": "",
            "error": "",
        }
        # A partial copy cannot be compared: its folder holds a hand-picked
        # subset, so a differing commit says nothing about those files.
        if repo["partial"]:
            entry["skipped"] = "Only selected files were downloaded."
            return entry
        if not repo["commit"]:
            entry["skipped"] = "No download record — the commit is unknown."
            return entry
        async with limit:
            try:
                resolved = await asyncio.to_thread(
                    hub.resolve, repo["repo_id"], repo["repo_type"], repo["revision"] or None
                )
            except hub.HubError as exc:
                entry["error"] = str(exc)
                return entry
        entry["remote_commit"] = resolved["sha"]
        entry["outdated"] = bool(resolved["sha"]) and resolved["sha"] != repo["commit"]
        return entry

    return list(await asyncio.gather(*(check(repo) for repo in repos)))


@app.get("/api/library/updates", dependencies=[Depends(require_auth)])
async def library_updates() -> dict[str, Any]:
    checked = await _check_updates()
    return {
        "repos": checked,
        "outdated": sum(1 for entry in checked if entry["outdated"]),
        "checked_at": time.time(),
    }


@app.post("/api/library/update", dependencies=[Depends(require_auth)])
async def library_update(body: UpdateBody) -> dict[str, Any]:
    if body.all_outdated:
        targets = [entry for entry in await _check_updates() if entry["outdated"]]
    else:
        if not valid_repo_id(body.repo_id):
            raise HTTPException(status_code=400, detail="Invalid repo ID")
        if body.repo_type not in REPO_TYPES:
            raise HTTPException(status_code=400, detail="Invalid repo type")
        repos = await asyncio.to_thread(storage.list_repos, body.repo_type, False)
        repo = next((r for r in repos if r["repo_id"] == body.repo_id), None)
        if repo is None:
            raise HTTPException(status_code=404, detail=f"{body.repo_id} is not in the library")
        if repo["partial"]:
            raise HTTPException(
                status_code=400,
                detail="Only selected files were downloaded — use Refresh from Hub to fetch that same selection.",
            )
        targets = [dict(repo, remote_commit="")]

    queued, skipped = [], []
    for entry in targets:
        active = any(
            job.kind == "download"
            and job.repo_id == entry["repo_id"]
            and job.repo_type == entry["repo_type"]
            and job.status in ACTIVE
            for job in manager.jobs.values()
        )
        if active:
            skipped.append(entry["repo_id"])
            continue
        await manager.add_download(
            repo_id=entry["repo_id"],
            repo_type=entry["repo_type"],
            revision=entry.get("revision") or "",
        )
        queued.append(entry["repo_id"])

    if not body.all_outdated and not queued:
        raise HTTPException(status_code=409, detail=f"{body.repo_id} is already queued")
    return {"queued": queued, "skipped": skipped}


@app.post("/api/library/delete", dependencies=[Depends(require_auth)])
async def library_delete(body: DeleteBody) -> dict[str, Any]:
    if not valid_repo_id(body.repo_id):
        raise HTTPException(status_code=400, detail="Invalid repo ID")
    if body.repo_type not in REPO_TYPES:
        raise HTTPException(status_code=400, detail="Invalid repo type")
    try:
        result = await asyncio.to_thread(storage.delete_repo, body.repo_type, body.repo_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc
    await ws_hub.broadcast({"type": "library"})
    return result


# ----------------------------------------------------------------------- Jobs


@app.get("/api/jobs", dependencies=[Depends(require_auth)])
async def list_jobs() -> dict[str, Any]:
    return {"jobs": manager.snapshot()}


@app.post("/api/jobs/download", dependencies=[Depends(require_auth)])
async def create_download(body: DownloadBody) -> dict[str, Any]:
    if not valid_repo_id(body.repo_id):
        raise HTTPException(status_code=400, detail="Invalid repo ID — expected org/name")
    if body.repo_type not in REPO_TYPES:
        raise HTTPException(status_code=400, detail="Invalid repo type")

    # Check against the Hub first: it catches typos and gated repos before a job
    # ever runs, and yields the canonical id (see hub.resolve).
    try:
        resolved = await asyncio.to_thread(hub.resolve, body.repo_id, body.repo_type, body.revision or None)
    except hub.HubError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    repo_id = resolved["repo_id"]

    duplicate = next(
        (
            j
            for j in manager.jobs.values()
            if j.kind == "download"
            and j.repo_id == repo_id
            and j.repo_type == body.repo_type
            and j.status in ACTIVE
        ),
        None,
    )
    if duplicate:
        raise HTTPException(status_code=409, detail=f"{repo_id} is already queued")

    job = await manager.add_download(
        repo_id=repo_id,
        repo_type=body.repo_type,
        revision=body.revision.strip(),
        files=[f for f in body.files if f.strip()],
        allow_patterns=[p for p in body.allow_patterns if p.strip()],
        ignore_patterns=[p for p in body.ignore_patterns if p.strip()],
    )
    if repo_id != body.repo_id:
        job.logs.append(
            {"t": time.time(), "level": "info", "msg": f"'{body.repo_id}' redirects to '{repo_id}' — using the canonical name."}
        )
    return job.to_dict()


@app.post("/api/jobs/upload", dependencies=[Depends(require_auth)])
async def create_upload(body: UploadBody) -> dict[str, Any]:
    if not valid_repo_id(body.repo_id):
        raise HTTPException(status_code=400, detail="Invalid repo ID — expected org/name")
    if body.repo_type not in REPO_TYPES:
        raise HTTPException(status_code=400, detail="Invalid repo type")

    src = Path(body.path)
    if not inside_data_dir(src):
        raise HTTPException(status_code=400, detail=f"Path must sit inside {DATA_DIR}")
    if not src.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {src}")
    if not (settings.get("hf_token") or "").strip():
        raise HTTPException(status_code=400, detail="No token stored — uploads need write access")

    job = await manager.add_upload(
        repo_id=body.repo_id,
        src=str(src.resolve()),
        repo_type=body.repo_type,
        private=body.private,
        create_repo=body.create_repo,
        commit_message=body.commit_message.strip(),
        revision=body.revision.strip(),
        allow_patterns=[p for p in body.allow_patterns if p.strip()],
        ignore_patterns=[p for p in body.ignore_patterns if p.strip()],
    )
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
async def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        job = await manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return job.to_dict()


@app.post("/api/jobs/{job_id}/retry", dependencies=[Depends(require_auth)])
async def retry_job(job_id: str) -> dict[str, Any]:
    try:
        job = await manager.retry(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return job.to_dict()


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_auth)])
async def delete_job(job_id: str) -> dict[str, Any]:
    try:
        await manager.remove(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return {"ok": True}


@app.post("/api/jobs/clear", dependencies=[Depends(require_auth)])
async def clear_jobs() -> dict[str, Any]:
    return {"removed": await manager.clear_finished()}


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    if not _valid_session(ws.cookies.get(COOKIE_NAME)):
        await ws.close(code=4401)
        return

    await ws_hub.connect(ws)
    try:
        await ws.send_json({"type": "jobs", "jobs": manager.snapshot()})
        while True:
            # The client only sends pings; everything else arrives by broadcast.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(ws.receive_text(), timeout=45)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        ws_hub.disconnect(ws)


# ------------------------------------------------------------------- Frontend


def _index_html() -> Response:
    """Serve index.html with versioned asset links.

    Without this the browser hangs on to CSS and JS from an older version and
    shows a mixture of both after an update. The page itself is never cached;
    the assets below it can be cached indefinitely.
    """
    html = (STATIC_DIR / "index.html").read_text()
    html = html.replace("/css/style.css", f"/css/style.css?v={__version__}")
    html = html.replace("/js/app.js", f"/js/app.js?v={__version__}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


# /index.html needs its own route: the static mount below would otherwise serve
# the file verbatim and skip the rewrite, handing out stale asset links.
@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def index() -> Response:
    return _index_html()


@app.exception_handler(404)
async def not_found(request: Request, exc: Any) -> Response:
    # Keep the original message for the API — otherwise every 404 raised in a
    # handler collapses into a useless "Not found".
    if request.url.path.startswith(("/api", "/ws")):
        return JSONResponse({"detail": getattr(exc, "detail", "Not found")}, status_code=404)
    return _index_html()


app.mount("/", StaticFiles(directory=STATIC_DIR, html=False), name="static")

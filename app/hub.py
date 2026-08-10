"""A thin layer over huggingface_hub for search, repo info and token checks.

Every call in here blocks (httpx sync), so the API layer runs them through
asyncio.to_thread.
"""

from __future__ import annotations

from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from .config import settings


class HubError(Exception):
    """An error whose message is safe to show in the interface as-is."""


def api(token: str | None = None) -> HfApi:
    return HfApi(
        token=token if token is not None else (settings.get("hf_token") or None),
        endpoint=settings.get("endpoint") or None,
        library_name="trove",
    )


def _ts(value: Any) -> float:
    try:
        return value.timestamp()
    except AttributeError:
        return 0.0


def whoami(token: str | None = None) -> dict[str, Any]:
    try:
        info = api(token).whoami()
    except HfHubHTTPError as exc:
        raise HubError(f"Token rejected, or the Hub is unreachable: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - any kind of network failure
        raise HubError(str(exc)) from exc

    orgs = [o.get("name") for o in info.get("orgs", []) if isinstance(o, dict)]
    return {"name": info.get("name", ""), "orgs": [o for o in orgs if o]}


def search(query: str, repo_type: str = "model", limit: int = 30) -> list[dict[str, Any]]:
    client = api()
    query = (query or "").strip()
    try:
        # As of huggingface_hub 1.x, `sort` is descending and the `direction`
        # argument is gone.
        if repo_type == "dataset":
            items = client.list_datasets(search=query or None, limit=limit, sort="downloads")
        elif repo_type == "space":
            items = client.list_spaces(search=query or None, limit=limit, sort="likes")
        else:
            items = client.list_models(search=query or None, limit=limit, sort="downloads")
        results = list(items)
    except Exception as exc:  # noqa: BLE001
        raise HubError(f"Search failed: {exc}") from exc

    return [
        {
            "repo_id": item.id,
            "repo_type": repo_type,
            "author": getattr(item, "author", "") or item.id.split("/")[0],
            "downloads": getattr(item, "downloads", 0) or 0,
            "likes": getattr(item, "likes", 0) or 0,
            "pipeline_tag": getattr(item, "pipeline_tag", None) or "",
            "library": getattr(item, "library_name", None) or "",
            "private": bool(getattr(item, "private", False)),
            "gated": getattr(item, "gated", False) or False,
            "updated_at": _ts(getattr(item, "last_modified", None)),
            "tags": [t for t in (getattr(item, "tags", None) or []) if ":" not in t][:6],
        }
        for item in results
    ]


def _fetch_info(
    repo_id: str,
    repo_type: str,
    revision: str | None,
    files_metadata: bool = False,
) -> Any:
    """repo_info with the Hub's errors turned into readable ones."""
    try:
        return api().repo_info(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision or None,
            files_metadata=files_metadata,
        )
    except GatedRepoError as exc:
        raise HubError(
            "This repo is gated. Request access on huggingface.co and store a token in Settings."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise HubError(f"Repo '{repo_id}' not found, or private and your token has no access.") from exc
    except RevisionNotFoundError as exc:
        raise HubError(f"Revision '{revision}' does not exist in '{repo_id}'.") from exc
    except Exception as exc:  # noqa: BLE001
        raise HubError(str(exc)) from exc


def resolve(repo_id: str, repo_type: str = "model", revision: str | None = None) -> dict[str, Any]:
    """Check that a repo exists and return its canonical id.

    Legacy short names such as `bert-base-uncased` redirect to
    `google-bert/...`. The Hub follows that redirect but the Xet token endpoint
    does not, so a download started under the short name dies partway through
    with a 404. Resolving at queue time avoids it.
    """
    info = _fetch_info(repo_id, repo_type, revision)
    return {"repo_id": info.id or repo_id, "sha": info.sha or ""}


def file_state(repo_id: str, repo_type: str = "model", revision: str | None = None) -> dict[str, Any]:
    """The commit plus one content hash per file.

    The hash is the git blob id for plain files and the LFS sha256 for large
    ones — which is exactly what huggingface_hub writes into its per-file
    `.metadata` when downloading, so local and remote compare directly.
    """
    info = _fetch_info(repo_id, repo_type, revision, files_metadata=True)
    files = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        files[sibling.rfilename] = (getattr(lfs, "sha256", "") if lfs else "") or sibling.blob_id or ""
    return {"repo_id": info.id or repo_id, "sha": info.sha or "", "files": files}


def repo_info(repo_id: str, repo_type: str = "model", revision: str | None = None) -> dict[str, Any]:
    info = _fetch_info(repo_id, repo_type, revision, files_metadata=True)

    files = []
    total = 0
    for sibling in info.siblings or []:
        size = sibling.size or (sibling.lfs.size if getattr(sibling, "lfs", None) else 0) or 0
        total += size
        files.append({"name": sibling.rfilename, "size": size})
    files.sort(key=lambda f: f["name"])

    return {
        "repo_id": info.id,
        "repo_type": repo_type,
        "sha": info.sha or "",
        "revision": revision or "main",
        "private": bool(getattr(info, "private", False)),
        "gated": getattr(info, "gated", False) or False,
        "downloads": getattr(info, "downloads", 0) or 0,
        "likes": getattr(info, "likes", 0) or 0,
        "pipeline_tag": getattr(info, "pipeline_tag", None) or "",
        "tags": getattr(info, "tags", None) or [],
        "updated_at": _ts(getattr(info, "last_modified", None)),
        "total_size": total,
        "files": files,
    }


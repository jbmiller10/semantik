"""HTTP client for calling Semantik WebUI APIs from the MCP server."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemantikAPIError(Exception):
    """Raised when a Semantik WebUI API request fails."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class SemantikAPIClient:
    """Async HTTP client for Semantik WebUI API."""

    def __init__(self, webui_url: str, auth_token: str) -> None:
        self.base_url = webui_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=30.0,
        )

    async def get_profiles(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        params = {"enabled": "true"} if enabled_only else None
        response = await self._client.get("/api/v2/mcp/profiles", params=params)
        self._raise_for_status(response, "GET", "/api/v2/mcp/profiles")
        payload = response.json()
        profiles = payload.get("profiles")
        if not isinstance(profiles, list):
            raise SemantikAPIError("Unexpected response from /api/v2/mcp/profiles (missing 'profiles' list)")
        return profiles

    async def search(self, **params: Any) -> dict[str, Any]:
        response = await self._client.post("/api/v2/search", json=params)
        self._raise_for_status(response, "POST", "/api/v2/search")
        data = response.json()
        if not isinstance(data, dict):
            raise SemantikAPIError("Unexpected response from /api/v2/search (expected object)")
        return data

    async def list_documents(
        self,
        collection_id: str,
        *,
        page: int = 1,
        per_page: int = 50,
        status: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if status is not None:
            params["status"] = status
        response = await self._client.get(f"/api/v2/collections/{collection_id}/documents", params=params)
        self._raise_for_status(response, "GET", f"/api/v2/collections/{collection_id}/documents")
        data = response.json()
        if not isinstance(data, dict):
            raise SemantikAPIError("Unexpected response from list documents endpoint (expected object)")
        return data

    async def get_document(self, collection_id: str, document_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/api/v2/collections/{collection_id}/documents/{document_id}")
        self._raise_for_status(response, "GET", f"/api/v2/collections/{collection_id}/documents/{document_id}")
        data = response.json()
        if not isinstance(data, dict):
            raise SemantikAPIError("Unexpected response from get document endpoint (expected object)")
        return data

    async def get_document_content(self, collection_id: str, document_id: str) -> tuple[bytes, str | None]:
        response = await self._client.get(f"/api/v2/collections/{collection_id}/documents/{document_id}/content")
        self._raise_for_status(response, "GET", f"/api/v2/collections/{collection_id}/documents/{document_id}/content")
        return response.content, response.headers.get("content-type")

    async def get_chunk(self, collection_id: str, chunk_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/api/v2/chunking/collections/{collection_id}/chunks/{chunk_id}")
        self._raise_for_status(response, "GET", f"/api/v2/chunking/collections/{collection_id}/chunks/{chunk_id}")
        data = response.json()
        if not isinstance(data, dict):
            raise SemantikAPIError("Unexpected response from get chunk endpoint (expected object)")
        return data

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _raise_for_status(response: httpx.Response, method: str, path: str) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail: Any
            try:
                detail = response.json()
            except Exception as json_err:
                logger.debug("Failed to parse error response as JSON: %s", json_err)
                detail = response.text
            raise SemantikAPIError(f"{method} {path} failed ({response.status_code}): {detail}") from exc
        except httpx.RequestError as exc:
            raise SemantikAPIError(f"{method} {path} failed (request error): {exc}") from exc

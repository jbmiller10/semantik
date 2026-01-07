"""HTTP client for calling Semantik WebUI APIs from the MCP server."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {502, 503, 504}

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds

T = TypeVar("T")


@dataclass(frozen=True)
class SemantikAPIError(Exception):
    """Raised when a Semantik WebUI API request fails."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class SemantikAPIClient:
    """Async HTTP client for Semantik WebUI API."""

    def __init__(
        self,
        webui_url: str,
        auth_token: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
    ) -> None:
        self.base_url = webui_url.rstrip("/")
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=30.0,
        )

    async def _with_retry(
        self,
        operation: Callable[[], Any],
        method: str,
        path: str,
    ) -> httpx.Response:
        """Execute an HTTP operation with retry logic for transient failures.

        Retries on:
        - 502, 503, 504 status codes (server errors)
        - Connection timeouts and network errors

        Uses exponential backoff: 1s, 2s, 4s between retries.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await operation()

                # Check if we should retry based on status code
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                    delay = self._base_delay * (2**attempt)
                    logger.debug(
                        "Retrying %s %s after %d status (attempt %d/%d, delay %.1fs)",
                        method,
                        path,
                        response.status_code,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                return response

            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = self._base_delay * (2**attempt)
                    logger.debug(
                        "Retrying %s %s after timeout (attempt %d/%d, delay %.1fs)",
                        method,
                        path,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise SemantikAPIError(f"{method} {path} failed (timeout): {exc}") from exc

            except httpx.ConnectError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = self._base_delay * (2**attempt)
                    logger.debug(
                        "Retrying %s %s after connection error (attempt %d/%d, delay %.1fs)",
                        method,
                        path,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise SemantikAPIError(f"{method} {path} failed (connection error): {exc}") from exc

        # Should not reach here, but handle edge case
        raise SemantikAPIError(f"{method} {path} failed after {self._max_retries} retries: {last_error}")

    async def get_profiles(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        params = {"enabled": "true"} if enabled_only else None
        path = "/api/v2/mcp/profiles"
        response = await self._with_retry(
            lambda: self._client.get(path, params=params),
            "GET",
            path,
        )
        self._raise_for_status(response, "GET", path)
        payload = response.json()
        profiles = payload.get("profiles")
        if not isinstance(profiles, list):
            raise SemantikAPIError("Unexpected response from /api/v2/mcp/profiles (missing 'profiles' list)")
        return profiles

    async def search(self, **params: Any) -> dict[str, Any]:
        path = "/api/v2/search"
        response = await self._with_retry(
            lambda: self._client.post(path, json=params),
            "POST",
            path,
        )
        self._raise_for_status(response, "POST", path)
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
        query_params: dict[str, Any] = {"page": page, "per_page": per_page}
        if status is not None:
            query_params["status"] = status
        path = f"/api/v2/collections/{collection_id}/documents"
        response = await self._with_retry(
            lambda: self._client.get(path, params=query_params),
            "GET",
            path,
        )
        self._raise_for_status(response, "GET", path)
        data = response.json()
        if not isinstance(data, dict):
            raise SemantikAPIError("Unexpected response from list documents endpoint (expected object)")
        return data

    async def get_document(self, collection_id: str, document_id: str) -> dict[str, Any]:
        path = f"/api/v2/collections/{collection_id}/documents/{document_id}"
        response = await self._with_retry(
            lambda: self._client.get(path),
            "GET",
            path,
        )
        self._raise_for_status(response, "GET", path)
        data = response.json()
        if not isinstance(data, dict):
            raise SemantikAPIError("Unexpected response from get document endpoint (expected object)")
        return data

    async def get_document_content(self, collection_id: str, document_id: str) -> tuple[bytes, str | None]:
        path = f"/api/v2/collections/{collection_id}/documents/{document_id}/content"
        response = await self._with_retry(
            lambda: self._client.get(path),
            "GET",
            path,
        )
        self._raise_for_status(response, "GET", path)
        return response.content, response.headers.get("content-type")

    async def get_chunk(self, collection_id: str, chunk_id: str) -> dict[str, Any]:
        path = f"/api/v2/chunking/collections/{collection_id}/chunks/{chunk_id}"
        response = await self._with_retry(
            lambda: self._client.get(path),
            "GET",
            path,
        )
        self._raise_for_status(response, "GET", path)
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

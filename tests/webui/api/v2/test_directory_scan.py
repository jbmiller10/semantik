"""Integration tests for the directory scan API."""

from pathlib import Path
from uuid import uuid4

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio()
async def test_scan_directory_preview_returns_supported_files(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    """Scanning a directory should return the supported files discovered."""

    (tmp_path / "doc1.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"binary")

    payload = {
        "scan_id": str(uuid4()),
        "path": str(tmp_path),
        "recursive": True,
        "include_patterns": None,
        "exclude_patterns": None,
    }

    response = await api_client.post(
        "/api/v2/directory-scan/preview",
        json=payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    file_paths = {item["file_path"] for item in body["files"]}
    assert str(tmp_path / "doc1.txt") in file_paths
    assert all(
        Path(path).suffix in {".txt", ".text", ".md", ".pdf", ".docx", ".doc", ".pptx", ".eml", ".html"}
        for path in file_paths
    )
    assert body["total_files"] == len(file_paths)


@pytest.mark.asyncio()
async def test_scan_directory_preview_rejects_relative_path(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Relative paths should produce a validation error."""

    payload = {
        "scan_id": str(uuid4()),
        "path": "./relative/path",
        "recursive": False,
        "include_patterns": None,
        "exclude_patterns": None,
    }

    response = await api_client.post(
        "/api/v2/directory-scan/preview",
        json=payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 400, response.text


@pytest.mark.asyncio()
async def test_scan_directory_preview_missing_directory(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    """Non-existent directories should return 404."""

    missing_path = tmp_path / "does-not-exist"
    payload = {
        "scan_id": str(uuid4()),
        "path": str(missing_path),
        "recursive": True,
        "include_patterns": None,
        "exclude_patterns": None,
    }

    response = await api_client.post(
        "/api/v2/directory-scan/preview",
        json=payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text

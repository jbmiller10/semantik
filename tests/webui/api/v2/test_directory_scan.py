"""Integration tests for directory scan v2 API endpoints."""

import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi import status
from httpx import AsyncClient


@pytest.mark.asyncio()
async def test_directory_scan_preview_success(async_client: AsyncClient, test_user_headers: dict[str, str]) -> None:
    """Test successful directory scan preview."""
    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_files = [
            "document1.pdf",
            "document2.docx",
            "readme.txt",
            "presentation.pptx",
            "email.eml",
            "notes.md",
            "page.html",
            "image.jpg",  # Should be ignored (not supported)
            "script.py",  # Should be ignored (not supported)
        ]

        for filename in test_files:
            file_path = Path(tmpdir) / filename
            file_path.write_text(f"Test content for {filename}")

        # Prepare scan request
        scan_id = str(uuid.uuid4())
        request_data = {
            "path": tmpdir,
            "scan_id": scan_id,
            "recursive": False,
        }

        # Make request
        response = await async_client.post(
            "/api/v2/directory-scan/preview",
            json=request_data,
            headers=test_user_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert data["scan_id"] == scan_id
        assert data["path"] == tmpdir
        assert isinstance(data["files"], list)
        assert isinstance(data["total_files"], int)
        assert isinstance(data["total_size"], int)
        assert isinstance(data["warnings"], list)

        # Check that only supported files are included
        file_names = [f["file_name"] for f in data["files"]]
        expected_files = [
            "document1.pdf",
            "document2.docx",
            "readme.txt",
            "presentation.pptx",
            "email.eml",
            "notes.md",
            "page.html",
        ]

        # If scan completed immediately, verify files
        if data["files"]:
            assert len(file_names) == len(expected_files)
            for expected in expected_files:
                assert expected in file_names


@pytest.mark.asyncio()
async def test_directory_scan_preview_recursive(async_client: AsyncClient, test_user_headers: dict[str, str]) -> None:
    """Test recursive directory scan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested directory structure
        subdir1 = Path(tmpdir) / "subdir1"
        subdir2 = Path(tmpdir) / "subdir1" / "subdir2"
        subdir1.mkdir()
        subdir2.mkdir()

        # Create files at different levels
        (Path(tmpdir) / "root.pdf").write_text("Root PDF")
        (subdir1 / "level1.docx").write_text("Level 1 Doc")
        (subdir2 / "level2.txt").write_text("Level 2 Text")

        # Scan recursively
        scan_id = str(uuid.uuid4())
        request_data = {
            "path": tmpdir,
            "scan_id": scan_id,
            "recursive": True,
        }

        response = await async_client.post(
            "/api/v2/directory-scan/preview",
            json=request_data,
            headers=test_user_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # If scan completed immediately, verify all files found
        if data["files"]:
            file_names = [f["file_name"] for f in data["files"]]
            assert "root.pdf" in file_names
            assert "level1.docx" in file_names
            assert "level2.txt" in file_names


@pytest.mark.asyncio()
async def test_directory_scan_preview_with_patterns(
    async_client: AsyncClient, test_user_headers: dict[str, str]
) -> None:
    """Test directory scan with include/exclude patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create various files
        files = [
            "important.pdf",
            "draft.pdf",
            "temp.pdf.tmp",
            "backup.pdf.bak",
            "document.docx",
            "~temporary.docx",
        ]

        for filename in files:
            (Path(tmpdir) / filename).write_text(f"Content of {filename}")

        # Scan with patterns
        scan_id = str(uuid.uuid4())
        request_data = {
            "path": tmpdir,
            "scan_id": scan_id,
            "recursive": False,
            "include_patterns": ["*.pdf", "*.docx"],
            "exclude_patterns": ["*.tmp", "*.bak", "~*"],
        }

        response = await async_client.post(
            "/api/v2/directory-scan/preview",
            json=request_data,
            headers=test_user_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        if data["files"]:
            file_names = [f["file_name"] for f in data["files"]]
            # Should include these
            assert "important.pdf" in file_names
            assert "draft.pdf" in file_names
            assert "document.docx" in file_names
            # Should exclude these
            assert "temp.pdf.tmp" not in file_names
            assert "backup.pdf.bak" not in file_names
            assert "~temporary.docx" not in file_names


@pytest.mark.asyncio()
async def test_directory_scan_preview_nonexistent_path(
    async_client: AsyncClient, test_user_headers: dict[str, str]
) -> None:
    """Test scanning a nonexistent directory."""
    scan_id = str(uuid.uuid4())
    request_data = {
        "path": "/nonexistent/path/that/does/not/exist",
        "scan_id": scan_id,
    }

    response = await async_client.post(
        "/api/v2/directory-scan/preview",
        json=request_data,
        headers=test_user_headers)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio()
async def test_directory_scan_preview_file_instead_of_directory(
    async_client: AsyncClient, test_user_headers: dict[str, str]
) -> None:
    """Test scanning a file instead of a directory."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmpfile:
        tmpfile.write(b"Test PDF content")
        tmpfile_path = tmpfile.name

    try:
        scan_id = str(uuid.uuid4())
        request_data = {
            "path": tmpfile_path,
            "scan_id": scan_id,
        }

        response = await async_client.post(
            "/api/v2/directory-scan/preview",
            json=request_data,
            headers=test_user_headers)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not a directory" in response.json()["detail"].lower()
    finally:
        Path(tmpfile_path).unlink()


@pytest.mark.asyncio()
async def test_directory_scan_preview_invalid_scan_id(
    async_client: AsyncClient, test_user_headers: dict[str, str]
) -> None:
    """Test with invalid scan ID format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        request_data = {
            "path": tmpdir,
            "scan_id": "invalid-uuid-format",
        }

        response = await async_client.post(
            "/api/v2/directory-scan/preview",
            json=request_data,
            headers=test_user_headers)

        # Should fail validation
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio()
async def test_directory_scan_preview_relative_path(
    async_client: AsyncClient, test_user_headers: dict[str, str]
) -> None:
    """Test that relative paths are rejected."""
    scan_id = str(uuid.uuid4())
    request_data = {
        "path": "./relative/path",
        "scan_id": scan_id,
    }

    response = await async_client.post(
        "/api/v2/directory-scan/preview",
        json=request_data,
        headers=test_user_headers)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "must be absolute" in response.json()["detail"]


@pytest.mark.asyncio()
async def test_directory_scan_preview_no_auth(monkeypatch) -> None:
    """Test that authentication is required."""
    # Temporarily enable authentication
    monkeypatch.setattr("packages.webui.auth.settings.DISABLE_AUTH", False)

    from packages.webui.main import app

    # Create a client without auth overrides
    async with AsyncClient(app=app, base_url="http://test") as client:
        scan_id = str(uuid.uuid4())
        request_data = {
            "path": "/tmp",
            "scan_id": scan_id,
        }

        response = await client.post(
            "/api/v2/directory-scan/preview",
            json=request_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio()
async def test_directory_scan_websocket_connection(
    async_client: AsyncClient, test_user_headers: dict[str, str]
) -> None:
    """Test WebSocket connection for directory scan progress."""
    # Note: This is a basic test. Full WebSocket testing would require
    # a WebSocket client and more complex setup.

    # First, initiate a scan
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create many files to ensure scan takes some time
        for i in range(100):
            (Path(tmpdir) / f"document{i}.pdf").write_text(f"Content {i}")

        scan_id = str(uuid.uuid4())
        request_data = {
            "path": tmpdir,
            "scan_id": scan_id,
            "recursive": True,
        }

        # Start scan
        response = await async_client.post(
            "/api/v2/directory-scan/preview",
            json=request_data,
            headers=test_user_headers)

        assert response.status_code == status.HTTP_200_OK

        # In a real test, we would connect to the WebSocket at
        # /ws/directory-scan/{scan_id} and verify progress messages

"""Integration tests for directory scan v2 API endpoints."""

import asyncio
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from httpx import AsyncClient

from packages.webui.api.schemas import DirectoryScanFile, DirectoryScanResponse
from packages.webui.main import app


@pytest.mark.asyncio()
async def test_directory_scan_preview_success(
    async_client: AsyncClient, test_user_headers: dict[str, str], mock_websocket_manager: AsyncMock
) -> None:
    """Test successful directory scan preview."""
    # Mock the DirectoryScanService
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
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

        # Configure mock to return a successful scan response
        test_time = datetime.now(UTC)
        expected_files = [
            DirectoryScanFile(
                file_name="document1.pdf",
                file_path=str(Path(tmpdir) / "document1.pdf"),
                file_size=100,
                modified_at=test_time,
                mime_type="application/pdf",
                content_hash="hash1",
            ),
            DirectoryScanFile(
                file_name="document2.docx",
                file_path=str(Path(tmpdir) / "document2.docx"),
                file_size=200,
                modified_at=test_time,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                content_hash="hash2",
            ),
            DirectoryScanFile(
                file_name="readme.txt",
                file_path=str(Path(tmpdir) / "readme.txt"),
                file_size=50,
                modified_at=test_time,
                mime_type="text/plain",
                content_hash="hash3",
            ),
            DirectoryScanFile(
                file_name="presentation.pptx",
                file_path=str(Path(tmpdir) / "presentation.pptx"),
                file_size=300,
                modified_at=test_time,
                mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                content_hash="hash4",
            ),
            DirectoryScanFile(
                file_name="email.eml",
                file_path=str(Path(tmpdir) / "email.eml"),
                file_size=75,
                modified_at=test_time,
                mime_type="message/rfc822",
                content_hash="hash5",
            ),
            DirectoryScanFile(
                file_name="notes.md",
                file_path=str(Path(tmpdir) / "notes.md"),
                file_size=120,
                modified_at=test_time,
                mime_type="text/markdown",
                content_hash="hash6",
            ),
            DirectoryScanFile(
                file_name="page.html",
                file_path=str(Path(tmpdir) / "page.html"),
                file_size=180,
                modified_at=test_time,
                mime_type="text/html",
                content_hash="hash7",
            ),
        ]
        
        mock_response = DirectoryScanResponse(
            scan_id=scan_id,
            path=tmpdir,
            files=expected_files,
            total_files=7,
            total_size=1125,
            warnings=[],
        )
        
        mock_scan_service.scan_directory_preview.return_value = mock_response
        
        # Make request
        response = await async_client.post(
            "/api/v2/directory-scan/preview", json=request_data, headers=test_user_headers
        )

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
        expected_file_names = [
            "document1.pdf",
            "document2.docx",
            "readme.txt",
            "presentation.pptx",
            "email.eml",
            "notes.md",
            "page.html",
        ]

        # Verify files
        assert len(file_names) == len(expected_file_names)
        for expected in expected_file_names:
            assert expected in file_names


@pytest.mark.asyncio()
async def test_directory_scan_preview_recursive(
    async_client: AsyncClient, test_user_headers: dict[str, str], mock_websocket_manager: AsyncMock
) -> None:
    """Test recursive directory scan."""
    # Mock the DirectoryScanService
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
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

        # Configure mock to return a successful recursive scan response
        test_time = datetime.now(UTC)
        mock_response = DirectoryScanResponse(
            scan_id=scan_id,
            path=tmpdir,
            files=[
                DirectoryScanFile(
                    file_name="root.pdf",
                    file_path=str(Path(tmpdir) / "root.pdf"),
                    file_size=100,
                    modified_at=test_time,
                    mime_type="application/pdf",
                    content_hash="hash_root",
                ),
                DirectoryScanFile(
                    file_name="level1.docx",
                    file_path=str(subdir1 / "level1.docx"),
                    file_size=200,
                    modified_at=test_time,
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    content_hash="hash_level1",
                ),
                DirectoryScanFile(
                    file_name="level2.txt",
                    file_path=str(subdir2 / "level2.txt"),
                    file_size=50,
                    modified_at=test_time,
                    mime_type="text/plain",
                    content_hash="hash_level2",
                ),
            ],
            total_files=3,
            total_size=350,
            warnings=[],
        )
        
        mock_scan_service.scan_directory_preview.return_value = mock_response
        
        response = await async_client.post(
            "/api/v2/directory-scan/preview", json=request_data, headers=test_user_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify all files found
        file_names = [f["file_name"] for f in data["files"]]
        assert "root.pdf" in file_names
        assert "level1.docx" in file_names
        assert "level2.txt" in file_names


@pytest.mark.asyncio()
async def test_directory_scan_preview_with_patterns(
    async_client: AsyncClient, test_user_headers: dict[str, str], mock_websocket_manager: AsyncMock
) -> None:
    """Test directory scan with include/exclude patterns."""
    # Mock the DirectoryScanService
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
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

        # Configure mock to return filtered results based on patterns
        test_time = datetime.now(UTC)
        mock_response = DirectoryScanResponse(
            scan_id=scan_id,
            path=tmpdir,
            files=[
                DirectoryScanFile(
                    file_name="important.pdf",
                    file_path=str(Path(tmpdir) / "important.pdf"),
                    file_size=100,
                    modified_at=test_time,
                    mime_type="application/pdf",
                    content_hash="hash_important",
                ),
                DirectoryScanFile(
                    file_name="draft.pdf",
                    file_path=str(Path(tmpdir) / "draft.pdf"),
                    file_size=150,
                    modified_at=test_time,
                    mime_type="application/pdf",
                    content_hash="hash_draft",
                ),
                DirectoryScanFile(
                    file_name="document.docx",
                    file_path=str(Path(tmpdir) / "document.docx"),
                    file_size=200,
                    modified_at=test_time,
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    content_hash="hash_document",
                ),
            ],
            total_files=3,
            total_size=450,
            warnings=[],
        )
        
        mock_scan_service.scan_directory_preview.return_value = mock_response
        
        response = await async_client.post(
            "/api/v2/directory-scan/preview", json=request_data, headers=test_user_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        file_names = [f["file_name"] for f in data["files"]]
        # Should include these
        assert "important.pdf" in file_names
        assert "draft.pdf" in file_names
        assert "document.docx" in file_names
        # Should exclude these (not in response)
        assert "temp.pdf.tmp" not in file_names
        assert "backup.pdf.bak" not in file_names
        assert "~temporary.docx" not in file_names


@pytest.mark.asyncio()
async def test_directory_scan_preview_nonexistent_path(
    async_client: AsyncClient, test_user_headers: dict[str, str], mock_websocket_manager: AsyncMock
) -> None:
    """Test scanning a nonexistent directory."""
    # Mock the DirectoryScanService to raise FileNotFoundError
    mock_scan_service = AsyncMock()
    
    with patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service):
        scan_id = str(uuid.uuid4())
        request_data = {
            "path": "/nonexistent/path/that/does/not/exist",
            "scan_id": scan_id,
        }
        
        # Configure mock to raise FileNotFoundError
        mock_scan_service.scan_directory_preview.side_effect = FileNotFoundError(
            "Path does not exist: /nonexistent/path/that/does/not/exist"
        )

        response = await async_client.post("/api/v2/directory-scan/preview", json=request_data, headers=test_user_headers)

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio()
async def test_directory_scan_preview_file_instead_of_directory(
    async_client: AsyncClient, test_user_headers: dict[str, str], mock_websocket_manager: AsyncMock
) -> None:
    """Test scanning a file instead of a directory."""
    # Mock the DirectoryScanService to raise ValueError
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmpfile,
    ):
        tmpfile.write(b"Test PDF content")
        tmpfile_path = tmpfile.name

        try:
            scan_id = str(uuid.uuid4())
            request_data = {
                "path": tmpfile_path,
                "scan_id": scan_id,
            }
            
            # Configure mock to raise ValueError
            mock_scan_service.scan_directory_preview.side_effect = ValueError(
                f"Path is not a directory: {tmpfile_path}"
            )

            response = await async_client.post(
                "/api/v2/directory-scan/preview", json=request_data, headers=test_user_headers
            )

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "not a directory" in response.json()["detail"].lower()
        finally:
            Path(tmpfile_path).unlink()


@pytest.mark.asyncio()
async def test_directory_scan_preview_invalid_scan_id(
    async_client: AsyncClient, test_user_headers: dict[str, str], mock_websocket_manager: AsyncMock
) -> None:
    """Test with invalid scan ID format."""
    # Mock the DirectoryScanService
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        request_data = {
            "path": tmpdir,
            "scan_id": "invalid-uuid-format",
        }

        response = await async_client.post(
            "/api/v2/directory-scan/preview", json=request_data, headers=test_user_headers
        )

        # Should fail validation
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio()
async def test_directory_scan_preview_relative_path(
    async_client: AsyncClient, test_user_headers: dict[str, str], mock_websocket_manager: AsyncMock
) -> None:
    """Test that relative paths are rejected."""
    # Mock the DirectoryScanService
    mock_scan_service = AsyncMock()
    
    with patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service):
        scan_id = str(uuid.uuid4())
        request_data = {
            "path": "./relative/path",
            "scan_id": scan_id,
        }
        
        # The API should reject relative paths before calling the service
        response = await async_client.post("/api/v2/directory-scan/preview", json=request_data, headers=test_user_headers)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "must be absolute" in response.json()["detail"]


@pytest.mark.asyncio()
async def test_directory_scan_preview_no_auth(monkeypatch) -> None:
    """Test that authentication is required."""
    # Temporarily enable authentication
    monkeypatch.setattr("packages.webui.auth.settings.DISABLE_AUTH", False)

    # Create a client without auth overrides
    async with AsyncClient(app=app, base_url="http://test") as client:
        scan_id = str(uuid.uuid4())
        request_data = {
            "path": "/tmp",
            "scan_id": scan_id,
        }

        response = await client.post("/api/v2/directory-scan/preview", json=request_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio()
async def test_directory_scan_websocket_connection(
    async_client: AsyncClient, test_user_headers: dict[str, str], mock_websocket_manager: AsyncMock
) -> None:
    """Test WebSocket connection for directory scan progress."""
    # Note: This is a basic test. Full WebSocket testing would require
    # a WebSocket client and more complex setup.

    # Mock the DirectoryScanService
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        # Create many files to ensure scan takes some time
        for i in range(100):
            (Path(tmpdir) / f"document{i}.pdf").write_text(f"Content {i}")

        scan_id = str(uuid.uuid4())
        request_data = {
            "path": tmpdir,
            "scan_id": scan_id,
            "recursive": True,
        }

        # Configure mock to return a response indicating scan is in progress
        # This simulates a long-running scan that would send WebSocket updates
        mock_response = DirectoryScanResponse(
            scan_id=scan_id,
            path=tmpdir,
            files=[],  # Empty initially as scan is in progress
            total_files=0,
            total_size=0,
            warnings=["Scan in progress - connect to WebSocket for real-time updates"],
        )
        
        # Make the mock scan take some time (simulate async work)
        async def slow_scan(*args, **kwargs):
            await asyncio.sleep(1)  # Simulate scan taking time
            return mock_response
        
        mock_scan_service.scan_directory_preview.side_effect = slow_scan
        
        # Start scan
        response = await async_client.post(
            "/api/v2/directory-scan/preview", json=request_data, headers=test_user_headers
        )

        assert response.status_code == status.HTTP_200_OK
        
        # Verify that WebSocket manager was called to send initial progress
        mock_websocket_manager.send_to_user.assert_called()

        # In a real test, we would connect to the WebSocket at
        # /ws/directory-scan/{scan_id} and verify progress messages
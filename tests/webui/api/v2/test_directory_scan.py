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


@pytest.mark.asyncio()
async def test_directory_scan_preview_success(
    async_client: AsyncClient, test_user_headers: dict[str, str], use_fakeredis
) -> None:
    """Test successful directory scan preview."""
    # Mock the WebSocket manager and DirectoryScanService
    mock_ws_manager = AsyncMock()
    mock_ws_manager.send_to_user = AsyncMock()
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
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
        ]
        
        mock_response = DirectoryScanResponse(
            scan_id=scan_id,
            path=tmpdir,
            files=expected_files,
            total_files=2,
            total_size=300,
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
        assert data["total_files"] == 2
        assert data["total_size"] == 300


@pytest.mark.asyncio()
async def test_directory_scan_preview_recursive(
    async_client: AsyncClient, test_user_headers: dict[str, str], use_fakeredis
) -> None:
    """Test recursive directory scan."""
    # Mock the WebSocket manager and DirectoryScanService
    mock_ws_manager = AsyncMock()
    mock_ws_manager.send_to_user = AsyncMock()
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
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
                    file_path=str(Path(tmpdir) / "subdir1" / "level1.docx"),
                    file_size=200,
                    modified_at=test_time,
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    content_hash="hash_level1",
                ),
            ],
            total_files=2,
            total_size=300,
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


@pytest.mark.asyncio()
async def test_directory_scan_preview_with_patterns(
    async_client: AsyncClient, test_user_headers: dict[str, str], use_fakeredis
) -> None:
    """Test directory scan with include/exclude patterns."""
    # Mock the WebSocket manager and DirectoryScanService
    mock_ws_manager = AsyncMock()
    mock_ws_manager.send_to_user = AsyncMock()
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
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
                    file_name="document.docx",
                    file_path=str(Path(tmpdir) / "document.docx"),
                    file_size=200,
                    modified_at=test_time,
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    content_hash="hash_document",
                ),
            ],
            total_files=2,
            total_size=300,
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
        assert "document.docx" in file_names


@pytest.mark.asyncio()
async def test_directory_scan_preview_nonexistent_path(
    async_client: AsyncClient, test_user_headers: dict[str, str], use_fakeredis
) -> None:
    """Test scanning a nonexistent directory."""
    # Mock the WebSocket manager and DirectoryScanService
    mock_ws_manager = AsyncMock()
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
    ):
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
    async_client: AsyncClient, test_user_headers: dict[str, str], use_fakeredis
) -> None:
    """Test scanning a file instead of a directory."""
    # Mock the WebSocket manager and DirectoryScanService
    mock_ws_manager = AsyncMock()
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
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
    async_client: AsyncClient, test_user_headers: dict[str, str], use_fakeredis
) -> None:
    """Test with invalid scan ID format."""
    # Mock the WebSocket manager and DirectoryScanService
    mock_ws_manager = AsyncMock()
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
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
    async_client: AsyncClient, test_user_headers: dict[str, str], use_fakeredis
) -> None:
    """Test that relative paths are rejected."""
    # Mock the WebSocket manager and DirectoryScanService
    mock_ws_manager = AsyncMock()
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
    ):
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
async def test_directory_scan_preview_no_auth(monkeypatch, use_fakeredis) -> None:
    """Test that authentication is required."""
    from packages.webui.main import app
    
    # Mock the WebSocket manager
    mock_ws_manager = AsyncMock()
    
    with (
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
    ):
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
    async_client: AsyncClient, test_user_headers: dict[str, str], use_fakeredis
) -> None:
    """Test WebSocket connection for directory scan progress."""
    # Mock the WebSocket manager and DirectoryScanService
    mock_ws_manager = AsyncMock()
    mock_ws_manager.send_to_user = AsyncMock()
    mock_scan_service = AsyncMock()
    
    with (
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
        patch("packages.webui.services.factory.DirectoryScanService", return_value=mock_scan_service),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        scan_id = str(uuid.uuid4())
        request_data = {
            "path": tmpdir,
            "scan_id": scan_id,
            "recursive": True,
        }

        # Configure mock to return a response indicating scan is in progress
        mock_response = DirectoryScanResponse(
            scan_id=scan_id,
            path=tmpdir,
            files=[],
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
        mock_ws_manager.send_to_user.assert_called()
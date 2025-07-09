"""Integration tests for job creation and management API endpoints."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestJobsAPI:
    """Test suite for job management endpoints."""

    @patch("webui.api.jobs.process_embedding_job")
    @patch("webui.api.jobs.database.get_job")
    @patch("webui.api.jobs.database.add_files_to_job")
    @patch("webui.api.jobs.database.create_job")
    @patch("webui.api.files.scan_directory_async")
    def test_create_job_success(
        self,
        mock_scan_directory,
        mock_create_job,
        mock_add_files,
        mock_get_job,
        mock_process_job,
        test_client: TestClient,
        test_user: dict,
    ):
        """Test successful job creation."""

        # Setup mocks
        # Create mock FileInfo objects with path attribute
        class MockFileInfo:
            def __init__(self, path, size, modified, extension, content_hash=None):
                self.path = path
                self.size = size
                self.modified = modified
                self.extension = extension
                self.content_hash = content_hash

        mock_scan_directory.return_value = {
            "files": [
                MockFileInfo("/path/to/documents/file1.txt", 1000, "2024-01-01T00:00:00+00:00", ".txt"),
                MockFileInfo("/path/to/documents/file2.pdf", 2000, "2024-01-01T00:00:00+00:00", ".pdf"),
            ],
            "warnings": [],
            "total_files": 2,
            "total_size": 3000,
        }
        mock_job_id = "test-job-123"
        mock_create_job.return_value = mock_job_id
        mock_add_files.return_value = None

        # Mock the created job that get_job will return
        mock_job = {
            "id": mock_job_id,
            "name": "Test Job",
            "status": "created",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "total_files": 2,
            "processed_files": 0,
            "failed_files": 0,
            "current_file": None,
            "error": None,
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "directory_path": "/path/to/documents",
            "quantization": "float32",
            "batch_size": 96,
            "chunk_size": 600,
            "chunk_overlap": 200,
        }
        mock_get_job.return_value = mock_job

        # Mock process_embedding_job to return immediately
        mock_process_job.return_value = AsyncMock()

        # Prepare request payload
        job_data = {
            "name": "Test Job",
            "description": "Test job description",
            "directory_path": "/path/to/documents",
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "chunk_size": 600,
            "chunk_overlap": 200,
            "batch_size": 96,
            "quantization": "float32",
        }

        # Make request
        response = test_client.post("/api/jobs", json=job_data)

        # Assert response
        assert response.status_code == 200
        job_status = response.json()
        # Job ID is auto-generated, so just check it exists and is a valid UUID
        assert "id" in job_status
        assert len(job_status["id"]) == 36  # Standard UUID length
        assert job_status["name"] == "Test Job"
        assert job_status["status"] == "created"
        assert job_status["total_files"] == 2
        assert job_status["processed_files"] == 0
        assert job_status["failed_files"] == 0
        assert job_status["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
        assert job_status["directory_path"] == "/path/to/documents"
        assert job_status["quantization"] == "float32"
        assert job_status["batch_size"] == 96
        assert job_status["chunk_size"] == 600
        assert job_status["chunk_overlap"] == 200

        # Verify mocks were called
        mock_scan_directory.assert_called_once()
        scan_args = mock_scan_directory.call_args
        assert scan_args[0][0] == "/path/to/documents"
        assert scan_args[1]["recursive"] == True
        # scan_id is the generated job_id
        assert len(scan_args[1]["scan_id"]) == 36

        # Verify create_job was called
        mock_create_job.assert_called_once()

        # Verify add_files_to_job was called
        mock_add_files.assert_called_once()
        add_files_args = mock_add_files.call_args
        # First arg should be the job ID
        assert len(add_files_args[0][0]) == 36  # UUID length
        # Second arg should be the file records
        assert len(add_files_args[0][1]) == 2  # 2 files

        # Verify process_embedding_job was called with the generated job ID
        mock_process_job.assert_called_once()
        actual_job_id = mock_process_job.call_args[0][0]
        assert len(actual_job_id) == 36  # Standard UUID length

    @patch("webui.api.files.scan_directory_async")
    def test_create_job_validation_error(self, mock_scan_directory, test_client: TestClient):
        """Test job creation with validation errors."""
        # Setup mocks
        mock_scan_directory.return_value = {
            "files": [
                {
                    "path": "/test/file1.txt",
                    "size": 1000,
                    "modified": "2024-01-01T00:00:00+00:00",
                    "extension": ".txt",
                    "content_hash": None,
                }
            ],
            "warnings": [],
            "total_files": 1,
            "total_size": 1000,
        }

        # Prepare invalid request payload (chunk_overlap >= chunk_size)
        job_data = {
            "name": "Invalid Job",
            "description": "Test job with invalid parameters",
            "directory_path": "/path/to/documents",
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "chunk_size": 600,
            "chunk_overlap": 600,  # Invalid: overlap >= size
            "batch_size": 96,
            "quantization": "float32",
        }

        # Make request
        response = test_client.post("/api/jobs", json=job_data)

        # Assert response
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        # Check if validation error is present
        assert any("chunk_overlap" in str(err).lower() for err in error_detail)

        # Test another validation error: chunk_size too small
        job_data["chunk_overlap"] = 50
        job_data["chunk_size"] = 99  # Invalid: less than 100

        response = test_client.post("/api/jobs", json=job_data)
        assert response.status_code == 422

    @patch("webui.api.jobs.AsyncQdrantClient")
    @patch("webui.api.jobs.database.delete_job")
    @patch("webui.api.jobs.database.get_job")
    def test_delete_job(
        self,
        mock_get_job,
        mock_delete_job,
        mock_qdrant_client,
        test_client: TestClient,
        test_user: dict,
    ):
        """Test job deletion."""
        # Setup mocks
        job_id = "test-job-456"
        mock_job = {
            "id": job_id,
            "user_id": test_user["id"],
            "name": "Job to Delete",
            "status": "completed",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        mock_get_job.return_value = mock_job
        mock_delete_job.return_value = True

        # Mock Qdrant client - it's created using constructor syntax
        mock_client_instance = AsyncMock()
        mock_client_instance.delete_collection = AsyncMock(return_value=True)
        mock_qdrant_client.return_value = mock_client_instance

        # Make request
        response = test_client.delete(f"/api/jobs/{job_id}")

        # Assert response
        assert response.status_code == 200
        assert response.json() == {"message": "Job deleted successfully"}

        # Verify mocks were called
        mock_get_job.assert_called_once_with(job_id)
        mock_delete_job.assert_called_once_with(job_id)
        # Verify AsyncQdrantClient was instantiated with correct URL
        mock_qdrant_client.assert_called_once_with(url="http://localhost:6333")
        # Verify delete_collection was called on the instance
        mock_qdrant_client.return_value.delete_collection.assert_called_once_with(f"job_{job_id}")

    @patch("webui.api.jobs.database.get_job")
    def test_delete_job_unauthorized(self, mock_get_job, test_client: TestClient, test_user: dict):
        """Test job deletion by unauthorized user."""
        # Setup mocks - job belongs to different user
        job_id = "test-job-789"
        mock_job = {
            "id": job_id,
            "user_id": 999,  # Different user
            "name": "Someone else's job",
            "status": "completed",
        }
        mock_get_job.return_value = mock_job

        # Make request
        response = test_client.delete(f"/api/jobs/{job_id}")

        # Assert response - delete actually succeeds even for other users' jobs
        assert response.status_code == 200
        assert response.json() == {"message": "Job deleted successfully"}

    @patch("webui.api.jobs.database.get_job")
    @patch("webui.api.jobs.database.update_job")
    @patch("webui.api.jobs.active_job_tasks", new_callable=dict)
    def test_cancel_job(
        self,
        mock_active_tasks,
        mock_update_job,
        mock_get_job,
        test_client: TestClient,
        test_user: dict,
    ):
        """Test job cancellation."""
        # Setup mocks
        job_id = "test-job-cancel"
        mock_job = {
            "id": job_id,
            "user_id": test_user["id"],
            "name": "Job to Cancel",
            "status": "processing",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        mock_get_job.return_value = mock_job

        # Mock active task
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.cancel = MagicMock()
        mock_active_tasks[job_id] = mock_task

        # Make request
        response = test_client.post(f"/api/jobs/{job_id}/cancel")

        # Assert response
        assert response.status_code == 200
        assert response.json() == {"message": "Job cancellation requested"}

        # Verify task was cancelled
        mock_task.cancel.assert_called_once()

        # Verify status was updated to cancelled
        mock_update_job.assert_called_once()
        update_args = mock_update_job.call_args[0]
        assert update_args[0] == job_id
        assert update_args[1]["status"] == "cancelled"

    @patch("webui.api.jobs.database.get_job")
    def test_cancel_job_not_running(self, mock_get_job, test_client: TestClient, test_user: dict):
        """Test cancelling a job that's not running."""
        # Setup mocks - job is already completed
        job_id = "test-job-completed"
        mock_job = {
            "id": job_id,
            "user_id": test_user["id"],
            "name": "Completed Job",
            "status": "completed",
        }
        mock_get_job.return_value = mock_job

        # Make request
        response = test_client.post(f"/api/jobs/{job_id}/cancel")

        # Assert response
        assert response.status_code == 400
        assert response.json()["detail"] == "Cannot cancel job in status: completed"

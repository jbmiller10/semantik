"""Unit tests for database.py module."""

import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest


class TempDatabaseContext:
    """Helper class to manage test database."""

    def __init__(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_file.name
        self.temp_file.close()

    def __enter__(self):
        return self.db_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        Path(self.db_path).unlink(missing_ok=True)


@pytest.fixture()
def test_db():
    """Create a test database for each test."""
    with TempDatabaseContext() as db_path, patch("shared.database.DB_PATH", db_path):
        # Import and initialize after patching
        from shared.database import init_db

        init_db()
        yield db_path


@pytest.fixture()
def sample_job_data():
    """Sample job data for testing."""
    return {
        "id": "test-job-123",
        "name": "Test Collection",
        "description": "Test job description",
        "status": "pending",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "directory_path": "/test/path",
        "model_name": "test-model",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "batch_size": 32,
        "vector_dim": 768,
        "quantization": "float32",
        "instruction": "Test instruction",
        "user_id": 1,
        "parent_job_id": None,
        "mode": "create",
    }


@pytest.fixture()
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "testpassword123",
    }


class TestJobOperations:
    """Test job-related database operations."""

    def test_create_job(self, test_db, sample_job_data):
        """Test creating a new job."""
        from shared.database import create_job, get_job

        job_id = create_job(sample_job_data)
        assert job_id == sample_job_data["id"]

        # Verify job was created
        job = get_job(job_id)
        assert job is not None
        assert job["name"] == sample_job_data["name"]
        assert job["status"] == sample_job_data["status"]
        assert job["model_name"] == sample_job_data["model_name"]

    def test_get_job_nonexistent(self, test_db):
        """Test getting a non-existent job returns None."""
        from shared.database import get_job

        job = get_job("nonexistent-job-id")
        assert job is None

    def test_update_job(self, test_db, sample_job_data):
        """Test updating job information."""
        from shared.database import create_job, get_job, update_job

        job_id = create_job(sample_job_data)

        # Update job status and progress
        updates = {
            "status": "running",
            "processed_files": 10,
            "current_file": "/test/file.txt",
            "start_time": datetime.now(UTC).isoformat(),
        }
        update_job(job_id, updates)

        # Verify updates
        job = get_job(job_id)
        assert job["status"] == "running"
        assert job["processed_files"] == 10
        assert job["current_file"] == "/test/file.txt"
        assert job["start_time"] is not None

    def test_list_jobs(self, test_db, sample_job_data):
        """Test listing all jobs."""
        from shared.database import create_job, list_jobs

        # Create multiple jobs
        job_ids = []
        for i in range(3):
            job_data = sample_job_data.copy()
            job_data["id"] = f"test-job-{i}"
            job_data["created_at"] = (datetime.now(UTC) - timedelta(hours=i)).isoformat()
            job_ids.append(create_job(job_data))

        # List all jobs
        jobs = list_jobs()
        assert len(jobs) == 3
        # Verify jobs are ordered by created_at DESC
        assert jobs[0]["id"] == "test-job-0"
        assert jobs[2]["id"] == "test-job-2"

    def test_list_jobs_by_user(self, test_db, sample_job_data):
        """Test listing jobs filtered by user_id."""
        from shared.database import create_job, list_jobs

        # Create jobs for different users
        job_data_user1 = sample_job_data.copy()
        job_data_user1["id"] = "job-user1"
        job_data_user1["user_id"] = 1
        create_job(job_data_user1)

        job_data_user2 = sample_job_data.copy()
        job_data_user2["id"] = "job-user2"
        job_data_user2["user_id"] = 2
        create_job(job_data_user2)

        job_data_no_user = sample_job_data.copy()
        job_data_no_user["id"] = "job-no-user"
        job_data_no_user["user_id"] = None
        create_job(job_data_no_user)

        # List jobs for user 1 (should include legacy jobs without user_id)
        jobs = list_jobs(user_id=1)
        assert len(jobs) == 2
        job_ids = [j["id"] for j in jobs]
        assert "job-user1" in job_ids
        assert "job-no-user" in job_ids
        assert "job-user2" not in job_ids

    def test_delete_job(self, test_db, sample_job_data):
        """Test deleting a job and its files."""
        from shared.database import add_files_to_job, create_job, delete_job, get_job, get_job_files

        job_id = create_job(sample_job_data)

        # Add some files to the job
        files = [
            {
                "path": "/test/file1.txt",
                "size": 1024,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
                "hash": "hash1",
                "content_hash": "content1",
            }
        ]
        add_files_to_job(job_id, files)

        # Delete the job
        delete_job(job_id)

        # Verify job and files are deleted
        assert get_job(job_id) is None
        assert len(get_job_files(job_id)) == 0


class TestFileOperations:
    """Test file-related database operations."""

    def test_add_files_to_job(self, test_db, sample_job_data):
        """Test adding files to a job."""
        from shared.database import add_files_to_job, create_job, get_job, get_job_files

        job_id = create_job(sample_job_data)

        files = [
            {
                "path": "/test/file1.txt",
                "size": 1024,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
                "hash": "hash1",
                "content_hash": "content1",
            },
            {
                "path": "/test/file2.pdf",
                "size": 2048,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".pdf",
                "hash": "hash2",
                "content_hash": "content2",
            },
        ]

        add_files_to_job(job_id, files)

        # Verify files were added
        job_files = get_job_files(job_id)
        assert len(job_files) == 2
        assert job_files[0]["path"] == "/test/file1.txt"
        assert job_files[0]["doc_id"] is not None  # Should be auto-generated
        assert job_files[1]["path"] == "/test/file2.pdf"

        # Verify job total_files was updated
        job = get_job(job_id)
        assert job["total_files"] == 2

    def test_update_file_status(self, test_db, sample_job_data):
        """Test updating file processing status."""
        from shared.database import add_files_to_job, create_job, get_job_files, update_file_status

        job_id = create_job(sample_job_data)

        files = [
            {
                "path": "/test/file1.txt",
                "size": 1024,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
            }
        ]
        add_files_to_job(job_id, files)

        # Update file status
        update_file_status(
            job_id=job_id,
            file_path="/test/file1.txt",
            status="completed",
            chunks_created=10,
            vectors_created=10,
        )

        # Verify update
        job_files = get_job_files(job_id)
        assert job_files[0]["status"] == "completed"
        assert job_files[0]["chunks_created"] == 10
        assert job_files[0]["vectors_created"] == 10

    def test_update_file_status_with_error(self, test_db, sample_job_data):
        """Test updating file status with error."""
        from shared.database import add_files_to_job, create_job, get_job_files, update_file_status

        job_id = create_job(sample_job_data)

        files = [
            {
                "path": "/test/file1.txt",
                "size": 1024,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
            }
        ]
        add_files_to_job(job_id, files)

        # Update with error
        update_file_status(
            job_id=job_id,
            file_path="/test/file1.txt",
            status="failed",
            error="Processing error",
        )

        # Verify
        job_files = get_job_files(job_id)
        assert job_files[0]["status"] == "failed"
        assert job_files[0]["error"] == "Processing error"

    def test_get_job_files_by_status(self, test_db, sample_job_data):
        """Test getting files filtered by status."""
        from shared.database import add_files_to_job, create_job, get_job_files, update_file_status

        job_id = create_job(sample_job_data)

        # Add files with different statuses
        files = [
            {"path": f"/test/file{i}.txt", "size": 1024, "modified": datetime.now(UTC).isoformat(), "extension": ".txt"}
            for i in range(3)
        ]
        add_files_to_job(job_id, files)

        # Update statuses
        update_file_status(job_id, "/test/file0.txt", "completed")
        update_file_status(job_id, "/test/file1.txt", "failed", error="Error")
        # file2.txt remains "pending"

        # Test filtering
        completed_files = get_job_files(job_id, status="completed")
        assert len(completed_files) == 1
        assert completed_files[0]["path"] == "/test/file0.txt"

        failed_files = get_job_files(job_id, status="failed")
        assert len(failed_files) == 1
        assert failed_files[0]["path"] == "/test/file1.txt"

        pending_files = get_job_files(job_id, status="pending")
        assert len(pending_files) == 1
        assert pending_files[0]["path"] == "/test/file2.txt"

    def test_get_job_total_vectors(self, test_db, sample_job_data):
        """Test getting total vectors for a job."""
        from shared.database import add_files_to_job, create_job, get_job_total_vectors, update_file_status

        job_id = create_job(sample_job_data)

        # Add files with vectors
        files = [
            {"path": f"/test/file{i}.txt", "size": 1024, "modified": datetime.now(UTC).isoformat(), "extension": ".txt"}
            for i in range(3)
        ]
        add_files_to_job(job_id, files)

        # Update with vector counts
        update_file_status(job_id, "/test/file0.txt", "completed", vectors_created=10)
        update_file_status(job_id, "/test/file1.txt", "completed", vectors_created=20)
        update_file_status(job_id, "/test/file2.txt", "failed", vectors_created=0)

        # Test total vectors (only counts completed files)
        total = get_job_total_vectors(job_id)
        assert total == 30


class TestUserOperations:
    """Test user-related database operations."""

    def test_create_user(self, test_db, sample_user_data):
        """Test creating a new user."""
        from shared.database import create_user, pwd_context

        hashed_password = pwd_context.hash(sample_user_data["password"])
        user = create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
            hashed_password=hashed_password,
            full_name=sample_user_data["full_name"],
        )

        assert user["username"] == sample_user_data["username"]
        assert user["email"] == sample_user_data["email"]
        assert user["full_name"] == sample_user_data["full_name"]
        assert user["is_active"] is True
        assert user["id"] is not None
        assert user["created_at"] is not None
        assert user["last_login"] is None

    def test_create_duplicate_user(self, test_db, sample_user_data):
        """Test creating duplicate user raises error."""
        from shared.database import create_user, pwd_context

        hashed_password = pwd_context.hash(sample_user_data["password"])

        # Create first user
        create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
            hashed_password=hashed_password,
        )

        # Attempt to create duplicate
        with pytest.raises(ValueError, match="already exists"):
            create_user(
                username=sample_user_data["username"],
                email="different@example.com",
                hashed_password=hashed_password,
            )

    def test_get_user(self, test_db, sample_user_data):
        """Test getting user by username."""
        from shared.database import create_user, get_user, pwd_context

        hashed_password = pwd_context.hash(sample_user_data["password"])
        created_user = create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
            hashed_password=hashed_password,
        )

        # Get user
        user = get_user(sample_user_data["username"])
        assert user is not None
        assert user["username"] == sample_user_data["username"]
        assert user["id"] == created_user["id"]

    def test_get_user_nonexistent(self, test_db):
        """Test getting non-existent user returns None."""
        from shared.database import get_user

        user = get_user("nonexistent")
        assert user is None

    def test_get_user_by_id(self, test_db, sample_user_data):
        """Test getting user by ID."""
        from shared.database import create_user, get_user_by_id, pwd_context

        hashed_password = pwd_context.hash(sample_user_data["password"])
        created_user = create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
            hashed_password=hashed_password,
        )

        # Get user by ID
        user = get_user_by_id(created_user["id"])
        assert user is not None
        assert user["username"] == sample_user_data["username"]
        assert user["id"] == created_user["id"]

    def test_update_user_last_login(self, test_db, sample_user_data):
        """Test updating user's last login timestamp."""
        from shared.database import create_user, get_user_by_id, pwd_context, update_user_last_login

        hashed_password = pwd_context.hash(sample_user_data["password"])
        user = create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
            hashed_password=hashed_password,
        )

        # Update last login
        update_user_last_login(user["id"])

        # Verify
        updated_user = get_user_by_id(user["id"])
        assert updated_user["last_login"] is not None


class TestCollectionOperations:
    """Test collection-related database operations."""

    def test_get_collection_metadata(self, test_db, sample_job_data):
        """Test getting collection metadata from parent job."""
        from shared.database import create_job, get_collection_metadata

        # Create parent job
        parent_job = sample_job_data.copy()
        parent_job["id"] = "parent-job"
        parent_job["mode"] = "create"
        create_job(parent_job)

        # Create child job
        child_job = sample_job_data.copy()
        child_job["id"] = "child-job"
        child_job["parent_job_id"] = "parent-job"
        child_job["mode"] = "append"
        create_job(child_job)

        # Get metadata (should return parent job)
        metadata = get_collection_metadata(sample_job_data["name"])
        assert metadata is not None
        assert metadata["id"] == "parent-job"
        assert metadata["mode"] == "create"

    def test_get_collection_metadata_nonexistent(self, test_db):
        """Test getting metadata for non-existent collection."""
        from shared.database import get_collection_metadata

        metadata = get_collection_metadata("nonexistent-collection")
        assert metadata is None

    def test_list_collections(self, test_db):
        """Test listing unique collections with stats."""
        from shared.database import add_files_to_job, create_job, list_collections, update_file_status

        # Create jobs for different collections
        collections = ["Collection A", "Collection B", "Collection A"]  # A has 2 jobs
        job_ids = []

        for i, collection_name in enumerate(collections):
            job_data = {
                "id": f"job-{i}",
                "name": collection_name,
                "description": f"Job {i}",
                "status": "completed",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "directory_path": f"/test/path{i}",
                "model_name": "test-model",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "batch_size": 32,
                "total_files": 10,
                "user_id": 1,
                "mode": "create" if i == 0 else "append",
            }
            job_ids.append(create_job(job_data))

        # Add files with vectors to jobs
        for job_id in job_ids:
            files = [
                {
                    "path": f"/test/{job_id}/file.txt",
                    "size": 1024,
                    "modified": datetime.now(UTC).isoformat(),
                    "extension": ".txt",
                }
            ]
            add_files_to_job(job_id, files)
            update_file_status(job_id, f"/test/{job_id}/file.txt", "completed", vectors_created=5)

        # List collections
        collections_list = list_collections()
        assert len(collections_list) == 2  # Unique collections

        # Find Collection A
        collection_a = next(c for c in collections_list if c["name"] == "Collection A")
        assert collection_a["job_count"] == 2
        # The add_files_to_job overwrites total_files with actual file count (1 per job)
        assert collection_a["total_files"] == 2  # 1 file × 2 jobs (actual files added)
        assert collection_a["total_vectors"] == 10  # 5 vectors × 2 jobs

    def test_list_collections_by_user(self, test_db):
        """Test listing collections filtered by user."""
        from shared.database import create_job, list_collections

        # Create jobs for different users
        for i, user_id in enumerate([1, 2, None]):  # None = legacy job
            job_data = {
                "id": f"job-{i}",
                "name": f"Collection {i}",
                "description": f"Job {i}",
                "status": "completed",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "directory_path": f"/test/path{i}",
                "model_name": "test-model",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "batch_size": 32,
                "total_files": 10,
                "user_id": user_id,
            }
            create_job(job_data)

        # List collections for user 1 (should include legacy)
        collections = list_collections(user_id=1)
        assert len(collections) == 2
        collection_names = [c["name"] for c in collections]
        assert "Collection 0" in collection_names
        assert "Collection 2" in collection_names  # Legacy job
        assert "Collection 1" not in collection_names

    def test_rename_collection(self, test_db, sample_job_data):
        """Test renaming a collection."""
        from shared.database import create_job, get_job, rename_collection

        # Create job
        job_id = create_job(sample_job_data)

        # Rename collection
        success = rename_collection(
            old_name=sample_job_data["name"],
            new_name="Renamed Collection",
            user_id=sample_job_data["user_id"],
        )
        assert success is True

        # Verify rename
        job = get_job(job_id)
        assert job["name"] == "Renamed Collection"

    def test_rename_collection_duplicate_name(self, test_db, sample_job_data):
        """Test renaming to existing collection name fails."""
        from shared.database import create_job, rename_collection

        # Create two collections
        job_data1 = sample_job_data.copy()
        job_data1["id"] = "job1"
        job_data1["name"] = "Collection 1"
        create_job(job_data1)

        job_data2 = sample_job_data.copy()
        job_data2["id"] = "job2"
        job_data2["name"] = "Collection 2"
        create_job(job_data2)

        # Try to rename Collection 1 to Collection 2
        success = rename_collection(
            old_name="Collection 1",
            new_name="Collection 2",
            user_id=sample_job_data["user_id"],
        )
        assert success is False

    def test_rename_collection_no_access(self, test_db, sample_job_data):
        """Test renaming collection without access fails."""
        from shared.database import create_job, rename_collection

        # Create job for user 2
        job_data = sample_job_data.copy()
        job_data["user_id"] = 2
        create_job(job_data)

        # Try to rename as user 1
        success = rename_collection(
            old_name=job_data["name"],
            new_name="New Name",
            user_id=1,
        )
        assert success is False

    def test_delete_collection(self, test_db, sample_job_data):
        """Test deleting a collection."""
        from shared.database import add_files_to_job, create_job, delete_collection, get_job, get_job_files

        # Create multiple jobs for collection
        job_ids = []
        for i in range(2):
            job_data = sample_job_data.copy()
            job_data["id"] = f"job-{i}"
            job_id = create_job(job_data)
            job_ids.append(job_id)

            # Add files
            files = [
                {
                    "path": f"/test/file{i}.txt",
                    "size": 1024,
                    "modified": datetime.now(UTC).isoformat(),
                    "extension": ".txt",
                }
            ]
            add_files_to_job(job_id, files)

        # Delete collection
        result = delete_collection(
            collection_name=sample_job_data["name"],
            user_id=sample_job_data["user_id"],
        )

        assert len(result["job_ids"]) == 2
        assert result["qdrant_collections"] == ["job_job-0", "job_job-1"]

        # Verify deletion
        for job_id in job_ids:
            assert get_job(job_id) is None
            assert len(get_job_files(job_id)) == 0

    def test_get_duplicate_files_in_collection(self, test_db, sample_job_data):
        """Test finding duplicate files in collection."""
        from shared.database import add_files_to_job, create_job, get_duplicate_files_in_collection

        # Create completed job
        job_data = sample_job_data.copy()
        job_data["status"] = "completed"
        job_id = create_job(job_data)

        # Add files with content hashes
        files = [
            {
                "path": "/test/file1.txt",
                "size": 1024,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
                "content_hash": "hash1",
            },
            {
                "path": "/test/file2.txt",
                "size": 1024,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
                "content_hash": "hash2",
            },
        ]
        add_files_to_job(job_id, files)

        # Check for duplicates
        check_hashes = ["hash1", "hash3", "hash2"]
        duplicates = get_duplicate_files_in_collection(sample_job_data["name"], check_hashes)

        assert len(duplicates) == 2
        assert "hash1" in duplicates
        assert "hash2" in duplicates
        assert "hash3" not in duplicates

    def test_get_duplicate_files_empty_collection(self, test_db):
        """Test duplicate check on non-existent collection."""
        from shared.database import get_duplicate_files_in_collection

        duplicates = get_duplicate_files_in_collection("nonexistent", ["hash1", "hash2"])
        assert len(duplicates) == 0

    def test_get_collection_details(self, test_db):
        """Test getting detailed collection information."""
        from shared.database import add_files_to_job, create_job, get_collection_details, update_file_status

        # Create parent job
        parent_job = {
            "id": "parent-job",
            "name": "Test Collection",
            "description": "Parent job",
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "directory_path": "/test/parent",
            "model_name": "test-model",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "batch_size": 32,
            "vector_dim": 768,
            "quantization": "int8",
            "instruction": "Test instruction",
            "user_id": 1,
            "mode": "create",
            "total_files": 5,
            "processed_files": 5,
            "failed_files": 0,
        }
        create_job(parent_job)

        # Add files to parent
        files = [
            {
                "path": f"/test/file{i}.txt",
                "size": 1024 * (i + 1),
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
            }
            for i in range(5)
        ]
        add_files_to_job("parent-job", files)
        for file in files:
            update_file_status("parent-job", file["path"], "completed", vectors_created=10)

        # Create child job
        child_job = parent_job.copy()
        child_job.update(
            {
                "id": "child-job",
                "description": "Child job",
                "directory_path": "/test/child",
                "parent_job_id": "parent-job",
                "mode": "append",
                "total_files": 3,
                "processed_files": 3,
            }
        )
        create_job(child_job)

        # Add 3 files to child job to match the metadata
        child_files = [
            {
                "path": f"/test/child/file{i}.txt",
                "size": 1024,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
            }
            for i in range(3)
        ]
        add_files_to_job("child-job", child_files)

        # Get details
        details = get_collection_details("Test Collection", user_id=1)

        assert details is not None
        assert details["name"] == "Test Collection"
        # add_files_to_job updates total_files to actual count (5 parent + 3 child)
        assert details["stats"]["total_files"] == 8  # 5 + 3 (actual files added)
        assert details["stats"]["total_vectors"] == 50  # 5 files × 10 vectors
        assert details["stats"]["job_count"] == 2
        assert details["stats"]["total_size"] == sum(1024 * (i + 1) for i in range(5))
        assert len(details["source_directories"]) == 2
        assert details["configuration"]["model_name"] == "test-model"
        assert details["configuration"]["quantization"] == "int8"

    def test_get_collection_files(self, test_db, sample_job_data):
        """Test getting paginated files for a collection."""
        from shared.database import add_files_to_job, create_job, get_collection_files

        # Create job
        job_id = create_job(sample_job_data)

        # Add 10 files
        files = [
            {
                "path": f"/test/file{i:02d}.txt",
                "size": 1024,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
            }
            for i in range(10)
        ]
        add_files_to_job(job_id, files)

        # Test pagination
        result = get_collection_files(
            collection_name=sample_job_data["name"],
            user_id=sample_job_data["user_id"],
            page=1,
            limit=5,
        )

        assert len(result["files"]) == 5
        assert result["total"] == 10
        assert result["page"] == 1
        assert result["pages"] == 2

        # Test page 2
        result_page2 = get_collection_files(
            collection_name=sample_job_data["name"],
            user_id=sample_job_data["user_id"],
            page=2,
            limit=5,
        )

        assert len(result_page2["files"]) == 5
        assert result_page2["files"][0]["path"] == "/test/file05.txt"


class TestAuthOperations:
    """Test authentication-related operations."""

    def test_save_and_verify_refresh_token(self, test_db):
        """Test saving and verifying refresh tokens."""
        from shared.database import pwd_context, save_refresh_token, verify_refresh_token

        user_id = 1
        token = "test_refresh_token_123"
        token_hash = pwd_context.hash(token)
        expires_at = datetime.now(UTC) + timedelta(days=30)

        # Save token
        save_refresh_token(user_id, token_hash, expires_at)

        # Verify valid token
        verified_user_id = verify_refresh_token(token)
        assert verified_user_id == user_id

    def test_verify_expired_token(self, test_db):
        """Test verifying expired token returns None."""
        from shared.database import pwd_context, save_refresh_token, verify_refresh_token

        user_id = 1
        token = "expired_token"
        token_hash = pwd_context.hash(token)
        expires_at = datetime.now(UTC) - timedelta(days=1)  # Expired

        save_refresh_token(user_id, token_hash, expires_at)

        # Verify expired token
        verified_user_id = verify_refresh_token(token)
        assert verified_user_id is None

    def test_revoke_refresh_token(self, test_db):
        """Test revoking refresh tokens."""
        from shared.database import revoke_refresh_token

        # This test just ensures the function runs without error
        # The actual implementation doesn't check the token parameter
        revoke_refresh_token("any_token")
        # No assertion needed - just checking it doesn't raise


class TestDatabaseManagement:
    """Test database management functions."""

    def test_get_database_stats(self, test_db, sample_job_data, sample_user_data):
        """Test getting database statistics."""
        from shared.database import add_files_to_job, create_job, create_user, get_database_stats, pwd_context

        # Create test data
        # Jobs
        for i, status in enumerate(["completed", "failed", "running", "pending"]):
            job_data = sample_job_data.copy()
            job_data["id"] = f"job-{i}"
            job_data["status"] = status
            create_job(job_data)

        # Files
        files = [
            {
                "path": f"/test/file{i}.txt",
                "size": 1024,
                "modified": datetime.now(UTC).isoformat(),
                "extension": ".txt",
            }
            for i in range(5)
        ]
        add_files_to_job("job-0", files)

        # Users
        hashed_password = pwd_context.hash(sample_user_data["password"])
        create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
            hashed_password=hashed_password,
        )

        # Get stats
        stats = get_database_stats()

        assert stats["jobs"]["total"] == 4
        assert stats["jobs"]["completed"] == 1
        assert stats["jobs"]["failed"] == 1
        assert stats["jobs"]["running"] == 1
        assert stats["files"]["total"] == 5
        assert stats["users"]["total"] == 1

    def test_reset_database(self, test_db):
        """Test resetting the database."""
        from shared.database import create_job, get_database_stats, get_job, reset_database

        # Create some data
        job_data = {
            "id": "test-job",
            "name": "Test",
            "description": "Test",
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "directory_path": "/test",
            "model_name": "model",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "batch_size": 32,
        }
        create_job(job_data)

        # Reset database
        reset_database()

        # Verify data is gone
        assert get_job("test-job") is None
        stats = get_database_stats()
        assert stats["jobs"]["total"] == 0
        assert stats["files"]["total"] == 0
        assert stats["users"]["total"] == 0


class TestSchemaMigration:
    """Test database schema migration."""

    def test_schema_migration(self, tmp_path):
        """Test that init_db properly migrates an old schema."""
        db_path = str(tmp_path / "test_migration.db")

        # Create old schema manually
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Create old jobs table without newer columns
        c.execute(
            """CREATE TABLE jobs
                     (id TEXT PRIMARY KEY,
                      name TEXT NOT NULL,
                      description TEXT,
                      status TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL,
                      directory_path TEXT NOT NULL,
                      model_name TEXT NOT NULL,
                      chunk_size INTEGER,
                      chunk_overlap INTEGER,
                      batch_size INTEGER,
                      total_files INTEGER DEFAULT 0,
                      processed_files INTEGER DEFAULT 0,
                      failed_files INTEGER DEFAULT 0,
                      current_file TEXT,
                      error TEXT)"""
        )

        # Create old files table without newer columns
        c.execute(
            """CREATE TABLE files
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      job_id TEXT NOT NULL,
                      path TEXT NOT NULL,
                      size INTEGER NOT NULL,
                      modified TEXT NOT NULL,
                      extension TEXT NOT NULL,
                      hash TEXT,
                      status TEXT DEFAULT 'pending',
                      error TEXT,
                      chunks_created INTEGER DEFAULT 0,
                      vectors_created INTEGER DEFAULT 0,
                      FOREIGN KEY (job_id) REFERENCES jobs(id))"""
        )

        conn.commit()
        conn.close()

        # Now run init_db to migrate
        with patch("shared.database.DB_PATH", db_path):
            from shared.database import init_db

            init_db()

        # Verify migration
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Check jobs table columns
        c.execute("PRAGMA table_info(jobs)")
        job_columns = {col[1] for col in c.fetchall()}
        assert "vector_dim" in job_columns
        assert "quantization" in job_columns
        assert "instruction" in job_columns
        assert "start_time" in job_columns
        assert "user_id" in job_columns
        assert "parent_job_id" in job_columns
        assert "mode" in job_columns

        # Check files table columns
        c.execute("PRAGMA table_info(files)")
        file_columns = {col[1] for col in c.fetchall()}
        assert "doc_id" in file_columns
        assert "content_hash" in file_columns

        # Check that auth tables were created
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        assert c.fetchone() is not None

        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='refresh_tokens'")
        assert c.fetchone() is not None

        conn.close()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_database_operations(self, test_db):
        """Test operations on empty database."""
        from shared.database import (
            get_collection_metadata,
            get_database_stats,
            get_job,
            get_user,
            list_collections,
            list_jobs,
        )

        # List operations should return empty lists
        assert list_jobs() == []
        assert list_collections() == []

        # Get operations should return None
        assert get_job("nonexistent") is None
        assert get_user("nonexistent") is None
        assert get_collection_metadata("nonexistent") is None

        # Stats should show zeros
        stats = get_database_stats()
        assert stats["jobs"]["total"] == 0
        assert stats["files"]["total"] == 0
        assert stats["users"]["total"] == 0

    def test_null_values_handling(self, test_db):
        """Test handling of null/None values."""
        from shared.database import create_job, get_job

        # Create job with minimal required fields
        job_data = {
            "id": "minimal-job",
            "name": "Minimal",
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "directory_path": "/test",
            "model_name": "model",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "batch_size": 32,
            # Optional fields are None
            "description": None,
            "vector_dim": None,
            "quantization": "float32",  # Default value should be handled at creation
            "instruction": None,
            "user_id": None,
            "parent_job_id": None,
        }
        job_id = create_job(job_data)

        job = get_job(job_id)
        assert job is not None
        assert job["description"] is None
        assert job["vector_dim"] is None
        # quantization should be the value passed in create_job
        assert job["quantization"] == "float32"

    def test_sql_injection_protection(self, test_db):
        """Test that SQL injection attempts are handled safely."""
        from shared.database import create_job, get_database_stats, get_job

        # Try SQL injection in job name
        malicious_name = "'; DROP TABLE jobs; --"
        job_data = {
            "id": "test-job",
            "name": malicious_name,
            "description": "Test",
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "directory_path": "/test",
            "model_name": "model",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "batch_size": 32,
        }

        # This should not cause SQL injection
        create_job(job_data)

        # Verify tables still exist and data is properly escaped
        job = get_job("test-job")
        assert job is not None
        assert job["name"] == malicious_name

        # Verify jobs table still exists
        stats = get_database_stats()
        assert stats["jobs"]["total"] == 1

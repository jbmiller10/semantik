"""
Unit tests for add-to-collection functionality
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from shared import database
from webui.api.files import compute_file_content_hash


class TestContentHashing:
    """Test content-based hashing for deduplication"""

    def test_compute_file_content_hash_regular_file(self, tmp_path):
        """Test hashing a regular file"""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, this is test content!"
        test_file.write_bytes(test_content)

        # Compute hash
        hash_result = compute_file_content_hash(test_file)

        # Verify hash is correct
        expected_hash = hashlib.sha256(test_content).hexdigest()
        assert hash_result == expected_hash

    def test_compute_file_content_hash_symlink(self, tmp_path):
        """Test hashing a symbolic link"""
        # Create a target file
        target_file = tmp_path / "target.txt"
        target_file.write_text("target content")

        # Create a symlink
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(target_file)

        # Compute hash
        hash_result = compute_file_content_hash(symlink)

        # Should start with "symlink:" prefix
        assert hash_result.startswith("symlink:")

    def test_compute_file_content_hash_permission_error(self, tmp_path):
        """Test handling permission errors"""
        # Create a file
        test_file = tmp_path / "restricted.txt"
        test_file.write_text("content")

        # Make it unreadable (Unix only)
        import os

        if os.name != "nt":
            # Skip test if running as root (root can read any file)
            if os.getuid() == 0:
                pytest.skip("Permission test not applicable when running as root")

            try:
                test_file.chmod(0o000)

                # Should return None on permission error
                hash_result = compute_file_content_hash(test_file)
                assert hash_result is None
            finally:
                # Always restore permissions for cleanup
                test_file.chmod(0o644)
        else:
            # Skip test on Windows
            pytest.skip("Permission test not applicable on Windows")

    def test_compute_file_content_hash_nonexistent_file(self):
        """Test handling non-existent files"""
        fake_path = Path("/does/not/exist/file.txt")
        hash_result = compute_file_content_hash(fake_path)
        assert hash_result is None


class TestDuplicateDetection:
    """Test duplicate file detection in collections"""

    def setup_method(self):
        """Setup test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.original_db_path = database.DB_PATH
        database.DB_PATH = self.temp_db.name
        database.init_db()

    def teardown_method(self):
        """Cleanup test database"""
        database.DB_PATH = self.original_db_path
        self.temp_db.close()
        Path(self.temp_db.name).unlink()

    def test_get_duplicate_files_empty_collection(self):
        """Test checking duplicates in non-existent collection"""
        duplicates = database.get_duplicate_files_in_collection("non_existent_collection", ["hash1", "hash2"])
        assert duplicates == set()

    def test_get_duplicate_files_with_matches(self):
        """Test finding duplicate files"""
        # Create a job
        import uuid
        job_id = f"test_job_{uuid.uuid4().hex[:8]}"
        database.create_job(
            {
                "id": job_id,
                "name": "test_collection",
                "description": "Test collection",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "total_files": 2,
                "model_name": "test_model",
                "directory_path": "/test",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "batch_size": 32,
                "user_id": 1,
            }
        )

        # Add files with content hashes
        database.add_files_to_job(
            job_id,
            [
                {
                    "path": "/test/file1.txt",
                    "size": 100,
                    "modified": "2024-01-01T00:00:00",
                    "extension": ".txt",
                    "content_hash": "hash1",
                },
                {
                    "path": "/test/file2.txt",
                    "size": 200,
                    "modified": "2024-01-01T00:00:00",
                    "extension": ".txt",
                    "content_hash": "hash2",
                },
            ],
        )

        # Check for duplicates
        duplicates = database.get_duplicate_files_in_collection("test_collection", ["hash1", "hash3", "hash2"])

        assert duplicates == {"hash1", "hash2"}

    def test_get_duplicate_files_incomplete_job(self):
        """Test that incomplete jobs are ignored"""
        # Create a failed job
        import uuid
        job_id = f"failed_job_{uuid.uuid4().hex[:8]}"
        database.create_job(
            {
                "id": job_id,
                "name": "incomplete_test_collection",
                "description": "Test collection",
                "status": "failed",  # Not completed
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "total_files": 1,
                "model_name": "test_model",
                "directory_path": "/test",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "batch_size": 32,
                "user_id": 1,
            }
        )

        # Add file
        database.add_files_to_job(
            job_id,
            [
                {
                    "path": "/test/file1.txt",
                    "size": 100,
                    "modified": "2024-01-01T00:00:00",
                    "extension": ".txt",
                    "content_hash": "hash1",
                }
            ],
        )

        # Should not find duplicates from failed job
        duplicates = database.get_duplicate_files_in_collection("incomplete_test_collection", ["hash1"])

        assert duplicates == set()


class TestSettingsInheritance:
    """Test settings inheritance for append mode"""

    def setup_method(self):
        """Setup test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.original_db_path = database.DB_PATH
        database.DB_PATH = self.temp_db.name
        database.init_db()

    def teardown_method(self):
        """Cleanup test database"""
        database.DB_PATH = self.original_db_path
        self.temp_db.close()
        Path(self.temp_db.name).unlink()

    def test_get_collection_metadata(self):
        """Test retrieving collection metadata"""
        # Create a parent job
        import uuid
        parent_job_id = f"parent_job_{uuid.uuid4().hex[:8]}"
        collection_name = f"metadata_test_collection_{uuid.uuid4().hex[:8]}"
        database.create_job(
            {
                "id": parent_job_id,
                "name": collection_name,
                "description": "Test collection",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "total_files": 1,
                "model_name": "test_model",
                "directory_path": "/test",
                "chunk_size": 600,
                "chunk_overlap": 200,
                "batch_size": 96,
                "quantization": "float32",
                "vector_dim": 768,
                "instruction": "Test instruction",
                "user_id": 1,
            }
        )

        # Get metadata
        metadata = database.get_collection_metadata(collection_name)

        assert metadata is not None
        assert metadata["id"] == parent_job_id
        assert metadata["model_name"] == "test_model"
        assert metadata["chunk_size"] == 600
        assert metadata["chunk_overlap"] == 200
        assert metadata["batch_size"] == 96
        assert metadata["quantization"] == "float32"
        assert metadata["vector_dim"] == 768
        assert metadata["instruction"] == "Test instruction"

    def test_get_collection_metadata_not_found(self):
        """Test metadata for non-existent collection"""
        metadata = database.get_collection_metadata("non_existent")
        assert metadata is None

    def test_append_job_inherits_settings(self):
        """Test that append jobs inherit parent settings"""
        # Create parent job
        import uuid  
        parent_job_id = f"parent_job_{uuid.uuid4().hex[:8]}"
        database.create_job(
            {
                "id": parent_job_id,
                "name": "test_collection",
                "description": "Test collection",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "total_files": 1,
                "model_name": "parent_model",
                "directory_path": "/parent",
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "batch_size": 64,
                "quantization": "int8",
                "vector_dim": 384,
                "instruction": "Parent instruction",
                "user_id": 1,
            }
        )

        # Create append job
        append_job_id = f"append_job_{uuid.uuid4().hex[:8]}"
        database.create_job(
            {
                "id": append_job_id,
                "name": "test_collection",
                "description": "Test collection",
                "status": "created",
                "created_at": "2024-01-02T00:00:00",
                "updated_at": "2024-01-02T00:00:00",
                "total_files": 0,
                "model_name": "parent_model",  # Should match parent
                "directory_path": "/append",
                "chunk_size": 1000,  # Should match parent
                "chunk_overlap": 100,  # Should match parent
                "batch_size": 64,  # Should match parent
                "quantization": "int8",  # Should match parent
                "vector_dim": 384,  # Should match parent
                "instruction": "Parent instruction",  # Should match parent
                "user_id": 1,
                "parent_job_id": parent_job_id,
                "mode": "append",
            }
        )

        # Get the append job
        append_job = database.get_job(append_job_id)

        assert append_job["parent_job_id"] == parent_job_id
        assert append_job["mode"] == "append"
        assert append_job["model_name"] == "parent_model"
        assert append_job["chunk_size"] == 1000
        assert append_job["chunk_overlap"] == 100
        assert append_job["batch_size"] == 64
        assert append_job["quantization"] == "int8"
        assert append_job["vector_dim"] == 384
        assert append_job["instruction"] == "Parent instruction"


class TestResourceLimits:
    """Test resource limit enforcement"""

    def test_scan_directory_file_warning(self, tmp_path):
        """Test that file count warnings are generated"""
        from webui.api.files import scan_directory

        # Create many files
        for i in range(5):
            (tmp_path / f"file{i}.txt").write_text(f"content {i}")

        # Should return warnings instead of raising error
        result = scan_directory(str(tmp_path))
        assert len(result["files"]) == 5
        assert result["total_files"] == 5
        # No warning for just 5 files
        assert len(result["warnings"]) == 0

    def test_scan_directory_size_warning(self, tmp_path):
        """Test that total size warnings are generated"""
        from webui.api.files import scan_directory

        # Create a large file (simulate)
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 1000)  # Small file for test

        # Mock stat to return large size
        original_stat = Path.stat

        def mock_stat(self, **kwargs):
            result = original_stat(self, **kwargs)
            if self.name == "large.txt":
                # Mock the file as being 60GB
                # Create a proper mock with all required attributes
                class MockStatResult:
                    st_size = 60 * 1024 * 1024 * 1024
                    st_mtime = result.st_mtime
                    st_atime = result.st_atime
                    st_ctime = result.st_ctime
                    st_mode = result.st_mode
                    st_ino = result.st_ino
                    st_dev = result.st_dev
                    st_nlink = result.st_nlink
                    st_uid = result.st_uid
                    st_gid = result.st_gid

                return MockStatResult()
            return result

        with patch.object(Path, "stat", mock_stat):
            result = scan_directory(str(tmp_path))
            assert len(result["files"]) == 1
            assert result["total_size"] == 60 * 1024 * 1024 * 1024
            # Should have a size warning
            assert len(result["warnings"]) == 1
            assert result["warnings"][0]["type"] == "high_total_size"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

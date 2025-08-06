"""Unit tests for packages/vecpipe/ingest_qdrant.py"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from qdrant_client.models import PointStruct

from packages.vecpipe.ingest_qdrant import move_file, process_parquet_file


class TestProcessParquetFile:
    """Test suite for process_parquet_file function"""

    @patch("packages.vecpipe.ingest_qdrant.pq.read_table")
    def test_process_parquet_file_success(self, mock_read_table) -> None:
        """Test successful processing of a parquet file"""
        # Create mock table with column structure
        mock_table = MagicMock()

        # Mock the column method to return mock columns
        id_column = MagicMock()
        id_column.to_pylist.return_value = ["id1", "id2", "id3"]

        vector_column = MagicMock()
        vector_column.to_pylist.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        payload_column = MagicMock()
        payload_column.to_pylist.return_value = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]

        mock_table.column.side_effect = lambda name: {
            "id": id_column,
            "vector": vector_column,
            "payload": payload_column,
        }[name]

        mock_read_table.return_value = mock_table

        # Mock Qdrant client
        mock_client = MagicMock()
        mock_client.upsert.return_value = True

        # Test with batch size of 2
        result = process_parquet_file("/test/file.parquet", mock_client, batch_size=2)  # Pass as string, not Path

        # Verify success
        assert result is True

        # Verify file was read
        mock_read_table.assert_called_once_with("/test/file.parquet")

        # Verify upsert was called twice (3 records with batch size 2)
        assert mock_client.upsert.call_count == 2

        # Verify first batch
        first_call = mock_client.upsert.call_args_list[0]
        assert first_call.kwargs["collection_name"] == "test_collection"  # Test environment collection
        points = first_call.kwargs["points"]
        assert len(points) == 2
        assert isinstance(points[0], PointStruct)
        assert points[0].id == "id1"
        assert points[0].vector == [0.1, 0.2]
        assert points[0].payload == {"text": "doc1"}

        # Verify second batch
        second_call = mock_client.upsert.call_args_list[1]
        points = second_call.kwargs["points"]
        assert len(points) == 1
        assert points[0].id == "id3"

    @patch("packages.vecpipe.ingest_qdrant.pq.read_table")
    def test_process_parquet_file_empty_file(self, mock_read_table) -> None:
        """Test processing an empty parquet file"""
        # Create mock table with empty columns
        mock_table = MagicMock()

        id_column = MagicMock()
        id_column.to_pylist.return_value = []

        vector_column = MagicMock()
        vector_column.to_pylist.return_value = []

        payload_column = MagicMock()
        payload_column.to_pylist.return_value = []

        mock_table.column.side_effect = lambda name: {
            "id": id_column,
            "vector": vector_column,
            "payload": payload_column,
        }[name]

        mock_read_table.return_value = mock_table

        mock_client = MagicMock()

        result = process_parquet_file("/test/empty.parquet", mock_client)

        # Should succeed but not call upsert
        assert result is True
        mock_client.upsert.assert_not_called()

    @patch("packages.vecpipe.ingest_qdrant.pq.read_table")
    def test_process_parquet_file_missing_columns(self, mock_read_table) -> None:
        """Test handling of parquet file with missing required columns"""
        # Create mock table that raises KeyError for missing column
        mock_table = MagicMock()

        id_column = MagicMock()
        id_column.to_pylist.return_value = ["id1"]

        payload_column = MagicMock()
        payload_column.to_pylist.return_value = [{"text": "doc1"}]

        def mock_column(name):
            if name == "vector":
                raise KeyError("Column 'vector' not found")
            return {"id": id_column, "payload": payload_column}[name]

        mock_table.column.side_effect = mock_column
        mock_read_table.return_value = mock_table

        mock_client = MagicMock()

        result = process_parquet_file("/test/invalid.parquet", mock_client)

        # Should fail
        assert result is False

    @patch("packages.vecpipe.ingest_qdrant.pq.read_table")
    @patch("packages.vecpipe.ingest_qdrant.time.sleep")
    def test_process_parquet_file_with_retries(self, mock_sleep, mock_read_table) -> None:
        """Test retry logic when upsert fails"""
        # Setup mock table
        mock_table = MagicMock()

        id_column = MagicMock()
        id_column.to_pylist.return_value = ["id1"]

        vector_column = MagicMock()
        vector_column.to_pylist.return_value = [[0.1, 0.2]]

        payload_column = MagicMock()
        payload_column.to_pylist.return_value = [{"text": "doc1"}]

        mock_table.column.side_effect = lambda name: {
            "id": id_column,
            "vector": vector_column,
            "payload": payload_column,
        }[name]

        mock_read_table.return_value = mock_table

        # Mock client that fails twice then succeeds
        mock_client = MagicMock()
        mock_client.upsert.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            True,  # Success on third attempt
        ]

        result = process_parquet_file("/test/file.parquet", mock_client)

        # Should eventually succeed
        assert result is True
        assert mock_client.upsert.call_count == 3

        # Verify exponential backoff was used
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)  # First retry: RETRY_DELAY * (2^0) = 2
        mock_sleep.assert_any_call(4)  # Second retry: RETRY_DELAY * (2^1) = 4

    @patch("packages.vecpipe.ingest_qdrant.pq.read_table")
    @patch("packages.vecpipe.ingest_qdrant.time.sleep")
    def test_process_parquet_file_max_retries_exceeded(self, mock_sleep, mock_read_table) -> None:
        """Test failure when max retries are exceeded"""
        # Setup mock table
        mock_table = MagicMock()

        id_column = MagicMock()
        id_column.to_pylist.return_value = ["id1"]

        vector_column = MagicMock()
        vector_column.to_pylist.return_value = [[0.1, 0.2]]

        payload_column = MagicMock()
        payload_column.to_pylist.return_value = [{"text": "doc1"}]

        mock_table.column.side_effect = lambda name: {
            "id": id_column,
            "vector": vector_column,
            "payload": payload_column,
        }[name]

        mock_read_table.return_value = mock_table

        # Mock client that always fails
        mock_client = MagicMock()
        mock_client.upsert.side_effect = Exception("Persistent error")

        result = process_parquet_file("/test/file.parquet", mock_client)

        # Should fail after max retries
        assert result is False
        assert mock_client.upsert.call_count == 5  # Initial + 4 retries

    @patch("packages.vecpipe.ingest_qdrant.pq.read_table")
    def test_process_parquet_file_read_error(self, mock_read_table) -> None:
        """Test handling of parquet read errors"""
        mock_read_table.side_effect = Exception("Corrupt file")
        mock_client = MagicMock()

        result = process_parquet_file("/test/corrupt.parquet", mock_client)

        assert result is False
        mock_client.upsert.assert_not_called()


class TestMoveFile:
    """Test suite for move_file function"""

    def test_move_file_success(self) -> None:
        """Test successful file move"""
        source = "/source/file.parquet"
        dest_dir = "/dest"

        with patch.object(Path, "mkdir") as mock_mkdir, patch.object(Path, "rename") as mock_rename:
            move_file(source, dest_dir)

            # Verify directory was created
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Verify file was renamed/moved
            mock_rename.assert_called_once()

            # Verify the destination path
            expected_dst = Path(dest_dir) / "file.parquet"
            actual_dst = mock_rename.call_args[0][0]
            assert str(actual_dst) == str(expected_dst)

    def test_move_file_creates_directory(self) -> None:
        """Test that destination directory is created if it doesn't exist"""
        source = "/source/file.parquet"
        dest_dir = "/dest"

        with patch.object(Path, "mkdir") as mock_mkdir, patch.object(Path, "rename") as mock_rename:
            move_file(source, dest_dir)

            # Verify directory creation was attempted
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Verify rename was called
            mock_rename.assert_called_once()

    def test_move_file_handles_error(self) -> None:
        """Test error handling during file move"""
        source = "/source/file.parquet"
        dest_dir = "/dest"

        with patch.object(Path, "mkdir"), patch.object(Path, "rename") as mock_rename:
            # Make rename raise an error
            mock_rename.side_effect = OSError("Permission denied")

            # Should raise since there's no error handling in move_file
            with pytest.raises(OSError, match="Permission denied"):
                move_file(source, dest_dir)


class TestFileHandlingIntegration:
    """Test the integration of process_parquet_file with file movement"""

    @patch("packages.vecpipe.ingest_qdrant.pq.read_table")
    @patch("packages.vecpipe.ingest_qdrant.move_file")
    def test_successful_processing_moves_to_loaded(self, mock_move_file, mock_read_table) -> None:
        """Test that successfully processed files are moved to loaded directory"""
        # Setup successful processing
        mock_df = pd.DataFrame({"id": ["id1"], "vector": [[0.1, 0.2]], "payload": [{"text": "doc1"}]})
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = mock_df
        mock_read_table.return_value = mock_table

        mock_client = MagicMock()
        mock_client.upsert.return_value = True

        file_path = "/data/file.parquet"

        # Process file
        result = process_parquet_file(file_path, mock_client)

        assert result is True

        # In actual usage, the calling code would handle the move
        # This test verifies the return value that triggers the move

    @patch("packages.vecpipe.ingest_qdrant.pq.read_table")
    @patch("packages.vecpipe.ingest_qdrant.move_file")
    def test_failed_processing_moves_to_rejects(self, mock_move_file, mock_read_table) -> None:
        """Test that failed files would be moved to rejects directory"""
        # Setup failed processing
        mock_read_table.side_effect = Exception("Read error")

        mock_client = MagicMock()
        file_path = "/data/file.parquet"

        # Process file
        result = process_parquet_file(file_path, mock_client)

        assert result is False

        # In actual usage, the calling code would handle the move
        # This test verifies the return value that triggers the move

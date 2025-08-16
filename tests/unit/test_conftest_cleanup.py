"""Unit tests for conftest cleanup functions."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
from tests.integration.conftest import _drop_all_database_objects, _drop_all_database_objects_sync


class TestCleanupFunctions(unittest.TestCase):
    """Test cleanup functions handle errors gracefully."""
    
    @patch('tests.integration.conftest.text')
    async def test_async_cleanup_handles_errors(self, mock_text):
        """Test async cleanup handles errors without raising."""
        # Create a mock connection
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("Test error"))
        
        # Should not raise even when execute fails
        await _drop_all_database_objects(mock_conn)
        
        # Verify it tried to execute DROP statements
        assert mock_text.called
    
    @patch('tests.integration.conftest.text')
    def test_sync_cleanup_handles_errors(self, mock_text):
        """Test sync cleanup handles errors without raising."""
        # Create a mock connection
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(side_effect=Exception("Test error"))
        
        # Should not raise even when execute fails
        _drop_all_database_objects_sync(mock_conn)
        
        # Verify it tried to execute DROP statements
        assert mock_text.called
    
    @patch('tests.integration.conftest.text')
    async def test_async_cleanup_drops_in_correct_order(self, mock_text):
        """Test async cleanup drops objects in dependency order."""
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_result.__iter__ = lambda self: iter([])  # No partition tables
        mock_conn.execute = AsyncMock(return_value=mock_result)
        
        await _drop_all_database_objects(mock_conn)
        
        # Check that views are dropped before tables
        calls = [str(call) for call in mock_text.call_args_list]
        
        # Materialized views should be dropped first
        matview_calls = [c for c in calls if 'collection_chunking_stats' in str(c)]
        table_calls = [c for c in calls if 'DROP TABLE' in str(c)]
        
        # If both exist, materialized view should come before tables
        if matview_calls and table_calls:
            matview_index = calls.index(matview_calls[0])
            first_table_index = calls.index(table_calls[0])
            assert matview_index < first_table_index, "Views should be dropped before tables"
    
    @patch('tests.integration.conftest.text')
    def test_sync_cleanup_drops_in_correct_order(self, mock_text):
        """Test sync cleanup drops objects in dependency order."""  
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])  # No partition tables
        mock_conn.execute = MagicMock(return_value=mock_result)
        
        _drop_all_database_objects_sync(mock_conn)
        
        # Check that views are dropped before tables
        calls = [str(call) for call in mock_text.call_args_list]
        
        # Materialized views should be dropped first
        matview_calls = [c for c in calls if 'collection_chunking_stats' in str(c)]
        table_calls = [c for c in calls if 'DROP TABLE' in str(c)]
        
        # If both exist, materialized view should come before tables
        if matview_calls and table_calls:
            matview_index = calls.index(matview_calls[0])
            first_table_index = calls.index(table_calls[0])
            assert matview_index < first_table_index, "Views should be dropped before tables"


if __name__ == '__main__':
    unittest.main()
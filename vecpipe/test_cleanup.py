#!/usr/bin/env python3
"""
Test script for cleanup service
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vecpipe.cleanup import QdrantCleanupService
from vecpipe.extract_chunks import FileChangeTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cleanup_dry_run():
    """Test cleanup service in dry-run mode"""
    logger.info("Testing cleanup service in dry-run mode...")
    
    # Create temporary file list
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        # Add some test files (these don't need to exist for dry-run)
        test_files = [
            "/data/documents/file1.pdf",
            "/data/documents/file2.docx",
            "/data/documents/file3.txt"
        ]
        f.write('\0'.join(test_files).encode('utf-8'))
        temp_file_list = f.name
    
    try:
        # Initialize service
        service = QdrantCleanupService()
        
        # Get current files
        current_files = service.get_current_files(temp_file_list)
        logger.info(f"Current files: {len(current_files)}")
        
        # Get collections
        collections = service.get_job_collections()
        logger.info(f"Collections to check: {collections}")
        
        # Run cleanup in dry-run mode
        result = service.cleanup_removed_files(current_files, dry_run=True)
        logger.info(f"Dry-run result: {result}")
        
        service.close()
        
    finally:
        # Clean up temp file
        os.unlink(temp_file_list)

def test_tracker_integration():
    """Test FileChangeTracker integration"""
    logger.info("Testing FileChangeTracker integration...")
    
    # Create temporary tracking DB
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_db = f.name
    
    try:
        # Initialize tracker with temp DB
        tracker = FileChangeTracker(db_path=temp_db)
        
        # Simulate some tracked files
        tracker.tracking_data["files"] = {
            "/data/old/removed1.pdf": {
                "doc_id": "test_doc_1",
                "hash": "abc123",
                "last_seen": "2024-01-01T00:00:00"
            },
            "/data/old/removed2.docx": {
                "doc_id": "test_doc_2", 
                "hash": "def456",
                "last_seen": "2024-01-01T00:00:00"
            },
            "/data/current/exists.txt": {
                "doc_id": "test_doc_3",
                "hash": "ghi789",
                "last_seen": "2024-01-01T00:00:00"
            }
        }
        tracker.save()
        
        # Current files (only one exists)
        current_files = ["/data/current/exists.txt"]
        
        # Get removed files
        removed = tracker.get_removed_files(current_files)
        logger.info(f"Removed files detected: {len(removed)}")
        for r in removed:
            logger.info(f"  - {r['path']} (doc_id: {r['doc_id']})")
        
        # Test remove_file method
        tracker.remove_file("/data/old/removed1.pdf")
        tracker.save()
        
        # Verify removal
        assert "/data/old/removed1.pdf" not in tracker.tracking_data["files"]
        logger.info("remove_file() method works correctly")
        
    finally:
        # Clean up
        os.unlink(temp_db)

def main():
    """Run all tests"""
    logger.info("Starting cleanup service tests...")
    
    # Test tracker integration
    test_tracker_integration()
    
    # Test cleanup service
    test_cleanup_dry_run()
    
    logger.info("All tests completed!")

if __name__ == "__main__":
    main()
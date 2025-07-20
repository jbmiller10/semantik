"""Database connection utilities for SQLite."""
import sqlite3
import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_sqlite_connection(db_path: str) -> sqlite3.Connection:
    """Get a SQLite connection with proper settings for concurrency.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Configured SQLite connection
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    
    try:
        c = conn.cursor()
        
        # Enable WAL mode for better concurrent access
        c.execute("PRAGMA journal_mode=WAL")
        
        # Set busy timeout to 30 seconds
        c.execute("PRAGMA busy_timeout=30000")
        
        # Additional settings for better concurrency
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA temp_store=MEMORY")
        c.execute("PRAGMA mmap_size=30000000000")
        
        # Enable foreign keys
        c.execute("PRAGMA foreign_keys=ON")
        
        c.close()
        
        logger.debug("SQLite connection configured with WAL mode and concurrency settings")
        
    except Exception as e:
        conn.close()
        raise e
    
    return conn
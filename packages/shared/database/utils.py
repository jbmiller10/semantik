"""Database utility functions."""

import logging

logger = logging.getLogger(__name__)


def parse_user_id(user_id: str) -> int:
    """Convert string user ID to integer for SQLite compatibility.

    The repository interface uses string IDs for consistency across different
    storage backends (some databases use UUIDs, others use integers).
    The SQLite implementation requires integer IDs.

    Args:
        user_id: User ID as string

    Returns:
        User ID as integer

    Raises:
        ValueError: If user_id is not a valid integer string
    """
    try:
        return int(user_id)
    except ValueError:
        logger.error(f"Invalid user_id format: {user_id}")
        raise ValueError(f"user_id must be a valid integer, got: {user_id}") from None

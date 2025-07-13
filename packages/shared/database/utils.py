"""Database utility functions."""

import logging

from .exceptions import InvalidUserIdError

logger = logging.getLogger(__name__)


def parse_user_id(user_id: str | int) -> int:
    """Parse user ID with better error messages.

    The repository interface uses string IDs for consistency across different
    storage backends (some databases use UUIDs, others use integers).
    The SQLite implementation requires integer IDs.

    Args:
        user_id: User ID as string or integer

    Returns:
        User ID as integer

    Raises:
        ValueError: If user_id is not a valid integer or numeric string
    """
    if isinstance(user_id, int):
        return user_id
    try:
        return int(user_id)
    except ValueError:
        logger.error(f"Invalid user_id format: '{user_id}' must be numeric")
        raise InvalidUserIdError(str(user_id)) from None

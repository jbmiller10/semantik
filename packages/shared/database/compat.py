"""Temporary compatibility layer for tests.

This module provides a 'database' object that exposes the legacy functions
to support existing tests that mock webui.api.*.database attributes.

TODO: Remove this once all tests are migrated to use the repository pattern.
"""

from . import (
    create_user,
    get_user,
    get_user_by_id,
    revoke_refresh_token,
    save_refresh_token,
    update_user_last_login,
    verify_refresh_token,
)


# Create a namespace object that exposes all legacy functions
class DatabaseCompat:
    """Compatibility namespace for legacy database functions."""

    # User operations
    create_user = create_user
    get_user = get_user
    get_user_by_id = get_user_by_id
    update_user_last_login = update_user_last_login

    # Auth operations
    save_refresh_token = save_refresh_token
    verify_refresh_token = verify_refresh_token
    revoke_refresh_token = revoke_refresh_token


# Create a single instance to be imported
database = DatabaseCompat()
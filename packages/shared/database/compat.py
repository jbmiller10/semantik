"""Temporary compatibility layer for tests.

This module provides a 'database' object that exposes the legacy functions
to support existing tests that mock webui.api.*.database attributes.

TODO: Remove this once all tests are migrated to use the repository pattern.
"""

from . import (
    add_files_to_job,
    create_job,
    create_user,
    delete_collection,
    delete_job,
    get_collection_details,
    get_collection_files,
    get_collection_metadata,
    get_duplicate_files_in_collection,
    get_job,
    get_job_files,
    get_job_total_vectors,
    get_user,
    get_user_by_id,
    list_collections,
    list_jobs,
    rename_collection,
    revoke_refresh_token,
    save_refresh_token,
    update_file_status,
    update_job,
    update_user_last_login,
    verify_refresh_token,
)

# Create a namespace object that exposes all legacy functions
class DatabaseCompat:
    """Compatibility namespace for legacy database functions."""
    
    # Job operations
    create_job = create_job
    get_job = get_job
    update_job = update_job
    delete_job = delete_job
    list_jobs = list_jobs
    
    # File operations
    add_files_to_job = add_files_to_job
    update_file_status = update_file_status
    get_job_files = get_job_files
    get_job_total_vectors = get_job_total_vectors
    get_duplicate_files_in_collection = get_duplicate_files_in_collection
    
    # User operations
    create_user = create_user
    get_user = get_user
    get_user_by_id = get_user_by_id
    update_user_last_login = update_user_last_login
    
    # Collection operations
    get_collection_metadata = get_collection_metadata
    list_collections = list_collections
    get_collection_details = get_collection_details
    get_collection_files = get_collection_files
    rename_collection = rename_collection
    delete_collection = delete_collection
    
    # Auth operations
    save_refresh_token = save_refresh_token
    verify_refresh_token = verify_refresh_token
    revoke_refresh_token = revoke_refresh_token


# Create a single instance to be imported
database = DatabaseCompat()
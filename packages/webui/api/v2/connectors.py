"""
Connector API v2 endpoints.

This module provides RESTful API endpoints for connector operations,
including preview/validation endpoints and the connector catalog.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends

from webui.api.schemas import (
    ErrorResponse,
    GitPreviewRequest,
    GitPreviewResponse,
    ImapPreviewRequest,
    ImapPreviewResponse,
)
from webui.auth import get_current_user
from webui.services.connector_registry import (
    get_connector_catalog,
    get_connector_definition,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/connectors", tags=["connectors-v2"])


@router.get(
    "",
    response_model=dict[str, Any],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def list_connectors(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> dict[str, Any]:
    """Get the connector catalog.

    Returns all available connector types with their configuration schemas,
    including field definitions, secrets, and UI metadata.
    """
    return {"connectors": get_connector_catalog()}


@router.get(
    "/{connector_type}",
    response_model=dict[str, Any],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Connector type not found"},
    },
)
async def get_connector(
    connector_type: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> dict[str, Any]:
    """Get the definition for a specific connector type.

    Returns the configuration schema, field definitions, secrets,
    and UI metadata for the specified connector type.
    """
    from fastapi import HTTPException

    definition = get_connector_definition(connector_type)
    if definition is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown connector type: {connector_type}",
        )
    return {"type": connector_type, "definition": definition}


@router.post(
    "/preview/git",
    response_model=GitPreviewResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def preview_git(
    request: GitPreviewRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> GitPreviewResponse:
    """Preview/validate a Git repository connection.

    Tests connectivity to the Git repository with the provided credentials.
    Returns available refs (branches/tags) if authentication succeeds.

    This endpoint does NOT clone the repository - it only validates
    access using git ls-remote.
    """
    from shared.connectors.git import GitConnector

    try:
        # Build config for the connector
        config = {
            "repo_url": request.repo_url,
            "ref": request.ref,
            "auth_method": request.auth_method,
            "include_globs": request.include_globs,
            "exclude_globs": request.exclude_globs,
        }

        # Create connector and set credentials
        connector = GitConnector(config)

        if request.auth_method == "https_token" and request.token:
            connector.set_credentials(token=request.token)
        elif request.auth_method == "ssh_key" and request.ssh_key:
            connector.set_credentials(
                ssh_key=request.ssh_key,
                ssh_passphrase=request.ssh_passphrase,
            )

        # Attempt authentication (uses git ls-remote)
        valid = await connector.authenticate()

        if valid:
            # Get refs from connector
            refs_found = connector.get_refs()

            return GitPreviewResponse(
                valid=True,
                repo_url=request.repo_url,
                ref=request.ref,
                refs_found=refs_found,
                error=None,
            )

        return GitPreviewResponse(
            valid=False,
            repo_url=request.repo_url,
            ref=request.ref,
            refs_found=[],
            error="Authentication failed - could not access repository",
        )

    except ValueError as e:
        # Config validation errors
        return GitPreviewResponse(
            valid=False,
            repo_url=request.repo_url,
            ref=request.ref,
            refs_found=[],
            error=str(e),
        )
    except Exception as e:
        logger.error("Git preview failed: %s", e, exc_info=True)
        return GitPreviewResponse(
            valid=False,
            repo_url=request.repo_url,
            ref=request.ref,
            refs_found=[],
            error="Connection failed. Check repository URL and credentials.",
        )


@router.post(
    "/preview/imap",
    response_model=ImapPreviewResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def preview_imap(
    request: ImapPreviewRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> ImapPreviewResponse:
    """Preview/validate an IMAP connection.

    Tests connectivity to the IMAP server with the provided credentials.
    Returns available mailboxes if authentication succeeds.

    This endpoint does NOT fetch emails - it only validates
    access and lists available mailboxes.
    """
    import asyncio
    import imaplib
    import ssl

    try:

        def _connect_and_list() -> list[str]:
            """Connect to IMAP and list mailboxes."""
            # Connect
            conn: imaplib.IMAP4
            if request.use_ssl:
                context = ssl.create_default_context()
                conn = imaplib.IMAP4_SSL(request.host, request.port, ssl_context=context)
            else:
                conn = imaplib.IMAP4(request.host, request.port)

            try:
                # Login
                conn.login(request.username, request.password)

                # List mailboxes
                status, data = conn.list()
                mailboxes: list[str] = []
                if status == "OK" and data:
                    for item in data:
                        if isinstance(item, bytes):
                            # Parse mailbox name from LIST response
                            # Format: (\\Flags) "/" "MailboxName"
                            parts = item.decode("utf-8", errors="replace").split(' "/" ')
                            if len(parts) >= 2:
                                name = parts[-1].strip('"')
                                mailboxes.append(name)
                            else:
                                # Try alternative delimiter
                                parts = item.decode("utf-8", errors="replace").split(' "." ')
                                if len(parts) >= 2:
                                    name = parts[-1].strip('"')
                                    mailboxes.append(name)

                return mailboxes

            finally:
                conn.logout()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        mailboxes_found = await loop.run_in_executor(None, _connect_and_list)

        return ImapPreviewResponse(
            valid=True,
            host=request.host,
            username=request.username,
            mailboxes_found=mailboxes_found,
            error=None,
        )

    except imaplib.IMAP4.error as e:
        logger.warning("IMAP authentication/protocol error: %s", e, exc_info=True)
        return ImapPreviewResponse(
            valid=False,
            host=request.host,
            username=request.username,
            mailboxes_found=[],
            error="IMAP authentication failed. Check username and password.",
        )
    except (OSError, TimeoutError) as e:
        logger.warning("IMAP connection error: %s", e, exc_info=True)
        return ImapPreviewResponse(
            valid=False,
            host=request.host,
            username=request.username,
            mailboxes_found=[],
            error="Connection failed. Check host, port, and SSL settings.",
        )
    except Exception as e:
        logger.error("IMAP preview failed: %s", e, exc_info=True)
        return ImapPreviewResponse(
            valid=False,
            host=request.host,
            username=request.username,
            mailboxes_found=[],
            error="An unexpected error occurred. Please try again.",
        )

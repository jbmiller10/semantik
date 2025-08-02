#!/usr/bin/env python3
"""
Example of how to integrate chunking exception handlers and correlation middleware
into the main FastAPI application.

This file demonstrates the changes needed in main.py to enable the new functionality.
"""

# Example modifications for packages/webui/main.py:

# Add these imports to the existing imports section:
# from .api.chunking_exception_handlers import register_chunking_exception_handlers
# from .middleware.correlation import CorrelationMiddleware, configure_logging_with_correlation

# In the lifespan function, add after other startup tasks:
# @asynccontextmanager
# async def lifespan(app: FastAPI) -> AsyncIterator[None]:
#     """Manage application lifespan events."""
#     # ... existing startup code ...
#
#     # Configure logging with correlation ID support
#     configure_logging_with_correlation()
#     logger.info("Configured logging with correlation ID support")
#
#     yield
#
#     # ... existing shutdown code ...

# In the create_app function, add after creating the app instance:
# def create_app() -> FastAPI:
#     """Create and configure the FastAPI application"""
#     app = FastAPI(
#         title="Document Embedding Web UI",
#         description="Create and search document embeddings",
#         version="1.1.0",
#         lifespan=lifespan,
#     )
#
#     # Add correlation middleware BEFORE other middleware
#     app.add_middleware(CorrelationMiddleware)
#
#     # ... existing CORS middleware configuration ...
#
#     # Register chunking exception handlers
#     register_chunking_exception_handlers(app)
#
#     # ... rest of the function ...


# Example usage in API endpoints:
import psutil
from fastapi import APIRouter, Depends

from packages.webui.api.chunking_exceptions import ChunkingMemoryError, ChunkingValidationError
from packages.webui.middleware.correlation import get_correlation_id

router = APIRouter(prefix="/api/v2/chunking", tags=["chunking"])


@router.post("/process")
async def process_document(
    document_id: str,
    correlation_id: str = Depends(get_correlation_id),
) -> dict[str, str]:
    """Example endpoint showing correlation ID usage."""
    # The correlation_id is automatically available

    # Example of raising a chunking exception
    if not document_id:
        raise ChunkingValidationError(
            detail="Document ID is required",
            correlation_id=correlation_id,
            field_errors={"document_id": ["This field is required"]},
        )

    return {
        "status": "processed",
        "correlation_id": correlation_id,
    }


# Example usage in services:


class ChunkingService:
    """Example service showing exception usage."""

    async def process_large_document(self, content: str) -> list[str]:  # noqa: ARG002
        """Process document with memory monitoring."""
        correlation_id = get_correlation_id()

        # Check memory usage
        memory_info = psutil.virtual_memory()
        if memory_info.percent > 90:
            raise ChunkingMemoryError(
                detail="Insufficient memory to process document",
                correlation_id=correlation_id,
                operation_id="doc-processing",
                memory_used=memory_info.used,
                memory_limit=memory_info.total,
                recovery_hint="Try processing smaller documents or wait for other operations to complete",
            )

        # Process document...
        return []

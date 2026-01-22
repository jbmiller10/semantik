"""LLM service API endpoints for vecpipe.

Provides endpoints for local LLM inference:
- /llm/generate: Generate text using local LLM
- /llm/models: List available local LLM models
- /llm/models/load: Preload a model into GPU memory
- /llm/health: Health check for LLM subsystem
"""

import logging
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from shared.config import settings
from vecpipe.search.auth import require_internal_api_key
from vecpipe.search.state import get_resources

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])

LLMQuantization = Literal["int4", "int8", "float16"]


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class LLMGenerateRequest(BaseModel):
    """Request body for text generation."""

    request_id: UUID | None = Field(
        default=None,
        description="Optional request ID for tracking (auto-generated if not provided)",
    )
    model_name: str = Field(
        ...,
        description="HuggingFace model ID (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')",
    )
    quantization: LLMQuantization = Field(
        default=settings.DEFAULT_LLM_QUANTIZATION,
        description="Quantization type: 'int4', 'int8', or 'float16'",
    )
    prompts: list[str] = Field(
        ...,
        min_length=1,
        description="List of prompts to process (batch-shaped, processed sequentially in Phase 1)",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt applied to all prompts",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = greedy)",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum tokens to generate per prompt",
    )


class LLMGenerateResponse(BaseModel):
    """Response body for text generation."""

    request_id: UUID
    contents: list[str]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    model_name: str


class LLMPreloadRequest(BaseModel):
    """Request body for model preloading."""

    model_name: str = Field(
        ...,
        description="HuggingFace model ID to preload",
    )
    quantization: LLMQuantization = Field(
        default=settings.DEFAULT_LLM_QUANTIZATION,
        description="Quantization type: 'int4', 'int8', or 'float16'",
    )


class LLMHealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    loaded_models: list[str]
    governor_enabled: bool


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _get_llm_manager() -> Any:
    """Get the LLM manager from resources, raising 503 if not initialized."""
    resources = get_resources()
    llm_mgr = resources.get("llm_mgr")
    if llm_mgr is None:
        raise HTTPException(
            status_code=503,
            detail="LLM manager not initialized. Local LLM support may be disabled.",
        )
    return llm_mgr


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=LLMGenerateResponse,
    dependencies=[Depends(require_internal_api_key)],
)
async def generate_text(request: LLMGenerateRequest) -> LLMGenerateResponse:
    """Generate text using local LLM.

    Processes prompts sequentially in Phase 1 (batch-shaped API for future parallelism).
    Requires X-Internal-Api-Key header.

    Returns:
        LLMGenerateResponse with generated contents and token counts

    Raises:
        HTTPException 503: LLM manager not initialized
        HTTPException 507: Insufficient GPU memory
        HTTPException 500: Generation failed
    """
    llm_manager = _get_llm_manager()

    try:
        # Phase 1: sequential processing; preserves "batch-only" API surface without
        # requiring true batched generation support.
        results = []
        for prompt in request.prompts:
            result = await llm_manager.generate(
                model_name=request.model_name,
                quantization=request.quantization,
                prompt=prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            results.append(result)

        return LLMGenerateResponse(
            request_id=request.request_id or uuid4(),
            contents=[r["content"] for r in results],
            prompt_tokens=[r["prompt_tokens"] for r in results],
            completion_tokens=[r["completion_tokens"] for r in results],
            model_name=request.model_name,
        )
    except RuntimeError as e:
        if "Insufficient GPU memory" in str(e):
            raise HTTPException(status_code=507, detail=str(e)) from e
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("LLM generation failed with unexpected error")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List available local LLM models with memory requirements.

    Returns curated models from the model registry. Does not require authentication.

    Returns:
        Dictionary with "models" key containing list of model info dicts
    """
    from shared.llm.model_registry import get_models_by_provider

    models = get_models_by_provider("local")
    return {"models": [m.to_dict() for m in models]}


@router.post(
    "/models/load",
    dependencies=[Depends(require_internal_api_key)],
)
async def preload_model(request: LLMPreloadRequest) -> dict[str, Any]:
    """Preload a model into GPU memory.

    Useful for warming up models before expected usage.
    Requires X-Internal-Api-Key header.

    Returns:
        Dictionary with status and model name

    Raises:
        HTTPException 503: LLM manager not initialized
        HTTPException 507: Insufficient GPU memory
        HTTPException 500: Loading failed
    """
    llm_manager = _get_llm_manager()

    try:
        await llm_manager._ensure_model_loaded(request.model_name, request.quantization)
        return {"status": "loaded", "model_name": request.model_name}
    except RuntimeError as e:
        if "Insufficient GPU memory" in str(e):
            raise HTTPException(status_code=507, detail=str(e)) from e
        logger.exception("Model preload failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Model preload failed with unexpected error")
        raise HTTPException(status_code=500, detail=f"Preload failed: {e}") from e


@router.get("/health", response_model=LLMHealthResponse)
async def llm_health_check() -> LLMHealthResponse:
    """Health check - verify LLM subsystem is operational.

    Returns status information even if LLM manager is not initialized.

    Returns:
        LLMHealthResponse with status, loaded models, and governor state
    """
    resources = get_resources()
    llm_mgr = resources.get("llm_mgr")

    if llm_mgr is None:
        return LLMHealthResponse(
            status="disabled",
            loaded_models=[],
            governor_enabled=False,
        )

    return LLMHealthResponse(
        status="ok",
        loaded_models=list(llm_mgr._models.keys()),
        governor_enabled=llm_mgr._governor is not None,
    )


# -----------------------------------------------------------------------------
# Future Streaming Placeholders (501 Not Implemented)
# -----------------------------------------------------------------------------


@router.post(
    "/generate/stream",
    dependencies=[Depends(require_internal_api_key)],
)
async def generate_text_stream(request: LLMGenerateRequest) -> dict[str, str]:
    """Stream text generation (placeholder for Phase 2).

    Returns 501 Not Implemented in Phase 1.
    """
    _ = request  # Avoid unused parameter warning
    raise HTTPException(
        status_code=501,
        detail="Streaming generation not implemented in Phase 1",
    )


@router.post(
    "/requests/{request_id}/cancel",
    dependencies=[Depends(require_internal_api_key)],
)
async def cancel_request(request_id: UUID) -> dict[str, str]:
    """Cancel an in-flight request (placeholder for Phase 2).

    Returns 501 Not Implemented in Phase 1.
    """
    _ = request_id  # Avoid unused parameter warning
    raise HTTPException(
        status_code=501,
        detail="Request cancellation not implemented in Phase 1",
    )


__all__ = ["router"]

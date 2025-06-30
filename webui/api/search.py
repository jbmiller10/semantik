"""
Search routes for the Web UI
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import httpx

from vecpipe.config import settings
from webui import database
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


# Request models
class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=100)
    job_id: Optional[str] = None


class HybridSearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=100)
    job_id: Optional[str] = None
    mode: str = Field(default="filter", description="Hybrid search mode: 'filter' or 'rerank'")
    keyword_mode: str = Field(default="any", description="Keyword matching: 'any' or 'all'")
    score_threshold: Optional[float] = None


@router.post("/search")
async def search(request: SearchRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Search for similar documents - proxies to REST API"""
    try:
        # Determine collection name
        collection_name = f"job_{request.job_id}" if request.job_id else "work_docs"

        # Get model name and settings from job if specified
        model_name = None  # Will use REST API default if not specified
        quantization = None  # Will use REST API default if not specified

        if request.job_id:
            job = database.get_job(request.job_id)
            if job:
                if job.get("model_name"):
                    model_name = job["model_name"]
                if job.get("quantization"):
                    quantization = job["quantization"]

        # Prepare search request for REST API
        search_params = {
            "query": request.query,
            "k": request.k,
            "collection": collection_name,
            "search_type": "semantic",
        }

        # Add optional parameters if specified
        if model_name:
            search_params["model_name"] = model_name
        if quantization:
            search_params["quantization"] = quantization

        # Call REST API search endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{settings.SEARCH_API_URL}/search", json=search_params, timeout=30.0)
            response.raise_for_status()

        # Transform REST API response to match WebUI JavaScript expectations
        api_response = response.json()

        # Convert flat results to payload format expected by JS
        transformed_results = []
        for result in api_response.get("results", []):
            transformed_results.append(
                {
                    "id": result.get("doc_id", ""),
                    "score": result["score"],
                    "payload": {
                        "path": result["path"],
                        "chunk_id": result["chunk_id"],
                        "doc_id": result.get("doc_id"),
                        "text": result.get("content", ""),
                        "metadata": result.get("metadata", {}),  # Include metadata with page_number
                    },
                }
            )

        return {"query": api_response["query"], "results": transformed_results, "collection": collection_name}

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Collection not found")
        raise HTTPException(status_code=502, detail=f"Search failed: {str(e)}")
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to search API: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable")
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/hybrid_search")
async def hybrid_search(request: HybridSearchRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Perform hybrid search combining vector similarity and text matching - proxies to REST API"""
    try:
        # Determine collection name
        collection_name = f"job_{request.job_id}" if request.job_id else "work_docs"

        # Get model name and settings from job if specified
        model_name = None
        quantization = None

        if request.job_id:
            job = database.get_job(request.job_id)
            if job:
                if job.get("model_name"):
                    model_name = job["model_name"]
                if job.get("quantization"):
                    quantization = job["quantization"]

        # Prepare hybrid search params for REST API
        search_params = {
            "q": request.query,
            "k": request.k,
            "collection": collection_name,
            "mode": request.mode,
            "keyword_mode": request.keyword_mode,
        }

        # Add optional parameters if specified
        if model_name:
            search_params["model_name"] = model_name
        if quantization:
            search_params["quantization"] = quantization
        if request.score_threshold is not None:
            search_params["score_threshold"] = request.score_threshold

        # Call REST API hybrid search endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.SEARCH_API_URL}/hybrid_search", params=search_params, timeout=30.0)
            response.raise_for_status()

        # The REST API returns the response in the format we need
        return response.json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Collection not found")
        raise HTTPException(status_code=502, detail=f"Hybrid search failed: {str(e)}")
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to search API: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable")
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

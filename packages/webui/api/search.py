"""
Search routes for the Web UI
"""

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from shared.config import settings
from shared.database.base import JobRepository
from shared.database.factory import create_job_repository
from shared.contracts.search import HybridSearchRequest, PreloadModelRequest
from shared.contracts.search import SearchRequest as SharedSearchRequest
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


# Create a custom SearchRequest that handles the webui-specific field mappings
class SearchRequest(SharedSearchRequest):
    """WebUI-specific search request that maps frontend fields."""

    # Override to ensure we accept both 'top_k' and 'k' from frontend
    class Config(SharedSearchRequest.Config):
        populate_by_name = True


@router.post("/search")
async def search(
    request: SearchRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    job_repo: JobRepository = Depends(create_job_repository),
) -> dict[str, Any]:
    """Unified search endpoint - handles both vector and hybrid search"""
    logger.info(
        f"Search request received: query='{request.query}', type={request.search_type}, "
        f"collection={request.collection}, top_k={request.k}, threshold={request.score_threshold}"
    )

    try:
        # Determine collection name and job_id
        if request.collection:
            collection_name = request.collection
            # Extract job_id from collection name if it follows the pattern
            if request.collection.startswith("job_") and not request.job_id:
                request.job_id = request.collection.replace("job_", "")
        elif request.job_id:
            collection_name = f"job_{request.job_id}"
        else:
            collection_name = "work_docs"

        # Get model name and settings from job if specified
        model_name = None
        quantization = None

        if request.job_id:
            job = await job_repo.get_job(request.job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            # Check if the current user owns the job
            # Allow access to legacy jobs without user_id
            if job.get("user_id") is not None and job.get("user_id") != current_user.get("id"):
                raise HTTPException(status_code=403, detail="Access denied to this collection")

            if job.get("model_name"):
                model_name = job["model_name"]
            if job.get("quantization"):
                quantization = job["quantization"]

        # Route based on search type
        if request.search_type == "hybrid":
            # Call hybrid search endpoint
            search_params: dict[str, str | int | float] = {
                "q": request.query,  # Hybrid endpoint expects 'q' not 'query'
                "k": request.k,
                "collection": collection_name,
                "mode": request.hybrid_mode,
                "keyword_mode": request.keyword_mode,
                "alpha": request.hybrid_alpha,
            }

            # Add optional parameters
            if model_name:
                search_params["model_name"] = model_name
            if quantization:
                search_params["quantization"] = quantization
            if request.score_threshold > 0:
                search_params["score_threshold"] = request.score_threshold

            # Call REST API hybrid search endpoint with GET
            logger.info(f"Calling hybrid search API with params: {search_params}")

            # Use longer timeout for hybrid search as well
            timeout = httpx.Timeout(
                timeout=60.0,  # Total timeout increased to 60 seconds
                connect=5.0,  # Connection timeout
                read=60.0,  # Read timeout for model loading
                write=5.0,  # Write timeout
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    response = await client.get(f"{settings.SEARCH_API_URL}/hybrid_search", params=search_params)
                    response.raise_for_status()
                except httpx.ReadTimeout:
                    # If first attempt times out, it might be due to model loading
                    # Log warning and retry with even longer timeout
                    logger.warning(
                        "Hybrid search request timed out, likely due to model loading. Retrying with longer timeout..."
                    )
                    extended_timeout = httpx.Timeout(timeout=120.0, connect=5.0, read=120.0, write=5.0)
                    async with httpx.AsyncClient(timeout=extended_timeout) as retry_client:
                        response = await retry_client.get(
                            f"{settings.SEARCH_API_URL}/hybrid_search", params=search_params
                        )
                        response.raise_for_status()

            # Transform hybrid search results to match frontend expectations
            api_response = response.json()
            logger.info(f"Hybrid search returned {len(api_response.get('results', []))} results")

            # Convert to format expected by frontend
            transformed_results = []
            for result in api_response.get("results", []):
                # Extract file name from path
                path = result.get("path", "")
                file_name = path.split("/")[-1] if path else "Unknown"

                # Get metadata
                metadata = result.get("metadata", {})

                # Extract job_id from collection name (format: job_{job_id})
                job_id_from_collection = (
                    collection_name.replace("job_", "") if collection_name.startswith("job_") else request.job_id
                )

                # Create result in format expected by SearchResults component
                transformed_result = {
                    "doc_id": result.get("doc_id", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "score": result.get("score", 0.0),
                    "content": result.get("content", ""),  # Hybrid search might not include content
                    "file_path": path,
                    "file_name": file_name,
                    "chunk_index": metadata.get("chunk_index", 0),
                    "total_chunks": metadata.get("total_chunks", 1),
                    "job_id": job_id_from_collection,
                    # Hybrid-specific fields
                    "matched_keywords": result.get("matched_keywords", []),
                    "keyword_score": result.get("keyword_score"),
                    "combined_score": result.get("combined_score"),
                }
                transformed_results.append(transformed_result)

            return {
                "query": api_response.get("query", request.query),
                "results": transformed_results,
                "collection": collection_name,
                "num_results": len(transformed_results),
                "search_type": request.search_type,
                "keywords_extracted": api_response.get("keywords_extracted", []),
                "search_mode": api_response.get("search_mode", request.hybrid_mode),
            }

        # Vector search
        vector_search_params: dict[str, str | int | bool | float] = {
            "query": request.query,
            "k": request.k,
            "collection": collection_name,
            "search_type": "semantic",
            "include_content": True,
        }

        # Add reranking parameters
        if request.use_reranker:
            vector_search_params["use_reranker"] = request.use_reranker
            if request.rerank_model:
                vector_search_params["rerank_model"] = request.rerank_model
            if request.rerank_quantization:
                vector_search_params["rerank_quantization"] = request.rerank_quantization

        # Add optional parameters
        if model_name:
            vector_search_params["model_name"] = model_name
        if quantization:
            vector_search_params["quantization"] = quantization
        if request.score_threshold > 0:
            vector_search_params["score_threshold"] = request.score_threshold

        # Call REST API search endpoint with POST
        logger.info(f"Calling vector search API with params: {vector_search_params}")

        # Use longer timeout for first request after model might be unloaded
        # This handles the case where model needs to be reloaded
        timeout = httpx.Timeout(
            timeout=60.0,  # Total timeout increased to 60 seconds
            connect=5.0,  # Connection timeout
            read=60.0,  # Read timeout for model loading
            write=5.0,  # Write timeout
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(f"{settings.SEARCH_API_URL}/search", json=vector_search_params)
                response.raise_for_status()
            except httpx.ReadTimeout:
                # If first attempt times out, it might be due to model loading
                # Log warning and retry with even longer timeout
                logger.warning("Search request timed out, likely due to model loading. Retrying with longer timeout...")
                extended_timeout = httpx.Timeout(timeout=120.0, connect=5.0, read=120.0, write=5.0)
                async with httpx.AsyncClient(timeout=extended_timeout) as retry_client:
                    response = await retry_client.post(f"{settings.SEARCH_API_URL}/search", json=vector_search_params)
                    response.raise_for_status()

        # Transform REST API response to match WebUI JavaScript expectations
        api_response = response.json()
        logger.info(f"Vector search returned {len(api_response.get('results', []))} results")

        # Convert to format expected by frontend
        transformed_results = []
        for result in api_response.get("results", []):
            # Extract file name from path
            path = result.get("path", "")
            file_name = path.split("/")[-1] if path else "Unknown"

            # Get metadata
            metadata = result.get("metadata", {})

            # Extract job_id from collection name (format: job_{job_id})
            job_id_from_collection = (
                collection_name.replace("job_", "") if collection_name.startswith("job_") else request.job_id
            )

            # Create result in format expected by SearchResults component
            transformed_result = {
                "doc_id": result.get("doc_id", ""),
                "chunk_id": result.get("chunk_id", ""),
                "score": result.get("score", 0.0),
                "content": result.get("content", ""),
                "file_path": path,
                "file_name": file_name,
                "chunk_index": metadata.get("chunk_index", 0),
                "total_chunks": metadata.get("total_chunks", 1),
                "job_id": job_id_from_collection,
                # Keep the payload format for backward compatibility
                "payload": {
                    "path": path,
                    "chunk_id": result.get("chunk_id", ""),
                    "doc_id": result.get("doc_id", ""),
                    "text": result.get("content", ""),
                    "metadata": metadata,
                },
            }
            transformed_results.append(transformed_result)

        # Build response with reranking metrics if available
        response_data = {
            "query": api_response["query"],
            "results": transformed_results,
            "collection": collection_name,
            "num_results": len(transformed_results),
            "search_type": request.search_type,
        }

        # Include reranking metrics if present
        if api_response.get("reranking_used"):
            response_data["reranking_used"] = api_response["reranking_used"]
            response_data["reranker_model"] = api_response.get("reranker_model")
            response_data["reranking_time_ms"] = api_response.get("reranking_time_ms")

        return response_data

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 507:
            # Insufficient memory error from search API
            error_detail = e.response.json().get("detail", {})
            if isinstance(error_detail, dict) and error_detail.get("error") == "insufficient_memory":
                raise HTTPException(
                    status_code=507,
                    detail={
                        "error": "insufficient_memory",
                        "message": error_detail.get("message", "Insufficient GPU memory for reranking"),
                        "suggestion": error_detail.get(
                            "suggestion", "Try using a smaller model or different quantization"
                        ),
                    },
                ) from e
        elif e.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="Collection not found. The embedding job may not have created any vectors yet. Please check the job status.",
            ) from e
        raise HTTPException(status_code=502, detail=f"Search failed: {str(e)}") from e
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to search API: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable") from e
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


# PreloadModelRequest is now imported from shared.contracts.search


@router.post("/preload_model")
async def preload_model(
    request: PreloadModelRequest, current_user: dict[str, Any] = Depends(get_current_user)  # noqa: ARG001
) -> dict[str, str]:
    """
    Preload a model to prevent timeout issues on first search.
    This is useful when you know a search is coming and want to ensure the model is ready.
    """
    logger.info(f"Preloading model: {request.model_name} with quantization: {request.quantization}")

    try:
        # Call the search API with a dummy query to force model loading
        preload_params = {
            "query": "preload",
            "k": 1,
            "collection": "work_docs",  # Use default collection
            "search_type": "semantic",
            "include_content": False,
            "model_name": request.model_name,
            "quantization": request.quantization,
        }

        # Use extended timeout for model loading
        timeout = httpx.Timeout(timeout=120.0, connect=5.0, read=120.0, write=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{settings.SEARCH_API_URL}/search", json=preload_params)
            response.raise_for_status()

        return {"status": "success", "message": f"Model {request.model_name} preloaded successfully"}

    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to preload model: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to preload model: {str(e)}") from e
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to search API: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable") from e
    except Exception as e:
        logger.error(f"Preload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preload model: {str(e)}") from e


@router.post("/hybrid_search")
async def hybrid_search(
    request: HybridSearchRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    job_repo: JobRepository = Depends(create_job_repository),
) -> dict[str, Any]:
    """Perform hybrid search combining vector similarity and text matching - proxies to REST API"""
    try:
        # Determine collection name
        collection_name = f"job_{request.job_id}" if request.job_id else "work_docs"

        # Get model name and settings from job if specified
        model_name = None
        quantization = None

        if request.job_id:
            job = await job_repo.get_job(request.job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            # Check if the current user owns the job
            # Allow access to legacy jobs without user_id
            if job.get("user_id") is not None and job.get("user_id") != current_user.get("id"):
                raise HTTPException(status_code=403, detail="Access denied to this collection")

            if job.get("model_name"):
                model_name = job["model_name"]
            if job.get("quantization"):
                quantization = job["quantization"]

        # Prepare hybrid search params for REST API
        search_params: dict[str, str | int | float | None] = {
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
        timeout = httpx.Timeout(timeout=60.0, connect=5.0, read=60.0, write=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{settings.SEARCH_API_URL}/hybrid_search", params=search_params)
            response.raise_for_status()

        # The REST API returns the response in the format we need
        result: dict[str, Any] = response.json()
        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Collection not found") from e
        raise HTTPException(status_code=502, detail=f"Hybrid search failed: {str(e)}") from e
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to search API: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable") from e
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}") from e

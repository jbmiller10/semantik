#!/usr/bin/env python3
"""
FastAPI search service (VS-040)
REST API for vector similarity search with Qwen3 support
"""

import asyncio
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prometheus_client import Counter, Histogram

from vecpipe.config import settings
from vecpipe.hybrid_search import HybridSearchEngine
from vecpipe.metrics import metrics_collector, registry, start_metrics_server
from vecpipe.search_utils import parse_search_results, search_qdrant
from webui.embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Create or get metrics for search API
def get_or_create_metric(metric_class, name, description, labels=None, **kwargs):
    """Create a metric or return existing one if already registered"""
    from prometheus_client import REGISTRY

    # Check if metric already exists
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]

    # Create new metric
    if labels:
        return metric_class(name, description, labels, registry=registry, **kwargs)
    else:
        return metric_class(name, description, registry=registry, **kwargs)


# Create metrics with duplicate protection
search_latency = get_or_create_metric(
    Histogram,
    "search_api_latency_seconds",
    "Search API request latency",
    ["endpoint", "search_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
search_requests = get_or_create_metric(
    Counter, "search_api_requests_total", "Total search API requests", ["endpoint", "search_type"]
)
search_errors = get_or_create_metric(
    Counter, "search_api_errors_total", "Total search API errors", ["endpoint", "error_type"]
)
embedding_generation_latency = get_or_create_metric(
    Histogram,
    "search_api_embedding_latency_seconds",
    "Embedding generation latency for search queries",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

# Constants
DEFAULT_K = 10
METRICS_PORT = int(os.getenv("METRICS_PORT", "9091"))

# Search instructions for different use cases
SEARCH_INSTRUCTIONS = {
    "semantic": "Represent this sentence for searching relevant passages:",
    "question": "Represent this question for retrieving supporting documents:",
    "code": "Represent this code query for finding similar code snippets:",
    "hybrid": "Generate a comprehensive embedding for multi-modal search:",
}


# Response models
class SearchResult(BaseModel):
    path: str
    chunk_id: str
    score: float
    doc_id: str | None = None
    content: str | None = None
    metadata: dict | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(DEFAULT_K, ge=1, le=100, description="Number of results")
    search_type: str = Field("semantic", description="Type of search: semantic, question, code, hybrid")
    model_name: str | None = Field(None, description="Override embedding model")
    quantization: str | None = Field(None, description="Override quantization: float32, float16, int8")
    filters: dict | None = Field(None, description="Metadata filters for search")
    include_content: bool = Field(False, description="Include chunk content in results")
    collection: str | None = Field(None, description="Collection name (e.g., job_123)")


class BatchSearchRequest(BaseModel):
    queries: list[str] = Field(..., description="List of search queries")
    k: int = Field(DEFAULT_K, ge=1, le=100, description="Number of results per query")
    search_type: str = Field("semantic", description="Type of search")
    model_name: str | None = Field(None, description="Override embedding model")
    quantization: str | None = Field(None, description="Override quantization")
    collection: str | None = Field(None, description="Collection name")


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    num_results: int
    search_type: str | None = None
    model_used: str | None = None
    embedding_time_ms: float | None = None
    search_time_ms: float | None = None


class BatchSearchResponse(BaseModel):
    responses: list[SearchResponse]
    total_time_ms: float


class HybridSearchResult(BaseModel):
    path: str
    chunk_id: str
    score: float
    doc_id: str | None = None
    matched_keywords: list[str] = []
    keyword_score: float | None = None
    combined_score: float | None = None
    metadata: dict[str, Any] | None = None


class HybridSearchResponse(BaseModel):
    query: str
    results: list[HybridSearchResult]
    num_results: int
    keywords_extracted: list[str]
    search_mode: str


# Global resources
qdrant_client = None
model_manager = None
embedding_service = None
executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global qdrant_client, model_manager, embedding_service, executor
    # Startup
    # Start metrics server
    start_metrics_server(METRICS_PORT)
    logger.info(f"Metrics server started on port {METRICS_PORT}")

    qdrant_client = httpx.AsyncClient(base_url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}", timeout=60.0)
    logger.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

    # Initialize model manager with lazy loading
    from vecpipe.model_manager import ModelManager

    unload_after = int(os.getenv("MODEL_UNLOAD_AFTER_SECONDS", "300"))  # 5 minutes default
    model_manager = ModelManager(unload_after_seconds=unload_after)
    logger.info(f"Initialized model manager with {unload_after}s inactivity timeout")

    # Initialize embedding service for backward compatibility
    embedding_service = EmbeddingService(mock_mode=settings.USE_MOCK_EMBEDDINGS)
    logger.info(f"Initialized embedding service (mock_mode={settings.USE_MOCK_EMBEDDINGS})")

    # Initialize thread pool executor for async operations
    executor = ThreadPoolExecutor(max_workers=4)

    yield

    # Shutdown
    await qdrant_client.aclose()
    if model_manager:
        model_manager.shutdown()
    if executor:
        executor.shutdown(wait=True)
    logger.info("Disconnected from Qdrant")


# Create FastAPI app
app = FastAPI(
    title="Document Vector Search API",
    description="Unified search API with vector similarity, hybrid search, and Qwen3 support",
    version="2.0.0",
    lifespan=lifespan,
)


def generate_mock_embedding(text: str, vector_dim: int = None) -> list[float]:
    """Generate mock embedding for testing (fallback when real embeddings unavailable)"""
    # If vector_dim not specified, try to get from collection info
    if vector_dim is None:
        vector_dim = 1024  # Default fallback

    # Generate deterministic "embedding" from text hash
    hash_bytes = hashlib.sha256(text.encode()).digest()
    values = []

    for i in range(0, len(hash_bytes), 4):
        chunk = hash_bytes[i : i + 4]
        if len(chunk) == 4:
            val = int.from_bytes(chunk, byteorder="big") / (2**32)
            values.append(val * 2 - 1)

    # Pad or truncate to vector_dim
    if len(values) < vector_dim:
        values.extend([0.0] * (vector_dim - len(values)))
    else:
        values = values[:vector_dim]

    # Normalize to unit length for proper cosine similarity
    norm = sum(v**2 for v in values) ** 0.5
    if norm > 0:
        values = [v / norm for v in values]
    else:
        # Handle edge case of all zeros
        values[0] = 1.0  # Set first element to 1 to ensure unit vector

    return values


async def generate_embedding_async(
    text: str, model_name: str = None, quantization: str = None, instruction: str = None
) -> list[float]:
    """Generate embedding using the model manager"""
    if settings.USE_MOCK_EMBEDDINGS:
        return generate_mock_embedding(text)

    if model_manager is None:
        raise RuntimeError("Model manager not initialized")

    # Use provided model/quantization or defaults
    model = model_name or settings.DEFAULT_EMBEDDING_MODEL
    quant = quantization or settings.DEFAULT_QUANTIZATION

    # Determine instruction for search queries
    if instruction is None:
        instruction = "Represent this sentence for searching relevant passages:"

    # Time the embedding generation
    start_time = time.time()

    # Use model manager for lazy loading and automatic unloading
    embedding = await model_manager.generate_embedding_async(text, model, quant, instruction)

    # Record embedding generation latency
    embedding_generation_latency.observe(time.time() - start_time)

    if embedding is None:
        raise RuntimeError(f"Failed to generate embedding for text: {text[:100]}...")

    return embedding


@app.get("/model/status")
async def model_status():
    """Get model manager status"""
    if model_manager:
        return model_manager.get_status()
    else:
        return {"error": "Model manager not initialized"}


@app.get("/")
async def root():
    """Health check endpoint with detailed status"""
    try:
        # Check Qdrant connection
        response = await qdrant_client.get(f"/collections/{settings.DEFAULT_COLLECTION}")
        response.raise_for_status()
        info = response.json()["result"]

        health_info = {
            "status": "healthy",
            "collection": {
                "name": settings.DEFAULT_COLLECTION,
                "points_count": info["points_count"],
                "vector_size": info["config"]["params"]["vectors"]["size"] if "config" in info else None,
            },
            "embedding_mode": "mock" if settings.USE_MOCK_EMBEDDINGS else "real",
        }

        if not settings.USE_MOCK_EMBEDDINGS and embedding_service:
            model_info = embedding_service.get_model_info(
                embedding_service.current_model_name or settings.DEFAULT_EMBEDDING_MODEL,
                embedding_service.current_quantization or settings.DEFAULT_QUANTIZATION,
            )

            health_info["embedding_service"] = {
                "current_model": embedding_service.current_model_name,
                "quantization": embedding_service.current_quantization,
                "device": embedding_service.device,
                "model_info": model_info,
            }

        return health_info
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results to return"),
    collection: str | None = Query(None, description="Collection name (e.g., job_123)"),
    search_type: str = Query("semantic", description="Type of search: semantic, question, code, hybrid"),
    model_name: str | None = Query(None, description="Override embedding model"),
    quantization: str | None = Query(None, description="Override quantization"),
):
    """
    Search for similar documents (GET endpoint for compatibility)

    - **q**: The search query text
    - **k**: Number of results to return (1-100, default 10)
    - **collection**: Optional collection name (defaults to work_docs)
    - **search_type**: Type of search (semantic, question, code, hybrid)
    - **model_name**: Override default embedding model
    - **quantization**: Override default quantization
    """
    # Create request object for unified handling
    request = SearchRequest(
        query=q, k=k, search_type=search_type, model_name=model_name, quantization=quantization, collection=collection
    )
    return await search_post(request)


@app.post("/search", response_model=SearchResponse)
async def search_post(request: SearchRequest = Body(...)):
    """
    Search for similar documents with advanced options

    Supports different search types:
    - **semantic**: General semantic search
    - **question**: Question-answering search
    - **code**: Code similarity search
    - **hybrid**: Multi-modal search
    """
    start_time = time.time()

    # Record request
    search_requests.labels(endpoint="/search", search_type=request.search_type).inc()

    try:
        # Determine collection name
        collection_name = request.collection if request.collection else settings.DEFAULT_COLLECTION

        # Get collection info to determine vector dimension
        vector_dim = 1024  # default
        try:
            response = await qdrant_client.get(f"/collections/{collection_name}")
            response.raise_for_status()
            collection_info = response.json()["result"]
            if "config" in collection_info and "params" in collection_info["config"]:
                vector_dim = collection_info["config"]["params"]["vectors"]["size"]
        except Exception as e:
            logger.warning(f"Could not get collection info for {collection_name}, using default dimension: {e}")

        # Try to get collection metadata to determine the correct model
        collection_model = None
        collection_quantization = None
        collection_instruction = None

        try:
            # Check if this is a job collection and get metadata
            from qdrant_client import QdrantClient

            sync_client = QdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            from webui.api.collection_metadata import get_collection_metadata

            metadata = get_collection_metadata(sync_client, collection_name)
            if metadata:
                collection_model = metadata.get("model_name")
                collection_quantization = metadata.get("quantization")
                collection_instruction = metadata.get("instruction")
                logger.info(
                    f"Found metadata for collection {collection_name}: model={collection_model}, quantization={collection_quantization}"
                )
        except Exception as e:
            logger.warning(f"Could not get collection metadata: {e}")

        # Use collection's model if found, otherwise fall back to request or defaults
        model_name = request.model_name or collection_model or settings.DEFAULT_EMBEDDING_MODEL
        quantization = request.quantization or collection_quantization or settings.DEFAULT_QUANTIZATION

        # Log warning if using different model than collection was created with
        if collection_model and model_name != collection_model:
            logger.warning(
                f"Collection {collection_name} was created with model {collection_model} but searching with {model_name}"
            )
        if collection_quantization and quantization != collection_quantization:
            logger.warning(
                f"Collection {collection_name} was created with quantization {collection_quantization} but searching with {quantization}"
            )

        # Get appropriate instruction for search type
        # Use collection's instruction if available and no specific search type requested
        if collection_instruction and request.search_type == "semantic":
            instruction = collection_instruction
        else:
            instruction = SEARCH_INSTRUCTIONS.get(request.search_type, SEARCH_INSTRUCTIONS["semantic"])

        # Generate query embedding
        embed_start = time.time()
        logger.info(
            f"Processing search query: '{request.query}' (k={request.k}, collection={collection_name}, type={request.search_type})"
        )

        if not settings.USE_MOCK_EMBEDDINGS:
            query_vector = await generate_embedding_async(request.query, model_name, quantization, instruction)
        else:
            query_vector = generate_mock_embedding(request.query, vector_dim)

        embed_time = (time.time() - embed_start) * 1000

        # Search in Qdrant
        search_start = time.time()

        # Handle filters if provided
        if request.filters:
            # Direct Qdrant search with filters
            search_request = {
                "vector": query_vector,
                "limit": request.k,
                "with_payload": True,
                "with_vector": False,
                "filter": request.filters,
            }

            response = await qdrant_client.post(f"/collections/{collection_name}/points/search", json=search_request)
            response.raise_for_status()
            qdrant_results = response.json()["result"]
        else:
            # Use shared utility for regular search
            qdrant_results = await search_qdrant(
                settings.QDRANT_HOST, settings.QDRANT_PORT, collection_name, query_vector, request.k
            )

        search_time = (time.time() - search_start) * 1000

        # Parse results
        results = []
        for point in qdrant_results:
            if isinstance(point, dict) and "payload" in point:
                # Direct Qdrant response format
                payload = point["payload"]
                result = SearchResult(
                    path=payload.get("path", ""),
                    chunk_id=payload.get("chunk_id", ""),
                    score=point["score"],
                    doc_id=payload.get("doc_id"),
                    content=payload.get("content") if request.include_content else None,
                    metadata=payload.get("metadata"),
                )
            else:
                # Parsed format from search_utils
                parsed_results = parse_search_results(qdrant_results)
                for r in parsed_results:
                    result = SearchResult(
                        path=r["path"],
                        chunk_id=r["chunk_id"],
                        score=r["score"],
                        doc_id=r.get("doc_id"),
                        metadata=r.get("metadata"),
                    )
                    results.append(result)
                break
            results.append(result)

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Search completed in {total_time:.2f}ms (embed: {embed_time:.2f}ms, search: {search_time:.2f}ms)")

        # Record search latency
        search_latency.labels(endpoint="/search", search_type=request.search_type).observe(time.time() - start_time)

        # Update resource metrics periodically
        metrics_collector.update_resource_metrics()

        return SearchResponse(
            query=request.query,
            results=results,
            num_results=len(results),
            search_type=request.search_type,
            model_used=f"{model_name}/{quantization}" if not settings.USE_MOCK_EMBEDDINGS else "mock",
            embedding_time_ms=embed_time,
            search_time_ms=search_time,
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"Qdrant error: {e}")
        search_errors.labels(endpoint="/search", error_type="qdrant_error").inc()
        raise HTTPException(status_code=502, detail="Vector database error")
    except RuntimeError as e:
        # Specific handling for embedding failures
        logger.error(f"Embedding generation failed: {e}")
        search_errors.labels(endpoint="/search", error_type="embedding_error").inc()
        raise HTTPException(status_code=503, detail=f"Embedding service error: {str(e)}. Check logs for details.")
    except Exception as e:
        logger.error(f"Search error: {e}")
        search_errors.labels(endpoint="/search", error_type="unknown_error").inc()
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results to return"),
    collection: str | None = Query(None, description="Collection name (e.g., job_123)"),
    mode: str = Query("filter", description="Hybrid search mode: 'filter' or 'rerank'"),
    keyword_mode: str = Query("any", description="Keyword matching: 'any' or 'all'"),
    score_threshold: float | None = Query(None, description="Minimum similarity score threshold"),
    model_name: str | None = Query(None, description="Override embedding model"),
    quantization: str | None = Query(None, description="Override quantization"),
):
    """
    Perform hybrid search combining vector similarity and text matching

    - **q**: The search query text
    - **k**: Number of results to return (1-100, default 10)
    - **collection**: Optional collection name (defaults to work_docs)
    - **mode**: Hybrid search mode - 'filter' uses Qdrant filters, 'rerank' does post-processing
    - **keyword_mode**: How to match keywords - 'any' matches any keyword, 'all' requires all keywords
    - **score_threshold**: Optional minimum similarity score threshold
    """
    start_time = time.time()
    search_requests.labels(endpoint="/hybrid_search", search_type="hybrid").inc()

    try:
        # Determine collection name
        collection_name = collection if collection else settings.DEFAULT_COLLECTION

        # Try to get collection metadata to determine the correct model
        collection_model = None
        collection_quantization = None

        try:
            # Check if this is a job collection and get metadata
            from qdrant_client import QdrantClient

            sync_client = QdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            from webui.api.collection_metadata import get_collection_metadata

            metadata = get_collection_metadata(sync_client, collection_name)
            if metadata:
                collection_model = metadata.get("model_name")
                collection_quantization = metadata.get("quantization")
                logger.info(
                    f"Found metadata for collection {collection_name}: model={collection_model}, quantization={collection_quantization}"
                )
        except Exception as e:
            logger.warning(f"Could not get collection metadata: {e}")

        # Use collection's model if found, otherwise fall back to request or defaults
        model_name = model_name or collection_model or settings.DEFAULT_EMBEDDING_MODEL
        quantization = quantization or collection_quantization or settings.DEFAULT_QUANTIZATION

        # Log warning if using different model than collection was created with
        if collection_model and model_name != collection_model:
            logger.warning(
                f"Collection {collection_name} was created with model {collection_model} but searching with {model_name}"
            )

        # Initialize hybrid search engine
        hybrid_engine = HybridSearchEngine(settings.QDRANT_HOST, settings.QDRANT_PORT, collection_name)

        # Extract keywords for response
        keywords = hybrid_engine.extract_keywords(q)

        # Generate query embedding
        logger.info(
            f"Processing hybrid search query: '{q}' (k={k}, collection={collection_name}, mode={mode}, model={model_name}, quantization={quantization})"
        )

        if not settings.USE_MOCK_EMBEDDINGS:
            query_vector = await generate_embedding_async(q, model_name, quantization)
        else:
            # Get vector dimension from collection
            vector_dim = 1024  # default
            try:
                response = await qdrant_client.get(f"/collections/{collection_name}")
                response.raise_for_status()
                collection_info = response.json()["result"]
                if "config" in collection_info and "params" in collection_info["config"]:
                    vector_dim = collection_info["config"]["params"]["vectors"]["size"]
            except Exception as e:
                logger.warning(f"Could not get collection info for {collection_name}, using default dimension: {e}")

            query_vector = generate_mock_embedding(q, vector_dim)

        # Perform hybrid search
        results = hybrid_engine.hybrid_search(
            query_vector=query_vector,
            query_text=q,
            limit=k,
            keyword_mode=keyword_mode,
            score_threshold=score_threshold,
            hybrid_mode=mode,
        )

        # Convert results to response format
        hybrid_results = []
        for r in results:
            result = HybridSearchResult(
                path=r["payload"].get("path", ""),
                chunk_id=r["payload"].get("chunk_id", ""),
                score=r["score"],
                doc_id=r["payload"].get("doc_id"),
                matched_keywords=r.get("matched_keywords", []),
                keyword_score=r.get("keyword_score"),
                combined_score=r.get("combined_score"),
                metadata=r["payload"].get("metadata"),
            )
            hybrid_results.append(result)

        logger.info(f"Found {len(hybrid_results)} results for hybrid query: '{q}'")

        # Record search latency
        search_latency.labels(endpoint="/hybrid_search", search_type="hybrid").observe(time.time() - start_time)

        return HybridSearchResponse(
            query=q,
            results=hybrid_results,
            num_results=len(hybrid_results),
            keywords_extracted=keywords,
            search_mode=mode,
        )

    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        search_errors.labels(endpoint="/hybrid_search", error_type="search_error").inc()
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")
    finally:
        if "hybrid_engine" in locals():
            hybrid_engine.close()


@app.post("/search/batch", response_model=BatchSearchResponse)
async def batch_search(request: BatchSearchRequest = Body(...)):
    """
    Batch search for multiple queries

    Efficiently processes multiple search queries in parallel
    """
    start_time = time.time()

    try:
        collection_name = request.collection if request.collection else settings.DEFAULT_COLLECTION
        model_name = request.model_name or settings.DEFAULT_EMBEDDING_MODEL
        quantization = request.quantization or settings.DEFAULT_QUANTIZATION
        instruction = SEARCH_INSTRUCTIONS.get(request.search_type, SEARCH_INSTRUCTIONS["semantic"])

        # Generate embeddings for all queries in batch
        logger.info(f"Generating embeddings for {len(request.queries)} queries")

        # Create tasks for parallel embedding generation
        embedding_tasks = [
            generate_embedding_async(query, model_name, quantization, instruction) for query in request.queries
        ]

        # Wait for all embeddings
        query_vectors = await asyncio.gather(*embedding_tasks)

        # Create search tasks
        search_tasks = [
            search_qdrant(settings.QDRANT_HOST, settings.QDRANT_PORT, collection_name, vector, request.k)
            for vector in query_vectors
        ]

        # Execute searches in parallel
        all_results = await asyncio.gather(*search_tasks)

        # Build responses
        responses = []
        for query, results in zip(request.queries, all_results, strict=False):
            parsed_results = []
            for point in results:
                if isinstance(point, dict) and "payload" in point:
                    payload = point["payload"]
                    parsed_results.append(
                        SearchResult(
                            path=payload.get("path", ""),
                            chunk_id=payload.get("chunk_id", ""),
                            score=point["score"],
                            doc_id=payload.get("doc_id"),
                        )
                    )
                else:
                    # Handle parsed format
                    parsed = parse_search_results(results)
                    for r in parsed:
                        parsed_results.append(
                            SearchResult(
                                path=r["path"], chunk_id=r["chunk_id"], score=r["score"], doc_id=r.get("doc_id")
                            )
                        )
                    break

            responses.append(
                SearchResponse(
                    query=query,
                    results=parsed_results,
                    num_results=len(parsed_results),
                    search_type=request.search_type,
                    model_used=f"{model_name}/{quantization}" if not settings.USE_MOCK_EMBEDDINGS else "mock",
                )
            )

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Batch search completed in {total_time:.2f}ms for {len(request.queries)} queries")

        return BatchSearchResponse(responses=responses, total_time_ms=total_time)

    except Exception as e:
        logger.error(f"Batch search error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")


@app.get("/keyword_search", response_model=HybridSearchResponse)
async def keyword_search(
    q: str = Query(..., description="Keywords to search for"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results to return"),
    collection: str | None = Query(None, description="Collection name (e.g., job_123)"),
    mode: str = Query("any", description="Keyword matching: 'any' or 'all'"),
):
    """
    Search using only keywords (no vector similarity)

    - **q**: Keywords to search for (space-separated)
    - **k**: Number of results to return (1-100, default 10)
    - **collection**: Optional collection name (defaults to work_docs)
    - **mode**: How to match keywords - 'any' matches any keyword, 'all' requires all keywords
    """
    try:
        # Determine collection name
        collection_name = collection if collection else settings.DEFAULT_COLLECTION

        # Initialize hybrid search engine
        hybrid_engine = HybridSearchEngine(settings.QDRANT_HOST, settings.QDRANT_PORT, collection_name)

        # Extract keywords
        keywords = hybrid_engine.extract_keywords(q)

        logger.info(
            f"Processing keyword search: '{q}' -> {keywords} (k={k}, collection={collection_name}, mode={mode})"
        )

        # Perform keyword-only search
        results = hybrid_engine.search_by_keywords(keywords=keywords, limit=k, mode=mode)

        # Convert results to response format
        hybrid_results = []
        for r in results:
            result = HybridSearchResult(
                path=r["payload"].get("path", ""),
                chunk_id=r["payload"].get("chunk_id", ""),
                score=0.0,  # No vector score for keyword-only search
                doc_id=r["payload"].get("doc_id"),
                matched_keywords=r.get("matched_keywords", []),
            )
            hybrid_results.append(result)

        logger.info(f"Found {len(hybrid_results)} results for keyword search: '{q}'")

        return HybridSearchResponse(
            query=q,
            results=hybrid_results,
            num_results=len(hybrid_results),
            keywords_extracted=keywords,
            search_mode="keywords_only",
        )

    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Keyword search error: {str(e)}")
    finally:
        if "hybrid_engine" in locals():
            hybrid_engine.close()


@app.get("/collection/info")
async def collection_info():
    """Get information about the vector collection"""
    try:
        response = await qdrant_client.get(f"/collections/{settings.DEFAULT_COLLECTION}")
        response.raise_for_status()
        return response.json()["result"]
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=502, detail="Failed to get collection info")


@app.get("/models")
async def list_models():
    """List available embedding models and their properties"""
    from webui.embedding_service import QUANTIZED_MODEL_INFO

    models = []
    for model_name, info in QUANTIZED_MODEL_INFO.items():
        models.append(
            {
                "name": model_name,
                "description": info.get("description", ""),
                "dimension": info.get("dimension"),
                "supports_quantization": info.get("supports_quantization", True),
                "recommended_quantization": info.get("recommended_quantization", "float32"),
                "memory_estimate": info.get("memory_estimate", {}),
                "is_qwen3": "Qwen3-Embedding" in model_name,
            }
        )

    return {
        "models": models,
        "current_model": embedding_service.current_model_name if embedding_service else None,
        "current_quantization": embedding_service.current_quantization if embedding_service else None,
    }


@app.post("/models/load")
async def load_model(
    model_name: str = Body(..., description="Model name to load"),
    quantization: str = Body("float32", description="Quantization type"),
):
    """Load a specific embedding model"""
    if settings.USE_MOCK_EMBEDDINGS:
        raise HTTPException(status_code=400, detail="Cannot load models when using mock embeddings")

    try:
        success = await asyncio.get_event_loop().run_in_executor(
            executor, embedding_service.load_model, model_name, quantization
        )

        if success:
            model_info = embedding_service.get_model_info(model_name, quantization)
            return {"status": "success", "model": model_name, "quantization": quantization, "info": model_info}
        else:
            raise HTTPException(status_code=400, detail="Failed to load model")

    except Exception as e:
        logger.error(f"Model load error: {e}")
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")


@app.get("/embedding/info")
async def embedding_info():
    """Get information about the embedding configuration"""
    info = {
        "mode": "mock" if settings.USE_MOCK_EMBEDDINGS else "real",
        "available": not settings.USE_MOCK_EMBEDDINGS and embedding_service is not None,
    }

    if not settings.USE_MOCK_EMBEDDINGS and embedding_service:
        info.update(
            {
                "current_model": embedding_service.current_model_name,
                "quantization": embedding_service.current_quantization,
                "device": embedding_service.device,
                "default_model": settings.DEFAULT_EMBEDDING_MODEL,
                "default_quantization": settings.DEFAULT_QUANTIZATION,
            }
        )

        # Get model details
        if embedding_service.current_model_name:
            model_info = embedding_service.get_model_info(
                embedding_service.current_model_name, embedding_service.current_quantization
            )
            info["model_details"] = model_info

    return info


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.SEARCH_API_PORT,
        reload=False,  # Disable reload to avoid metrics registration issues
        log_level="info",
    )

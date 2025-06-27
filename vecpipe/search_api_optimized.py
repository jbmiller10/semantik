#!/usr/bin/env python3
"""
Optimized FastAPI search service with Qwen3 support
REST API for vector similarity search with real embeddings
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Union
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel, Field
import httpx
import uvicorn
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from webui.embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.1.173")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "work_docs")
DEFAULT_K = 10
DEFAULT_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEFAULT_QUANTIZATION = os.getenv("DEFAULT_QUANTIZATION", "float16")

# Search instructions for different use cases
SEARCH_INSTRUCTIONS = {
    "semantic": "Represent this sentence for searching relevant passages:",
    "question": "Represent this question for retrieving supporting documents:",
    "code": "Represent this code query for finding similar code snippets:",
    "hybrid": "Generate a comprehensive embedding for multi-modal search:"
}

# Response models
class SearchResult(BaseModel):
    path: str
    chunk_id: str
    score: float
    doc_id: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(DEFAULT_K, ge=1, le=100, description="Number of results")
    search_type: str = Field("semantic", description="Type of search: semantic, question, code, hybrid")
    model_name: Optional[str] = Field(None, description="Override embedding model")
    quantization: Optional[str] = Field(None, description="Override quantization: float32, float16, int8")
    filters: Optional[Dict] = Field(None, description="Metadata filters for search")
    include_content: bool = Field(False, description="Include chunk content in results")

class BatchSearchRequest(BaseModel):
    queries: List[str] = Field(..., description="List of search queries")
    k: int = Field(DEFAULT_K, ge=1, le=100, description="Number of results per query")
    search_type: str = Field("semantic", description="Type of search")
    model_name: Optional[str] = Field(None, description="Override embedding model")
    quantization: Optional[str] = Field(None, description="Override quantization")

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    num_results: int
    search_type: str
    model_used: str
    embedding_time_ms: Optional[float] = None
    search_time_ms: Optional[float] = None

class BatchSearchResponse(BaseModel):
    responses: List[SearchResponse]
    total_time_ms: float

# Global resources
qdrant_client = None
embedding_service = None
executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global qdrant_client, embedding_service, executor
    
    # Startup
    qdrant_client = httpx.AsyncClient(
        base_url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        timeout=60.0
    )
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    
    # Pre-load default model
    if embedding_service.load_model(DEFAULT_MODEL, DEFAULT_QUANTIZATION):
        logger.info(f"Pre-loaded default model: {DEFAULT_MODEL} with {DEFAULT_QUANTIZATION}")
    else:
        logger.warning("Failed to pre-load default model")
    
    # Create thread pool for CPU-bound embedding operations
    executor = ThreadPoolExecutor(max_workers=4)
    
    yield
    
    # Shutdown
    await qdrant_client.aclose()
    executor.shutdown(wait=True)
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Qwen3-Optimized Vector Search API",
    description="High-performance document search using Qwen3 embeddings",
    version="2.0.0",
    lifespan=lifespan
)

async def generate_embedding_async(text: str, model_name: str, quantization: str, instruction: Optional[str] = None) -> List[float]:
    """Generate embedding asynchronously using thread pool"""
    loop = asyncio.get_event_loop()
    
    # Run embedding generation in thread pool
    embedding = await loop.run_in_executor(
        executor,
        embedding_service.generate_single_embedding,
        text,
        model_name,
        quantization,
        instruction
    )
    
    if embedding is None:
        raise ValueError("Failed to generate embedding")
    
    return embedding

async def search_qdrant(query_vector: List[float], k: int, filters: Optional[Dict] = None) -> List[Dict]:
    """Search Qdrant with optional filters"""
    search_request = {
        "vector": query_vector,
        "limit": k,
        "with_payload": True,
        "with_vector": False
    }
    
    # Add filters if provided
    if filters:
        search_request["filter"] = filters
    
    response = await qdrant_client.post(
        f"/collections/{COLLECTION_NAME}/points/search",
        json=search_request
    )
    response.raise_for_status()
    
    return response.json()['result']

@app.get("/")
async def root():
    """Health check endpoint with detailed status"""
    try:
        # Check Qdrant connection
        response = await qdrant_client.get(f"/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        info = response.json()['result']
        
        # Get embedding service info
        model_info = embedding_service.get_model_info(
            embedding_service.current_model_name or DEFAULT_MODEL,
            embedding_service.current_quantization or DEFAULT_QUANTIZATION
        )
        
        return {
            "status": "healthy",
            "collection": {
                "name": COLLECTION_NAME,
                "points_count": info['points_count'],
                "vector_size": info['config']['params']['vectors']['size']
            },
            "embedding_service": {
                "current_model": embedding_service.current_model_name,
                "quantization": embedding_service.current_quantization,
                "device": embedding_service.device,
                "model_info": model_info
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest = Body(...)):
    """
    Search for similar documents with Qwen3 embeddings
    
    Supports different search types:
    - **semantic**: General semantic search
    - **question**: Question-answering search
    - **code**: Code similarity search
    - **hybrid**: Multi-modal search
    """
    import time
    start_time = time.time()
    
    try:
        # Select model and quantization
        model_name = request.model_name or DEFAULT_MODEL
        quantization = request.quantization or DEFAULT_QUANTIZATION
        
        # Get appropriate instruction for search type
        instruction = SEARCH_INSTRUCTIONS.get(request.search_type, SEARCH_INSTRUCTIONS["semantic"])
        
        # Generate query embedding
        embed_start = time.time()
        logger.info(f"Generating embedding for query: '{request.query}' with {model_name}/{quantization}")
        
        query_vector = await generate_embedding_async(
            request.query,
            model_name,
            quantization,
            instruction
        )
        embed_time = (time.time() - embed_start) * 1000
        
        # Search in Qdrant
        search_start = time.time()
        search_results = await search_qdrant(query_vector, request.k, request.filters)
        search_time = (time.time() - search_start) * 1000
        
        # Parse results
        results = []
        for point in search_results:
            payload = point['payload']
            result = SearchResult(
                path=payload.get('path', ''),
                chunk_id=payload.get('chunk_id', ''),
                score=point['score'],
                doc_id=payload.get('doc_id'),
                content=payload.get('content') if request.include_content else None,
                metadata=payload.get('metadata')
            )
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Search completed in {total_time:.2f}ms (embed: {embed_time:.2f}ms, search: {search_time:.2f}ms)")
        
        return SearchResponse(
            query=request.query,
            results=results,
            num_results=len(results),
            search_type=request.search_type,
            model_used=f"{model_name}/{quantization}",
            embedding_time_ms=embed_time,
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/batch", response_model=BatchSearchResponse)
async def batch_search(request: BatchSearchRequest = Body(...)):
    """
    Batch search for multiple queries
    
    Efficiently processes multiple search queries in parallel
    """
    import time
    start_time = time.time()
    
    try:
        model_name = request.model_name or DEFAULT_MODEL
        quantization = request.quantization or DEFAULT_QUANTIZATION
        instruction = SEARCH_INSTRUCTIONS.get(request.search_type, SEARCH_INSTRUCTIONS["semantic"])
        
        # Generate embeddings for all queries in batch
        logger.info(f"Generating embeddings for {len(request.queries)} queries")
        
        # Create tasks for parallel embedding generation
        embedding_tasks = [
            generate_embedding_async(query, model_name, quantization, instruction)
            for query in request.queries
        ]
        
        # Wait for all embeddings
        query_vectors = await asyncio.gather(*embedding_tasks)
        
        # Create search tasks
        search_tasks = [
            search_qdrant(vector, request.k)
            for vector in query_vectors
        ]
        
        # Execute searches in parallel
        all_results = await asyncio.gather(*search_tasks)
        
        # Build responses
        responses = []
        for query, results in zip(request.queries, all_results):
            parsed_results = []
            for point in results:
                payload = point['payload']
                parsed_results.append(SearchResult(
                    path=payload.get('path', ''),
                    chunk_id=payload.get('chunk_id', ''),
                    score=point['score'],
                    doc_id=payload.get('doc_id')
                ))
            
            responses.append(SearchResponse(
                query=query,
                results=parsed_results,
                num_results=len(parsed_results),
                search_type=request.search_type,
                model_used=f"{model_name}/{quantization}"
            ))
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Batch search completed in {total_time:.2f}ms for {len(request.queries)} queries")
        
        return BatchSearchResponse(
            responses=responses,
            total_time_ms=total_time
        )
        
    except Exception as e:
        logger.error(f"Batch search error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available embedding models and their properties"""
    from webui.embedding_service import QUANTIZED_MODEL_INFO
    
    models = []
    for model_name, info in QUANTIZED_MODEL_INFO.items():
        models.append({
            "name": model_name,
            "description": info.get("description", ""),
            "dimension": info.get("dimension"),
            "supports_quantization": info.get("supports_quantization", True),
            "recommended_quantization": info.get("recommended_quantization", "float32"),
            "memory_estimate": info.get("memory_estimate", {}),
            "is_qwen3": "Qwen3-Embedding" in model_name
        })
    
    return {
        "models": models,
        "current_model": embedding_service.current_model_name,
        "current_quantization": embedding_service.current_quantization
    }

@app.post("/models/load")
async def load_model(
    model_name: str = Body(..., description="Model name to load"),
    quantization: str = Body("float32", description="Quantization type")
):
    """Load a specific embedding model"""
    try:
        success = await asyncio.get_event_loop().run_in_executor(
            executor,
            embedding_service.load_model,
            model_name,
            quantization
        )
        
        if success:
            model_info = embedding_service.get_model_info(model_name, quantization)
            return {
                "status": "success",
                "model": model_name,
                "quantization": quantization,
                "info": model_info
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to load model")
            
    except Exception as e:
        logger.error(f"Model load error: {e}")
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")

@app.get("/collection/info")
async def collection_info():
    """Get detailed information about the vector collection"""
    try:
        response = await qdrant_client.get(f"/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        return response.json()['result']
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=502, detail="Failed to get collection info")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "search_api_optimized:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
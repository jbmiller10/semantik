#!/usr/bin/env python3
"""
FastAPI search service (VS-040)
REST API for vector similarity search with Qwen3 support
"""

import os
import sys
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import httpx
import hashlib
import uvicorn

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from webui.embedding_service import EmbeddingService
from vecpipe.search_utils import search_qdrant, parse_search_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.1.173")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "work_docs"
DEFAULT_K = 10
USE_MOCK_EMBEDDINGS = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"
DEFAULT_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEFAULT_QUANTIZATION = os.getenv("DEFAULT_QUANTIZATION", "float16")

# Response models
class SearchResult(BaseModel):
    path: str
    chunk_id: str
    score: float
    doc_id: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    num_results: int

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
        timeout=30.0
    )
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # Initialize embedding service if not using mock
    if not USE_MOCK_EMBEDDINGS:
        embedding_service = EmbeddingService()
        # Pre-load default model
        if not embedding_service.load_model(DEFAULT_MODEL, DEFAULT_QUANTIZATION):
            logger.error(f"Failed to load embedding model: {DEFAULT_MODEL}")
            raise RuntimeError(f"Cannot start API: Failed to load embedding model {DEFAULT_MODEL}. "
                             f"Either fix the model loading issue or set USE_MOCK_EMBEDDINGS=true")
        logger.info(f"Successfully loaded embedding model: {DEFAULT_MODEL} with {DEFAULT_QUANTIZATION}")
        # Create thread pool for CPU-bound operations
        executor = ThreadPoolExecutor(max_workers=4)
    else:
        logger.info("Using mock embeddings (USE_MOCK_EMBEDDINGS=true)")
    
    yield
    
    # Shutdown
    await qdrant_client.aclose()
    if executor:
        executor.shutdown(wait=True)
    logger.info("Disconnected from Qdrant")

# Create FastAPI app
app = FastAPI(
    title="Document Vector Search API",
    description="Search documents using vector similarity with Qwen3 support",
    version="1.1.0",
    lifespan=lifespan
)

def generate_mock_embedding(text: str, vector_dim: int = None) -> List[float]:
    """Generate mock embedding for testing (fallback when real embeddings unavailable)"""
    # If vector_dim not specified, try to get from collection info
    if vector_dim is None:
        vector_dim = 1024  # Default fallback
    
    # Generate deterministic "embedding" from text hash
    hash_bytes = hashlib.sha256(text.encode()).digest()
    values = []
    
    for i in range(0, len(hash_bytes), 4):
        chunk = hash_bytes[i:i+4]
        if len(chunk) == 4:
            val = int.from_bytes(chunk, byteorder='big') / (2**32)
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

async def generate_embedding_async(text: str, instruction: str = None) -> List[float]:
    """Generate embedding using the embedding service"""
    if USE_MOCK_EMBEDDINGS:
        return generate_mock_embedding(text)
    
    if embedding_service is None:
        raise RuntimeError("Embedding service not initialized")
    
    # Use real embeddings
    loop = asyncio.get_event_loop()
    
    # Determine instruction for search queries
    if instruction is None:
        instruction = "Represent this sentence for searching relevant passages:"
    
    embedding = await loop.run_in_executor(
        executor,
        embedding_service.generate_single_embedding,
        text,
        DEFAULT_MODEL,
        DEFAULT_QUANTIZATION,
        instruction
    )
    
    if embedding is None:
        raise RuntimeError(f"Failed to generate embedding for text: {text[:100]}...")
    
    return embedding

@app.get("/")
async def root():
    """Health check endpoint"""
    try:
        # Check Qdrant connection
        response = await qdrant_client.get(f"/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        info = response.json()['result']
        
        health_info = {
            "status": "healthy",
            "collection": COLLECTION_NAME,
            "points_count": info['points_count'],
            "embedding_mode": "mock" if USE_MOCK_EMBEDDINGS else "real"
        }
        
        if not USE_MOCK_EMBEDDINGS and embedding_service:
            health_info["embedding_model"] = embedding_service.current_model_name
            health_info["quantization"] = embedding_service.current_quantization
        
        return health_info
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results to return"),
    collection: Optional[str] = Query(None, description="Collection name (e.g., job_123)")
):
    """
    Search for similar documents
    
    - **q**: The search query text
    - **k**: Number of results to return (1-100, default 10)
    - **collection**: Optional collection name (defaults to work_docs)
    """
    try:
        # Determine collection name
        collection_name = collection if collection else COLLECTION_NAME
        
        # Get collection info to determine vector dimension
        vector_dim = 1024  # default
        try:
            response = await qdrant_client.get(f"/collections/{collection_name}")
            response.raise_for_status()
            collection_info = response.json()['result']
            if 'config' in collection_info and 'params' in collection_info['config']:
                vector_dim = collection_info['config']['params']['vectors']['size']
        except Exception as e:
            logger.warning(f"Could not get collection info for {collection_name}, using default dimension: {e}")
        
        # Generate query embedding
        logger.info(f"Processing search query: '{q}' (k={k}, collection={collection_name}, vector_dim={vector_dim})")
        
        # Generate query embedding
        if not USE_MOCK_EMBEDDINGS:
            query_vector = await generate_embedding_async(q)
        else:
            query_vector = generate_mock_embedding(q, vector_dim)
        
        # Search in Qdrant using shared utility
        qdrant_results = await search_qdrant(
            QDRANT_HOST,
            QDRANT_PORT,
            collection_name,
            query_vector,
            k
        )
        
        # Parse results using shared utility
        parsed_results = parse_search_results(qdrant_results)
        
        results = []
        for r in parsed_results:
            result = SearchResult(
                path=r['path'],
                chunk_id=r['chunk_id'],
                score=r['score'],
                doc_id=r.get('doc_id')
            )
            results.append(result)
        
        logger.info(f"Found {len(results)} results for query: '{q}'")
        
        return SearchResponse(
            query=q,
            results=results,
            num_results=len(results)
        )
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Qdrant error: {e}")
        raise HTTPException(status_code=502, detail="Vector database error")
    except RuntimeError as e:
        # Specific handling for embedding failures
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Embedding service error: {str(e)}. Check logs for details."
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/collection/info")
async def collection_info():
    """Get information about the vector collection"""
    try:
        response = await qdrant_client.get(f"/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        return response.json()['result']
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=502, detail="Failed to get collection info")

@app.get("/embedding/info")
async def embedding_info():
    """Get information about the embedding configuration"""
    info = {
        "mode": "mock" if USE_MOCK_EMBEDDINGS else "real",
        "available": not USE_MOCK_EMBEDDINGS and embedding_service is not None
    }
    
    if not USE_MOCK_EMBEDDINGS and embedding_service:
        info.update({
            "current_model": embedding_service.current_model_name,
            "quantization": embedding_service.current_quantization,
            "device": embedding_service.device,
            "default_model": DEFAULT_MODEL,
            "default_quantization": DEFAULT_QUANTIZATION
        })
        
        # Get model details
        if embedding_service.current_model_name:
            model_info = embedding_service.get_model_info(
                embedding_service.current_model_name,
                embedding_service.current_quantization
            )
            info["model_details"] = model_info
    
    return info

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "search_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
#!/usr/bin/env python3
"""
FastAPI search service (VS-040)
REST API for vector similarity search
"""

import os
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import httpx
import hashlib
import uvicorn

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
VECTOR_DIM = 1024
DEFAULT_K = 10

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

# Global client
qdrant_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global qdrant_client
    # Startup
    qdrant_client = httpx.AsyncClient(
        base_url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        timeout=30.0
    )
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    yield
    
    # Shutdown
    await qdrant_client.aclose()
    logger.info("Disconnected from Qdrant")

# Create FastAPI app
app = FastAPI(
    title="Document Vector Search API",
    description="Search documents using vector similarity",
    version="1.0.0",
    lifespan=lifespan
)

def generate_mock_embedding(text: str) -> List[float]:
    """Generate mock embedding for testing (same as in embed_chunks_simple.py)"""
    # Generate deterministic "embedding" from text hash
    hash_bytes = hashlib.sha256(text.encode()).digest()
    values = []
    
    for i in range(0, len(hash_bytes), 4):
        chunk = hash_bytes[i:i+4]
        if len(chunk) == 4:
            val = int.from_bytes(chunk, byteorder='big') / (2**32)
            values.append(val * 2 - 1)
    
    # Pad or truncate to VECTOR_DIM
    if len(values) < VECTOR_DIM:
        values.extend([0.0] * (VECTOR_DIM - len(values)))
    else:
        values = values[:VECTOR_DIM]
    
    # Normalize
    norm = sum(v**2 for v in values) ** 0.5
    if norm > 0:
        values = [v / norm for v in values]
    
    return values

@app.get("/")
async def root():
    """Health check endpoint"""
    try:
        # Check Qdrant connection
        response = await qdrant_client.get(f"/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        info = response.json()['result']
        
        return {
            "status": "healthy",
            "collection": COLLECTION_NAME,
            "points_count": info['points_count']
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    k: int = Query(DEFAULT_K, ge=1, le=100, description="Number of results to return")
):
    """
    Search for similar documents
    
    - **q**: The search query text
    - **k**: Number of results to return (1-100, default 10)
    """
    try:
        # Generate query embedding
        logger.info(f"Processing search query: '{q}' (k={k})")
        query_vector = generate_mock_embedding(q)
        
        # Search in Qdrant
        search_request = {
            "vector": query_vector,
            "limit": k,
            "with_payload": True
        }
        
        response = await qdrant_client.post(
            f"/collections/{COLLECTION_NAME}/points/search",
            json=search_request
        )
        response.raise_for_status()
        
        # Parse results
        search_results = response.json()['result']
        
        results = []
        for point in search_results:
            payload = point['payload']
            result = SearchResult(
                path=payload['path'],
                chunk_id=payload['chunk_id'],
                score=point['score'],
                doc_id=payload.get('doc_id')
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
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

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

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "search_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
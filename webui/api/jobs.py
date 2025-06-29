"""
Job management routes and WebSocket handlers for the Web UI
"""

import os
import sys
import logging
import asyncio
import uuid
import hashlib
import gc
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, validator
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vecpipe.config import settings
from vecpipe.extract_chunks import extract_text, TokenChunker
from webui import database
from webui.auth import get_current_user
from webui.embedding_service import embedding_service, POPULAR_MODELS

logger = logging.getLogger(__name__)

# Constants
JOBS_DIR = str(settings.JOBS_DIR)
OUTPUT_DIR = str(settings.OUTPUT_DIR)
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.text', '.pptx', '.eml', '.md', '.html']

# Create necessary directories
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Request/Response models
class CreateJobRequest(BaseModel):
    name: str
    description: str = ""
    directory_path: str
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    chunk_size: int = 600
    chunk_overlap: int = 200
    batch_size: int = 96
    vector_dim: Optional[int] = None
    quantization: str = "float32"
    instruction: Optional[str] = None
    job_id: Optional[str] = None  # Allow pre-generated job_id for WebSocket connection
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v <= 0:
            raise ValueError('chunk_size must be positive')
        if v < 100:
            raise ValueError('chunk_size must be at least 100 tokens')
        if v > 50000:
            raise ValueError('chunk_size must not exceed 50000 tokens')
        return v
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        if v < 0:
            raise ValueError('chunk_overlap cannot be negative')
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError(f'chunk_overlap ({v}) must be less than chunk_size ({values["chunk_size"]})')
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError('batch_size must be positive')
        if v > 1000:
            raise ValueError('batch_size must not exceed 1000')
        return v

class JobStatus(BaseModel):
    id: str
    name: str
    status: str  # created, scanning, processing, completed, failed
    created_at: str
    updated_at: str
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    current_file: Optional[str] = None
    error: Optional[str] = None
    model_name: str
    directory_path: str
    quantization: Optional[str] = None
    batch_size: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
    
    async def send_update(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, job_id)

manager = ConnectionManager()

# Global task tracking for cancellation support
active_job_tasks: Dict[str, asyncio.Task] = {}

# Background task executor - increased workers for parallel processing
executor = ThreadPoolExecutor(max_workers=8)

def extract_text_thread_safe(filepath: str) -> str:
    """Thread-safe version of extract_text that uses the unified extraction"""
    return extract_text(filepath)

# Import scan function from files module to avoid circular import  
from webui.api.files import scan_directory_async


async def update_metrics_continuously():
    """Background task to update resource metrics continuously"""
    while True:
        try:
            from vecpipe.metrics import metrics_collector
            metrics_collector.last_update = 0  # Force update
            metrics_collector.update_resource_metrics()
        except:
            pass
        await asyncio.sleep(.5)  # Update every 1 second for smoother metrics

async def process_embedding_job(job_id: str):
    """Process an embedding job asynchronously"""
    metrics_task = None  # Initialize to avoid undefined reference
    
    # Import metrics if available
    try:
        from vecpipe.metrics import (
            record_file_processed, record_file_failed,
            record_chunks_created, record_embeddings_generated,
            metrics_collector, embedding_batch_duration
        )
        METRICS_TRACKING = True
        # Set shorter update interval for webui
        metrics_collector.update_interval = .5  # 1 second for smoother updates
        
        # Start background metrics updater
        metrics_task = asyncio.create_task(update_metrics_continuously())
    except ImportError:
        METRICS_TRACKING = False
        logger.warning("Metrics tracking not available for embedding job")
    
    try:
        # Update job status and set start time
        database.update_job(job_id, {
            'status': 'processing',
            'start_time': datetime.now().isoformat()
        })
        
        # Get job details
        job = database.get_job(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")
        
        # Get pending files
        files = database.get_job_files(job_id, status='pending')
        
        # Send initial update with total files
        await manager.send_update(job_id, {
            "type": "job_started",
            "total_files": len(files)
        })
        
        # Initialize Qdrant client
        qdrant = QdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        
        for file_idx, file_row in enumerate(files):
            try:
                # Update current file
                database.update_job(job_id, {'current_file': file_row['path']})
                
                # Send progress update
                await manager.send_update(job_id, {
                    "type": "file_processing",
                    "current_file": file_row['path'],
                    "processed_files": file_idx,
                    "total_files": len(files),
                    "status": "Processing"
                })
                
                # Yield control to event loop to keep UI responsive
                await asyncio.sleep(0)
                
                # Extract text and create chunks
                logger.info(f"Processing file: {file_row['path']}")
                
                # Add memory tracking
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Memory before extraction: {initial_memory:.2f} MB")
                
                logger.info(f"Starting text extraction for: {file_row['path']}")
                # Run text extraction in thread pool to avoid blocking
                try:
                    loop = asyncio.get_event_loop()
                    # Use asyncio timeout instead of signal-based timeout
                    text = await asyncio.wait_for(
                        loop.run_in_executor(
                            executor,
                            lambda: extract_text_thread_safe(file_row['path'])
                        ),
                        timeout=300  # 5 minute timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Text extraction timed out after 300 seconds for {file_row['path']}")
                
                memory_after_extract = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Memory after extraction: {memory_after_extract:.2f} MB (delta: {memory_after_extract - initial_memory:.2f} MB)")
                logger.info(f"Extracted text length: {len(text)} characters")
                
                doc_id = hashlib.md5(file_row['path'].encode()).hexdigest()[:16]
                
                logger.info(f"Starting chunking for: {file_row['path']} with chunk_size={job['chunk_size']}, overlap={job['chunk_overlap']}")
                
                # Create a chunker with job-specific settings
                chunker = TokenChunker(
                    chunk_size=job['chunk_size'] or 600,
                    chunk_overlap=job['chunk_overlap'] or 200
                )
                # Run chunking in thread pool to avoid blocking
                chunks = await loop.run_in_executor(
                    executor,
                    chunker.chunk_text,
                    text,
                    doc_id
                )
                
                memory_after_chunk = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Memory after chunking: {memory_after_chunk:.2f} MB (delta: {memory_after_chunk - memory_after_extract:.2f} MB)")
                logger.info(f"Created {len(chunks)} chunks")
                
                # Record chunks created
                if METRICS_TRACKING:
                    record_chunks_created(len(chunks))
                
                # Free the original text immediately after chunking
                del text
                
                # Generate embeddings
                texts = [chunk['text'] for chunk in chunks]
                
                # Use unified embedding service - run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Time the embedding generation
                import time
                embed_start_time = time.time()
                
                embeddings_array = await loop.run_in_executor(
                    executor,
                    embedding_service.generate_embeddings,
                    texts, 
                    job['model_name'],
                    job['quantization'] or 'float32',
                    job['batch_size'],
                    False,  # show_progress
                    job['instruction']
                )
                
                # Record embedding time
                if METRICS_TRACKING:
                    embed_duration = time.time() - embed_start_time
                    logger.info(f"Embedding generation took {embed_duration:.3f} seconds for {len(texts)} texts")
                    embedding_batch_duration.observe(embed_duration)
                
                # Free texts list after embedding generation
                del texts
                
                if embeddings_array is None:
                    raise Exception("Failed to generate embeddings")
                
                # Record embeddings generated
                if METRICS_TRACKING:
                    record_embeddings_generated(len(embeddings_array))
                
                embeddings = embeddings_array.tolist()
                
                # Free the numpy array after conversion
                del embeddings_array
                
                # Handle dimension override if specified
                target_dim = job['vector_dim']
                if target_dim and len(embeddings) > 0:
                    model_dim = len(embeddings[0])
                    if target_dim != model_dim:
                        logger.info(f"Adjusting embeddings from {model_dim} to {target_dim} dimensions")
                        adjusted_embeddings = []
                        for emb in embeddings:
                            if target_dim < model_dim:
                                # Truncate
                                adjusted = emb[:target_dim]
                            else:
                                # Pad with zeros
                                adjusted = emb + [0.0] * (target_dim - model_dim)
                            
                            # Renormalize
                            norm = sum(x**2 for x in adjusted) ** 0.5
                            if norm > 0:
                                adjusted = [x / norm for x in adjusted]
                            adjusted_embeddings.append(adjusted)
                        embeddings = adjusted_embeddings
                
                # Prepare points for Qdrant in batches to avoid memory spikes
                UPLOAD_BATCH_SIZE = 100
                total_points = len(chunks)
                
                for batch_start in range(0, total_points, UPLOAD_BATCH_SIZE):
                    batch_end = min(batch_start + UPLOAD_BATCH_SIZE, total_points)
                    points = []
                    
                    for i in range(batch_start, batch_end):
                        chunk = chunks[i]
                        embedding = embeddings[i]
                        point = {
                            "id": str(uuid.uuid4()),
                            "vector": embedding,
                            "payload": {
                                "job_id": job_id,
                                "doc_id": doc_id,
                                "chunk_id": chunk['chunk_id'],
                                "path": file_row['path'],
                                "content": chunk['text']  # Store full text for hybrid search
                            }
                        }
                        points.append(point)
                    
                    # Upload batch to Qdrant
                    if points:
                        point_structs = [
                            PointStruct(
                                id=point["id"],
                                vector=point["vector"],
                                payload=point["payload"]
                            )
                            for point in points
                        ]
                        qdrant.upsert(
                            collection_name=f"job_{job_id}",
                            points=point_structs
                        )
                    
                    # Free the points batch
                    del points
                
                # Update database after all batches uploaded
                database.update_file_status(job_id, file_row['path'], 'completed', 
                                          vectors_created=total_points,
                                          chunks_created=len(chunks))
                # Get current job to update processed files count
                current_job = database.get_job(job_id)
                database.update_job(job_id, {'processed_files': current_job['processed_files'] + 1})
                
                # Record file processed
                if METRICS_TRACKING:
                    record_file_processed('embedding')
                
                # Send file completed update
                await manager.send_update(job_id, {
                    "type": "file_completed",
                    "processed_files": current_job['processed_files'] + 1,
                    "total_files": len(files)
                })
                
                # Free chunks and embeddings after upload
                del chunks
                del embeddings
                
                # Force garbage collection after each file
                gc.collect()
                
                # Update resource metrics periodically (force update)
                if METRICS_TRACKING:
                    # Force update by resetting the last update time
                    metrics_collector.last_update = 0
                    metrics_collector.update_resource_metrics()
                
                # Yield control to event loop after processing each file
                await asyncio.sleep(0)
                
            except Exception as e:
                logger.error(f"Failed to process file {file_row['path']}: {e}")
                # Record file failed
                if METRICS_TRACKING:
                    record_file_failed('embedding', type(e).__name__)
                # File status already updated in the database.update_file_status call above
        
        # Mark job as completed
        database.update_job(job_id, {'status': 'completed', 'current_file': None})
        
        await manager.send_update(job_id, {
            "type": "job_completed",
            "message": "Job completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        database.update_job(job_id, {'status': 'failed', 'error': str(e)})
        
        await manager.send_update(job_id, {
            "type": "error",
            "message": str(e)
        })
    
    finally:
        qdrant.close()
        # Cancel metrics updater task if it exists
        if metrics_task:
            metrics_task.cancel()
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass

# API Routes
@router.get("/new-id")
async def get_new_job_id(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Generate a new job ID for WebSocket connection"""
    return {"job_id": str(uuid.uuid4())}

@router.post("", response_model=JobStatus)
async def create_job(request: CreateJobRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Create a new embedding job"""
    # Accept job_id from request if provided, otherwise generate new one
    job_id = request.job_id if request.job_id else str(uuid.uuid4())
    
    try:
        # Scan directory first - use async version to avoid blocking UI
        files = await scan_directory_async(request.directory_path, recursive=True, scan_id=job_id)
        
        if not files:
            raise HTTPException(status_code=400, detail="No supported files found in directory")
        
        # Create job record
        now = datetime.now().isoformat()
        job_data = {
            'id': job_id,
            'name': request.name,
            'description': request.description,
            'status': 'created',
            'created_at': now,
            'updated_at': now,
            'directory_path': request.directory_path,
            'model_name': request.model_name,
            'chunk_size': request.chunk_size,
            'chunk_overlap': request.chunk_overlap,
            'batch_size': request.batch_size,
            'vector_dim': request.vector_dim,
            'quantization': request.quantization,
            'instruction': request.instruction
        }
        database.create_job(job_data)
        
        # Create file records
        file_records = [{
            'path': f.path,
            'size': f.size,
            'modified': f.modified,
            'extension': f.extension,
            'hash': getattr(f, 'hash', None)
        } for f in files]
        database.add_files_to_job(job_id, file_records)
        
        # Create Qdrant collection for this job
        qdrant = QdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        
        # Determine vector size
        vector_size = request.vector_dim
        if not vector_size:
            # Try to get from POPULAR_MODELS first
            if request.model_name in POPULAR_MODELS:
                vector_size = POPULAR_MODELS[request.model_name].get('dim') or POPULAR_MODELS[request.model_name].get('dimension')
            
            # If still not found, get from actual model
            if not vector_size:
                try:
                    model_info = embedding_service.get_model_info(request.model_name, request.quantization)
                    if not model_info.get('error'):
                        vector_size = model_info['embedding_dim']
                    else:
                        logger.warning(f"Could not get model info: {model_info.get('error')}")
                        vector_size = 1024  # Default fallback
                except Exception as e:
                    logger.warning(f"Error getting model info: {e}")
                    vector_size = 1024  # Default fallback
        
        # Update job with actual vector dimension
        database.update_job(job_id, {'vector_dim': vector_size})
            
        # Create collection using official client
        collection_name = f"job_{job_id}"
        try:
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
                optimizers_config={
                    "indexing_threshold": 20000,
                    "memmap_threshold": 0
                }
            )
            
            # Store metadata about this collection
            from webui.api.collection_metadata import store_collection_metadata
            store_collection_metadata(
                qdrant=qdrant,
                collection_name=collection_name,
                model_name=request.model_name,
                quantization=request.quantization,
                vector_dim=vector_size,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                instruction=request.instruction
            )
            
        except Exception as e:
            logger.warning(f"Collection {collection_name} might already exist: {e}")
        
        # Start processing in background with cancellation support
        task = asyncio.create_task(process_embedding_job(job_id))
        active_job_tasks[job_id] = task
        
        return JobStatus(
            id=job_id,
            name=request.name,
            status='created',
            created_at=now,
            updated_at=now,
            total_files=len(files),
            processed_files=0,
            failed_files=0,
            model_name=request.model_name,
            directory_path=request.directory_path,
            quantization=request.quantization,
            batch_size=request.batch_size,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("", response_model=List[JobStatus])
async def list_jobs(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all jobs"""
    jobs = database.list_jobs()
    
    result = []
    for job in jobs:
        result.append(JobStatus(
            id=job['id'],
            name=job['name'],
            status=job['status'],
            created_at=job['created_at'],
            updated_at=job['updated_at'],
            total_files=job.get('total_files', 0),
            processed_files=job.get('processed_files', 0),
            failed_files=job.get('failed_files', 0),
            current_file=job.get('current_file'),
            error=job.get('error'),
            model_name=job['model_name'],
            directory_path=job['directory_path'],
            quantization=job.get('quantization'),
            batch_size=job.get('batch_size'),
            chunk_size=job.get('chunk_size'),
            chunk_overlap=job.get('chunk_overlap')
        ))
    
    return result

@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Cancel a running job"""
    # Check current job status
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] not in ['created', 'scanning', 'processing']:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in status: {job['status']}")
    
    # Update job status to cancelled
    database.update_job(job_id, {'status': 'cancelled'})
    
    # Cancel the task if it's running
    if job_id in active_job_tasks:
        active_job_tasks[job_id].cancel()
        
    return {"message": "Job cancellation requested"}

@router.delete("/{job_id}")
async def delete_job(job_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Delete a job and its associated collection"""
    # Check if job exists
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete from Qdrant
    collection_name = f"job_{job_id}"
    try:
        async_client = AsyncQdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        await async_client.delete_collection(collection_name)
    except Exception as e:
        logger.warning(f"Failed to delete Qdrant collection: {e}")
    
    # Delete from database
    database.delete_job(job_id)
    
    return {"message": "Job deleted successfully"}

@router.get("/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get job details"""
    job = database.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(
        id=job['id'],
        name=job['name'],
        status=job['status'],
        created_at=job['created_at'],
        updated_at=job['updated_at'],
        total_files=job['total_files'],
        processed_files=job['processed_files'],
        failed_files=job['failed_files'],
        current_file=job['current_file'],
        error=job['error'],
        model_name=job['model_name'],
        directory_path=job['directory_path'],
        quantization=job.get('quantization'),
        batch_size=job.get('batch_size'),
        chunk_size=job.get('chunk_size'),
        chunk_overlap=job.get('chunk_overlap')
    )


# WebSocket handler - export this separately so it can be mounted at the app level
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket for real-time job updates"""
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
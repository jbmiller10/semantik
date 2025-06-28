#!/usr/bin/env python3
"""
Web UI for Document Embedding System
Provides interface for creating and searching embeddings
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import uuid
from concurrent.futures import ThreadPoolExecutor
import gc

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, validator
import httpx
from tqdm import tqdm
import hashlib
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from vecpipe.extract_chunks import extract_text, chunk_text, process_file
from vecpipe.search_utils import search_qdrant
from vecpipe.hybrid_search import HybridSearchEngine
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Import the unified embedding service
from webui.embedding_service import embedding_service, POPULAR_MODELS

# Import authentication
from webui.auth import (
    UserCreate, UserLogin, Token, User, get_current_user,
    authenticate_user, create_access_token, create_refresh_token,
    get_password_hash
)

# Import database operations
from webui import database

# Constants
JOBS_DIR = "/var/embeddings/jobs"
OUTPUT_DIR = "/var/embeddings/output"
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.text', '.pptx', '.eml', '.md', '.html']
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.1.173")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Create necessary directories
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Request/Response models
class ScanDirectoryRequest(BaseModel):
    path: str
    recursive: bool = True

class FileInfo(BaseModel):
    path: str
    size: int
    modified: str
    extension: str
    hash: Optional[str] = None

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


# Create FastAPI app
app = FastAPI(
    title="Document Embedding Web UI",
    description="Create and search document embeddings",
    version="1.0.0"
)

# Start metrics server if metrics port is configured
METRICS_PORT = int(os.getenv("WEBUI_METRICS_PORT", "9092"))
METRICS_AVAILABLE = False
generate_latest = None
registry = None
if METRICS_PORT:
    try:
        from vecpipe.metrics import start_metrics_server, generate_latest, registry
        start_metrics_server(METRICS_PORT)
        logger.info(f"Metrics server started on port {METRICS_PORT}")
        METRICS_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")

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
    # The new extract_text function in extract_chunks.py is already thread-safe
    # as it doesn't use signals in the unstructured implementation
    return extract_text(filepath)

def scan_directory(path: str, recursive: bool = True) -> List[FileInfo]:
    """Scan directory for supported files"""
    files = []
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    # Determine search pattern
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in path_obj.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                stat = file_path.stat()
                files.append(FileInfo(
                    path=str(file_path),
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    extension=file_path.suffix.lower()
                ))
            except Exception as e:
                logger.warning(f"Failed to stat file {file_path}: {e}")
    
    return files

async def scan_directory_async(path: str, recursive: bool = True, scan_id: str = None) -> List[FileInfo]:
    """Scan directory for supported files with progress updates"""
    files = []
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    # First, count total files to scan
    total_files = 0
    scanned_files = 0
    
    # Determine search pattern
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    # Count phase
    for _ in path_obj.glob(pattern):
        total_files += 1
        if total_files % 100 == 0 and scan_id:
            await manager.send_update(f"scan_{scan_id}", {
                "type": "counting",
                "count": total_files
            })
    
    # Scan phase
    for file_path in path_obj.glob(pattern):
        scanned_files += 1
        
        # Send progress update every 10 files or at specific percentages
        if scan_id and (scanned_files % 10 == 0 or scanned_files == total_files):
            await manager.send_update(f"scan_{scan_id}", {
                "type": "progress",
                "scanned": scanned_files,
                "total": total_files,
                "current_path": str(file_path)
            })
        
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                stat = file_path.stat()
                files.append(FileInfo(
                    path=str(file_path),
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    extension=file_path.suffix.lower()
                ))
            except Exception as e:
                logger.warning(f"Failed to stat file {file_path}: {e}")
    
    return files

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
            metrics_collector, TimingContext, embedding_batch_duration
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
        qdrant = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
        
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
                # Note: extract_text uses signals which don't work in threads,
                # so we need to handle this differently
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
                from vecpipe.extract_chunks import TokenChunker
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
                
                # Update chunks created (we'll need to update the file status later)
                
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
                import gc
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

# Authentication Routes
@app.post("/api/auth/register", response_model=User)
async def register(user_data: UserCreate):
    """Register a new user"""
    try:
        hashed_password = get_password_hash(user_data.password)
        user = database.create_user(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login", response_model=Token)
async def login(login_data: UserLogin):
    """Login and receive access token"""
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": user["username"]})
    refresh_token = create_refresh_token(data={"sub": user["username"]})
    
    # Save refresh token
    from datetime import timedelta
    expires_at = datetime.utcnow() + timedelta(days=30)
    # Hash the token for storage
    from webui.auth import pwd_context
    token_hash = pwd_context.hash(refresh_token)
    database.save_refresh_token(user["id"], token_hash, expires_at)
    
    return Token(access_token=access_token, refresh_token=refresh_token)

@app.post("/api/auth/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token"""
    user_id = database.verify_refresh_token(refresh_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    # Get user by ID
    user = database.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Create new tokens
    access_token = create_access_token(data={"sub": user["username"]})
    new_refresh_token = create_refresh_token(data={"sub": user["username"]})
    
    # Revoke old refresh token and save new one
    database.revoke_refresh_token(refresh_token)
    from datetime import timedelta
    expires_at = datetime.utcnow() + timedelta(days=30)
    # Hash the new token for storage
    from webui.auth import pwd_context
    token_hash = pwd_context.hash(new_refresh_token)
    database.save_refresh_token(user["id"], token_hash, expires_at)
    
    return Token(access_token=access_token, refresh_token=new_refresh_token)

@app.post("/api/auth/logout")
async def logout(refresh_token: str = None, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout and revoke refresh token"""
    if refresh_token:
        database.revoke_refresh_token(refresh_token)
    return {"message": "Logged out successfully"}

@app.get("/api/auth/me", response_model=User)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user info"""
    return User(**current_user)

# API Routes (Public - no auth required for root pages)
@app.get("/")
async def root():
    """Serve the main UI"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, 'static', 'index.html'))

@app.get("/login.html")
async def login_page():
    """Serve the login page"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, 'static', 'login.html'))

@app.get("/settings")
async def settings_page():
    """Serve the settings page"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, 'static', 'settings.html'))

@app.post("/api/settings/reset-database")
async def reset_database_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Reset the database"""
    try:
        # Get all job IDs before reset
        jobs = database.list_jobs()
        job_ids = [job['id'] for job in jobs]
        
        # Delete Qdrant collections for all jobs
        async_client = AsyncQdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
        for job_id in job_ids:
            collection_name = f"job_{job_id}"
            try:
                await async_client.delete_collection(collection_name)
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection_name}: {e}")
        
        # Delete all parquet files
        try:
            parquet_files = glob.glob(os.path.join(OUTPUT_DIR, "*.parquet"))
            for pf in parquet_files:
                os.remove(pf)
                logger.info(f"Deleted parquet file: {pf}")
        except Exception as e:
            logger.warning(f"Failed to delete parquet files: {e}")
        
        # Reset database
        database.reset_database()
        
        return {"status": "success", "message": "Database reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings/stats")
async def get_database_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get database statistics"""
    # Get stats from database module
    stats = database.get_database_stats()
    
    # Get database file size
    db_size = os.path.getsize(database.DB_PATH) if os.path.exists(database.DB_PATH) else 0
    
    # Get total parquet files size
    parquet_files = glob.glob(os.path.join(OUTPUT_DIR, "*.parquet"))
    parquet_size = sum(os.path.getsize(f) for f in parquet_files)
    
    return {
        "job_count": stats['jobs']['total'],
        "file_count": stats['files']['total'],
        "database_size_mb": round(db_size / 1024 / 1024, 2),
        "parquet_files_count": len(parquet_files),
        "parquet_size_mb": round(parquet_size / 1024 / 1024, 2)
    }

@app.post("/api/scan-directory")
async def scan_directory_endpoint(request: ScanDirectoryRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Scan a directory for supported files"""
    try:
        files = scan_directory(request.path, request.recursive)
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/jobs/new-id")
async def get_new_job_id(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Generate a new job ID for WebSocket connection"""
    return {"job_id": str(uuid.uuid4())}

@app.post("/api/jobs", response_model=JobStatus)
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
        qdrant = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
        
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
        try:
            qdrant.create_collection(
                collection_name=f"job_{job_id}",
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
                optimizers_config={
                    "indexing_threshold": 20000,
                    "memmap_threshold": 0
                }
            )
        except Exception as e:
            logger.warning(f"Collection job_{job_id} might already exist: {e}")
        
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
            directory_path=request.directory_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get available embedding models"""
    return {
        "models": POPULAR_MODELS,
        "current_device": embedding_service.device,
        "using_real_embeddings": True  # Always true with unified service
    }

@app.get("/api/jobs", response_model=List[JobStatus])
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
            directory_path=job['directory_path']
        ))
    
    return result

@app.post("/api/jobs/{job_id}/cancel")
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

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Delete a job and its associated collection"""
    # Check if job exists
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete from Qdrant
    collection_name = f"job_{job_id}"
    try:
        async_client = AsyncQdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")
        await async_client.delete_collection(collection_name)
    except Exception as e:
        logger.warning(f"Failed to delete Qdrant collection: {e}")
    
    # Delete from database
    database.delete_job(job_id)
    
    return {"message": "Job deleted successfully"}

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
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
        directory_path=job['directory_path']
    )

@app.post("/api/search")
async def search(request: SearchRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Search for similar documents"""
    try:
        # Determine collection name
        collection_name = f"job_{request.job_id}" if request.job_id else "work_docs"
        
        # Generate query embedding using the appropriate model
        # Get model name from job if specified
        model_name = "BAAI/bge-large-en-v1.5"  # default
        quantization = "float32"  # default
        instruction = None  # default
        
        if request.job_id:
            job = database.get_job(request.job_id)
            if job:
                if job.get('model_name'):
                    model_name = job['model_name']
                if job.get('quantization'):
                    quantization = job['quantization']
                if job.get('instruction'):
                    instruction = job['instruction']
        
        # Generate query embedding with same settings as job
        query_vector = embedding_service.generate_single_embedding(
            request.query, model_name, quantization=quantization, instruction=instruction
        )
            
        if not query_vector:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Search using shared utility
        results = await search_qdrant(
            QDRANT_HOST,
            QDRANT_PORT,
            collection_name,
            query_vector,
            request.k
        )
        
        return {
            "query": request.query,
            "results": results,
            "collection": collection_name
        }
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Collection not found")
        raise HTTPException(status_code=502, detail="Search failed")

@app.post("/api/hybrid_search")
async def hybrid_search(request: HybridSearchRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Perform hybrid search combining vector similarity and text matching"""
    try:
        # Determine collection name
        collection_name = f"job_{request.job_id}" if request.job_id else "work_docs"
        
        # Initialize hybrid search engine
        hybrid_engine = HybridSearchEngine(QDRANT_HOST, QDRANT_PORT, collection_name)
        
        # Get model name and settings from job if specified
        model_name = "BAAI/bge-large-en-v1.5"  # default
        quantization = "float32"  # default
        instruction = None  # default
        
        if request.job_id:
            job = database.get_job(request.job_id)
            if job:
                if job.get('model_name'):
                    model_name = job['model_name']
                if job.get('quantization'):
                    quantization = job['quantization']
                if job.get('instruction'):
                    instruction = job['instruction']
        
        # Generate query embedding with same settings as job
        query_vector = embedding_service.generate_single_embedding(
            request.query, model_name, quantization=quantization, instruction=instruction
        )
            
        if not query_vector:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Extract keywords
        keywords = hybrid_engine.extract_keywords(request.query)
        
        # Convert query_vector to list if it's a numpy array
        if hasattr(query_vector, 'tolist'):
            query_vector_list = query_vector.tolist()
        else:
            query_vector_list = query_vector
        
        # Perform hybrid search
        results = hybrid_engine.hybrid_search(
            query_vector=query_vector_list,
            query_text=request.query,
            limit=request.k,
            keyword_mode=request.keyword_mode,
            score_threshold=request.score_threshold,
            hybrid_mode=request.mode
        )
        
        # Close the engine
        hybrid_engine.close()
        
        return {
            "query": request.query,
            "results": results,
            "collection": collection_name,
            "keywords_extracted": keywords,
            "search_mode": request.mode
        }
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket for real-time job updates"""
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)

@app.websocket("/ws/scan/{scan_id}")
async def scan_websocket(websocket: WebSocket, scan_id: str):
    """WebSocket for real-time scan progress"""
    await manager.connect(websocket, f"scan_{scan_id}")
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "scan":
                path = data.get("path")
                recursive = data.get("recursive", True)
                
                try:
                    # Send initial status
                    await manager.send_update(f"scan_{scan_id}", {
                        "type": "started",
                        "path": path
                    })
                    
                    # Perform scan with progress updates
                    files = await scan_directory_async(path, recursive, scan_id)
                    
                    # Send completion
                    await manager.send_update(f"scan_{scan_id}", {
                        "type": "completed",
                        "files": [f.dict() for f in files],
                        "count": len(files)
                    })
                except Exception as e:
                    await manager.send_update(f"scan_{scan_id}", {
                        "type": "error",
                        "error": str(e)
                    })
            elif data.get("action") == "cancel":
                # Handle cancellation
                await manager.send_update(f"scan_{scan_id}", {
                    "type": "cancelled"
                })
                break
    except WebSocketDisconnect:
        manager.disconnect(websocket, f"scan_{scan_id}")

# Metrics endpoint
@app.get("/api/metrics")
async def get_metrics(current_user: User = Depends(get_current_user)):
    """Get current Prometheus metrics"""
    if not METRICS_AVAILABLE:
        return {"error": "Metrics not available", "metrics_port": METRICS_PORT}
    
    try:
        # Generate metrics in Prometheus format
        metrics_data = generate_latest(registry)
        return {
            "available": True,
            "metrics_port": METRICS_PORT,
            "data": metrics_data.decode('utf-8')
        }
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return {"error": str(e), "metrics_port": METRICS_PORT}

# Mount static files with proper path resolution
base_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
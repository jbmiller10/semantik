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

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import httpx
from tqdm import tqdm
import sqlite3
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
from vecpipe.embed_chunks_simple import generate_mock_embeddings
from vecpipe.ingest_qdrant import QdrantClient

# Try to import enhanced embedding service with quantization support
try:
    from webui.embedding_service_v2 import enhanced_embedding_service as embedding_service, QUANTIZED_MODEL_INFO as POPULAR_MODELS
    USE_ENHANCED_EMBEDDINGS = True
    logger.info("Using enhanced embedding service with quantization support")
except Exception as e:
    logger.warning(f"Could not load enhanced embedding service: {e}. Using standard service.")
    from webui.embedding_service import embedding_service, POPULAR_MODELS
    USE_ENHANCED_EMBEDDINGS = False

# Constants
DB_PATH = "/var/embeddings/webui.db"
JOBS_DIR = "/var/embeddings/jobs"
OUTPUT_DIR = "/var/embeddings/output"
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.text']
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.1.173")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Create necessary directories
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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

# Database initialization
def init_db():
    """Initialize SQLite database for job tracking"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if tables exist and need migration
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
    jobs_exists = c.fetchone() is not None
    
    if jobs_exists:
        # Check for missing columns and add them
        c.execute("PRAGMA table_info(jobs)")
        columns = [col[1] for col in c.fetchall()]
        
        if 'vector_dim' not in columns:
            logger.info("Migrating database: adding vector_dim column")
            c.execute("ALTER TABLE jobs ADD COLUMN vector_dim INTEGER")
        
        if 'quantization' not in columns:
            logger.info("Migrating database: adding quantization column")
            c.execute("ALTER TABLE jobs ADD COLUMN quantization TEXT DEFAULT 'float32'")
        
        if 'instruction' not in columns:
            logger.info("Migrating database: adding instruction column")
            c.execute("ALTER TABLE jobs ADD COLUMN instruction TEXT")
    
    # Jobs table
    c.execute('''CREATE TABLE IF NOT EXISTS jobs
                 (id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  description TEXT,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  directory_path TEXT NOT NULL,
                  model_name TEXT NOT NULL,
                  chunk_size INTEGER,
                  chunk_overlap INTEGER,
                  batch_size INTEGER,
                  vector_dim INTEGER,
                  quantization TEXT,
                  instruction TEXT,
                  total_files INTEGER DEFAULT 0,
                  processed_files INTEGER DEFAULT 0,
                  failed_files INTEGER DEFAULT 0,
                  current_file TEXT,
                  error TEXT)''')
    
    # Files table
    c.execute('''CREATE TABLE IF NOT EXISTS files
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL,
                  path TEXT NOT NULL,
                  size INTEGER NOT NULL,
                  modified TEXT NOT NULL,
                  extension TEXT NOT NULL,
                  hash TEXT,
                  status TEXT DEFAULT 'pending',
                  error TEXT,
                  chunks_created INTEGER DEFAULT 0,
                  vectors_created INTEGER DEFAULT 0,
                  FOREIGN KEY (job_id) REFERENCES jobs(id))''')
    
    # Create indices
    c.execute('''CREATE INDEX IF NOT EXISTS idx_files_job_id ON files(job_id)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)''')
    
    conn.commit()
    conn.close()

def reset_database():
    """Reset the database by dropping all tables and recreating them"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Drop tables
    c.execute("DROP TABLE IF EXISTS files")
    c.execute("DROP TABLE IF EXISTS jobs")
    
    conn.commit()
    conn.close()
    
    # Recreate tables
    init_db()
    logger.info("Database reset successfully")

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(
    title="Document Embedding Web UI",
    description="Create and search document embeddings",
    version="1.0.0"
)

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

# Background task executor
executor = ThreadPoolExecutor(max_workers=4)

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

async def process_embedding_job(job_id: str):
    """Process an embedding job asynchronously"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    try:
        # Update job status
        c.execute("UPDATE jobs SET status='processing', updated_at=? WHERE id=?",
                 (datetime.now().isoformat(), job_id))
        conn.commit()
        
        # Get job details
        job = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        
        # Get pending files
        files = c.execute("SELECT * FROM files WHERE job_id=? AND status='pending'",
                         (job_id,)).fetchall()
        
        # Initialize Qdrant client
        qdrant = QdrantClient(QDRANT_HOST, QDRANT_PORT)
        
        for file_idx, file_row in enumerate(files):
            try:
                # Update current file
                c.execute("UPDATE jobs SET current_file=?, updated_at=? WHERE id=?",
                         (file_row['path'], datetime.now().isoformat(), job_id))
                conn.commit()
                
                # Send progress update
                await manager.send_update(job_id, {
                    "type": "progress",
                    "current_file": file_row['path'],
                    "processed": file_idx,
                    "total": len(files)
                })
                
                # Extract text and create chunks
                logger.info(f"Processing file: {file_row['path']}")
                text = extract_text(file_row['path'])
                doc_id = hashlib.md5(file_row['path'].encode()).hexdigest()[:16]
                chunks = chunk_text(text, doc_id)
                
                # Update chunks created
                c.execute("UPDATE files SET chunks_created=? WHERE id=?",
                         (len(chunks), file_row['id']))
                
                # Generate embeddings
                texts = [chunk['text'] for chunk in chunks]
                
                # Use real embedding service with quantization if available
                if USE_ENHANCED_EMBEDDINGS:
                    embeddings_array = embedding_service.generate_embeddings(
                        texts, 
                        job['model_name'],
                        quantization=job['quantization'] or 'float32',
                        batch_size=job['batch_size'],
                        show_progress=False,
                        instruction=job['instruction']
                    )
                else:
                    embeddings_array = embedding_service.generate_embeddings(
                        texts, 
                        job['model_name'],
                        batch_size=job['batch_size'],
                        instruction=job['instruction']
                    )
                
                if embeddings_array is None:
                    raise Exception("Failed to generate embeddings")
                
                embeddings = embeddings_array.tolist()
                
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
                
                # Prepare points for Qdrant
                points = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    point = {
                        "id": str(uuid.uuid4()),
                        "vector": embedding,
                        "payload": {
                            "job_id": job_id,
                            "doc_id": doc_id,
                            "chunk_id": chunk['chunk_id'],
                            "path": file_row['path'],
                            "text": chunk['text'][:200]  # Store preview
                        }
                    }
                    points.append(point)
                
                # Upload to Qdrant
                if points:
                    success = qdrant.upload_points(f"job_{job_id}", points)
                    if success:
                        c.execute("UPDATE files SET status='completed', vectors_created=? WHERE id=?",
                                 (len(points), file_row['id']))
                        c.execute("UPDATE jobs SET processed_files = processed_files + 1 WHERE id=?",
                                 (job_id,))
                    else:
                        raise Exception("Failed to upload to Qdrant")
                
                conn.commit()
                
            except Exception as e:
                logger.error(f"Failed to process file {file_row['path']}: {e}")
                c.execute("UPDATE files SET status='failed', error=? WHERE id=?",
                         (str(e), file_row['id']))
                c.execute("UPDATE jobs SET failed_files = failed_files + 1 WHERE id=?",
                         (job_id,))
                conn.commit()
        
        # Mark job as completed
        c.execute("UPDATE jobs SET status='completed', current_file=NULL, updated_at=? WHERE id=?",
                 (datetime.now().isoformat(), job_id))
        conn.commit()
        
        await manager.send_update(job_id, {
            "type": "completed",
            "message": "Job completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        c.execute("UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?",
                 (str(e), datetime.now().isoformat(), job_id))
        conn.commit()
        
        await manager.send_update(job_id, {
            "type": "error",
            "message": str(e)
        })
    
    finally:
        qdrant.close()
        conn.close()

# API Routes
@app.get("/")
async def root():
    """Serve the main UI"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, 'static', 'index.html'))

@app.get("/settings")
async def settings_page():
    """Serve the settings page"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, 'static', 'settings.html'))

@app.post("/api/settings/reset-database")
async def reset_database_endpoint():
    """Reset the database"""
    try:
        # Get all job IDs before reset
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        job_ids = [row[0] for row in c.execute("SELECT id FROM jobs").fetchall()]
        conn.close()
        
        # Delete Qdrant collections for all jobs
        for job_id in job_ids:
            collection_name = f"job_{job_id}"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.delete(
                        f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}",
                        timeout=30.0
                    )
                    response.raise_for_status()
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
        reset_database()
        
        return {"status": "success", "message": "Database reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings/stats")
async def get_database_stats():
    """Get database statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Get counts
        job_count = c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        file_count = c.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        
        # Get database file size
        db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        
        # Get total parquet files size
        parquet_files = glob.glob(os.path.join(OUTPUT_DIR, "*.parquet"))
        parquet_size = sum(os.path.getsize(f) for f in parquet_files)
        
        return {
            "job_count": job_count,
            "file_count": file_count,
            "database_size_mb": round(db_size / 1024 / 1024, 2),
            "parquet_files_count": len(parquet_files),
            "parquet_size_mb": round(parquet_size / 1024 / 1024, 2)
        }
    finally:
        conn.close()

@app.post("/api/scan-directory")
async def scan_directory_endpoint(request: ScanDirectoryRequest):
    """Scan a directory for supported files"""
    try:
        files = scan_directory(request.path, request.recursive)
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/jobs", response_model=JobStatus)
async def create_job(request: CreateJobRequest, background_tasks: BackgroundTasks):
    """Create a new embedding job"""
    job_id = str(uuid.uuid4())
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Scan directory first
        files = scan_directory(request.directory_path, recursive=True)
        
        if not files:
            raise HTTPException(status_code=400, detail="No supported files found in directory")
        
        # Create job record
        now = datetime.now().isoformat()
        c.execute('''INSERT INTO jobs 
                    (id, name, description, status, created_at, updated_at, 
                     directory_path, model_name, chunk_size, chunk_overlap, 
                     batch_size, vector_dim, quantization, instruction, total_files)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (job_id, request.name, request.description, 'created', now, now,
                  request.directory_path, request.model_name, request.chunk_size,
                  request.chunk_overlap, request.batch_size, request.vector_dim,
                  request.quantization, request.instruction, len(files)))
        
        # Create file records
        for f in files:
            c.execute('''INSERT INTO files 
                        (job_id, path, size, modified, extension)
                        VALUES (?, ?, ?, ?, ?)''',
                     (job_id, f.path, f.size, f.modified, f.extension))
        
        conn.commit()
        
        # Create Qdrant collection for this job
        qdrant = QdrantClient(QDRANT_HOST, QDRANT_PORT)
        
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
        c.execute("UPDATE jobs SET vector_dim=? WHERE id=?", (vector_size, job_id))
        conn.commit()
            
        # Create collection via HTTP API
        collection_config = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            },
            "optimizers_config": {
                "indexing_threshold": 20000,
                "memmap_threshold": 0
            }
        }
        
        try:
            response = httpx.put(
                f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/job_{job_id}",
                json=collection_config,
                timeout=30.0
            )
            response.raise_for_status()
        except:
            logger.warning(f"Collection job_{job_id} might already exist")
        
        qdrant.close()
        
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
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/models")
async def get_models():
    """Get available embedding models"""
    return {
        "models": POPULAR_MODELS,
        "current_device": embedding_service.device,
        "using_real_embeddings": USE_ENHANCED_EMBEDDINGS
    }

@app.get("/api/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all jobs"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    jobs = c.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    
    result = []
    for job in jobs:
        result.append(JobStatus(
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
        ))
    
    conn.close()
    return result

@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    try:
        # Check current job status
        job = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job['status'] not in ['created', 'scanning', 'processing']:
            raise HTTPException(status_code=400, detail=f"Cannot cancel job in status: {job['status']}")
        
        # Update job status to cancelled
        c.execute("UPDATE jobs SET status='cancelled', updated_at=? WHERE id=?",
                 (datetime.now().isoformat(), job_id))
        conn.commit()
        
        # Cancel the task if it's running
        if job_id in active_job_tasks:
            active_job_tasks[job_id].cancel()
            
        return {"message": "Job cancellation requested"}
        
    finally:
        conn.close()

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated collection"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Check if job exists
        job = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Delete from Qdrant
        collection_name = f"job_{job_id}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}",
                    timeout=30.0
                )
                response.raise_for_status()
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant collection: {e}")
        
        # Delete from database
        c.execute("DELETE FROM files WHERE job_id=?", (job_id,))
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        conn.commit()
        
        return {"message": "Job deleted successfully"}
        
    finally:
        conn.close()

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get job details"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    job = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    conn.close()
    
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
async def search(request: SearchRequest):
    """Search for similar documents"""
    try:
        # Determine collection name
        collection_name = f"job_{request.job_id}" if request.job_id else "work_docs"
        
        # Generate query embedding using the appropriate model
        # Get model name from job if specified
        model_name = "BAAI/bge-large-en-v1.5"  # default
        if request.job_id:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            job = c.execute("SELECT model_name FROM jobs WHERE id=?", (request.job_id,)).fetchone()
            if job:
                model_name = job[0]
            conn.close()
        
        # Get quantization settings and instruction from job if available
        quantization = "float32"  # default
        instruction = None  # default
        if request.job_id:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            job = c.execute("SELECT quantization, instruction FROM jobs WHERE id=?", (request.job_id,)).fetchone()
            if job:
                if job[0]:
                    quantization = job[0]
                if job[1]:
                    instruction = job[1]
            conn.close()
        
        # Generate query embedding with same settings as job
        if USE_ENHANCED_EMBEDDINGS:
            query_vector = embedding_service.generate_single_embedding(
                request.query, model_name, quantization=quantization, instruction=instruction
            )
        else:
            query_vector = embedding_service.generate_single_embedding(request.query, model_name, instruction=instruction)
            
        if not query_vector:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Search via HTTP API
        search_request = {
            "vector": query_vector,
            "limit": request.k,
            "with_payload": True
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}/points/search",
                json=search_request,
                timeout=30.0
            )
            response.raise_for_status()
        
        results = response.json()['result']
        
        return {
            "query": request.query,
            "results": results,
            "collection": collection_name
        }
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Collection not found")
        raise HTTPException(status_code=502, detail="Search failed")

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

# Mount static files with proper path resolution
base_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
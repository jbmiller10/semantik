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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vecpipe.extract_chunks import extract_text, chunk_text, process_file
from vecpipe.embed_chunks_simple import generate_mock_embeddings
from vecpipe.ingest_qdrant import QdrantClient
from webui.embedding_service import embedding_service, POPULAR_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "/var/embeddings/webui.db"
JOBS_DIR = "/var/embeddings/jobs"
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.text']
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.1.173")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Create necessary directories
os.makedirs(JOBS_DIR, exist_ok=True)
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
    model_name: str = "BAAI/bge-large-en-v1.5"
    chunk_size: int = 600
    chunk_overlap: int = 200
    batch_size: int = 96

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
                
                # Use real embedding service
                embeddings_array = embedding_service.generate_embeddings(
                    texts, 
                    job['model_name'],
                    batch_size=job['batch_size']
                )
                
                if embeddings_array is None:
                    raise Exception("Failed to generate embeddings")
                
                embeddings = embeddings_array.tolist()
                
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
    return FileResponse('webui/static/index.html')

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
                     batch_size, total_files)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (job_id, request.name, request.description, 'created', now, now,
                  request.directory_path, request.model_name, request.chunk_size,
                  request.chunk_overlap, request.batch_size, len(files)))
        
        # Create file records
        for f in files:
            c.execute('''INSERT INTO files 
                        (job_id, path, size, modified, extension)
                        VALUES (?, ?, ?, ?, ?)''',
                     (job_id, f.path, f.size, f.modified, f.extension))
        
        conn.commit()
        
        # Create Qdrant collection for this job
        qdrant = QdrantClient(QDRANT_HOST, QDRANT_PORT)
        # Create collection via HTTP API
        collection_config = {
            "vectors": {
                "size": 1024,
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
        
        # Start processing in background
        background_tasks.add_task(process_embedding_job, job_id)
        
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
        "current_device": embedding_service.device
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
        
        query_vector = embedding_service.generate_single_embedding(request.query, model_name)
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

# Mount static files
app.mount("/static", StaticFiles(directory="webui/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
#!/usr/bin/env python3
"""
Simplified Web UI for Document Embedding System
Uses file-based storage instead of SQLite
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
import hashlib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vecpipe.extract_chunks import extract_text, chunk_text, process_file
from vecpipe.embed_chunks_simple import generate_mock_embeddings
from vecpipe.ingest_qdrant import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
JOBS_DIR = "/var/embeddings/jobs"
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.text']
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.1.173")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Create necessary directories
os.makedirs(JOBS_DIR, exist_ok=True)

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

# Job management functions
def save_job(job_data: dict):
    """Save job data to file"""
    job_file = os.path.join(JOBS_DIR, f"{job_data['id']}.json")
    with open(job_file, 'w') as f:
        json.dump(job_data, f, indent=2)

def load_job(job_id: str) -> Optional[dict]:
    """Load job data from file"""
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if os.path.exists(job_file):
        with open(job_file, 'r') as f:
            return json.load(f)
    return None

def list_jobs() -> List[dict]:
    """List all jobs"""
    jobs = []
    for job_file in sorted(Path(JOBS_DIR).glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(job_file, 'r') as f:
                jobs.append(json.load(f))
        except:
            pass
    return jobs

def update_job(job_id: str, updates: dict):
    """Update job data"""
    job = load_job(job_id)
    if job:
        job.update(updates)
        job['updated_at'] = datetime.now().isoformat()
        save_job(job)

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
    try:
        # Update job status
        update_job(job_id, {"status": "processing"})
        
        # Get job details
        job = load_job(job_id)
        if not job:
            raise Exception("Job not found")
        
        # Get file list
        files = job.get('files', [])
        
        # Initialize Qdrant client
        qdrant = QdrantClient(QDRANT_HOST, QDRANT_PORT)
        
        processed_files = job.get('processed_files', 0)
        failed_files = job.get('failed_files', 0)
        
        for file_idx, file_info in enumerate(files[processed_files:], start=processed_files):
            try:
                # Update current file
                update_job(job_id, {
                    "current_file": file_info['path'],
                    "processed_files": file_idx
                })
                
                # Send progress update
                await manager.send_update(job_id, {
                    "type": "progress",
                    "current_file": file_info['path'],
                    "processed": file_idx,
                    "total": len(files)
                })
                
                # Extract text and create chunks
                logger.info(f"Processing file: {file_info['path']}")
                text = extract_text(file_info['path'])
                doc_id = hashlib.md5(file_info['path'].encode()).hexdigest()[:16]
                chunks = chunk_text(text, doc_id)
                
                # Generate embeddings
                texts = [chunk['text'] for chunk in chunks]
                embeddings = generate_mock_embeddings(texts)  # Replace with real embeddings
                
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
                            "path": file_info['path'],
                            "text": chunk['text'][:200]  # Store preview
                        }
                    }
                    points.append(point)
                
                # Upload to Qdrant
                if points:
                    success = qdrant.upload_points(f"job_{job_id}", points)
                    if not success:
                        raise Exception("Failed to upload to Qdrant")
                
            except Exception as e:
                logger.error(f"Failed to process file {file_info['path']}: {e}")
                failed_files += 1
                update_job(job_id, {"failed_files": failed_files})
        
        # Mark job as completed
        update_job(job_id, {
            "status": "completed",
            "current_file": None,
            "processed_files": len(files)
        })
        
        await manager.send_update(job_id, {
            "type": "completed",
            "message": "Job completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        update_job(job_id, {
            "status": "failed",
            "error": str(e)
        })
        
        await manager.send_update(job_id, {
            "type": "error",
            "message": str(e)
        })
    
    finally:
        qdrant.close()

# API Routes
@app.get("/")
async def root():
    """Serve the main UI"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, 'static', 'index.html')
    return FileResponse(index_path)

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
    
    try:
        # Scan directory first
        files = scan_directory(request.directory_path, recursive=True)
        
        if not files:
            raise HTTPException(status_code=400, detail="No supported files found in directory")
        
        # Create job record
        now = datetime.now().isoformat()
        job_data = {
            "id": job_id,
            "name": request.name,
            "description": request.description,
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "directory_path": request.directory_path,
            "model_name": request.model_name,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "batch_size": request.batch_size,
            "total_files": len(files),
            "processed_files": 0,
            "failed_files": 0,
            "current_file": None,
            "error": None,
            "files": [f.dict() for f in files]
        }
        
        save_job(job_data)
        
        # Create Qdrant collection for this job
        try:
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
            
            response = httpx.put(
                f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/job_{job_id}",
                json=collection_config,
                timeout=30.0
            )
            response.raise_for_status()
        except:
            logger.warning(f"Collection job_{job_id} might already exist")
        
        # Start processing in background
        background_tasks.add_task(process_embedding_job, job_id)
        
        return JobStatus(**job_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs", response_model=List[JobStatus])
async def get_jobs():
    """List all jobs"""
    jobs = list_jobs()
    return [JobStatus(**job) for job in jobs]

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get job details"""
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated collection"""
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        # Delete Qdrant collection
        collection_name = f"job_{job_id}"
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}",
                timeout=30.0
            )
            # It's OK if collection doesn't exist
            if response.status_code not in [200, 404]:
                response.raise_for_status()
        
        # Delete job file
        job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            os.remove(job_file)
        
        # Delete associated parquet files
        parquet_files = glob.glob(os.path.join(OUTPUT_DIR, f"*{job_id}*.parquet"))
        for pf in parquet_files:
            try:
                os.remove(pf)
            except:
                pass
        
        logger.info(f"Deleted job {job_id} and associated data")
        
        return {"status": "deleted", "job_id": job_id}
        
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search(request: SearchRequest):
    """Search for similar documents"""
    try:
        # Determine collection name
        collection_name = f"job_{request.job_id}" if request.job_id else "work_docs"
        
        # Generate query embedding
        query_vector = generate_mock_embeddings([request.query])[0]
        
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
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
# Project Semantik: Service Decoupling and Architecture Refactoring Plan

 ## Executive Summary

Project Semantik currently suffers from significant architectural debt due to improper package dependencies and service coupling. This comprehensive plan outlines a complete refactoring to establish proper service boundaries, eliminate circular dependencies, and create a maintainable architecture for future growth.

**Duration**: 4 weeks  
**Risk Level**: Medium (mitigated through comprehensive testing)  
**Impact**: Fundamental architecture improvement with no functional changes

## Current State Analysis

### Package Structure and Dependencies

```
packages/
├── vecpipe/                 # Core search engine (low-level)
│   ├── model_manager.py     → imports from webui.embedding_service
│   ├── search_api.py        → imports from webui.embedding_service, webui.api.collection_metadata
│   ├── cleanup.py           → directly accesses webui SQLite database
│   ├── config.py            ← imported by webui components
│   ├── metrics.py           ← imported by webui.embedding_service
│   └── extract_chunks.py    ← imported by webui.api.jobs
│
└── webui/                   # Web UI and API (high-level)
    ├── embedding_service.py → imports from vecpipe.metrics
    ├── database.py          → imports from vecpipe.config
    ├── main.py             → imports from vecpipe.config
    └── api/
        ├── jobs.py         → imports from vecpipe.config, vecpipe.extract_chunks
        ├── search.py       → imports from vecpipe.config
        └── collection_metadata.py ← imported by vecpipe.search_api
```

### Critical Code Examples

#### 1. Embedding Service Circular Dependency
```python
# packages/webui/embedding_service.py:22
from vecpipe.metrics import Counter, registry

# packages/vecpipe/model_manager.py:13
from webui.embedding_service import EmbeddingService
```

#### 2. Direct Database Access Violation
```python
# packages/vecpipe/cleanup.py:60-84
def cleanup_orphaned_collections():
    db_path = os.path.join(PROJECT_ROOT, "data", "webui.db")
    if DOCKER_ENV:
        db_path = "/app/data/webui.db"
    
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT job_id FROM jobs WHERE status = 'completed'"))
        job_ids = [row[0] for row in result]
```

#### 3. Configuration Ownership Confusion
```python
# packages/vecpipe/config.py - Defines settings
class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    # ... more settings

# packages/webui/database.py:19 - Uses vecpipe settings
from vecpipe.config import settings
DATABASE_URL = f"sqlite:///{settings.DATA_DIR}/webui.db"
```

### Docker Service Architecture

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    
  vecpipe:
    build: .
    depends_on: [qdrant]
    volumes:
      - ./data:/app/data  # Shared SQLite access (problematic)
    environment:
      - QDRANT_URL=http://qdrant:6333
      
  webui:
    build: .
    depends_on: [vecpipe, qdrant]
    volumes:
      - ./data:/app/data  # Shared SQLite access
    ports: ["8080:8080"]
    environment:
      - VECPIPE_URL=http://vecpipe:8000
      - QDRANT_URL=http://qdrant:6333
```

## Identified Problems

### 1. Architectural Violations
- **Dependency Inversion**: Low-level vecpipe depends on high-level webui
- **Circular Dependencies**: webui ↔ vecpipe create tight coupling
- **Service Boundary Violations**: Direct database access across packages

### 2. Shared State Issues
- Both services access SQLite database directly via shared volume
- No clear ownership of database schema
- Configuration scattered across packages

### 3. Testing Complexity
- Mock requirements are complex due to cross-package imports
- Integration tests require both packages to be functional
- No clear service contracts to test against

### 4. Deployment Constraints
- Cannot deploy services independently
- Shared volume creates single point of failure
- No ability to scale services separately

## Proposed Architecture

### New Package Structure

```
packages/
├── shared/                      # Common utilities and contracts
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base.py             # Base configuration class
│   │   ├── vecpipe.py          # Vecpipe-specific config
│   │   └── webui.py            # WebUI-specific config
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base class
│   │   ├── dense.py            # Dense embedding implementation
│   │   └── service.py          # Main embedding service
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── prometheus.py       # Metrics implementation
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py           # SQLAlchemy models
│   │   └── repository.py       # Repository pattern
│   ├── text_processing/
│   │   ├── __init__.py
│   │   ├── extraction.py       # Text extraction utilities
│   │   └── chunking.py         # Text chunking logic
│   └── contracts/
│       ├── __init__.py
│       ├── search.py           # Search API contracts
│       └── jobs.py             # Jobs API contracts
│
├── vecpipe/                    # Core search engine
│   ├── __init__.py
│   ├── model_manager.py        # Uses shared.embedding
│   ├── search_api.py           # Implements shared.contracts.search
│   └── maintenance.py          # Replaces cleanup.py
│
└── webui/                      # Web application
    ├── __init__.py
    ├── main.py                 # FastAPI application
    ├── database.py             # Uses shared.database
    └── api/
        ├── jobs.py             # Implements shared.contracts.jobs
        └── search.py           # Proxies to vecpipe
```

### Service Communication

```
┌─────────────┐     HTTP API      ┌─────────────┐
│   WebUI     │ ←───────────────→ │  Vecpipe    │
│  (port 8080)│                   │ (port 8000) │
└──────┬──────┘                   └──────┬──────┘
       │                                 │
       │         ┌─────────────┐         │
       └────────→│   Shared    │←────────┘
                 │  Libraries   │
                 └─────────────┘
```

## Implementation Plan

### Phase 1: Foundation (Week 1)

#### Task 1.1: Create Comprehensive Test Suite
**Duration**: 2 days  
**Dependencies**: None  

Create tests that capture current behavior before any refactoring:

```python
# tests/refactoring/test_current_behavior.py
import pytest
import requests
import time
from pathlib import Path
from sqlalchemy import create_engine, text

class TestCurrentSystemBehavior:
    """Capture exact current behavior for regression testing"""
    
    BASE_URL = "http://localhost:8080"
    
    def test_complete_embedding_pipeline(self, test_documents):
        """Test document ingestion through search"""
        # 1. Create job
        response = requests.post(
            f"{self.BASE_URL}/api/jobs",
            json={"directory_path": str(test_documents)}
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        # 2. Wait for completion
        start_time = time.time()
        while time.time() - start_time < 60:
            response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
            if response.json()["status"] == "completed":
                break
            time.sleep(1)
        
        # 3. Verify embeddings created
        response = requests.post(
            f"{self.BASE_URL}/api/search",
            json={"query": "test content", "top_k": 5}
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) > 0
        
    def test_cleanup_process(self, mock_orphaned_collection):
        """Test cleanup behavior"""
        # Create orphaned collection in Qdrant
        # Run cleanup
        # Verify collection removed
        pass
        
    def test_database_state_consistency(self):
        """Capture database access patterns"""
        engine = create_engine("sqlite:///data/webui.db")
        with engine.connect() as conn:
            # Document all tables and relationships
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            assert set(row[0] for row in tables) == {"jobs", "files", "users", "tokens"}
```

#### Task 1.2: Create Shared Package Structure
**Duration**: 1 day  
**Dependencies**: None  

```bash
#!/bin/bash
# scripts/create_shared_package.sh

# Create directory structure
mkdir -p packages/shared/{config,embedding,metrics,database,text_processing,contracts}

# Create __init__.py files
find packages/shared -type d -exec touch {}/__init__.py \;

# Update pyproject.toml
cat >> pyproject.toml << 'EOF'

[[tool.poetry.packages]]
include = "shared"
from = "packages"
EOF

# Install updated package structure
poetry install
```

#### Task 1.3: Set Up Migration Tooling
**Duration**: 1 day  
**Dependencies**: Task 1.2  

Create automated tools for the migration:

```python
# scripts/refactoring/update_imports.py
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple

class ImportUpdater:
    """Automated import path updater"""
    
    IMPORT_MAPPINGS = {
        # Embedding service moves
        r"from webui\.embedding_service import": "from shared.embedding.service import",
        r"from packages\.webui\.embedding_service import": "from shared.embedding.service import",
        
        # Config moves
        r"from vecpipe\.config import": "from shared.config import",
        r"from packages\.vecpipe\.config import": "from shared.config import",
        
        # Metrics moves
        r"from vecpipe\.metrics import": "from shared.metrics import",
        r"from packages\.vecpipe\.metrics import": "from shared.metrics import",
        
        # Text processing moves
        r"from vecpipe\.extract_chunks import TokenChunker": "from shared.text_processing.chunking import TokenChunker",
        r"from vecpipe\.extract_chunks import extract_text": "from shared.text_processing.extraction import extract_text",
    }
    
    def update_file(self, file_path: Path) -> List[str]:
        """Update imports in a single file"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        changes = []
        for pattern, replacement in self.IMPORT_MAPPINGS.items():
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                changes.append(f"{pattern} -> {replacement}")
        
        if changes:
            with open(file_path, 'w') as f:
                f.write(content)
                
        return changes
    
    def update_directory(self, directory: Path) -> Dict[str, List[str]]:
        """Update all Python files in directory"""
        results = {}
        for py_file in directory.rglob("*.py"):
            changes = self.update_file(py_file)
            if changes:
                results[str(py_file)] = changes
        return results
```

### Phase 2: Core Component Migration (Week 2)

#### Task 2.1: Extract Configuration
**Duration**: 1 day  
**Dependencies**: Task 1.3  

Create unified configuration system:

```python
# packages/shared/config/base.py
from pydantic import BaseSettings
from typing import Optional
import os

class BaseConfig(BaseSettings):
    """Base configuration shared by all services"""
    
    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Paths
    DATA_DIR: str = "/app/data" if os.getenv("DOCKER_ENV") else "./data"
    LOG_DIR: str = "/app/logs" if os.getenv("DOCKER_ENV") else "./logs"
    
    # Database
    DATABASE_URL: str = ""
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "work_docs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"sqlite:///{self.DATA_DIR}/webui.db"

# packages/shared/config/vecpipe.py
from .base import BaseConfig

class VecpipeConfig(BaseConfig):
    """Vecpipe-specific configuration"""
    
    # Embedding settings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_BATCH_SIZE: int = 32
    MAX_EMBEDDING_TOKENS: int = 512
    
    # Search settings
    SEARCH_TOP_K: int = 10
    SEARCH_SCORE_THRESHOLD: float = 0.0
    
    # Performance
    MODEL_CACHE_DIR: str = "/app/.cache/huggingface"
    USE_GPU: bool = True
    GPU_MEMORY_FRACTION: float = 0.9

# packages/shared/config/webui.py
from .base import BaseConfig

class WebuiConfig(BaseConfig):
    """WebUI-specific configuration"""
    
    # API settings
    API_PREFIX: str = "/api"
    CORS_ORIGINS: list = ["http://localhost:3000"]
    
    # Security
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    
    # External services
    VECPIPE_URL: str = "http://localhost:8000"
    
    # UI settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list = [".txt", ".md", ".pdf", ".docx"]
```

Move config and update imports:
```bash
# Move config file
mv packages/vecpipe/config.py packages/shared/config/legacy.py

# Update imports
python scripts/refactoring/update_imports.py --directory packages/
```

#### Task 2.2: Migrate Embedding Service
**Duration**: 2 days  
**Dependencies**: Task 2.1  

Create embedding abstraction and migrate service:

```python
# packages/shared/embedding/base.py
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
import numpy as np

class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services"""
    
    @abstractmethod
    async def initialize(self, model_name: str, **kwargs) -> None:
        """Initialize the embedding model"""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    async def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""
        pass

# packages/shared/embedding/dense.py
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from .base import BaseEmbeddingService
from ..metrics import embedding_counter, embedding_histogram

logger = logging.getLogger(__name__)

class DenseEmbeddingService(BaseEmbeddingService):
    """Dense embedding service using sentence-transformers"""
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.model_name: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self, model_name: str, **kwargs) -> None:
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"Loading model {model_name} on {self.device}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_name = model_name
            
            # Apply optimizations
            if kwargs.get("use_fp16", False) and self.device == "cuda":
                self.model.half()
                
            logger.info(f"Model loaded: dim={self.get_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if not self.model:
            raise RuntimeError("Model not initialized")
            
        embedding_counter.labels(model=self.model_name).inc(len(texts))
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with embedding_histogram.labels(model=self.model_name).time():
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)
    
    async def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embeddings = await self.embed_texts([text], batch_size=1)
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if not self.model:
            raise RuntimeError("Model not initialized")
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        if not self.model:
            return {"status": "not_initialized"}
            
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.get_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "dtype": str(next(self.model.parameters()).dtype),
        }
    
    async def cleanup(self) -> None:
        """Clean up model resources"""
        if self.model:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model resources cleaned up")
```

Move and update the service:
```bash
# Move embedding service
mv packages/webui/embedding_service.py packages/shared/embedding/legacy_service.py

# Create new service.py that uses the refactored code
# Update all imports
python scripts/refactoring/update_imports.py --directory packages/
```

#### Task 2.3: Extract Shared Components
**Duration**: 2 days  
**Dependencies**: Task 2.2  

Move metrics, text processing, and other shared utilities:

```python
# packages/shared/metrics/prometheus.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Create a shared registry
registry = CollectorRegistry()

# Define metrics
embedding_counter = Counter(
    'embeddings_generated_total',
    'Total number of embeddings generated',
    ['model'],
    registry=registry
)

embedding_histogram = Histogram(
    'embedding_generation_seconds',
    'Time spent generating embeddings',
    ['model'],
    registry=registry
)

search_counter = Counter(
    'searches_total',
    'Total number of searches performed',
    ['collection'],
    registry=registry
)

search_histogram = Histogram(
    'search_duration_seconds',
    'Search query duration',
    ['collection'],
    registry=registry
)

active_jobs_gauge = Gauge(
    'active_embedding_jobs',
    'Number of active embedding jobs',
    registry=registry
)

# packages/shared/text_processing/extraction.py
import magic
import pypdf
import docx
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def extract_text(file_path: str, file_type: Optional[str] = None) -> str:
    """Extract text from various file formats"""
    if file_type is None:
        file_type = magic.from_file(file_path, mime=True)
    
    try:
        if file_type == 'application/pdf':
            return extract_pdf_text(file_path)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return extract_docx_text(file_path)
        elif file_type.startswith('text/'):
            return extract_plain_text(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return ""
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {e}")
        raise

def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF"""
    text_parts = []
    with open(file_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        for page in pdf_reader.pages:
            text_parts.append(page.extract_text())
    return '\n'.join(text_parts)

def extract_docx_text(file_path: str) -> str:
    """Extract text from DOCX"""
    doc = docx.Document(file_path)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def extract_plain_text(file_path: str) -> str:
    """Extract text from plain text file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

# packages/shared/text_processing/chunking.py
from typing import List, Optional
import tiktoken
import logging

logger = logging.getLogger(__name__)

class TokenChunker:
    """Split text into token-based chunks"""
    
    def __init__(self, 
                 encoding_name: str = "cl100k_base",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position
            start += self.chunk_size - self.chunk_overlap
            
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
```

### Phase 3: Service Boundary Implementation (Week 3)

#### Task 3.1: Create Database Abstraction
**Duration**: 2 days  
**Dependencies**: Task 2.3  

Implement repository pattern for database access:

```python
# packages/shared/database/models.py
from sqlalchemy import Column, String, DateTime, Text, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    
    job_id = Column(String, primary_key=True)
    directory_path = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    user_id = Column(String, nullable=True)
    total_files = Column(Float, default=0)
    processed_files = Column(Float, default=0)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    files = relationship("File", back_populates="job", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "job_id": self.job_id,
            "directory_path": self.directory_path,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "progress": self.processed_files / self.total_files if self.total_files > 0 else 0,
            "metadata": self.metadata or {}
        }

class File(Base):
    __tablename__ = "files"
    
    file_id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.job_id"))
    file_path = Column(String, nullable=False)
    file_hash = Column(String, nullable=True)
    file_size = Column(Float, nullable=True)
    status = Column(String, default="pending")
    error_message = Column(Text, nullable=True)
    chunks_created = Column(Float, default=0)
    
    # Relationships
    job = relationship("Job", back_populates="files")

# packages/shared/database/repository.py
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from .models import Job, File
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class JobRepository:
    """Repository for job-related database operations"""
    
    def __init__(self, session: Session):
        self.session = session
        
    def create(self, directory_path: str, user_id: Optional[str] = None, 
              metadata: Optional[Dict[str, Any]] = None) -> Job:
        """Create a new job"""
        job = Job(
            job_id=str(uuid.uuid4()),
            directory_path=directory_path,
            user_id=user_id,
            status="pending",
            metadata=metadata or {}
        )
        self.session.add(job)
        self.session.commit()
        logger.info(f"Created job {job.job_id} for {directory_path}")
        return job
        
    def get_by_id(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.session.query(Job).filter_by(job_id=job_id).first()
        
    def list_all(self, 
                 status: Optional[str] = None,
                 user_id: Optional[str] = None,
                 limit: int = 100,
                 offset: int = 0) -> List[Job]:
        """List jobs with optional filtering"""
        query = self.session.query(Job)
        
        if status:
            query = query.filter_by(status=status)
        if user_id:
            query = query.filter_by(user_id=user_id)
            
        return query.order_by(Job.created_at.desc()).limit(limit).offset(offset).all()
        
    def update_status(self, job_id: str, status: str, 
                     error_message: Optional[str] = None) -> bool:
        """Update job status"""
        job = self.get_by_id(job_id)
        if not job:
            return False
            
        job.status = status
        job.error_message = error_message
        
        if status == "completed":
            job.completed_at = datetime.utcnow()
        elif status == "failed" and not job.completed_at:
            job.completed_at = datetime.utcnow()
            
        self.session.commit()
        logger.info(f"Updated job {job_id} status to {status}")
        return True
        
    def update_progress(self, job_id: str, processed_files: int, total_files: int) -> bool:
        """Update job progress"""
        job = self.get_by_id(job_id)
        if not job:
            return False
            
        job.processed_files = processed_files
        job.total_files = total_files
        self.session.commit()
        return True
        
    def delete(self, job_id: str) -> bool:
        """Delete job and all associated data"""
        job = self.get_by_id(job_id)
        if not job:
            return False
            
        self.session.delete(job)
        self.session.commit()
        logger.info(f"Deleted job {job_id}")
        return True
        
    def get_all_job_ids(self) -> List[str]:
        """Get all job IDs (for cleanup operations)"""
        return [job.job_id for job in self.session.query(Job.job_id).all()]

class FileRepository:
    """Repository for file-related database operations"""
    
    def __init__(self, session: Session):
        self.session = session
        
    def create_batch(self, job_id: str, file_paths: List[str]) -> List[File]:
        """Create multiple file records"""
        files = []
        for path in file_paths:
            file_record = File(
                file_id=str(uuid.uuid4()),
                job_id=job_id,
                file_path=path,
                status="pending"
            )
            files.append(file_record)
            self.session.add(file_record)
            
        self.session.commit()
        logger.info(f"Created {len(files)} file records for job {job_id}")
        return files
        
    def update_status(self, file_id: str, status: str, 
                     error_message: Optional[str] = None,
                     chunks_created: Optional[int] = None) -> bool:
        """Update file processing status"""
        file_record = self.session.query(File).filter_by(file_id=file_id).first()
        if not file_record:
            return False
            
        file_record.status = status
        file_record.error_message = error_message
        if chunks_created is not None:
            file_record.chunks_created = chunks_created
            
        self.session.commit()
        return True
```

#### Task 3.2: Implement Service Contracts
**Duration**: 2 days  
**Dependencies**: Task 3.1  

Define API contracts for service communication:

```python
# packages/shared/contracts/search.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class SearchRequest(BaseModel):
    """Search API request model"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(10, ge=1, le=100)
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    collection: Optional[str] = "work_docs"
    filters: Optional[Dict[str, Any]] = None
    
    @validator('query')
    def clean_query(cls, v):
        return v.strip()

class SearchResult(BaseModel):
    """Individual search result"""
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    highlights: Optional[List[str]] = None

class SearchResponse(BaseModel):
    """Search API response model"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    collection: str
    
    class Config:
        schema_extra = {
            "example": {
                "query": "quantum computing",
                "results": [{
                    "doc_id": "doc_123",
                    "score": 0.95,
                    "text": "Quantum computing leverages quantum mechanics...",
                    "metadata": {
                        "source": "quantum_intro.pdf",
                        "page": 1,
                        "job_id": "job_456"
                    }
                }],
                "total_results": 42,
                "search_time_ms": 23.5,
                "collection": "work_docs"
            }
        }

# packages/shared/contracts/jobs.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CreateJobRequest(BaseModel):
    """Create job request model"""
    directory_path: str = Field(..., min_length=1)
    scan_subdirs: bool = Field(True)
    file_extensions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('directory_path')
    def validate_path(cls, v):
        # Add path validation logic
        return v.strip()

class JobResponse(BaseModel):
    """Job information response"""
    job_id: str
    directory_path: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    progress: float = Field(0.0, ge=0.0, le=1.0)
    total_files: int = 0
    processed_files: int = 0
    metadata: Dict[str, Any] = {}

class JobListResponse(BaseModel):
    """List of jobs response"""
    jobs: List[JobResponse]
    total: int
    page: int = 1
    page_size: int = 100

# packages/shared/contracts/errors.py
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "details": {
                    "field": "query",
                    "error": "Field required"
                }
            }
        }
```

#### Task 3.3: Replace Direct Database Access
**Duration**: 1 day  
**Dependencies**: Task 3.2  

Update cleanup.py to use repository pattern:

```python
# packages/vecpipe/maintenance.py (replaces cleanup.py)
from typing import List, Set
import asyncio
import logging
from qdrant_client import QdrantClient
from shared.database.repository import JobRepository
from shared.config import VecpipeConfig
from shared.contracts.errors import ErrorResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

class MaintenanceService:
    """Service for maintenance operations"""
    
    def __init__(self, config: VecpipeConfig):
        self.config = config
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)
        
        # Create database session
        engine = create_engine(config.DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        self.db_session = SessionLocal()
        self.job_repo = JobRepository(self.db_session)
        
    async def cleanup_orphaned_collections(self) -> Dict[str, Any]:
        """Remove Qdrant collections without corresponding jobs"""
        try:
            # Get all collections from Qdrant
            collections_response = self.qdrant_client.get_collections()
            all_collections = {c.name for c in collections_response.collections}
            
            # Get all job IDs from database
            job_ids = set(self.job_repo.get_all_job_ids())
            
            # Add job-specific collection names
            job_collections = {f"job_{job_id}" for job_id in job_ids}
            job_collections.add(self.config.QDRANT_COLLECTION)  # Default collection
            
            # Find orphaned collections
            orphaned = all_collections - job_collections
            
            # Delete orphaned collections
            deleted = []
            errors = []
            
            for collection_name in orphaned:
                try:
                    self.qdrant_client.delete_collection(collection_name)
                    deleted.append(collection_name)
                    logger.info(f"Deleted orphaned collection: {collection_name}")
                except Exception as e:
                    errors.append({
                        "collection": collection_name,
                        "error": str(e)
                    })
                    logger.error(f"Failed to delete collection {collection_name}: {e}")
                    
            return {
                "orphaned_count": len(orphaned),
                "deleted_count": len(deleted),
                "deleted_collections": deleted,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise
        finally:
            self.db_session.close()
            
    async def optimize_collections(self) -> Dict[str, Any]:
        """Optimize Qdrant collections"""
        results = {}
        
        try:
            collections_response = self.qdrant_client.get_collections()
            
            for collection in collections_response.collections:
                try:
                    # Trigger optimization
                    self.qdrant_client.update_collection(
                        collection_name=collection.name,
                        optimizer_config={
                            "deleted_threshold": 0.2,
                            "vacuum_min_vector_number": 1000,
                            "default_segment_number": 2
                        }
                    )
                    results[collection.name] = "optimized"
                except Exception as e:
                    results[collection.name] = f"error: {str(e)}"
                    
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
```

### Phase 4: Docker and Testing Updates (Week 4)

#### Task 4.1: Update Docker Configuration
**Duration**: 1 day  
**Dependencies**: Task 3.3  

Update Docker configuration for new architecture:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copy packages
COPY packages/ /app/packages/

# Set Python path
ENV PYTHONPATH=/app/packages:$PYTHONPATH
ENV DOCKER_ENV=1

# Create directories
RUN mkdir -p /app/data /app/logs /app/.cache

# Entry point varies by service
# For vecpipe:
# CMD ["uvicorn", "vecpipe.search_api:app", "--host", "0.0.0.0", "--port", "8000"]
# For webui:
# CMD ["uvicorn", "webui.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Update docker-compose.yml:
```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__LOG_LEVEL=INFO

  vecpipe:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["uvicorn", "vecpipe.search_api:app", "--host", "0.0.0.0", "--port", "8000"]
    depends_on:
      - qdrant
    environment:
      - QDRANT_URL=http://qdrant:6333
      - DATABASE_URL=postgresql://user:pass@db:5432/semantik  # Future: PostgreSQL
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data:ro  # Read-only access for now
      - model_cache:/app/.cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  webui:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["uvicorn", "webui.main:app", "--host", "0.0.0.0", "--port", "8080"]
    depends_on:
      - vecpipe
      - qdrant
    ports:
      - "8080:8080"
    environment:
      - VECPIPE_URL=http://vecpipe:8000
      - QDRANT_URL=http://qdrant:6333
      - DATABASE_URL=sqlite:///app/data/webui.db  # Will migrate to PostgreSQL
      - SECRET_KEY=change-me-in-production
    volumes:
      - ./data:/app/data
      - model_cache:/app/.cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_data:
  model_cache:
```

#### Task 4.2: Update CI/CD Pipeline
**Duration**: 1 day  
**Dependencies**: Task 4.1  

Update GitHub Actions and testing configuration:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHONPATH: /home/runner/work/semantik/semantik/packages

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
          
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH
        
    - name: Install dependencies
      run: |
        poetry install
        
    - name: Run linting
      run: |
        poetry run ruff check packages/
        poetry run mypy packages/
        
    - name: Run tests
      run: |
        poetry run pytest tests/ \
          --cov=packages/shared \
          --cov=packages/vecpipe \
          --cov=packages/webui \
          --cov-report=xml \
          --cov-report=term-missing
          
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        
  integration-test:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker images
      run: |
        docker-compose build
        
    - name: Run integration tests
      run: |
        docker-compose up -d
        sleep 30  # Wait for services to start
        docker-compose exec -T webui pytest tests/integration/
        
    - name: Show logs on failure
      if: failure()
      run: |
        docker-compose logs
```

#### Task 4.3: Comprehensive Testing
**Duration**: 2 days  
**Dependencies**: Task 4.2  

Run full test suite and validate refactoring:

```python
# tests/refactoring/test_architecture_validation.py
import pytest
import ast
from pathlib import Path
from typing import Set, Dict, List

class TestArchitectureValidation:
    """Validate the refactored architecture"""
    
    def test_no_circular_dependencies(self):
        """Ensure no circular dependencies between packages"""
        dependencies = self.analyze_package_dependencies()
        
        # Check vecpipe doesn't import from webui
        assert not any(
            imp.startswith('webui') or imp.startswith('packages.webui')
            for imp in dependencies.get('vecpipe', [])
        ), "vecpipe should not import from webui"
        
        # Check webui doesn't import from vecpipe (except shared)
        webui_imports = dependencies.get('webui', [])
        vecpipe_imports = [imp for imp in webui_imports 
                          if imp.startswith(('vecpipe', 'packages.vecpipe'))]
        assert not vecpipe_imports, f"webui should not import from vecpipe: {vecpipe_imports}"
        
    def test_shared_package_independence(self):
        """Ensure shared package doesn't depend on app packages"""
        dependencies = self.analyze_package_dependencies()
        shared_imports = dependencies.get('shared', [])
        
        forbidden_imports = [
            imp for imp in shared_imports
            if imp.startswith(('webui', 'vecpipe', 'packages.webui', 'packages.vecpipe'))
        ]
        
        assert not forbidden_imports, f"shared package has forbidden imports: {forbidden_imports}"
        
    def test_database_access_patterns(self):
        """Ensure database is only accessed through repository"""
        # Find all files that import sqlalchemy or sqlite3
        db_imports = []
        
        for py_file in Path('packages').rglob('*.py'):
            with open(py_file) as f:
                content = f.read()
                
            if 'sqlalchemy' in content or 'sqlite3' in content:
                # Exclude shared/database package
                if not str(py_file).startswith('packages/shared/database'):
                    db_imports.append(str(py_file))
                    
        assert not db_imports, f"Direct database access found in: {db_imports}"
        
    def analyze_package_dependencies(self) -> Dict[str, List[str]]:
        """Analyze import dependencies for each package"""
        dependencies = {'shared': [], 'vecpipe': [], 'webui': []}
        
        for package in dependencies.keys():
            package_path = Path(f'packages/{package}')
            imports = set()
            
            for py_file in package_path.rglob('*.py'):
                with open(py_file) as f:
                    tree = ast.parse(f.read(), str(py_file))
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.add(node.module)
                        
            dependencies[package] = sorted(list(imports))
            
        return dependencies

# tests/refactoring/test_api_contracts.py
from shared.contracts.search import SearchRequest, SearchResponse
from shared.contracts.jobs import CreateJobRequest, JobResponse
import pytest
from pydantic import ValidationError

class TestAPIContracts:
    """Test API contract validation"""
    
    def test_search_request_validation(self):
        """Test search request validation"""
        # Valid request
        request = SearchRequest(query="test query", top_k=5)
        assert request.query == "test query"
        assert request.top_k == 5
        
        # Invalid requests
        with pytest.raises(ValidationError):
            SearchRequest(query="")  # Empty query
            
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=200)  # top_k too large
            
    def test_job_request_validation(self):
        """Test job creation request validation"""
        # Valid request
        request = CreateJobRequest(directory_path="/data/docs")
        assert request.scan_subdirs is True  # Default value
        
        # Invalid request
        with pytest.raises(ValidationError):
            CreateJobRequest(directory_path="")  # Empty path
```

#### Task 4.4: Performance Benchmarking
**Duration**: 1 day  
**Dependencies**: Task 4.3  

Ensure no performance regression:

```python
# tests/performance/benchmark_refactoring.py
import time
import asyncio
import statistics
from typing import List
import numpy as np

class BenchmarkSuite:
    """Benchmark critical operations before/after refactoring"""
    
    async def benchmark_embedding_generation(self):
        """Benchmark embedding generation performance"""
        from shared.embedding.service import EmbeddingService
        from shared.config import VecpipeConfig
        
        service = EmbeddingService()
        config = VecpipeConfig()
        await service.initialize(config.EMBEDDING_MODEL)
        
        # Test data
        texts = [f"This is test document number {i} with some content." 
                for i in range(1000)]
        
        # Warm up
        await service.embed_texts(texts[:10])
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            embeddings = await service.embed_texts(texts)
            end = time.time()
            times.append(end - start)
            
        await service.cleanup()
        
        return {
            "mean_time": statistics.mean(times),
            "std_dev": statistics.stdev(times),
            "texts_per_second": len(texts) / statistics.mean(times)
        }
        
    async def benchmark_search_latency(self):
        """Benchmark search operation latency"""
        import aiohttp
        
        queries = [
            "quantum computing",
            "machine learning",
            "distributed systems",
            "database optimization",
            "cloud architecture"
        ]
        
        async with aiohttp.ClientSession() as session:
            latencies = []
            
            for query in queries * 10:  # Run each query 10 times
                start = time.time()
                async with session.post(
                    "http://localhost:8000/search",
                    json={"query": query, "top_k": 10}
                ) as response:
                    await response.json()
                end = time.time()
                latencies.append((end - start) * 1000)  # ms
                
        return {
            "mean_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99)
        }
```

### Phase 5: Finalization (Week 4 continued)

#### Task 5.1: Documentation Update
**Duration**: 1 day  
**Dependencies**: Task 4.4  

Update all documentation:

```markdown
# Architecture Documentation Updates

## New Package Structure

### Shared Package
The `shared` package contains all common utilities and contracts:
- **config/**: Centralized configuration management
- **embedding/**: Embedding service with abstraction for future extensions
- **metrics/**: Prometheus metrics collection
- **database/**: SQLAlchemy models and repository pattern
- **text_processing/**: Text extraction and chunking utilities
- **contracts/**: API contract definitions using Pydantic

### Service Communication
- WebUI → Vecpipe: HTTP REST API calls
- Both services → Shared: Direct imports
- Database access: Only through repository pattern

### Configuration
Each service has its own configuration class:
- `VecpipeConfig`: Search engine specific settings
- `WebuiConfig`: Web application specific settings
- Both inherit from `BaseConfig` for common settings
```

#### Task 5.2: Final Cleanup
**Duration**: 0.5 days  
**Dependencies**: Task 5.1  

Remove old code and update imports:

```bash
#!/bin/bash
# scripts/final_cleanup.sh

# Remove old files that were moved
rm -f packages/webui/embedding_service.py
rm -f packages/vecpipe/config.py
rm -f packages/vecpipe/metrics.py
rm -f packages/vecpipe/cleanup.py

# Update any remaining imports
find packages -name "*.py" -exec grep -l "from vecpipe.config" {} \; | \
  xargs sed -i 's/from vecpipe.config/from shared.config/g'

# Format all code
poetry run black packages/
poetry run isort packages/

# Run final validation
poetry run mypy packages/
poetry run pytest tests/
```

## Success Criteria

1. **No Circular Dependencies**: Validated by architecture tests
2. **All Tests Pass**: Unit, integration, and E2E tests
3. **Performance Maintained**: <5% regression in benchmarks
4. **Clean Service Boundaries**: Vecpipe and webui only communicate via APIs
5. **Extensibility Proven**: Can add new embedding service implementation

## Post-Refactoring Opportunities

With the clean architecture in place:

1. **Add PostgreSQL**: Replace SQLite with PostgreSQL for production
2. **Implement Sparse Embeddings**: Easy with `BaseEmbeddingService`
3. **Add Caching Layer**: Redis between services
4. **Horizontal Scaling**: Deploy multiple vecpipe instances
5. **API Versioning**: Add versioned endpoints with contract evolution

This refactoring establishes a solid foundation for Project Semantik's future growth while maintaining all current functionality.

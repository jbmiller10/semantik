# Model Manager Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a Model Manager feature in Settings that allows users to browse, download, and delete ML models (embedding, LLM, reranker, SPLADE).

**Architecture:** New Models tab with horizontal sub-tabs per model type. Backend uses HuggingFace Hub library for cache management. Celery tasks for background downloads with progress via Redis. Custom models stored in database, curated models in YAML registry.

**Tech Stack:** FastAPI, SQLAlchemy, Celery, huggingface_hub, React, TanStack Query, Zustand

---

## Phase 1: Backend Foundation

### Task 1: Create Unified Model Registry YAML

**Files:**
- Create: `packages/shared/models/model_registry.yaml`
- Create: `packages/shared/models/__init__.py`
- Create: `packages/shared/models/registry.py`

**Step 1: Create the model registry YAML**

Create `packages/shared/models/model_registry.yaml`:

```yaml
# Unified Model Registry
# All model types in one place for the Model Manager UI

schema_version: "1.0"

embedding:
  - id: "Qwen/Qwen3-Embedding-0.6B"
    name: "Qwen3 Embedding 0.6B"
    description: "Small model, instruction-aware (1024d)"
    dimension: 1024
    max_sequence_length: 32768
    pooling_method: "last_token"
    is_asymmetric: true
    default_query_instruction: "Given a web search query, retrieve relevant passages that answer the query"
    download_size_mb: 1200
    memory_mb:
      float32: 2400
      float16: 1200
      int8: 600

  - id: "Qwen/Qwen3-Embedding-4B"
    name: "Qwen3 Embedding 4B"
    description: "Medium model, MTEB top performer (2560d)"
    dimension: 2560
    max_sequence_length: 32768
    pooling_method: "last_token"
    is_asymmetric: true
    default_query_instruction: "Given a web search query, retrieve relevant passages that answer the query"
    download_size_mb: 8000
    memory_mb:
      float32: 16000
      float16: 8000
      int8: 4000

  - id: "Qwen/Qwen3-Embedding-8B"
    name: "Qwen3 Embedding 8B"
    description: "Large model, MTEB #1 (4096d)"
    dimension: 4096
    max_sequence_length: 32768
    pooling_method: "last_token"
    is_asymmetric: true
    default_query_instruction: "Given a web search query, retrieve relevant passages that answer the query"
    download_size_mb: 16000
    memory_mb:
      float32: 32000
      float16: 16000
      int8: 8000

  - id: "sentence-transformers/all-MiniLM-L6-v2"
    name: "all-MiniLM-L6-v2"
    description: "Fast, lightweight model for general use (384d)"
    dimension: 384
    max_sequence_length: 256
    pooling_method: "mean"
    is_asymmetric: false
    download_size_mb: 90
    memory_mb:
      float32: 90
      float16: 45
      int8: 23

  - id: "BAAI/bge-large-en-v1.5"
    name: "BGE Large English"
    description: "State-of-the-art English embeddings (1024d)"
    dimension: 1024
    max_sequence_length: 512
    pooling_method: "mean"
    is_asymmetric: true
    query_prefix: "Represent this sentence for searching relevant passages: "
    download_size_mb: 1300
    memory_mb:
      float32: 1300
      float16: 650
      int8: 325

llm:
  - id: "Qwen/Qwen2.5-0.5B-Instruct"
    name: "Qwen 2.5 0.5B"
    description: "Tiny model for simple tasks"
    context_window: 32768
    download_size_mb: 1000
    memory_mb:
      float16: 1300
      int8: 800
      int4: 500

  - id: "Qwen/Qwen2.5-1.5B-Instruct"
    name: "Qwen 2.5 1.5B"
    description: "Small model, good for HyDE"
    context_window: 32768
    download_size_mb: 3000
    memory_mb:
      float16: 3500
      int8: 2000
      int4: 1200

  - id: "Qwen/Qwen2.5-3B-Instruct"
    name: "Qwen 2.5 3B"
    description: "Medium model, good balance"
    context_window: 32768
    download_size_mb: 6000
    memory_mb:
      float16: 7000
      int8: 4000
      int4: 2500

  - id: "Qwen/Qwen2.5-7B-Instruct"
    name: "Qwen 2.5 7B"
    description: "Large model, best quality"
    context_window: 32768
    download_size_mb: 14000
    memory_mb:
      float16: 15000
      int8: 8500
      int4: 5000

reranker:
  - id: "Qwen/Qwen3-Reranker-0.6B"
    name: "Qwen3 Reranker 0.6B"
    description: "Fast reranker for Qwen3 0.6B embeddings"
    download_size_mb: 1200
    memory_mb:
      float16: 1200
      int8: 600

  - id: "Qwen/Qwen3-Reranker-4B"
    name: "Qwen3 Reranker 4B"
    description: "Balanced reranker for Qwen3 4B embeddings"
    download_size_mb: 8000
    memory_mb:
      float16: 8000
      int8: 4000

  - id: "Qwen/Qwen3-Reranker-8B"
    name: "Qwen3 Reranker 8B"
    description: "High-quality reranker for Qwen3 8B embeddings"
    download_size_mb: 16000
    memory_mb:
      float16: 16000
      int8: 8000

splade:
  - id: "naver/splade-cocondenser-ensembledistil"
    name: "SPLADE CoCondenser"
    description: "Standard SPLADE model for hybrid search"
    download_size_mb: 400
    memory_mb:
      float32: 500
      float16: 300
```

**Step 2: Create the registry loader module**

Create `packages/shared/models/__init__.py`:

```python
"""Unified model registry for all model types."""

from shared.models.registry import (
    ModelInfo,
    ModelType,
    get_curated_models,
    get_model_info,
    list_models_by_type,
)

__all__ = [
    "ModelInfo",
    "ModelType",
    "get_curated_models",
    "get_model_info",
    "list_models_by_type",
]
```

Create `packages/shared/models/registry.py`:

```python
"""Model registry loader with caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent / "model_registry.yaml"


class ModelType(str, Enum):
    """Supported model types."""

    EMBEDDING = "embedding"
    LLM = "llm"
    RERANKER = "reranker"
    SPLADE = "splade"


@dataclass
class ModelInfo:
    """Model metadata from registry."""

    id: str
    name: str
    description: str
    model_type: ModelType
    download_size_mb: int
    memory_mb: dict[str, int] = field(default_factory=dict)

    # Embedding-specific
    dimension: int | None = None
    max_sequence_length: int | None = None
    pooling_method: str | None = None
    is_asymmetric: bool = False
    query_prefix: str | None = None
    document_prefix: str | None = None
    default_query_instruction: str | None = None

    # LLM-specific
    context_window: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type.value,
            "download_size_mb": self.download_size_mb,
            "memory_mb": self.memory_mb,
            "dimension": self.dimension,
            "max_sequence_length": self.max_sequence_length,
            "pooling_method": self.pooling_method,
            "is_asymmetric": self.is_asymmetric,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "default_query_instruction": self.default_query_instruction,
            "context_window": self.context_window,
        }


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, list[dict[str, Any]]]:
    """Load and cache the model registry YAML."""
    if not REGISTRY_PATH.exists():
        logger.warning("Model registry not found at %s", REGISTRY_PATH)
        return {}

    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f)

    # Remove schema_version from the dict
    data.pop("schema_version", None)
    return data


def _parse_model(model_type: ModelType, data: dict[str, Any]) -> ModelInfo:
    """Parse a model entry from the registry."""
    return ModelInfo(
        id=data["id"],
        name=data["name"],
        description=data["description"],
        model_type=model_type,
        download_size_mb=data.get("download_size_mb", 0),
        memory_mb=data.get("memory_mb", {}),
        dimension=data.get("dimension"),
        max_sequence_length=data.get("max_sequence_length"),
        pooling_method=data.get("pooling_method"),
        is_asymmetric=data.get("is_asymmetric", False),
        query_prefix=data.get("query_prefix"),
        document_prefix=data.get("document_prefix"),
        default_query_instruction=data.get("default_query_instruction"),
        context_window=data.get("context_window"),
    )


def get_curated_models() -> list[ModelInfo]:
    """Get all curated models from the registry."""
    registry = _load_registry()
    models: list[ModelInfo] = []

    for type_key, model_list in registry.items():
        try:
            model_type = ModelType(type_key)
        except ValueError:
            logger.warning("Unknown model type in registry: %s", type_key)
            continue

        for model_data in model_list:
            try:
                models.append(_parse_model(model_type, model_data))
            except (KeyError, TypeError) as e:
                logger.warning("Invalid model entry: %s - %s", model_data.get("id", "unknown"), e)

    return models


def list_models_by_type(model_type: ModelType) -> list[ModelInfo]:
    """Get all curated models of a specific type."""
    return [m for m in get_curated_models() if m.model_type == model_type]


def get_model_info(model_id: str) -> ModelInfo | None:
    """Get info for a specific model by ID."""
    for model in get_curated_models():
        if model.id == model_id:
            return model
    return None
```

**Step 3: Run tests to verify the registry loads**

Run: `uv run pytest -xvs -k "test_model_registry" 2>/dev/null || echo "No tests yet - will add in next task"`

**Step 4: Commit**

```bash
git add packages/shared/models/
git commit -m "feat(models): add unified model registry with YAML config"
```

---

### Task 2: Database Model for Custom Models

**Files:**
- Modify: `packages/shared/database/models.py`
- Create: `alembic/versions/YYYYMMDD_add_custom_models_table.py`

**Step 1: Add CustomModel to database models**

Add to `packages/shared/database/models.py` after the existing model classes (find a good location near PluginConfig):

```python
class CustomModel(Base):
    """User-defined custom models for the model manager."""

    __tablename__ = "custom_models"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String(255), nullable=False)  # HuggingFace model ID
    model_type = Column(String(50), nullable=False)  # embedding, llm, reranker, splade
    config = Column(JSON, nullable=False, default=dict)  # User overrides
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Composite unique constraint
    __table_args__ = (
        UniqueConstraint("user_id", "model_id", name="uq_custom_models_user_model"),
        Index("ix_custom_models_user_id", "user_id"),
        Index("ix_custom_models_model_type", "model_type"),
    )
```

**Step 2: Create Alembic migration**

Run: `uv run alembic revision -m "add_custom_models_table"`

Then edit the generated file with:

```python
"""add_custom_models_table

Revision ID: <generated>
Revises: <previous>
Create Date: 2026-01-19
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "<generated>"
down_revision = "<previous>"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "custom_models",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("user_id", "model_id", name="uq_custom_models_user_model"),
    )
    op.create_index("ix_custom_models_user_id", "custom_models", ["user_id"])
    op.create_index("ix_custom_models_model_type", "custom_models", ["model_type"])


def downgrade() -> None:
    op.drop_index("ix_custom_models_model_type", table_name="custom_models")
    op.drop_index("ix_custom_models_user_id", table_name="custom_models")
    op.drop_table("custom_models")
```

**Step 3: Run migration**

Run: `uv run alembic upgrade head`

**Step 4: Commit**

```bash
git add packages/shared/database/models.py alembic/versions/
git commit -m "feat(db): add custom_models table for user-defined models"
```

---

### Task 3: Custom Model Repository

**Files:**
- Create: `packages/shared/database/repositories/custom_model_repository.py`
- Modify: `packages/shared/database/repositories/__init__.py`

**Step 1: Create the repository**

Create `packages/shared/database/repositories/custom_model_repository.py`:

```python
"""Repository for CustomModel operations."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError
from shared.database.models import CustomModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class CustomModelRepository:
    """Repository for custom model operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def list_by_user(
        self,
        user_id: str,
        *,
        model_type: str | None = None,
    ) -> list[CustomModel]:
        """List custom models for a user with optional type filter."""
        try:
            stmt = select(CustomModel).where(CustomModel.user_id == user_id)
            if model_type is not None:
                stmt = stmt.where(CustomModel.model_type == model_type)
            stmt = stmt.order_by(CustomModel.created_at.desc())
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as exc:
            logger.error("Failed to list custom models for user %s: %s", user_id, exc, exc_info=True)
            raise DatabaseOperationError("list", "CustomModel", str(exc)) from exc

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get(self, custom_model_id: str) -> CustomModel | None:
        """Get a custom model by ID."""
        try:
            result = await self.session.execute(
                select(CustomModel).where(CustomModel.id == custom_model_id)
            )
            return result.scalar_one_or_none()
        except Exception as exc:
            logger.error("Failed to get custom model %s: %s", custom_model_id, exc, exc_info=True)
            raise DatabaseOperationError("get", "CustomModel", str(exc)) from exc

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_by_model_id(self, user_id: str, model_id: str) -> CustomModel | None:
        """Get a custom model by user and model ID."""
        try:
            result = await self.session.execute(
                select(CustomModel).where(
                    CustomModel.user_id == user_id,
                    CustomModel.model_id == model_id,
                )
            )
            return result.scalar_one_or_none()
        except Exception as exc:
            logger.error("Failed to get custom model %s for user %s: %s", model_id, user_id, exc, exc_info=True)
            raise DatabaseOperationError("get", "CustomModel", str(exc)) from exc

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def create(
        self,
        *,
        user_id: str,
        model_id: str,
        model_type: str,
        config: dict[str, Any] | None = None,
    ) -> CustomModel:
        """Create a new custom model entry."""
        try:
            custom_model = CustomModel(
                id=str(uuid.uuid4()),
                user_id=user_id,
                model_id=model_id,
                model_type=model_type,
                config=config or {},
            )
            self.session.add(custom_model)
            await self.session.flush()
            return custom_model
        except Exception as exc:
            logger.error("Failed to create custom model %s: %s", model_id, exc, exc_info=True)
            raise DatabaseOperationError("create", "CustomModel", str(exc)) from exc

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def update_config(
        self,
        custom_model_id: str,
        config: dict[str, Any],
    ) -> CustomModel:
        """Update the config for a custom model."""
        try:
            model = await self.get(custom_model_id)
            if model is None:
                raise EntityNotFoundError("CustomModel", custom_model_id)
            model.config = config
            await self.session.flush()
            return model
        except EntityNotFoundError:
            raise
        except Exception as exc:
            logger.error("Failed to update custom model %s: %s", custom_model_id, exc, exc_info=True)
            raise DatabaseOperationError("update", "CustomModel", str(exc)) from exc

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete(self, custom_model_id: str) -> bool:
        """Delete a custom model. Returns True if deleted, False if not found."""
        try:
            result = await self.session.execute(
                delete(CustomModel).where(CustomModel.id == custom_model_id)
            )
            await self.session.flush()
            return result.rowcount > 0
        except Exception as exc:
            logger.error("Failed to delete custom model %s: %s", custom_model_id, exc, exc_info=True)
            raise DatabaseOperationError("delete", "CustomModel", str(exc)) from exc
```

**Step 2: Export from repositories __init__.py**

Add to `packages/shared/database/repositories/__init__.py`:

```python
from shared.database.repositories.custom_model_repository import CustomModelRepository
```

And add to `__all__`:

```python
"CustomModelRepository",
```

**Step 3: Commit**

```bash
git add packages/shared/database/repositories/
git commit -m "feat(db): add CustomModelRepository for user-defined models"
```

---

### Task 4: HuggingFace Cache Manager

**Files:**
- Create: `packages/shared/models/cache_manager.py`

**Step 1: Create the cache manager**

Create `packages/shared/models/cache_manager.py`:

```python
"""HuggingFace model cache management utilities."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, scan_cache_dir
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """Information about a cached model."""

    model_id: str
    size_on_disk_bytes: int
    last_accessed: float | None
    revisions: list[str]

    @property
    def size_on_disk_mb(self) -> int:
        """Size in megabytes."""
        return self.size_on_disk_bytes // (1024 * 1024)


@dataclass
class ModelExistsResult:
    """Result of checking if a model exists on HuggingFace."""

    exists: bool
    error: str | None = None
    model_info: dict[str, Any] | None = None


class HuggingFaceCacheManager:
    """Manages HuggingFace model cache operations."""

    def __init__(self, cache_dir: str | None = None):
        """Initialize the cache manager.

        Args:
            cache_dir: Custom cache directory. Defaults to HF_HOME or ~/.cache/huggingface/hub
        """
        self.cache_dir = cache_dir or os.environ.get(
            "HF_HOME",
            os.path.expanduser("~/.cache/huggingface/hub"),
        )
        self._api = HfApi()

    def list_cached_models(self) -> list[CachedModel]:
        """List all models in the HuggingFace cache.

        Returns:
            List of CachedModel with size and revision info.
        """
        try:
            cache_info = scan_cache_dir(self.cache_dir)
        except Exception as e:
            logger.warning("Failed to scan HuggingFace cache: %s", e)
            return []

        models: list[CachedModel] = []

        for repo in cache_info.repos:
            # Only include model repos (not datasets)
            if repo.repo_type != "model":
                continue

            # Calculate total size across all revisions
            total_size = sum(rev.size_on_disk for rev in repo.revisions)
            revisions = [rev.commit_hash[:8] for rev in repo.revisions]

            # Get last accessed time from most recent revision
            last_accessed = None
            if repo.revisions:
                last_accessed = max(
                    (rev.last_accessed for rev in repo.revisions if rev.last_accessed),
                    default=None,
                )

            models.append(
                CachedModel(
                    model_id=repo.repo_id,
                    size_on_disk_bytes=total_size,
                    last_accessed=last_accessed,
                    revisions=revisions,
                )
            )

        return models

    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is in the cache.

        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-Embedding-0.6B")

        Returns:
            True if the model is cached locally.
        """
        cached = self.list_cached_models()
        return any(m.model_id == model_id for m in cached)

    def get_cached_model_info(self, model_id: str) -> CachedModel | None:
        """Get info about a specific cached model.

        Args:
            model_id: HuggingFace model ID

        Returns:
            CachedModel info or None if not cached.
        """
        cached = self.list_cached_models()
        for model in cached:
            if model.model_id == model_id:
                return model
        return None

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the cache.

        Args:
            model_id: HuggingFace model ID to delete

        Returns:
            True if deleted, False if not found.
        """
        try:
            cache_info = scan_cache_dir(self.cache_dir)
        except Exception as e:
            logger.error("Failed to scan cache for deletion: %s", e)
            return False

        for repo in cache_info.repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                # Delete all revisions for this model
                delete_strategy = cache_info.delete_revisions(
                    *[rev.commit_hash for rev in repo.revisions]
                )
                logger.info(
                    "Deleting model %s: %d files, %.2f MB",
                    model_id,
                    delete_strategy.expected_freed_size_str,
                    delete_strategy.expected_freed_size / (1024 * 1024),
                )
                delete_strategy.execute()
                return True

        logger.warning("Model %s not found in cache", model_id)
        return False

    def check_model_exists_on_hub(self, model_id: str) -> ModelExistsResult:
        """Check if a model exists on HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID to check

        Returns:
            ModelExistsResult with exists flag and optional error/info.
        """
        try:
            info = self._api.model_info(model_id)
            return ModelExistsResult(
                exists=True,
                model_info={
                    "id": info.id,
                    "downloads": info.downloads,
                    "likes": info.likes,
                    "tags": info.tags,
                },
            )
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                return ModelExistsResult(exists=False, error="Model not found on HuggingFace Hub")
            return ModelExistsResult(exists=False, error=str(e))
        except Exception as e:
            return ModelExistsResult(exists=False, error=str(e))

    def get_total_cache_size_mb(self) -> int:
        """Get total size of all cached models in MB."""
        cached = self.list_cached_models()
        total_bytes = sum(m.size_on_disk_bytes for m in cached)
        return total_bytes // (1024 * 1024)
```

**Step 2: Commit**

```bash
git add packages/shared/models/cache_manager.py
git commit -m "feat(models): add HuggingFace cache manager for model downloads"
```

---

### Task 5: Model Download Celery Task

**Files:**
- Create: `packages/webui/tasks/model_download.py`
- Modify: `packages/webui/celery_app.py` (add task import)

**Step 1: Create the download task**

Create `packages/webui/tasks/model_download.py`:

```python
"""Celery task for background model downloads."""

from __future__ import annotations

import logging
import time
from typing import Any

from celery import shared_task
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from webui.celery_app import celery_app

logger = logging.getLogger(__name__)

# Redis key prefix for download progress
DOWNLOAD_PROGRESS_PREFIX = "model_download:"
DOWNLOAD_PROGRESS_TTL = 3600  # 1 hour


def _update_progress(
    task_id: str,
    model_id: str,
    status: str,
    bytes_downloaded: int = 0,
    bytes_total: int = 0,
    error: str | None = None,
) -> None:
    """Update download progress in Redis."""
    from webui.redis_client import get_redis_client

    redis = get_redis_client()
    key = f"{DOWNLOAD_PROGRESS_PREFIX}{task_id}"

    progress = {
        "task_id": task_id,
        "model_id": model_id,
        "status": status,
        "bytes_downloaded": bytes_downloaded,
        "bytes_total": bytes_total,
        "error": error,
        "updated_at": time.time(),
    }

    redis.hset(key, mapping={k: str(v) if v is not None else "" for k, v in progress.items()})
    redis.expire(key, DOWNLOAD_PROGRESS_TTL)


def get_download_progress(task_id: str) -> dict[str, Any] | None:
    """Get download progress from Redis."""
    from webui.redis_client import get_redis_client

    redis = get_redis_client()
    key = f"{DOWNLOAD_PROGRESS_PREFIX}{task_id}"

    data = redis.hgetall(key)
    if not data:
        return None

    return {
        "task_id": data.get(b"task_id", b"").decode(),
        "model_id": data.get(b"model_id", b"").decode(),
        "status": data.get(b"status", b"").decode(),
        "bytes_downloaded": int(data.get(b"bytes_downloaded", b"0")),
        "bytes_total": int(data.get(b"bytes_total", b"0")),
        "error": data.get(b"error", b"").decode() or None,
    }


@celery_app.task(bind=True, name="download_model")
def download_model_task(self, model_id: str) -> dict[str, Any]:
    """Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-Embedding-0.6B")

    Returns:
        Dict with status and details.
    """
    task_id = self.request.id
    logger.info("Starting download for model: %s (task: %s)", model_id, task_id)

    _update_progress(task_id, model_id, "downloading")

    try:
        # Track progress via callback
        bytes_downloaded = 0
        bytes_total = 0

        def progress_callback(progress: Any) -> None:
            nonlocal bytes_downloaded, bytes_total
            # huggingface_hub provides progress as tqdm-compatible object
            if hasattr(progress, "n") and hasattr(progress, "total"):
                bytes_downloaded = progress.n
                bytes_total = progress.total or 0
                _update_progress(task_id, model_id, "downloading", bytes_downloaded, bytes_total)

        # Download the model
        local_dir = snapshot_download(
            repo_id=model_id,
            # Let HF use default cache location
        )

        logger.info("Model %s downloaded to %s", model_id, local_dir)
        _update_progress(task_id, model_id, "completed", bytes_downloaded, bytes_total)

        return {
            "status": "completed",
            "model_id": model_id,
            "local_dir": local_dir,
        }

    except HfHubHTTPError as e:
        error_msg = f"HuggingFace Hub error: {e}"
        logger.error("Failed to download model %s: %s", model_id, error_msg)
        _update_progress(task_id, model_id, "failed", error=error_msg)
        return {"status": "failed", "model_id": model_id, "error": error_msg}

    except Exception as e:
        error_msg = str(e)
        logger.error("Failed to download model %s: %s", model_id, error_msg, exc_info=True)
        _update_progress(task_id, model_id, "failed", error=error_msg)
        return {"status": "failed", "model_id": model_id, "error": error_msg}
```

**Step 2: Import in celery_app.py**

Add to the imports in `packages/webui/celery_app.py`:

```python
# Import tasks to register them
import webui.tasks.model_download  # noqa: F401
```

**Step 3: Commit**

```bash
git add packages/webui/tasks/model_download.py packages/webui/celery_app.py
git commit -m "feat(tasks): add Celery task for background model downloads"
```

---

### Task 6: Model Manager API Schemas

**Files:**
- Create: `packages/webui/api/v2/models_schemas.py`

**Step 1: Create the schemas**

Create `packages/webui/api/v2/models_schemas.py`:

```python
"""Pydantic schemas for the Model Manager API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuantizationMemory(BaseModel):
    """Memory estimates per quantization level."""

    float32: int | None = None
    float16: int | None = None
    int8: int | None = None
    int4: int | None = None


class ModelResponse(BaseModel):
    """Model information for API responses."""

    id: str = Field(..., description="HuggingFace model ID")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Model description")
    model_type: str = Field(..., description="Model type: embedding, llm, reranker, splade")
    download_size_mb: int = Field(..., description="Download size in MB")
    memory_mb: QuantizationMemory = Field(default_factory=QuantizationMemory, description="VRAM estimates per quantization")

    # Installation status
    is_installed: bool = Field(..., description="Whether the model is cached locally")
    size_on_disk_mb: int | None = Field(None, description="Actual size on disk if installed")

    # Usage info (only for installed models)
    used_by_collections: list[str] = Field(default_factory=list, description="Collection names using this model")

    # Whether this is a curated or custom model
    is_custom: bool = Field(False, description="Whether this is a user-added custom model")
    custom_model_db_id: str | None = Field(None, description="Database ID if custom model")

    # Embedding-specific fields
    dimension: int | None = None
    max_sequence_length: int | None = None
    pooling_method: str | None = None
    is_asymmetric: bool = False
    query_prefix: str | None = None
    document_prefix: str | None = None
    default_query_instruction: str | None = None

    # LLM-specific fields
    context_window: int | None = None


class ModelListResponse(BaseModel):
    """Response for listing models."""

    models: list[ModelResponse]
    total_cache_size_mb: int = Field(..., description="Total size of all cached models")


class DownloadModelRequest(BaseModel):
    """Request to start downloading a model."""

    model_id: str = Field(..., description="HuggingFace model ID to download")


class DownloadProgressResponse(BaseModel):
    """Progress information for a model download."""

    task_id: str
    model_id: str
    status: str = Field(..., description="pending, downloading, completed, failed")
    bytes_downloaded: int = 0
    bytes_total: int = 0
    error: str | None = None


class DownloadStartResponse(BaseModel):
    """Response when starting a download."""

    task_id: str
    model_id: str
    status: str = "pending"


class DeleteModelResponse(BaseModel):
    """Response from deleting a model."""

    success: bool
    model_id: str
    message: str
    freed_mb: int = 0


class AddCustomModelRequest(BaseModel):
    """Request to add a custom model to the registry."""

    model_id: str = Field(..., description="HuggingFace model ID")
    model_type: str = Field(..., description="Model type: embedding, llm, reranker, splade")

    # Optional overrides
    dimension: int | None = Field(None, description="Embedding dimension (auto-detected if not provided)")
    pooling_method: str | None = Field(None, description="Pooling method: mean, cls, last_token")
    is_asymmetric: bool = Field(False, description="Whether model needs different query/doc handling")
    query_prefix: str | None = Field(None, description="Prefix for queries (BGE/E5 style)")
    document_prefix: str | None = Field(None, description="Prefix for documents")
    default_query_instruction: str | None = Field(None, description="Default instruction (Qwen style)")


class AddCustomModelResponse(BaseModel):
    """Response from adding a custom model."""

    success: bool
    custom_model_id: str
    model_id: str
    message: str


class DeleteCustomModelResponse(BaseModel):
    """Response from removing a custom model from registry."""

    success: bool
    message: str
```

**Step 2: Commit**

```bash
git add packages/webui/api/v2/models_schemas.py
git commit -m "feat(api): add Pydantic schemas for Model Manager API"
```

---

### Task 7: Model Manager API Router

**Files:**
- Create: `packages/webui/api/v2/models.py`
- Modify: `packages/webui/api/v2/__init__.py` (add router)

**Step 1: Create the API router**

Create `packages/webui/api/v2/models.py`:

```python
"""Model Manager API endpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from shared.database import get_db
from shared.database.repositories.custom_model_repository import CustomModelRepository
from shared.models import ModelType, get_curated_models, get_model_info
from shared.models.cache_manager import HuggingFaceCacheManager
from webui.api.schemas import ErrorResponse
from webui.api.v2.models_schemas import (
    AddCustomModelRequest,
    AddCustomModelResponse,
    DeleteCustomModelResponse,
    DeleteModelResponse,
    DownloadModelRequest,
    DownloadProgressResponse,
    DownloadStartResponse,
    ModelListResponse,
    ModelResponse,
    QuantizationMemory,
)
from webui.auth import get_current_user
from webui.celery_app import celery_app
from webui.tasks.model_download import get_download_progress

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/models", tags=["models-v2"])

# Singleton cache manager
_cache_manager: HuggingFaceCacheManager | None = None


def _get_cache_manager() -> HuggingFaceCacheManager:
    """Get or create the cache manager singleton."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = HuggingFaceCacheManager()
    return _cache_manager


async def _get_collections_using_model(
    db: AsyncSession,
    model_id: str,
    user_id: str,
) -> list[str]:
    """Get collection names using a specific model."""
    from shared.database.repositories.collection_repository import CollectionRepository

    repo = CollectionRepository(db)
    collections = await repo.list_by_owner(user_id)

    # Filter collections using this embedding model
    using_model = [c.name for c in collections if c.embedding_model == model_id]
    return using_model


def _model_info_to_response(
    model: Any,
    is_installed: bool,
    size_on_disk_mb: int | None,
    used_by: list[str],
    is_custom: bool = False,
    custom_model_db_id: str | None = None,
) -> ModelResponse:
    """Convert model info to API response."""
    memory = model.memory_mb if hasattr(model, "memory_mb") else {}

    return ModelResponse(
        id=model.id,
        name=model.name,
        description=model.description,
        model_type=model.model_type.value if hasattr(model.model_type, "value") else model.model_type,
        download_size_mb=model.download_size_mb if hasattr(model, "download_size_mb") else 0,
        memory_mb=QuantizationMemory(**memory) if isinstance(memory, dict) else QuantizationMemory(),
        is_installed=is_installed,
        size_on_disk_mb=size_on_disk_mb,
        used_by_collections=used_by,
        is_custom=is_custom,
        custom_model_db_id=custom_model_db_id,
        dimension=getattr(model, "dimension", None),
        max_sequence_length=getattr(model, "max_sequence_length", None),
        pooling_method=getattr(model, "pooling_method", None),
        is_asymmetric=getattr(model, "is_asymmetric", False),
        query_prefix=getattr(model, "query_prefix", None),
        document_prefix=getattr(model, "document_prefix", None),
        default_query_instruction=getattr(model, "default_query_instruction", None),
        context_window=getattr(model, "context_window", None),
    )


@router.get(
    "",
    response_model=ModelListResponse,
    responses={401: {"model": ErrorResponse, "description": "Unauthorized"}},
)
async def list_models(
    model_type: str | None = None,
    installed_only: bool = False,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ModelListResponse:
    """List all models (curated + custom) with installation status."""
    user_id = current_user["id"]
    cache_mgr = _get_cache_manager()

    # Get cached models for install status
    cached_models = {m.model_id: m for m in cache_mgr.list_cached_models()}

    # Get curated models
    curated = get_curated_models()
    if model_type:
        try:
            mt = ModelType(model_type)
            curated = [m for m in curated if m.model_type == mt]
        except ValueError:
            raise HTTPException(400, f"Invalid model_type: {model_type}")

    # Get custom models
    custom_repo = CustomModelRepository(db)
    custom_models = await custom_repo.list_by_user(user_id, model_type=model_type)

    # Build response
    models: list[ModelResponse] = []

    # Add curated models
    for model in curated:
        cached = cached_models.get(model.id)
        is_installed = cached is not None
        size_on_disk = cached.size_on_disk_mb if cached else None

        if installed_only and not is_installed:
            continue

        used_by = await _get_collections_using_model(db, model.id, user_id) if is_installed else []

        models.append(
            _model_info_to_response(model, is_installed, size_on_disk, used_by)
        )

    # Add custom models
    for custom in custom_models:
        cached = cached_models.get(custom.model_id)
        is_installed = cached is not None
        size_on_disk = cached.size_on_disk_mb if cached else None

        if installed_only and not is_installed:
            continue

        used_by = await _get_collections_using_model(db, custom.model_id, user_id) if is_installed else []

        # Build a simple object for the response helper
        class CustomModelInfo:
            def __init__(self, cm):
                self.id = cm.model_id
                self.name = cm.model_id.split("/")[-1]  # Use repo name as display name
                self.description = "Custom model"
                self.model_type = cm.model_type
                self.download_size_mb = 0  # Unknown for custom
                self.memory_mb = cm.config.get("memory_mb", {})
                self.dimension = cm.config.get("dimension")
                self.max_sequence_length = cm.config.get("max_sequence_length")
                self.pooling_method = cm.config.get("pooling_method", "mean")
                self.is_asymmetric = cm.config.get("is_asymmetric", False)
                self.query_prefix = cm.config.get("query_prefix")
                self.document_prefix = cm.config.get("document_prefix")
                self.default_query_instruction = cm.config.get("default_query_instruction")
                self.context_window = cm.config.get("context_window")

        models.append(
            _model_info_to_response(
                CustomModelInfo(custom),
                is_installed,
                size_on_disk,
                used_by,
                is_custom=True,
                custom_model_db_id=custom.id,
            )
        )

    return ModelListResponse(
        models=models,
        total_cache_size_mb=cache_mgr.get_total_cache_size_mb(),
    )


@router.post(
    "/download",
    response_model=DownloadStartResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid model ID"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def start_download(
    request: DownloadModelRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> DownloadStartResponse:
    """Start downloading a model in the background."""
    cache_mgr = _get_cache_manager()

    # Check if already installed
    if cache_mgr.is_model_cached(request.model_id):
        raise HTTPException(400, f"Model {request.model_id} is already installed")

    # Verify model exists on HuggingFace
    result = cache_mgr.check_model_exists_on_hub(request.model_id)
    if not result.exists:
        raise HTTPException(400, result.error or f"Model {request.model_id} not found")

    # Dispatch Celery task
    task = celery_app.send_task("download_model", args=[request.model_id])

    logger.info("Started download task %s for model %s", task.id, request.model_id)

    return DownloadStartResponse(
        task_id=task.id,
        model_id=request.model_id,
        status="pending",
    )


@router.get(
    "/download/{task_id}",
    response_model=DownloadProgressResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_download_status(
    task_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> DownloadProgressResponse:
    """Get the progress of a model download."""
    progress = get_download_progress(task_id)

    if progress is None:
        raise HTTPException(404, f"Download task {task_id} not found")

    return DownloadProgressResponse(**progress)


@router.delete(
    "/{model_id:path}",
    response_model=DeleteModelResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Model in use"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
async def delete_model(
    model_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeleteModelResponse:
    """Delete a model from the local cache."""
    user_id = current_user["id"]
    cache_mgr = _get_cache_manager()

    # Check if installed
    cached = cache_mgr.get_cached_model_info(model_id)
    if cached is None:
        raise HTTPException(404, f"Model {model_id} is not installed")

    # Check if in use
    used_by = await _get_collections_using_model(db, model_id, user_id)
    if used_by:
        raise HTTPException(
            400,
            f"Cannot delete: model is used by collections: {', '.join(used_by)}",
        )

    # Delete
    freed_mb = cached.size_on_disk_mb
    success = cache_mgr.delete_model(model_id)

    if not success:
        raise HTTPException(500, f"Failed to delete model {model_id}")

    return DeleteModelResponse(
        success=True,
        model_id=model_id,
        message=f"Model deleted, freed {freed_mb} MB",
        freed_mb=freed_mb,
    )


@router.post(
    "/custom",
    response_model=AddCustomModelResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid model"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def add_custom_model(
    request: AddCustomModelRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AddCustomModelResponse:
    """Add a custom model to the user's registry."""
    user_id = current_user["id"]
    cache_mgr = _get_cache_manager()

    # Validate model type
    valid_types = ["embedding", "llm", "reranker", "splade"]
    if request.model_type not in valid_types:
        raise HTTPException(400, f"Invalid model_type. Must be one of: {valid_types}")

    # Check if model exists on HuggingFace
    result = cache_mgr.check_model_exists_on_hub(request.model_id)
    if not result.exists:
        raise HTTPException(400, result.error or f"Model {request.model_id} not found on HuggingFace")

    # Check if already in registry (curated or custom)
    if get_model_info(request.model_id) is not None:
        raise HTTPException(400, f"Model {request.model_id} is already in the curated registry")

    custom_repo = CustomModelRepository(db)
    existing = await custom_repo.get_by_model_id(user_id, request.model_id)
    if existing:
        raise HTTPException(400, f"Model {request.model_id} is already in your custom models")

    # Build config from overrides
    config = {}
    if request.dimension is not None:
        config["dimension"] = request.dimension
    if request.pooling_method is not None:
        config["pooling_method"] = request.pooling_method
    if request.is_asymmetric:
        config["is_asymmetric"] = True
    if request.query_prefix is not None:
        config["query_prefix"] = request.query_prefix
    if request.document_prefix is not None:
        config["document_prefix"] = request.document_prefix
    if request.default_query_instruction is not None:
        config["default_query_instruction"] = request.default_query_instruction

    # Create custom model entry
    custom_model = await custom_repo.create(
        user_id=user_id,
        model_id=request.model_id,
        model_type=request.model_type,
        config=config,
    )
    await db.commit()

    return AddCustomModelResponse(
        success=True,
        custom_model_id=custom_model.id,
        model_id=request.model_id,
        message=f"Custom model {request.model_id} added to registry",
    )


@router.delete(
    "/custom/{custom_model_id}",
    response_model=DeleteCustomModelResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Custom model not found"},
    },
)
async def delete_custom_model(
    custom_model_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    db: AsyncSession = Depends(get_db),
) -> DeleteCustomModelResponse:
    """Remove a custom model from the user's registry."""
    custom_repo = CustomModelRepository(db)

    deleted = await custom_repo.delete(custom_model_id)
    if not deleted:
        raise HTTPException(404, f"Custom model {custom_model_id} not found")

    await db.commit()

    return DeleteCustomModelResponse(
        success=True,
        message="Custom model removed from registry",
    )
```

**Step 2: Register the router in __init__.py**

Add to `packages/webui/api/v2/__init__.py` with the other router imports:

```python
from webui.api.v2.models import router as models_router
```

And add to the router list that gets included in main.py (or wherever routers are registered).

**Step 3: Commit**

```bash
git add packages/webui/api/v2/models.py packages/webui/api/v2/__init__.py
git commit -m "feat(api): add Model Manager API endpoints"
```

---

## Phase 2: Frontend Implementation

### Task 8: Frontend Types

**Files:**
- Create: `apps/webui-react/src/types/model.ts`

**Step 1: Create the types file**

Create `apps/webui-react/src/types/model.ts`:

```typescript
/**
 * Model Manager type definitions
 */

/**
 * Model types supported by the manager
 */
export type ModelType = 'embedding' | 'llm' | 'reranker' | 'splade';

/**
 * Download status
 */
export type DownloadStatus = 'pending' | 'downloading' | 'completed' | 'failed';

/**
 * Memory estimates per quantization level
 */
export interface QuantizationMemory {
  float32?: number;
  float16?: number;
  int8?: number;
  int4?: number;
}

/**
 * Model information from the API
 */
export interface Model {
  id: string;
  name: string;
  description: string;
  model_type: ModelType;
  download_size_mb: number;
  memory_mb: QuantizationMemory;

  // Installation status
  is_installed: boolean;
  size_on_disk_mb: number | null;

  // Usage info
  used_by_collections: string[];

  // Custom model info
  is_custom: boolean;
  custom_model_db_id: string | null;

  // Embedding-specific
  dimension?: number | null;
  max_sequence_length?: number | null;
  pooling_method?: string | null;
  is_asymmetric?: boolean;
  query_prefix?: string | null;
  document_prefix?: string | null;
  default_query_instruction?: string | null;

  // LLM-specific
  context_window?: number | null;
}

/**
 * Response from listing models
 */
export interface ModelListResponse {
  models: Model[];
  total_cache_size_mb: number;
}

/**
 * Request to download a model
 */
export interface DownloadModelRequest {
  model_id: string;
}

/**
 * Response when starting a download
 */
export interface DownloadStartResponse {
  task_id: string;
  model_id: string;
  status: string;
}

/**
 * Download progress information
 */
export interface DownloadProgress {
  task_id: string;
  model_id: string;
  status: DownloadStatus;
  bytes_downloaded: number;
  bytes_total: number;
  error?: string | null;
}

/**
 * Response from deleting a model
 */
export interface DeleteModelResponse {
  success: boolean;
  model_id: string;
  message: string;
  freed_mb: number;
}

/**
 * Request to add a custom model
 */
export interface AddCustomModelRequest {
  model_id: string;
  model_type: ModelType;
  dimension?: number | null;
  pooling_method?: string | null;
  is_asymmetric?: boolean;
  query_prefix?: string | null;
  document_prefix?: string | null;
  default_query_instruction?: string | null;
}

/**
 * Response from adding a custom model
 */
export interface AddCustomModelResponse {
  success: boolean;
  custom_model_id: string;
  model_id: string;
  message: string;
}

/**
 * Model type display labels
 */
export const MODEL_TYPE_LABELS: Record<ModelType, string> = {
  embedding: 'Embedding Models',
  llm: 'Local LLMs',
  reranker: 'Rerankers',
  splade: 'SPLADE Models',
};

/**
 * Model type tab order
 */
export const MODEL_TYPE_ORDER: ModelType[] = ['embedding', 'llm', 'reranker', 'splade'];

/**
 * Filter status options
 */
export type ModelFilterStatus = 'all' | 'installed' | 'available';

/**
 * Filters for model list
 */
export interface ModelFilters {
  model_type: ModelType;
  status: ModelFilterStatus;
  search: string;
}

/**
 * Format bytes to human readable size
 */
export function formatSize(mb: number): string {
  if (mb >= 1000) {
    return `${(mb / 1000).toFixed(1)} GB`;
  }
  return `${mb} MB`;
}

/**
 * Get download progress percentage
 */
export function getProgressPercent(progress: DownloadProgress): number {
  if (progress.bytes_total === 0) return 0;
  return Math.round((progress.bytes_downloaded / progress.bytes_total) * 100);
}
```

**Step 2: Commit**

```bash
git add apps/webui-react/src/types/model.ts
git commit -m "feat(ui): add Model Manager TypeScript types"
```

---

### Task 9: Frontend API Service

**Files:**
- Create: `apps/webui-react/src/services/api/v2/models.ts`

**Step 1: Create the API service**

Create `apps/webui-react/src/services/api/v2/models.ts`:

```typescript
import apiClient from './client';
import type {
  ModelListResponse,
  DownloadModelRequest,
  DownloadStartResponse,
  DownloadProgress,
  DeleteModelResponse,
  AddCustomModelRequest,
  AddCustomModelResponse,
  ModelType,
} from '../../../types/model';

/**
 * Models API client for the Model Manager
 */
export const modelsApi = {
  /**
   * List all models with optional filters
   */
  list: (params?: { model_type?: ModelType; installed_only?: boolean }) =>
    apiClient.get<ModelListResponse>('/api/v2/models', { params }),

  /**
   * Start downloading a model
   */
  download: (request: DownloadModelRequest) =>
    apiClient.post<DownloadStartResponse>('/api/v2/models/download', request),

  /**
   * Get download progress
   */
  getDownloadProgress: (taskId: string) =>
    apiClient.get<DownloadProgress>(`/api/v2/models/download/${taskId}`),

  /**
   * Delete a model from cache
   */
  delete: (modelId: string) =>
    apiClient.delete<DeleteModelResponse>(`/api/v2/models/${encodeURIComponent(modelId)}`),

  /**
   * Add a custom model to registry
   */
  addCustom: (request: AddCustomModelRequest) =>
    apiClient.post<AddCustomModelResponse>('/api/v2/models/custom', request),

  /**
   * Remove a custom model from registry
   */
  deleteCustom: (customModelId: string) =>
    apiClient.delete<{ success: boolean; message: string }>(
      `/api/v2/models/custom/${customModelId}`
    ),
};
```

**Step 2: Commit**

```bash
git add apps/webui-react/src/services/api/v2/models.ts
git commit -m "feat(ui): add Models API service client"
```

---

### Task 10: Frontend Hooks

**Files:**
- Create: `apps/webui-react/src/hooks/useModels.ts`

**Step 1: Create the hooks file**

Create `apps/webui-react/src/hooks/useModels.ts`:

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState, useEffect, useCallback } from 'react';
import { modelsApi } from '../services/api/v2/models';
import type {
  Model,
  ModelType,
  DownloadProgress,
  AddCustomModelRequest,
} from '../types/model';
import { useUIStore } from '../stores/uiStore';

/**
 * Query key factory for model queries
 */
export const modelKeys = {
  all: ['models'] as const,
  list: (modelType?: ModelType, installedOnly?: boolean) =>
    [...modelKeys.all, 'list', modelType, installedOnly] as const,
  download: (taskId: string) => [...modelKeys.all, 'download', taskId] as const,
};

/**
 * Hook to fetch models with optional filtering
 */
export function useModels(modelType?: ModelType, installedOnly?: boolean) {
  return useQuery({
    queryKey: modelKeys.list(modelType, installedOnly),
    queryFn: async () => {
      const response = await modelsApi.list({
        model_type: modelType,
        installed_only: installedOnly,
      });
      return response.data;
    },
    staleTime: 30 * 1000, // 30 seconds
  });
}

/**
 * Hook to download a model with progress tracking
 */
export function useDownloadModel() {
  const queryClient = useQueryClient();
  const addToast = useUIStore((state) => state.addToast);
  const [activeDownloads, setActiveDownloads] = useState<Map<string, DownloadProgress>>(
    new Map()
  );

  const startDownload = useMutation({
    mutationFn: async (modelId: string) => {
      const response = await modelsApi.download({ model_id: modelId });
      return response.data;
    },
    onSuccess: (data) => {
      // Initialize progress tracking
      setActiveDownloads((prev) => {
        const next = new Map(prev);
        next.set(data.model_id, {
          task_id: data.task_id,
          model_id: data.model_id,
          status: 'pending',
          bytes_downloaded: 0,
          bytes_total: 0,
        });
        return next;
      });

      addToast({
        title: 'Download started',
        message: `Downloading ${data.model_id}...`,
        type: 'info',
      });
    },
    onError: (error: Error) => {
      addToast({
        title: 'Download failed',
        message: error.message,
        type: 'error',
      });
    },
  });

  // Poll for progress updates
  const pollProgress = useCallback(async (taskId: string, modelId: string) => {
    try {
      const response = await modelsApi.getDownloadProgress(taskId);
      const progress = response.data;

      setActiveDownloads((prev) => {
        const next = new Map(prev);
        next.set(modelId, progress);
        return next;
      });

      if (progress.status === 'completed') {
        addToast({
          title: 'Download complete',
          message: `${modelId} downloaded successfully`,
          type: 'success',
        });
        // Invalidate models list to refresh
        queryClient.invalidateQueries({ queryKey: modelKeys.all });
        // Remove from active downloads after a delay
        setTimeout(() => {
          setActiveDownloads((prev) => {
            const next = new Map(prev);
            next.delete(modelId);
            return next;
          });
        }, 2000);
        return false; // Stop polling
      }

      if (progress.status === 'failed') {
        addToast({
          title: 'Download failed',
          message: progress.error || 'Unknown error',
          type: 'error',
        });
        return false; // Stop polling
      }

      return true; // Continue polling
    } catch {
      return false; // Stop polling on error
    }
  }, [addToast, queryClient]);

  // Effect to poll active downloads
  useEffect(() => {
    const intervals: Map<string, ReturnType<typeof setInterval>> = new Map();

    activeDownloads.forEach((progress, modelId) => {
      if (progress.status === 'pending' || progress.status === 'downloading') {
        if (!intervals.has(modelId)) {
          const interval = setInterval(async () => {
            const shouldContinue = await pollProgress(progress.task_id, modelId);
            if (!shouldContinue) {
              clearInterval(interval);
              intervals.delete(modelId);
            }
          }, 1000);
          intervals.set(modelId, interval);
        }
      }
    });

    return () => {
      intervals.forEach((interval) => clearInterval(interval));
    };
  }, [activeDownloads, pollProgress]);

  return {
    startDownload: startDownload.mutate,
    isStarting: startDownload.isPending,
    activeDownloads,
    getProgress: (modelId: string) => activeDownloads.get(modelId),
  };
}

/**
 * Hook to delete a model
 */
export function useDeleteModel() {
  const queryClient = useQueryClient();
  const addToast = useUIStore((state) => state.addToast);

  return useMutation({
    mutationFn: async (modelId: string) => {
      const response = await modelsApi.delete(modelId);
      return response.data;
    },
    onSuccess: (data) => {
      addToast({
        title: 'Model deleted',
        message: data.message,
        type: 'success',
      });
      queryClient.invalidateQueries({ queryKey: modelKeys.all });
    },
    onError: (error: Error) => {
      addToast({
        title: 'Delete failed',
        message: error.message,
        type: 'error',
      });
    },
  });
}

/**
 * Hook to add a custom model
 */
export function useAddCustomModel() {
  const queryClient = useQueryClient();
  const addToast = useUIStore((state) => state.addToast);

  return useMutation({
    mutationFn: async (request: AddCustomModelRequest) => {
      const response = await modelsApi.addCustom(request);
      return response.data;
    },
    onSuccess: (data) => {
      addToast({
        title: 'Custom model added',
        message: data.message,
        type: 'success',
      });
      queryClient.invalidateQueries({ queryKey: modelKeys.all });
    },
    onError: (error: Error) => {
      addToast({
        title: 'Failed to add model',
        message: error.message,
        type: 'error',
      });
    },
  });
}

/**
 * Hook to delete a custom model from registry
 */
export function useDeleteCustomModel() {
  const queryClient = useQueryClient();
  const addToast = useUIStore((state) => state.addToast);

  return useMutation({
    mutationFn: async (customModelId: string) => {
      const response = await modelsApi.deleteCustom(customModelId);
      return response.data;
    },
    onSuccess: () => {
      addToast({
        title: 'Custom model removed',
        message: 'Model removed from registry',
        type: 'success',
      });
      queryClient.invalidateQueries({ queryKey: modelKeys.all });
    },
    onError: (error: Error) => {
      addToast({
        title: 'Failed to remove model',
        message: error.message,
        type: 'error',
      });
    },
  });
}
```

**Step 2: Commit**

```bash
git add apps/webui-react/src/hooks/useModels.ts
git commit -m "feat(ui): add Model Manager React hooks"
```

---

### Task 11: ModelCard Component

**Files:**
- Create: `apps/webui-react/src/components/settings/models/ModelCard.tsx`

**Step 1: Create the component**

Create `apps/webui-react/src/components/settings/models/ModelCard.tsx`:

```typescript
import React, { useState } from 'react';
import { Download, Trash2, AlertCircle, Check, Loader2 } from 'lucide-react';
import type { Model, DownloadProgress } from '../../../types/model';
import { formatSize, getProgressPercent } from '../../../types/model';

interface ModelCardProps {
  model: Model;
  downloadProgress?: DownloadProgress;
  onDownload: (modelId: string) => void;
  onDelete: (modelId: string) => void;
  isDeleting?: boolean;
}

export function ModelCard({
  model,
  downloadProgress,
  onDownload,
  onDelete,
  isDeleting,
}: ModelCardProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const isDownloading = downloadProgress?.status === 'downloading' || downloadProgress?.status === 'pending';
  const downloadFailed = downloadProgress?.status === 'failed';
  const isInUse = model.used_by_collections.length > 0;

  const handleDelete = () => {
    if (isInUse) {
      return;
    }
    setShowDeleteConfirm(true);
  };

  const confirmDelete = () => {
    onDelete(model.id);
    setShowDeleteConfirm(false);
  };

  return (
    <div className="border border-[var(--border)] rounded-lg p-4 bg-[var(--bg-secondary)]">
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-[var(--text-primary)] truncate">{model.name}</h3>
          <p className="text-sm text-[var(--text-muted)] truncate">{model.id}</p>
        </div>
        <div className="ml-2 flex-shrink-0">
          {model.is_installed ? (
            <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-full bg-green-500/20 text-green-400">
              <Check className="w-3 h-3 mr-1" />
              Installed
            </span>
          ) : (
            <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-full bg-gray-500/20 text-[var(--text-muted)]">
              Not Installed
            </span>
          )}
        </div>
      </div>

      {/* Description */}
      <p className="text-sm text-[var(--text-secondary)] mb-3">{model.description}</p>

      {/* Size and Usage Info */}
      <div className="text-sm text-[var(--text-muted)] mb-3 space-y-1">
        <div>
          Size: {model.is_installed && model.size_on_disk_mb
            ? formatSize(model.size_on_disk_mb)
            : formatSize(model.download_size_mb)}
        </div>
        {model.is_installed && model.used_by_collections.length > 0 && (
          <div className="text-amber-400">
            Used by: {model.used_by_collections.join(', ')}
          </div>
        )}
        {model.is_custom && (
          <div className="text-blue-400">Custom model</div>
        )}
      </div>

      {/* Quantization Memory Table */}
      {Object.keys(model.memory_mb).length > 0 && (
        <div className="mb-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-[var(--text-muted)]">
                <th className="text-left font-normal pb-1">Quantization</th>
                <th className="text-right font-normal pb-1">VRAM</th>
              </tr>
            </thead>
            <tbody className="text-[var(--text-secondary)]">
              {model.memory_mb.float16 && (
                <tr>
                  <td>float16</td>
                  <td className="text-right">~{formatSize(model.memory_mb.float16)}</td>
                </tr>
              )}
              {model.memory_mb.int8 && (
                <tr>
                  <td>int8</td>
                  <td className="text-right">~{formatSize(model.memory_mb.int8)}</td>
                </tr>
              )}
              {model.memory_mb.int4 && (
                <tr>
                  <td>int4</td>
                  <td className="text-right">~{formatSize(model.memory_mb.int4)}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Download Progress */}
      {isDownloading && downloadProgress && (
        <div className="mb-3">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
            <span>Downloading...</span>
            <span>
              {formatSize(Math.round(downloadProgress.bytes_downloaded / (1024 * 1024)))} /
              {formatSize(Math.round(downloadProgress.bytes_total / (1024 * 1024)))}
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${getProgressPercent(downloadProgress)}%` }}
            />
          </div>
        </div>
      )}

      {/* Error Message */}
      {downloadFailed && downloadProgress?.error && (
        <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <span>{downloadProgress.error}</span>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-end gap-2">
        {showDeleteConfirm ? (
          <>
            <button
              onClick={() => setShowDeleteConfirm(false)}
              className="px-3 py-1.5 text-sm border border-[var(--border)] rounded hover:bg-[var(--bg-tertiary)]"
            >
              Cancel
            </button>
            <button
              onClick={confirmDelete}
              disabled={isDeleting}
              className="px-3 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 flex items-center gap-1"
            >
              {isDeleting && <Loader2 className="w-3 h-3 animate-spin" />}
              Confirm Delete
            </button>
          </>
        ) : model.is_installed ? (
          <button
            onClick={handleDelete}
            disabled={isInUse || isDeleting}
            title={isInUse ? 'Cannot delete: model is in use' : 'Delete model'}
            className="px-3 py-1.5 text-sm border border-[var(--border)] rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
          >
            <Trash2 className="w-4 h-4" />
            Delete
          </button>
        ) : (
          <button
            onClick={() => onDownload(model.id)}
            disabled={isDownloading}
            className="px-3 py-1.5 text-sm bg-[var(--bg-tertiary)] border border-[var(--border)] rounded hover:bg-gray-600 disabled:opacity-50 flex items-center gap-1"
          >
            {isDownloading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            {downloadFailed ? 'Retry' : 'Download'}
          </button>
        )}
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add apps/webui-react/src/components/settings/models/ModelCard.tsx
git commit -m "feat(ui): add ModelCard component for model display"
```

---

### Task 12: AddCustomModelModal Component

**Files:**
- Create: `apps/webui-react/src/components/settings/models/AddCustomModelModal.tsx`

**Step 1: Create the component**

Create `apps/webui-react/src/components/settings/models/AddCustomModelModal.tsx`:

```typescript
import React, { useState } from 'react';
import { X, ChevronDown, ChevronRight, Loader2 } from 'lucide-react';
import type { ModelType, AddCustomModelRequest } from '../../../types/model';
import { MODEL_TYPE_LABELS } from '../../../types/model';

interface AddCustomModelModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (request: AddCustomModelRequest) => void;
  isSubmitting: boolean;
  defaultModelType?: ModelType;
}

export function AddCustomModelModal({
  isOpen,
  onClose,
  onSubmit,
  isSubmitting,
  defaultModelType = 'embedding',
}: AddCustomModelModalProps) {
  const [modelId, setModelId] = useState('');
  const [modelType, setModelType] = useState<ModelType>(defaultModelType);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced options
  const [dimension, setDimension] = useState<string>('');
  const [poolingMethod, setPoolingMethod] = useState<string>('mean');
  const [isAsymmetric, setIsAsymmetric] = useState(false);
  const [queryPrefix, setQueryPrefix] = useState('');
  const [documentPrefix, setDocumentPrefix] = useState('');
  const [defaultInstruction, setDefaultInstruction] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const request: AddCustomModelRequest = {
      model_id: modelId.trim(),
      model_type: modelType,
    };

    // Add advanced options if provided
    if (dimension) {
      request.dimension = parseInt(dimension, 10);
    }
    if (poolingMethod !== 'mean') {
      request.pooling_method = poolingMethod;
    }
    if (isAsymmetric) {
      request.is_asymmetric = true;
      if (queryPrefix) request.query_prefix = queryPrefix;
      if (documentPrefix) request.document_prefix = documentPrefix;
      if (defaultInstruction) request.default_query_instruction = defaultInstruction;
    }

    onSubmit(request);
  };

  const resetForm = () => {
    setModelId('');
    setModelType(defaultModelType);
    setShowAdvanced(false);
    setDimension('');
    setPoolingMethod('mean');
    setIsAsymmetric(false);
    setQueryPrefix('');
    setDocumentPrefix('');
    setDefaultInstruction('');
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={handleClose} />

      {/* Modal */}
      <div className="relative bg-[var(--bg-primary)] border border-[var(--border)] rounded-lg shadow-xl w-full max-w-lg mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[var(--border)]">
          <h2 className="text-lg font-medium text-[var(--text-primary)]">Add Custom Model</h2>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-[var(--bg-tertiary)] rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Model ID */}
          <div>
            <label className="block text-sm font-medium text-[var(--text-primary)] mb-1">
              Model ID *
            </label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="e.g., BAAI/bge-small-en-v1.5"
              required
              className="w-full px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-gray-400"
            />
            <p className="mt-1 text-xs text-[var(--text-muted)]">
              Enter a HuggingFace model ID
            </p>
          </div>

          {/* Model Type */}
          <div>
            <label className="block text-sm font-medium text-[var(--text-primary)] mb-1">
              Model Type *
            </label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as ModelType)}
              className="w-full px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400"
            >
              {Object.entries(MODEL_TYPE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          {/* Advanced Options Toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
          >
            {showAdvanced ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
            Advanced Options
          </button>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="space-y-4 pl-4 border-l-2 border-[var(--border)]">
              {/* Dimension (embedding only) */}
              {modelType === 'embedding' && (
                <div>
                  <label className="block text-sm font-medium text-[var(--text-primary)] mb-1">
                    Dimension
                  </label>
                  <input
                    type="number"
                    value={dimension}
                    onChange={(e) => setDimension(e.target.value)}
                    placeholder="Auto-detect"
                    className="w-full px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-gray-400"
                  />
                </div>
              )}

              {/* Pooling Method (embedding only) */}
              {modelType === 'embedding' && (
                <div>
                  <label className="block text-sm font-medium text-[var(--text-primary)] mb-1">
                    Pooling Method
                  </label>
                  <select
                    value={poolingMethod}
                    onChange={(e) => setPoolingMethod(e.target.value)}
                    className="w-full px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400"
                  >
                    <option value="mean">Mean</option>
                    <option value="cls">CLS</option>
                    <option value="last_token">Last Token</option>
                  </select>
                </div>
              )}

              {/* Asymmetric Mode (embedding only) */}
              {modelType === 'embedding' && (
                <div>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={isAsymmetric}
                      onChange={(e) => setIsAsymmetric(e.target.checked)}
                      className="rounded border-[var(--border)]"
                    />
                    <span className="text-sm text-[var(--text-primary)]">
                      Asymmetric Mode
                    </span>
                  </label>
                  <p className="mt-1 text-xs text-[var(--text-muted)]">
                    Enable if queries and documents need different processing
                  </p>
                </div>
              )}

              {/* Asymmetric Options */}
              {modelType === 'embedding' && isAsymmetric && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-[var(--text-primary)] mb-1">
                      Query Prefix
                    </label>
                    <input
                      type="text"
                      value={queryPrefix}
                      onChange={(e) => setQueryPrefix(e.target.value)}
                      placeholder="e.g., query: "
                      className="w-full px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-gray-400"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-[var(--text-primary)] mb-1">
                      Document Prefix
                    </label>
                    <input
                      type="text"
                      value={documentPrefix}
                      onChange={(e) => setDocumentPrefix(e.target.value)}
                      placeholder="Usually empty"
                      className="w-full px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-gray-400"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-[var(--text-primary)] mb-1">
                      Default Query Instruction
                    </label>
                    <input
                      type="text"
                      value={defaultInstruction}
                      onChange={(e) => setDefaultInstruction(e.target.value)}
                      placeholder="For Qwen-style models"
                      className="w-full px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-gray-400"
                    />
                  </div>
                </>
              )}
            </div>
          )}

          {/* Footer */}
          <div className="flex justify-end gap-2 pt-4 border-t border-[var(--border)]">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 text-sm border border-[var(--border)] rounded hover:bg-[var(--bg-tertiary)]"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!modelId.trim() || isSubmitting}
              className="px-4 py-2 text-sm bg-gray-200 dark:bg-white text-gray-900 rounded hover:bg-gray-300 dark:hover:bg-gray-100 disabled:opacity-50 flex items-center gap-1"
            >
              {isSubmitting && <Loader2 className="w-4 h-4 animate-spin" />}
              Add Model
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add apps/webui-react/src/components/settings/models/AddCustomModelModal.tsx
git commit -m "feat(ui): add AddCustomModelModal component"
```

---

### Task 13: ModelsTab Component

**Files:**
- Create: `apps/webui-react/src/components/settings/models/ModelsTab.tsx`
- Create: `apps/webui-react/src/components/settings/models/index.ts`

**Step 1: Create the main tab component**

Create `apps/webui-react/src/components/settings/models/ModelsTab.tsx`:

```typescript
import React, { useState, useMemo } from 'react';
import { Search, Plus, HardDrive } from 'lucide-react';
import { useModels, useDownloadModel, useDeleteModel, useAddCustomModel } from '../../../hooks/useModels';
import type { ModelType, ModelFilterStatus, Model } from '../../../types/model';
import { MODEL_TYPE_ORDER, MODEL_TYPE_LABELS, formatSize } from '../../../types/model';
import { ModelCard } from './ModelCard';
import { AddCustomModelModal } from './AddCustomModelModal';

export function ModelsTab() {
  const [activeType, setActiveType] = useState<ModelType>('embedding');
  const [filterStatus, setFilterStatus] = useState<ModelFilterStatus>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);

  const { data, isLoading, error } = useModels(activeType);
  const { startDownload, activeDownloads, getProgress } = useDownloadModel();
  const deleteModel = useDeleteModel();
  const addCustomModel = useAddCustomModel();

  // Filter models based on status and search
  const filteredModels = useMemo(() => {
    if (!data?.models) return [];

    let models = data.models;

    // Filter by status
    if (filterStatus === 'installed') {
      models = models.filter((m) => m.is_installed);
    } else if (filterStatus === 'available') {
      models = models.filter((m) => !m.is_installed);
    }

    // Filter by search
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      models = models.filter(
        (m) =>
          m.name.toLowerCase().includes(query) ||
          m.id.toLowerCase().includes(query) ||
          m.description.toLowerCase().includes(query)
      );
    }

    return models;
  }, [data?.models, filterStatus, searchQuery]);

  const handleAddCustomModel = (request: Parameters<typeof addCustomModel.mutate>[0]) => {
    addCustomModel.mutate(request, {
      onSuccess: () => {
        setShowAddModal(false);
      },
    });
  };

  return (
    <div className="space-y-4">
      {/* Header with cache info */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-medium text-[var(--text-primary)]">Models</h2>
          <p className="text-sm text-[var(--text-muted)]">
            Manage embedding, LLM, reranker, and SPLADE models
          </p>
        </div>
        {data && (
          <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
            <HardDrive className="w-4 h-4" />
            <span>Total cache: {formatSize(data.total_cache_size_mb)}</span>
          </div>
        )}
      </div>

      {/* Model Type Tabs */}
      <div className="flex gap-1 border-b border-[var(--border)]">
        {MODEL_TYPE_ORDER.map((type) => (
          <button
            key={type}
            onClick={() => setActiveType(type)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeType === type
                ? 'text-[var(--text-primary)] border-b-2 border-gray-400 dark:border-white'
                : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
            }`}
          >
            {MODEL_TYPE_LABELS[type]}
          </button>
        ))}
      </div>

      {/* Search and Filters */}
      <div className="flex gap-4 items-center">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search models..."
            className="w-full pl-9 pr-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-gray-400"
          />
        </div>

        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value as ModelFilterStatus)}
          className="px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border)] rounded text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400"
        >
          <option value="all">All</option>
          <option value="installed">Installed</option>
          <option value="available">Available</option>
        </select>

        <button
          onClick={() => setShowAddModal(true)}
          className="px-4 py-2 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded hover:bg-gray-600 flex items-center gap-2 text-sm"
        >
          <Plus className="w-4 h-4" />
          Add Custom
        </button>
      </div>

      {/* Error State */}
      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded text-red-400">
          Failed to load models: {error.message}
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-400" />
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !error && filteredModels.length === 0 && (
        <div className="text-center py-12 text-[var(--text-muted)]">
          {searchQuery || filterStatus !== 'all'
            ? 'No models match your filters'
            : 'No models available'}
        </div>
      )}

      {/* Model Grid */}
      {!isLoading && !error && filteredModels.length > 0 && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredModels.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              downloadProgress={getProgress(model.id)}
              onDownload={startDownload}
              onDelete={(modelId) => deleteModel.mutate(modelId)}
              isDeleting={deleteModel.isPending}
            />
          ))}
        </div>
      )}

      {/* Add Custom Model Modal */}
      <AddCustomModelModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onSubmit={handleAddCustomModel}
        isSubmitting={addCustomModel.isPending}
        defaultModelType={activeType}
      />
    </div>
  );
}
```

**Step 2: Create index export**

Create `apps/webui-react/src/components/settings/models/index.ts`:

```typescript
export { ModelsTab } from './ModelsTab';
export { ModelCard } from './ModelCard';
export { AddCustomModelModal } from './AddCustomModelModal';
```

**Step 3: Commit**

```bash
git add apps/webui-react/src/components/settings/models/
git commit -m "feat(ui): add ModelsTab component for Settings page"
```

---

### Task 14: Integrate ModelsTab into Settings Page

**Files:**
- Modify: `apps/webui-react/src/pages/SettingsPage.tsx`

**Step 1: Add Models tab to SettingsPage**

Find the tabs definition in `SettingsPage.tsx` and add the Models tab. The exact changes depend on how the file is structured, but typically:

1. Import the ModelsTab component:
```typescript
import { ModelsTab } from '../components/settings/models';
```

2. Add to the tabs array/config:
```typescript
{ id: 'models', label: 'Models', icon: Box, component: ModelsTab }
```

3. Add the Box icon import from lucide-react if not already present.

**Step 2: Run the frontend build to verify**

Run: `cd apps/webui-react && npm run build`

**Step 3: Commit**

```bash
git add apps/webui-react/src/pages/SettingsPage.tsx
git commit -m "feat(ui): integrate ModelsTab into Settings page"
```

---

## Phase 3: Testing & Documentation

### Task 15: Backend Unit Tests

**Files:**
- Create: `tests/unit/test_model_registry.py`
- Create: `tests/unit/test_model_cache_manager.py`

**Step 1: Create registry tests**

Create `tests/unit/test_model_registry.py`:

```python
"""Tests for the unified model registry."""

import pytest

from shared.models import ModelType, get_curated_models, get_model_info, list_models_by_type


class TestModelRegistry:
    """Tests for model registry functions."""

    def test_get_curated_models_returns_list(self):
        """Should return a list of ModelInfo objects."""
        models = get_curated_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_curated_models_have_required_fields(self):
        """All models should have required fields."""
        models = get_curated_models()
        for model in models:
            assert model.id
            assert model.name
            assert model.description
            assert model.model_type in ModelType

    def test_list_models_by_type_embedding(self):
        """Should filter by embedding type."""
        models = list_models_by_type(ModelType.EMBEDDING)
        assert all(m.model_type == ModelType.EMBEDDING for m in models)
        assert len(models) > 0

    def test_list_models_by_type_llm(self):
        """Should filter by LLM type."""
        models = list_models_by_type(ModelType.LLM)
        assert all(m.model_type == ModelType.LLM for m in models)

    def test_get_model_info_found(self):
        """Should return model info for known model."""
        model = get_model_info("Qwen/Qwen3-Embedding-0.6B")
        assert model is not None
        assert model.id == "Qwen/Qwen3-Embedding-0.6B"
        assert model.dimension == 1024

    def test_get_model_info_not_found(self):
        """Should return None for unknown model."""
        model = get_model_info("unknown/model")
        assert model is None

    def test_embedding_models_have_dimension(self):
        """Embedding models should have dimension."""
        models = list_models_by_type(ModelType.EMBEDDING)
        for model in models:
            assert model.dimension is not None
            assert model.dimension > 0

    def test_llm_models_have_context_window(self):
        """LLM models should have context window."""
        models = list_models_by_type(ModelType.LLM)
        for model in models:
            assert model.context_window is not None
            assert model.context_window > 0
```

**Step 2: Run tests**

Run: `uv run pytest tests/unit/test_model_registry.py -v`

**Step 3: Commit**

```bash
git add tests/unit/test_model_registry.py
git commit -m "test: add unit tests for model registry"
```

---

### Task 16: Frontend Component Tests

**Files:**
- Create: `apps/webui-react/src/components/settings/models/__tests__/ModelCard.test.tsx`

**Step 1: Create component tests**

Create `apps/webui-react/src/components/settings/models/__tests__/ModelCard.test.tsx`:

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ModelCard } from '../ModelCard';
import type { Model } from '../../../../types/model';

const mockModel: Model = {
  id: 'test/model',
  name: 'Test Model',
  description: 'A test model',
  model_type: 'embedding',
  download_size_mb: 1000,
  memory_mb: { float16: 1000, int8: 500 },
  is_installed: false,
  size_on_disk_mb: null,
  used_by_collections: [],
  is_custom: false,
  custom_model_db_id: null,
  dimension: 768,
};

describe('ModelCard', () => {
  it('renders model name and description', () => {
    render(
      <ModelCard
        model={mockModel}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
      />
    );

    expect(screen.getByText('Test Model')).toBeInTheDocument();
    expect(screen.getByText('A test model')).toBeInTheDocument();
  });

  it('shows Download button for uninstalled models', () => {
    render(
      <ModelCard
        model={mockModel}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
      />
    );

    expect(screen.getByText('Download')).toBeInTheDocument();
  });

  it('shows Delete button for installed models', () => {
    const installedModel = { ...mockModel, is_installed: true, size_on_disk_mb: 1000 };
    render(
      <ModelCard
        model={installedModel}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
      />
    );

    expect(screen.getByText('Delete')).toBeInTheDocument();
  });

  it('calls onDownload when Download is clicked', () => {
    const onDownload = vi.fn();
    render(
      <ModelCard
        model={mockModel}
        onDownload={onDownload}
        onDelete={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText('Download'));
    expect(onDownload).toHaveBeenCalledWith('test/model');
  });

  it('disables Delete for models in use', () => {
    const inUseModel = {
      ...mockModel,
      is_installed: true,
      used_by_collections: ['Collection 1'],
    };
    render(
      <ModelCard
        model={inUseModel}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
      />
    );

    const deleteButton = screen.getByText('Delete').closest('button');
    expect(deleteButton).toBeDisabled();
  });

  it('shows download progress when downloading', () => {
    render(
      <ModelCard
        model={mockModel}
        downloadProgress={{
          task_id: '123',
          model_id: 'test/model',
          status: 'downloading',
          bytes_downloaded: 500 * 1024 * 1024,
          bytes_total: 1000 * 1024 * 1024,
        }}
        onDownload={vi.fn()}
        onDelete={vi.fn()}
      />
    );

    expect(screen.getByText('Downloading...')).toBeInTheDocument();
  });
});
```

**Step 2: Run tests**

Run: `cd apps/webui-react && npm test -- --run`

**Step 3: Commit**

```bash
git add apps/webui-react/src/components/settings/models/__tests__/
git commit -m "test: add unit tests for ModelCard component"
```

---

### Task 17: Update Documentation

**Files:**
- Modify: `docs/plans/2026-01-19-model-manager-design.md` (add implementation status)

**Step 1: Add implementation status to design doc**

Add a section at the top of the design doc:

```markdown
## Implementation Status

- [x] Phase 1: Backend Foundation
  - [x] Unified model registry YAML
  - [x] CustomModel database table
  - [x] CustomModelRepository
  - [x] HuggingFace cache manager
  - [x] Model download Celery task
  - [x] API schemas and router
- [x] Phase 2: Frontend Implementation
  - [x] TypeScript types
  - [x] API service
  - [x] React hooks
  - [x] ModelCard component
  - [x] AddCustomModelModal component
  - [x] ModelsTab component
  - [x] Settings page integration
- [x] Phase 3: Testing & Documentation
  - [x] Backend unit tests
  - [x] Frontend component tests
  - [x] Documentation updates
```

**Step 2: Commit**

```bash
git add docs/plans/2026-01-19-model-manager-design.md
git commit -m "docs: update model manager design with implementation status"
```

---

## Final Verification

### Task 18: End-to-End Verification

**Step 1: Run full test suite**

```bash
make check
```

**Step 2: Start the application and manually test**

```bash
make docker-dev-up
```

Then:
1. Navigate to Settings > Models
2. Verify model list loads
3. Test downloading a small model
4. Test adding a custom model
5. Test deleting a model (ensure in-use protection works)

**Step 3: Final commit with any fixes**

```bash
git add -A
git commit -m "fix: address issues found during e2e testing"
```

---

## Summary

This plan creates:

**Backend (14 files):**
- `packages/shared/models/model_registry.yaml` - Curated model definitions
- `packages/shared/models/__init__.py` - Module exports
- `packages/shared/models/registry.py` - Registry loader
- `packages/shared/models/cache_manager.py` - HuggingFace cache management
- `packages/shared/database/models.py` - CustomModel SQLAlchemy model (modified)
- `packages/shared/database/repositories/custom_model_repository.py` - Repository
- `packages/webui/api/v2/models_schemas.py` - Pydantic schemas
- `packages/webui/api/v2/models.py` - API router
- `packages/webui/tasks/model_download.py` - Celery download task
- Alembic migration for custom_models table

**Frontend (7 files):**
- `apps/webui-react/src/types/model.ts` - TypeScript types
- `apps/webui-react/src/services/api/v2/models.ts` - API service
- `apps/webui-react/src/hooks/useModels.ts` - React hooks
- `apps/webui-react/src/components/settings/models/ModelCard.tsx`
- `apps/webui-react/src/components/settings/models/AddCustomModelModal.tsx`
- `apps/webui-react/src/components/settings/models/ModelsTab.tsx`
- `apps/webui-react/src/components/settings/models/index.ts`

**Tests:**
- `tests/unit/test_model_registry.py`
- `apps/webui-react/src/components/settings/models/__tests__/ModelCard.test.tsx`

# CORE-002: Relocate Core Utilities to shared Package - Implementation Plan

## Current State
- The shared package structure has been created by CORE-001, but no files have been migrated yet
- Files to migrate are still in their original locations:
  - `packages/vecpipe/config.py` - Configuration settings
  - `packages/vecpipe/metrics.py` - Prometheus metrics 
  - `packages/vecpipe/extract_chunks.py` - Text extraction and chunking utilities
- 9 files import from vecpipe config/metrics
- 6 files import from webui.embedding_service (cross-package dependency)

## Migration Tasks

### 1. Migrate Configuration (packages/vecpipe/config.py → shared/config/)
- Create `packages/shared/config/base.py` with BaseConfig class
- Create `packages/shared/config/vecpipe.py` with VecpipeConfig class  
- Create `packages/shared/config/webui.py` with WebuiConfig class
- Move settings from vecpipe/config.py into appropriate classes
- Create `packages/shared/config/__init__.py` to export settings

### 2. Migrate Metrics (packages/vecpipe/metrics.py → shared/metrics/)
- Create `packages/shared/metrics/prometheus.py`
- Move all Prometheus metric definitions and helper functions
- Create `packages/shared/metrics/__init__.py` to export metrics

### 3. Migrate Text Processing (packages/vecpipe/extract_chunks.py → shared/text_processing/)
- Create `packages/shared/text_processing/extraction.py`
  - Move `extract_text()` and `extract_and_serialize()` functions
- Create `packages/shared/text_processing/chunking.py`  
  - Move `TokenChunker` class
- Create `packages/shared/text_processing/__init__.py` to export utilities

### 4. Create Import Update Script
- Create `scripts/refactoring/update_imports.py` with mappings:
  - `from vecpipe.config import` → `from shared.config import`
  - `from vecpipe.metrics import` → `from shared.metrics.prometheus import`
  - `from vecpipe.extract_chunks import TokenChunker` → `from shared.text_processing.chunking import TokenChunker`
  - `from vecpipe.extract_chunks import extract_text` → `from shared.text_processing.extraction import extract_text`

### 5. Execute Migration
- Run the import update script on all Python files
- Remove or rename the original files to prevent confusion
- Ensure all __init__.py files properly export the relocated modules

### 6. Validation
- Run `make format` to ensure code formatting
- Run `make lint` to check for issues
- Run `make type-check` to verify type hints
- Run `make test` to ensure all tests pass

## Files to be Modified
- Create: 10 new files in shared package
- Update imports in: ~15 files across vecpipe and webui packages
- Remove/rename: 3 original files after migration

This plan follows the REFACTORING_PLAN.md specifications and will establish the shared package as the central location for cross-cutting concerns.
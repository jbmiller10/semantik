# syntax=docker/dockerfile:1
ARG PYTHON_VERSION="3.12"
ARG NODE_VERSION="20"

# ============================================
# Stage 1: Build React Frontend
# ============================================
FROM node:${NODE_VERSION}-alpine AS frontend-builder
WORKDIR /build

# Copy package files for better caching
COPY apps/webui-react/package*.json ./apps/webui-react/
WORKDIR /build/apps/webui-react
RUN npm ci

# Copy frontend source
COPY apps/webui-react/ ./

# Build will output to /build/packages/webui/static due to vite config
RUN npm run build

# ============================================
# Stage 2: Python Dependencies Builder
# ============================================
FROM python:${PYTHON_VERSION}-slim AS python-builder
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN python -m venv $POETRY_HOME && \
    $POETRY_HOME/bin/pip install poetry==$POETRY_VERSION

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies (without creating virtual env since we're in container)
ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry install --no-root --only main

# ============================================
# Stage 3: Final Runtime Image
# ============================================
FROM python:${PYTHON_VERSION}-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for unstructured document processing
    libmagic1 \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    # Required for some Python packages
    libpq5 \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
# Python installations use major.minor version in paths (e.g., python3.12 not python3.12.11)
COPY --from=python-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY packages/ ./packages/

# Copy built frontend to webui static directory
COPY --from=frontend-builder /build/packages/webui/static ./packages/webui/static/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data/jobs \
    /app/data/ingest \
    /app/data/extract \
    /app/data/loaded \
    /app/data/rejects \
    /app/data/output \
    && chown -R appuser:appuser /app/data /app/logs

USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/packages

# Create entrypoint script
COPY --chown=appuser:appuser docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Default to running the webui service
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["webui"]
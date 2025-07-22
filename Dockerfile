# syntax=docker/dockerfile:1
ARG PYTHON_VERSION="3.11"
ARG NODE_VERSION="20"
ARG CUDA_VERSION="12.1.0"

# ============================================
# Stage 1: Build React Frontend
# ============================================
FROM node:${NODE_VERSION}-alpine AS frontend-builder
WORKDIR /build

# Copy package files for better caching
COPY apps/webui-react/package*.json ./apps/webui-react/
WORKDIR /build/apps/webui-react
RUN npm install

# Copy frontend source
COPY apps/webui-react/ ./

# Build will output to /build/packages/webui/static due to vite config
RUN npm run build

# ============================================
# Stage 2: Python Dependencies Builder
# ============================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS python-builder
ARG PYTHON_VERSION
WORKDIR /app

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

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
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS runtime
ARG PYTHON_VERSION

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set CUDA environment variables early for bitsandbytes
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    # Required for unstructured document processing
    libmagic1 \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    # Required for PostgreSQL
    libpq5 \
    # Required for bitsandbytes (INT8 quantization)
    libblas3 \
    liblapack3 \
    libcusparse11 \
    libcublas11 \
    # C compiler for bitsandbytes JIT compilation
    gcc \
    g++ \
    # Required for healthchecks
    wget \
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

WORKDIR /app

# Copy Python packages from builder
# Poetry installs to dist-packages on Ubuntu with system Python
COPY --from=python-builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY packages/ ./packages/

# Copy alembic configuration and migrations
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Copy built frontend to webui static directory
COPY --from=frontend-builder /build/packages/webui/static ./packages/webui/static/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /app/alembic /app/alembic.ini

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data/operations \
    /app/data/ingest \
    /app/data/extract \
    /app/data/loaded \
    /app/data/rejects \
    /app/data/output \
    && chown -R appuser:appuser /app/data /app/logs

# Create symbolic links for CUDA libraries if needed
RUN ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so || true && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcusparse.so.11 /usr/local/cuda/lib64/libcusparse.so.11 || true && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so.11 || true && \
    ldconfig

# Test bitsandbytes installation (as root for library access)
RUN python -c "import bitsandbytes; print('Bitsandbytes loaded successfully')" || \
    (echo "WARNING: Bitsandbytes test failed, INT8 quantization may not work" && exit 0)

USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/packages
# Ensure CUDA libraries are available for bitsandbytes
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
# C compiler for bitsandbytes JIT compilation
ENV CC=gcc
ENV CXX=g++

# Create entrypoint script
COPY --chown=appuser:appuser docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Default to running the webui service
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["webui"]
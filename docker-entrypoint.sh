#!/bin/bash
set -e

# Set Python path to include the packages directory
export PYTHONPATH="/app/packages:${PYTHONPATH}"

# Set C compiler for bitsandbytes JIT compilation if not already set
if [ -z "$CC" ]; then
    export CC=gcc
    export CXX=g++
fi

# Function to run strict environment validation via scripts.validate_env
run_strict_env_validation() {
    local service=$1
    local flower_enabled=${2:-true}

    if ! FLOWER_ENABLED="$flower_enabled" python /app/scripts/validate_env.py --strict; then
        echo "ERROR: Environment validation failed for service '$service'." >&2
        echo "Fix the configuration issues reported above and try again." >&2
        exit 1
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    echo "Waiting for $service_name at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if python -c "import socket; socket.create_connection(('$host', $port), timeout=1).close()" 2>/dev/null; then
            echo "$service_name is ready!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: $service_name failed to start after $max_attempts attempts"
    return 1
}

# Determine which service to run based on the first argument
SERVICE=${1:-webui}

case "$SERVICE" in
    webui)
        run_strict_env_validation "webui" false
        echo "Starting WebUI service..."
        
        # Wait for Search API to be ready
        if [ "${WAIT_FOR_SEARCH_API:-true}" = "true" ]; then
            # Use Docker service name if available, otherwise fallback to localhost
            SEARCH_HOST="${SEARCH_API_HOST:-vecpipe}"
            wait_for_service "$SEARCH_HOST" 8000 "Search API"
        fi
        
        # Run database migrations using Alembic
        echo "Running database migrations..."
        alembic upgrade head
        
        # Start the WebUI service
        if [ "${WEBUI_RELOAD:-false}" = "true" ]; then
            echo "Starting WebUI in development mode with auto-reload..."
            exec uvicorn webui.main:app \
                --host 0.0.0.0 \
                --port "${WEBUI_PORT:-8080}" \
                --reload \
                --log-level "${LOG_LEVEL:-info}"
        else
            exec uvicorn webui.main:app \
                --host 0.0.0.0 \
                --port "${WEBUI_PORT:-8080}" \
                --workers "${WEBUI_WORKERS:-1}" \
                --log-level "${LOG_LEVEL:-info}"
        fi
        ;;
        
    vecpipe)
        run_strict_env_validation "vecpipe" false

        if [ -z "${QDRANT_HOST:-}" ] || [ -z "${QDRANT_PORT:-}" ]; then
            echo "ERROR: Vecpipe requires both QDRANT_HOST and QDRANT_PORT to be set." >&2
            echo "Provide the Qdrant connection details via environment variables and try again." >&2
            exit 1
        fi

        echo "Starting Search API service..."

        # Wait for Qdrant to be ready
        if [ "${WAIT_FOR_QDRANT:-true}" = "true" ]; then
            wait_for_service "${QDRANT_HOST:-localhost}" "${QDRANT_PORT:-6333}" "Qdrant"
        fi

        # Ensure Hugging Face cache is writable and clean up stale lock files
        CACHE_ROOT="${HF_HOME:-/app/.cache/huggingface}"
        if ! mkdir -p "$CACHE_ROOT/hub" 2>/dev/null; then
            echo "Warning: unable to create $CACHE_ROOT, falling back to /tmp/huggingface-cache"
            CACHE_ROOT="/tmp/huggingface-cache"
            export HF_HOME="$CACHE_ROOT"
            export TRANSFORMERS_CACHE="$CACHE_ROOT"
            mkdir -p "$CACHE_ROOT/hub"
        fi
        find "$CACHE_ROOT" -name "*.lock" -type f -delete 2>/dev/null || true

        # Start the Search API service
        exec python -m vecpipe.search_api
        ;;
        
    worker)
        run_strict_env_validation "worker" false
        echo "Starting Celery worker..."
        # Choose a sensible default: all available CPU cores minus one, capped by CELERY_MAX_CONCURRENCY (if set)
        # This avoids oversubscribing memory when the container can see many host CPUs.
        if [ -z "${CELERY_CONCURRENCY:-}" ]; then
            if command -v nproc >/dev/null 2>&1; then
                _cores=$(nproc)
            else
                _cores=$(python - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)
            fi

            if [ "${_cores}" -gt 1 ] 2>/dev/null; then
                CELERY_CONCURRENCY=$((_cores - 1))
            else
                CELERY_CONCURRENCY=1
            fi

            # Optional safety cap
            if [ -n "${CELERY_MAX_CONCURRENCY:-}" ]; then
                # shellcheck disable=SC2072
                if [ "${CELERY_CONCURRENCY}" -gt "${CELERY_MAX_CONCURRENCY}" ]; then
                    CELERY_CONCURRENCY=${CELERY_MAX_CONCURRENCY}
                fi
            fi
        fi

        echo "Using Celery concurrency=${CELERY_CONCURRENCY}"
        exec celery -A webui.celery_app worker --loglevel=info --concurrency="${CELERY_CONCURRENCY}"
        ;;
        
    flower)
        run_strict_env_validation "flower"

        if [ -z "${FLOWER_USERNAME:-}" ] || [ -z "${FLOWER_PASSWORD:-}" ]; then
            echo "ERROR: Flower requires FLOWER_USERNAME and FLOWER_PASSWORD to be set." >&2
            echo "Run 'make wizard' to generate secure credentials and re-run the container." >&2
            exit 1
        fi
        echo "Starting Flower monitoring..."
        exec celery -A webui.celery_app flower \
            --broker=redis://redis:6379/0 \
            --basic_auth="${FLOWER_USERNAME}:${FLOWER_PASSWORD}"
        ;;
        
    *)
        echo "Unknown service: $SERVICE"
        echo "Usage: $0 [webui|vecpipe|worker|flower]"
        exit 1
        ;;
esac

#!/bin/bash
set -e

# Set Python path to include the packages directory
export PYTHONPATH="/app/packages:${PYTHONPATH}"

# Set C compiler for bitsandbytes JIT compilation if not already set
if [ -z "$CC" ]; then
    export CC=gcc
    export CXX=g++
fi

# Function to validate required environment variables
validate_env_vars() {
    local service=$1
    local missing_vars=()
    
    case "$service" in
        webui)
            # Critical environment variables for webui
            if [ -z "$JWT_SECRET_KEY" ] || [ "$JWT_SECRET_KEY" = "CHANGE_THIS_TO_A_STRONG_SECRET_KEY" ]; then
                echo "ERROR: JWT_SECRET_KEY must be set to a secure value for webui service"
                echo "Generate one with: openssl rand -hex 32"
                exit 1
            fi
            ;;
        vecpipe)
            # Validate Qdrant connection
            if [ -z "$QDRANT_HOST" ]; then
                missing_vars+=("QDRANT_HOST")
            fi
            if [ -z "$QDRANT_PORT" ]; then
                missing_vars+=("QDRANT_PORT")
            fi
            ;;
    esac
    
    # Check for common required variables
    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "ERROR: Missing required environment variables for $service:"
        printf '%s\n' "${missing_vars[@]}"
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

# Validate environment variables for the service
validate_env_vars "$SERVICE"

case "$SERVICE" in
    webui)
        echo "Starting WebUI service..."
        
        # Wait for Search API to be ready
        if [ "${WAIT_FOR_SEARCH_API:-true}" = "true" ]; then
            # Use Docker service name if available, otherwise fallback to localhost
            SEARCH_HOST="${SEARCH_API_HOST:-vecpipe}"
            wait_for_service "$SEARCH_HOST" 8000 "Search API"
        fi
        
        # Run database migrations if needed
        echo "Setting up database..."
        python -c "from shared.database import init_db; init_db()"
        
        # Start the WebUI service
        exec uvicorn webui.main:app \
            --host 0.0.0.0 \
            --port "${WEBUI_PORT:-8080}" \
            --workers "${WEBUI_WORKERS:-1}"
        ;;
        
    vecpipe)
        echo "Starting Search API service..."
        
        # Wait for Qdrant to be ready
        if [ "${WAIT_FOR_QDRANT:-true}" = "true" ]; then
            wait_for_service "${QDRANT_HOST:-localhost}" "${QDRANT_PORT:-6333}" "Qdrant"
        fi
        
        # Start the Search API service
        exec python -m vecpipe.search_api
        ;;
        
    worker)
        echo "Starting background worker..."
        # This could be used for future background processing tasks
        exec python -m vecpipe.worker
        ;;
        
    *)
        echo "Unknown service: $SERVICE"
        echo "Usage: $0 [webui|vecpipe|worker]"
        exit 1
        ;;
esac
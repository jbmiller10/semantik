#!/bin/bash
set -e

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
        echo "Starting WebUI service..."
        
        # Wait for Search API to be ready
        if [ "${WAIT_FOR_SEARCH_API:-true}" = "true" ]; then
            # Use Docker service name if available, otherwise fallback to localhost
            SEARCH_HOST="${SEARCH_API_HOST:-vecpipe}"
            wait_for_service "$SEARCH_HOST" 8000 "Search API"
        fi
        
        # Run database migrations if needed
        echo "Setting up database..."
        python -c "from webui.database import create_tables; create_tables()"
        
        # Start the WebUI service
        exec uvicorn packages.webui.main:app \
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
        exec python -m packages.vecpipe.search_api
        ;;
        
    worker)
        echo "Starting background worker..."
        # This could be used for future background processing tasks
        exec python -m packages.vecpipe.worker
        ;;
        
    *)
        echo "Unknown service: $SERVICE"
        echo "Usage: $0 [webui|vecpipe|worker]"
        exit 1
        ;;
esac
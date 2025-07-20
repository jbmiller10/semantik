#!/bin/bash
# Convenience script to run docker compose with CUDA support

# Check if docker compose v2 is available
if docker compose version >/dev/null 2>&1; then
    docker compose -f docker-compose.yml -f docker-compose.cuda.yml "$@"
else
    echo "Error: Docker Compose v2 is required"
    exit 1
fi
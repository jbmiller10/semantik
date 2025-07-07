#!/bin/bash
# Validation script for Docker setup

echo "=== Docker Setup Validation ==="
echo

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed"
    docker --version
else
    echo "✗ Docker is not installed"
    echo "  Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo "✓ Docker Compose is installed"
    if command -v docker-compose &> /dev/null; then
        docker-compose --version
    else
        docker compose version
    fi
else
    echo "✗ Docker Compose is not installed"
    echo "  Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if .env file exists
if [ -f .env ]; then
    echo "✓ .env file exists"
else
    echo "✗ .env file not found"
    echo "  Creating from template..."
    cp .env.docker.example .env
    echo "  Please edit .env with your configuration"
fi

# Validate docker-compose.yml
echo
echo "Validating docker-compose.yml..."
if docker-compose config > /dev/null 2>&1 || docker compose config > /dev/null 2>&1; then
    echo "✓ docker-compose.yml is valid"
else
    echo "✗ docker-compose.yml validation failed"
    exit 1
fi

# Check required directories
echo
echo "Checking required directories..."
for dir in data logs; do
    if [ -d "$dir" ]; then
        echo "✓ Directory '$dir' exists"
    else
        echo "✗ Directory '$dir' not found, creating..."
        mkdir -p "$dir"
    fi
done

# Check Docker daemon
echo
echo "Checking Docker daemon..."
if docker info > /dev/null 2>&1; then
    echo "✓ Docker daemon is running"
else
    echo "✗ Docker daemon is not running"
    echo "  Please start Docker"
    exit 1
fi

echo
echo "=== Validation Complete ==="
echo
echo "To start Semantik, run:"
echo "  make docker-up"
echo "or"
echo "  docker-compose up -d"
echo
echo "The application will be available at:"
echo "  - WebUI: http://localhost:8080"
echo "  - Search API: http://localhost:8000"
echo "  - Qdrant: http://localhost:6333"
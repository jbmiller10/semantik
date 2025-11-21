#!/bin/bash
# Start local development environment with webui running locally
# and supporting services in Docker

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Semantik local development environment...${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env from .env.docker.example...${NC}"
    cp .env.docker.example .env
fi

# Check if .env.local exists
if [ ! -f .env.local ]; then
    echo -e "${YELLOW}Creating .env.local from .env.local.example...${NC}"
    cp .env.local.example .env.local
    echo -e "${RED}IMPORTANT: Please edit .env.local and update:${NC}"
    echo "  - POSTGRES_PASSWORD to match your .env file"
    echo "  - JWT_SECRET_KEY to match your .env file"
    echo "Then run this script again."
    exit 1
fi

# Start Docker services (backend only, without webui)
echo -e "${GREEN}Starting Docker services (Qdrant, PostgreSQL, Redis, VecPipe, Worker)...${NC}"
docker compose --profile backend up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 5

# Check if services are healthy
echo -e "${GREEN}Checking service health...${NC}"
docker compose --profile backend ps

# Export environment variables from .env.local
echo -e "${GREEN}Loading local environment variables...${NC}"
export $(cat .env.local | grep -v '^#' | xargs)

# Run database migrations
echo -e "${GREEN}Running database migrations...${NC}"
uv run alembic upgrade head

# Start the webui locally with hot reload
echo -e "${GREEN}Starting WebUI locally with hot reload...${NC}"
echo -e "${YELLOW}WebUI will be available at: http://localhost:8080${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"

# Run with uv to ensure correct environment
PYTHONPATH=packages:${PYTHONPATH:-} uv run uvicorn webui.main:app --host 0.0.0.0 --port 8080 --reload

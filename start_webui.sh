#!/bin/bash
# Start the Document Embedding Web UI

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a # automatically export all variables
  source .env
  set +a
fi

# Use PROJECT_ROOT if set, otherwise use current directory
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname $(readlink -f $0))}"
cd "$PROJECT_ROOT"

# Kill any existing process on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

echo "Starting Document Embedding Web UI..."
echo "Access the interface at: http://localhost:8080"
echo ""

# Run the web UI
poetry run uvicorn webui.app:app --host 0.0.0.0 --port 8080
#!/bin/bash
# Script to run the Web UI

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a # automatically export all variables
  source .env
  set +a
fi

# Run the web UI with poetry
# Use PROJECT_ROOT if set, otherwise use current directory
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname $(readlink -f $0))}"
cd "$PROJECT_ROOT"
poetry run python -m uvicorn webui.main:app --host 0.0.0.0 --port 8080 --reload
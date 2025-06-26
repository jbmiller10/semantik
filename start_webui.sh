#!/bin/bash
# Start the Document Embedding Web UI

cd /root/document-embedding-project

# Kill any existing process on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

echo "Starting Document Embedding Web UI..."
echo "Access the interface at: http://localhost:8080"
echo ""

# Run the web UI with SQLite-based job tracking
python3 -m uvicorn webui.app:app --host 0.0.0.0 --port 8080
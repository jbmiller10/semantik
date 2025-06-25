#!/bin/bash
# Start the Document Embedding Web UI

cd /root/document-embedding-project

# Kill any existing process on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

echo "Starting Document Embedding Web UI..."
echo "Access the interface at: http://localhost:8080"
echo ""

# Run the simplified version (no SQLite dependency)
python3 webui/app_simple.py
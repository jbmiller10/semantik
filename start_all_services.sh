#!/bin/bash
# Start all services for the document embedding system

echo "Starting Document Embedding Services..."
echo "======================================"

# Check if services are already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is already in use (Search API)"
    echo "   Please stop the existing service first"
    exit 1
fi

if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8080 is already in use (WebUI)"
    echo "   Please stop the existing service first"
    exit 1
fi

# Start Search API in background
echo "Starting Search API on port 8000..."
poetry run python -m packages.vecpipe.search_api > search_api.log 2>&1 &
SEARCH_PID=$!
echo "Search API started with PID: $SEARCH_PID"

# Wait a bit for Search API to initialize
echo "Waiting for Search API to initialize..."
sleep 5

# Check if Search API started successfully
if ! kill -0 $SEARCH_PID 2>/dev/null; then
    echo "❌ Search API failed to start. Check search_api.log for details"
    exit 1
fi

# Start WebUI
echo "Starting WebUI on port 8080..."
poetry run uvicorn packages.webui.app:app --host 0.0.0.0 --port 8080 > webui.log 2>&1 &
WEBUI_PID=$!
echo "WebUI started with PID: $WEBUI_PID"

# Save PIDs to file for easy shutdown
echo $SEARCH_PID > .search_api.pid
echo $WEBUI_PID > .webui.pid

echo ""
echo "✅ Services started successfully!"
echo "======================================"
echo "Search API: http://localhost:8000"
echo "WebUI:      http://localhost:8080"
echo ""
echo "Logs:"
echo "  Search API: tail -f search_api.log"
echo "  WebUI:      tail -f webui.log"
echo ""
echo "To stop all services, run: ./stop_all_services.sh"
echo ""
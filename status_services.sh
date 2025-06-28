#!/bin/bash
# Check status of all services

echo "Document Embedding Services Status"
echo "=================================="
echo ""

# Function to check service status
check_service() {
    local service_name=$1
    local port=$2
    local pid_file=$3
    
    echo -n "$service_name: "
    
    # Check by PID file
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 $PID 2>/dev/null; then
            echo "✅ Running (PID: $PID)"
        else
            echo "❌ Not running (stale PID file)"
        fi
    else
        # Check by port
        PORT_PID=$(lsof -t -i:$port -sTCP:LISTEN 2>/dev/null)
        if [ ! -z "$PORT_PID" ]; then
            echo "✅ Running on port $port (PID: $PORT_PID)"
        else
            echo "❌ Not running"
        fi
    fi
}

# Check services
check_service "Search API (port 8000)" 8000 ".search_api.pid"
check_service "WebUI (port 8080)     " 8080 ".webui.pid"

echo ""
echo "URLs:"
echo "  Search API: http://localhost:8000"
echo "  WebUI:      http://localhost:8080"
echo ""

# Check if log files exist
if [ -f "search_api.log" ] || [ -f "webui.log" ]; then
    echo "Log files:"
    [ -f "search_api.log" ] && echo "  Search API: search_api.log ($(wc -l < search_api.log) lines)"
    [ -f "webui.log" ] && echo "  WebUI: webui.log ($(wc -l < webui.log) lines)"
    echo ""
fi
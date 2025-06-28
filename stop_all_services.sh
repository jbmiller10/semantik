#!/bin/bash
# Stop all services for the document embedding system

echo "Stopping Document Embedding Services..."
echo "======================================"

# Function to stop a service
stop_service() {
    local service_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 $PID 2>/dev/null; then
            echo "Stopping $service_name (PID: $PID)..."
            kill $PID
            sleep 1
            
            # Force kill if still running
            if kill -0 $PID 2>/dev/null; then
                echo "Force stopping $service_name..."
                kill -9 $PID
            fi
            
            echo "✅ $service_name stopped"
        else
            echo "⚠️  $service_name not running (stale PID file)"
        fi
        rm -f "$pid_file"
    else
        echo "⚠️  No PID file found for $service_name"
    fi
}

# Stop services
stop_service "Search API" ".search_api.pid"
stop_service "WebUI" ".webui.pid"

# Also try to stop by port if PID files are missing
echo ""
echo "Checking for services on ports..."

# Check port 8000 (Search API)
SEARCH_PID=$(lsof -t -i:8000 -sTCP:LISTEN)
if [ ! -z "$SEARCH_PID" ]; then
    echo "Found process on port 8000 (PID: $SEARCH_PID), stopping..."
    kill $SEARCH_PID
fi

# Check port 8080 (WebUI)
WEBUI_PID=$(lsof -t -i:8080 -sTCP:LISTEN)
if [ ! -z "$WEBUI_PID" ]; then
    echo "Found process on port 8080 (PID: $WEBUI_PID), stopping..."
    kill $WEBUI_PID
fi

echo ""
echo "✅ All services stopped"
echo "======================================"
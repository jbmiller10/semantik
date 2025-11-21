#!/bin/bash
set -e

echo "Starting Semantik development environment..."

# Function to kill all background processes on exit
cleanup() {
    echo "Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup EXIT INT TERM

# Start the backend services
echo "Starting backend services..."
uv run uvicorn webui.main:app --host 0.0.0.0 --port 8080 --reload &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 3

# Start the frontend dev server
echo "Starting frontend dev server..."
cd apps/webui-react
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo "Development servers running:"
echo "  Backend API: http://localhost:8080"
echo "  Frontend Dev: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for any process to exit
wait

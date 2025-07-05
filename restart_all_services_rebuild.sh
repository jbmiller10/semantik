#!/bin/bash
# Restart all services with UI rebuild

echo "Restarting Document Embedding Services with UI Rebuild..."
echo "========================================================"
echo ""

# Stop existing services
echo "📛 Stopping existing services..."
./stop_all_services.sh

# Wait a moment for ports to be released
echo ""
echo "⏳ Waiting for ports to be released..."
sleep 2

# Start services with rebuild
echo ""
echo "🚀 Starting services with UI rebuild..."
./start_all_services_rebuild.sh
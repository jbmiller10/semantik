#!/bin/bash
set -e

echo "Building VecPipe project..."

# Build the React frontend
echo "Building frontend..."
cd apps/webui-react
npm install
npm run build
cd ../..

echo "Build complete! Frontend assets are in packages/webui/static/"
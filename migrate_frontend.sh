#!/bin/bash
# Script to migrate from old static files to new SvelteKit frontend

echo "Starting frontend migration..."

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend && npm install || exit 1
cd ..

# Build the frontend
echo "Building SvelteKit frontend..."
cd frontend && npm run build || exit 1
cd ..

# Backup old static files
echo "Backing up old static files..."
if [ -d "webui/static" ]; then
    mv webui/static webui/static_backup_$(date +%Y%m%d_%H%M%S)
fi

# Copy new build output
echo "Copying new build files..."
cp -r frontend/build webui/static

echo "Migration complete!"
echo ""
echo "The old static files have been backed up to webui/static_backup_*"
echo "The new SvelteKit frontend is now in webui/static"
echo ""
echo "To run the application:"
echo "  ./run_webui.sh"
echo ""
echo "To run frontend in development mode:"
echo "  make frontend-dev"
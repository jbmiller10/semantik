#!/bin/bash
# Clean up old/duplicate log files

echo "Cleaning up old log files..."

# Remove duplicate search API log files (keeping only search_api.log)
rm -f search_api_new.log
rm -f search_api_mock.log
rm -f search_api_real.log
rm -f search_api_startup.log

echo "✅ Removed duplicate search API log files"

# List remaining log files
echo ""
echo "Remaining log files:"
ls -la *.log 2>/dev/null || echo "No log files found"

echo ""
echo "Log file cleanup complete!"
#!/bin/bash
# Manifest generation script (VS-010)
# Finds all eligible documents and creates null-delimited file list

# Configuration
ROOTS=(
    "/mnt/zfs/docs/dirA"
    "/mnt/zfs/docs/dirB" 
    "/mnt/zfs/docs/dirC"
)
OUTPUT_FILE="/var/embeddings/filelist.null"
TEMP_FILE="/var/embeddings/filelist.null.tmp"

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Start
log "Starting manifest generation..."

# Find all eligible files
# Using -print0 for null-delimited output
# Case-insensitive matching for extensions
find "${ROOTS[@]}" -type f \
    \( -iname '*.pdf' -o -iname '*.docx' -o -iname '*.doc' -o -iname '*.txt' -o -iname '*.text' \) \
    -print0 > "$TEMP_FILE" 2>/dev/null || true

# Count files
FILE_COUNT=$(tr -cd '\0' < "$TEMP_FILE" | wc -c)

# Move temp file to final location
mv "$TEMP_FILE" "$OUTPUT_FILE"

log "Manifest generation complete. Found $FILE_COUNT files."
log "Output saved to: $OUTPUT_FILE"

# Exit successfully
exit 0
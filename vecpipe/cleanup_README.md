# Qdrant Vector Cleanup Service

## Overview

The cleanup service automatically removes vectors from Qdrant collections when their source documents are deleted from the filesystem. This prevents "ghost" search results that point to non-existent files.

## Components

### 1. `cleanup.py` - Main cleanup script
- Identifies deleted documents using `FileChangeTracker`
- Removes corresponding vectors from all Qdrant collections
- Logs cleanup actions and statistics
- Supports dry-run mode for testing

### 2. `cleanup.service` - Systemd service unit
- Runs the cleanup script as a oneshot service
- Configured with resource limits (2GB memory, 50% CPU)
- Runs as `embeduser` for security

### 3. `cleanup.timer` - Systemd timer unit
- Triggers cleanup daily at 2 AM
- Includes randomized delay to prevent load spikes
- Persistent - runs missed executions on startup

## Installation

1. Copy systemd units to system directory:
```bash
sudo cp deploy/systemd/cleanup.service /etc/systemd/system/
sudo cp deploy/systemd/cleanup.timer /etc/systemd/system/
```

2. Reload systemd and enable timer:
```bash
sudo systemctl daemon-reload
sudo systemctl enable cleanup.timer
sudo systemctl start cleanup.timer
```

3. Check timer status:
```bash
sudo systemctl status cleanup.timer
sudo systemctl list-timers cleanup.timer
```

## Manual Execution

### Dry run (no deletions):
```bash
python3 /opt/vecpipe/cleanup.py --dry-run
```

### Full cleanup:
```bash
python3 /opt/vecpipe/cleanup.py
```

### Custom file list:
```bash
python3 /opt/vecpipe/cleanup.py --file-list /path/to/filelist.null
```

## How It Works

1. **File Detection**: Reads current file list from `/var/embeddings/filelist.null`
2. **Change Tracking**: Uses `FileChangeTracker` to compare against previously seen files
3. **Collection Discovery**: Queries `webui.db` to find all job collections
4. **Vector Deletion**: For each deleted file:
   - Extracts the `doc_id` from tracking database
   - Deletes all points with matching `doc_id` from each collection
   - Updates tracking database to remove the file entry

## Logging

- Service logs: `journalctl -u cleanup.service`
- Cleanup history: `/var/embeddings/cleanup.log` (JSON format)
- Error log: Standard systemd journal

## Testing

Run the test script to verify functionality:
```bash
python3 /opt/vecpipe/test_cleanup.py
```

This performs:
- FileChangeTracker integration tests
- Dry-run cleanup simulation
- Collection discovery test

## Monitoring

Check cleanup history:
```bash
tail -f /var/embeddings/cleanup.log | jq '.'
```

Example output:
```json
{
  "timestamp": "2024-01-15T02:00:00.123456",
  "removed_files": 5,
  "deleted_points": 127,
  "by_collection": {
    "work_docs": 42,
    "job_abc123": 85
  },
  "dry_run": false
}
```

## Troubleshooting

1. **No files detected**: Ensure `/var/embeddings/filelist.null` exists and is up-to-date
2. **Connection errors**: Check Qdrant is running and accessible
3. **Permission errors**: Verify `embeduser` has read access to required files
4. **Missing collections**: Collections may have been deleted manually - this is logged but not an error

## Configuration

Environment variables (set in systemd service):
- `QDRANT_HOST`: Qdrant server address (default: 192.168.1.173)
- `QDRANT_PORT`: Qdrant server port (default: 6333)

File paths:
- WebUI database: `/var/embeddings/webui.db`
- File tracking: `/var/embeddings/file_tracking.json`
- File list: `/var/embeddings/filelist.null`
- Cleanup log: `/var/embeddings/cleanup.log`
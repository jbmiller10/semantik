#!/usr/bin/env python3
"""
Manual cleanup script for temporary image directories
Can be run via cron or manually to clean up expired sessions
"""

import os
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
TEMP_IMAGE_DIR = Path("/tmp/webui_temp_images")
TTL_HOURS = 1  # Clean up directories older than this

def cleanup_old_directories():
    """Remove temporary image directories older than TTL"""
    
    if not TEMP_IMAGE_DIR.exists():
        print(f"Temporary image directory does not exist: {TEMP_IMAGE_DIR}")
        return
    
    current_time = time.time()
    ttl_seconds = TTL_HOURS * 3600
    
    cleaned_count = 0
    error_count = 0
    total_size = 0
    
    print(f"Scanning {TEMP_IMAGE_DIR} for directories older than {TTL_HOURS} hour(s)...")
    
    # Iterate through all subdirectories
    for session_dir in TEMP_IMAGE_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        
        try:
            # Get directory modification time
            dir_mtime = session_dir.stat().st_mtime
            age_seconds = current_time - dir_mtime
            age_hours = age_seconds / 3600
            
            if age_seconds > ttl_seconds:
                # Calculate directory size before deletion
                dir_size = sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file())
                total_size += dir_size
                
                # Remove the directory
                shutil.rmtree(session_dir)
                cleaned_count += 1
                
                print(f"  ✓ Removed {session_dir.name} (age: {age_hours:.1f}h, size: {dir_size:,} bytes)")
            else:
                print(f"  - Keeping {session_dir.name} (age: {age_hours:.1f}h)")
                
        except Exception as e:
            error_count += 1
            print(f"  ✗ Error cleaning {session_dir.name}: {e}")
    
    # Summary
    print(f"\nCleanup complete:")
    print(f"  Directories removed: {cleaned_count}")
    print(f"  Total space freed: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
    if error_count > 0:
        print(f"  Errors encountered: {error_count}")

def main():
    """Main function"""
    print(f"Temporary Image Cleanup Script")
    print(f"==============================")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TTL: {TTL_HOURS} hour(s)")
    print()
    
    cleanup_old_directories()

if __name__ == "__main__":
    main()
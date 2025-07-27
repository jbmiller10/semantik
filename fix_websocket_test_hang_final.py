#!/usr/bin/env python3
"""
Final fix for the websocket test hang issue.

The problem: test_consume_updates_redis_none is hanging because there's a global
singleton ws_manager that might have tasks running from previous tests.
"""

# Add this to the beginning of test_websocket_manager.py


# Force cleanup of the singleton at import time
try:
    from packages.webui.websocket_manager import ws_manager

    # Cancel all tasks in the singleton
    for task_id, task in list(ws_manager.consumer_tasks.items()):
        if not task.done():
            task.cancel()

    # Clear all state
    ws_manager.consumer_tasks.clear()
    ws_manager.connections.clear()
    ws_manager.redis = None

    print("Cleaned up ws_manager singleton")
except Exception as e:
    print(f"Error cleaning up ws_manager: {e}")

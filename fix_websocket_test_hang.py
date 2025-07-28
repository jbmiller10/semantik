#!/usr/bin/env python3
"""
Debug script to identify what's causing the test hang in websocket_manager tests.
"""

import asyncio


async def check_running_tasks() -> None:
    """Check for any running asyncio tasks."""
    tasks = asyncio.all_tasks()
    print(f"Total active tasks: {len(tasks)}")
    for task in tasks:
        print(f"  Task: {task}")
        print(f"    Done: {task.done()}")
        if hasattr(task, "get_coro"):
            coro = task.get_coro()
            if hasattr(coro, "__name__"):
                print(f"    Coroutine: {coro.__name__}")
        print()


if __name__ == "__main__":
    # Run a basic check
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(check_running_tasks())
    loop.close()

    print("Script completed successfully.")

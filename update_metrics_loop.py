#!/usr/bin/env python3
"""Manually update metrics in a loop"""
import time
from vecpipe.metrics import metrics_collector

print("Starting metrics update loop...")
while True:
    try:
        # Force immediate update
        metrics_collector.last_update = 0
        metrics_collector.update_resource_metrics()
        print(".", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped")
        break
    except Exception as e:
        print(f"\nError: {e}")
    time.sleep(1)
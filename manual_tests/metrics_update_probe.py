#!/usr/bin/env python3
"""
Manual probe for the Prometheus metrics updater.

This script was migrated out of pytest on 2025-10-16. Run it from the repo root
to inspect CPU/memory gauges and verify the collector loop while a local stack
is running.
"""

from __future__ import annotations

import time

import psutil
import requests

from packages.shared.metrics.prometheus import metrics_collector


def probe_metrics_update() -> None:
    """Exercise the metrics collector and print sampled Prometheus output."""
    print("Testing metrics update...")
    print(f"Initial CPU percent (no interval): {psutil.cpu_percent(interval=None)}")
    print(f"CPU percent (1s interval): {psutil.cpu_percent(interval=1)}")

    mem = psutil.virtual_memory()
    print(f"Memory: {mem.percent}% ({mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB)")

    print("\nTesting metrics collector...")
    metrics_collector.update_resource_metrics(force=True)
    print("Metrics updated successfully")

    time.sleep(2)

    try:
        response = requests.get("http://localhost:9092/metrics", timeout=5)
    except requests.RequestException as exc:
        print(f"Error fetching metrics: {exc}")
        return

    if response.status_code != 200:
        print(f"Unexpected status from Prometheus endpoint: {response.status_code}")
        return

    for line in response.text.splitlines():
        if "embedding_cpu_utilization_percent" in line and not line.startswith("#"):
            print(f"Prometheus CPU metric: {line}")
        elif "embedding_memory_utilization_percent" in line and not line.startswith("#"):
            print(f"Prometheus Memory metric: {line}")
        elif "embedding_memory_used_bytes" in line and not line.startswith("#"):
            print(f"Prometheus Memory used: {line}")
        elif "embedding_memory_total_bytes" in line and not line.startswith("#"):
            print(f"Prometheus Memory total: {line}")


if __name__ == "__main__":
    probe_metrics_update()

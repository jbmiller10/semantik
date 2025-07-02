#!/usr/bin/env python3
"""Test metrics update functionality"""
import time

import psutil

from packages.vecpipe.metrics import metrics_collector

print("Testing metrics update...")
print(f"Initial CPU percent (no interval): {psutil.cpu_percent(interval=None)}")
print(f"CPU percent (1s interval): {psutil.cpu_percent(interval=1)}")

mem = psutil.virtual_memory()
print(f"Memory: {mem.percent}% ({mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB)")

# Test metrics collector
print("\nTesting metrics collector...")
metrics_collector.update_resource_metrics(force=True)
print("Metrics updated successfully")

# Give it a moment
time.sleep(2)

# Check metrics via Prometheus
import requests

try:
    response = requests.get("http://localhost:9092/metrics")
    if response.status_code == 200:
        lines = response.text.split("\n")
        for line in lines:
            if "embedding_cpu_utilization_percent" in line and not line.startswith("#"):
                print(f"Prometheus CPU metric: {line}")
            elif "embedding_memory_utilization_percent" in line and not line.startswith("#"):
                print(f"Prometheus Memory metric: {line}")
            elif "embedding_memory_used_bytes" in line and not line.startswith("#"):
                print(f"Prometheus Memory used: {line}")
            elif "embedding_memory_total_bytes" in line and not line.startswith("#"):
                print(f"Prometheus Memory total: {line}")
except Exception as e:
    print(f"Error fetching metrics: {e}")

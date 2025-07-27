#!/usr/bin/env python3
"""Test metrics directly from the registry"""
import time

from prometheus_client import generate_latest

from packages.shared.metrics.prometheus import (
    cpu_utilization,
    memory_total,
    memory_used,
    memory_utilization,
    metrics_collector,
    registry,
)

print("Testing direct metrics update...")

# Force update
metrics_collector.update_resource_metrics(force=True)

# Wait a bit
time.sleep(2)

# Check gauge values directly
print(f"CPU utilization gauge value: {cpu_utilization._value.get()}")
print(f"Memory utilization gauge value: {memory_utilization._value.get()}")
print(f"Memory used gauge value: {memory_used._value.get()}")
print(f"Memory total gauge value: {memory_total._value.get()}")

# Generate metrics
print("\nGenerating metrics from registry...")
metrics_data = generate_latest(registry).decode("utf-8")
lines = metrics_data.split("\n")
for line in lines:
    if "embedding_cpu_utilization_percent" in line and not line.startswith("#"):
        print(f"Registry CPU metric: {line}")
    elif "embedding_memory_utilization_percent" in line and not line.startswith("#"):
        print(f"Registry Memory metric: {line}")
    elif "embedding_memory_used_bytes" in line and not line.startswith("#"):
        print(f"Registry Memory used: {line}")
    elif "embedding_memory_total_bytes" in line and not line.startswith("#"):
        print(f"Registry Memory total: {line}")

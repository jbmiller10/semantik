#!/usr/bin/env python3
import time

import psutil

from packages.vecpipe.metrics import cpu_utilization, memory_utilization, metrics_collector

# First, let's check current system metrics
print(f"Actual CPU: {psutil.cpu_percent(interval=0.1)}%")
print(f"Actual Memory: {psutil.virtual_memory().percent}%")

# Force update with interval = 0
print("\nForcing metrics update...")
metrics_collector.last_update = 0
metrics_collector.update_interval = 0
metrics_collector.update_resource_metrics()

# Check the gauge values using collect
print("\nChecking gauge values...")
for metric in cpu_utilization.collect():
    for sample in metric.samples:
        if sample.name == "embedding_cpu_utilization_percent":
            print(f"CPU gauge value: {sample.value}")

for metric in memory_utilization.collect():
    for sample in metric.samples:
        if sample.name == "embedding_memory_utilization_percent":
            print(f"Memory gauge value: {sample.value}")

# Also check using the registry
from vecpipe.metrics import generate_latest, registry

metrics_text = generate_latest(registry).decode("utf-8")
print("\nMetrics from registry:")
for line in metrics_text.split("\n"):
    if "cpu_utilization_percent" in line and not line.startswith("#"):
        print(f"CPU line: {line}")
    if "memory_utilization_percent" in line and not line.startswith("#"):
        print(f"Memory line: {line}")

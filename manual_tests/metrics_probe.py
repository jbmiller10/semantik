#!/usr/bin/env python3
"""
Manual diagnostic helper for the /api/metrics endpoint.

This script mirrors the pre-2025-10-16 pytest smoke harness but lives outside
the automated suite. Use it when you need ad-hoc inspection of the metrics API
running on localhost:8080.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import requests


def probe_metrics_endpoint(base_url: str = "http://localhost:8080") -> None:
    """Print a formatted snapshot of /api/metrics for manual debugging."""
    metrics_url = f"{base_url}/api/metrics"

    print(f"Testing metrics endpoint at: {metrics_url}")
    print(f"Time: {datetime.now(UTC)}")
    print("-" * 60)

    try:
        response = requests.get(metrics_url, timeout=5)
    except requests.RequestException as exc:
        print(f"Request to {metrics_url} failed: {exc}")
        return

    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print("-" * 60)

    print("Raw Response:")
    print(response.text)
    print("-" * 60)

    if response.status_code != 200:
        print("Non-200 status received; skipping JSON parsing.")
        return

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        print(f"Failed to parse JSON: {exc}")
        return

    print("Parsed JSON:")
    print(json.dumps(data, indent=2))
    print("-" * 60)

    metrics = data.get("data", "")
    if metrics:
        non_zero_lines = [line for line in metrics.splitlines() if line and not line.startswith("#") and " 0" not in line]
        if non_zero_lines:
            print("Non-zero metric lines:")
            for line in non_zero_lines:
                print(line)
        else:
            print("WARNING: All reported metrics appear to be zero.")


if __name__ == "__main__":
    probe_metrics_endpoint()

#!/usr/bin/env python3
"""
Manual harness to exercise the search APIs via localhost.

Migrated out of pytest on 2025-10-16. Use it to triage latency/auth issues while
developing search flows.
"""

from __future__ import annotations

import time

import requests


def probe_search() -> None:
    """Hit both the search API and the webui gateway."""
    print("Testing search API directly...")
    try:
        start = time.time()
        response = requests.post(
            "http://localhost:8000/search",
            json={"query": "test", "collection_name": "test_collection", "k": 10},
            timeout=10,
        )
        elapsed = time.time() - start
        print(f"Search API response in {elapsed:.2f}s: {response.status_code}")
        if response.ok:
            results = response.json().get("results", [])
            print(f"Results: {len(results)} results found")
    except requests.RequestException as exc:
        print(f"Search API error: {exc}")

    print("\nTesting through webui backend (requires auth)...")
    try:
        login_response = requests.post(
            "http://localhost:8080/api/auth/login", json={"username": "admin", "password": "admin"}, timeout=10
        )
    except requests.RequestException as exc:
        print(f"Login error: {exc}")
        return

    if not login_response.ok:
        print(f"Login failed: {login_response.text}")
        return

    token = login_response.json().get("access_token")
    if not token:
        print("Login response missing token")
        return

    print("Login successful, got token")

    try:
        start = time.time()
        search_response = requests.post(
            "http://localhost:8080/api/search",
            json={
                "query": "test",
                "collection": "test_collection",
                "top_k": 10,
                "score_threshold": 0.5,
                "search_type": "vector",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=35,
        )
        elapsed = time.time() - start
        print(f"Webui search response in {elapsed:.2f}s: {search_response.status_code}")
        if search_response.ok:
            results = search_response.json().get("results", [])
            print(f"Results: {len(results)} results found")
        else:
            print(f"Error: {search_response.text}")
    except requests.Timeout:
        print("Request timed out!")
    except requests.RequestException as exc:
        print(f"Webui error: {exc}")


if __name__ == "__main__":
    probe_search()

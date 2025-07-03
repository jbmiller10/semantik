#!/usr/bin/env python3
"""Test script to debug search timeout issue"""
import json
import time

import requests

# First, let's test the search API directly
print("Testing search API directly...")
try:
    start = time.time()
    response = requests.post(
        "http://localhost:8000/search", json={"query": "test", "collection_name": "job_33", "k": 10}, timeout=10
    )
    elapsed = time.time() - start
    print(f"Search API response in {elapsed:.2f}s: {response.status_code}")
    if response.ok:
        print(f"Results: {len(response.json()['results'])} results found")
except Exception as e:
    print(f"Search API error: {e}")

# Now let's test through the webui backend
print("\nTesting through webui backend (requires auth)...")
# First login to get a token
try:
    login_response = requests.post(
        "http://localhost:8080/api/auth/login", json={"username": "admin", "password": "admin"}
    )
    if login_response.ok:
        token = login_response.json()["access_token"]
        print("Login successful, got token")

        # Now make the search request
        start = time.time()
        search_response = requests.post(
            "http://localhost:8080/api/search",
            json={
                "query": "test",
                "collection": "job_33",
                "top_k": 10,
                "score_threshold": 0.5,
                "search_type": "vector",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=35,  # Slightly more than the 30s timeout in webui
        )
        elapsed = time.time() - start
        print(f"Webui search response in {elapsed:.2f}s: {search_response.status_code}")
        if search_response.ok:
            print(f"Results: {len(search_response.json()['results'])} results found")
        else:
            print(f"Error: {search_response.text}")
    else:
        print(f"Login failed: {login_response.text}")
except requests.exceptions.Timeout:
    print("Request timed out!")
except Exception as e:
    print(f"Webui error: {e}")

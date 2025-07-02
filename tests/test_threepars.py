#!/usr/bin/env python3
"""Test script to verify document viewing works for threepars user"""

import json

import requests

# Base URL
BASE_URL = "http://localhost:8080"

# Login as threepars
print("1. Logging in as threepars...")
login_data = {"username": "threepars", "password": "puddin123"}
response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
if response.status_code == 200:
    token = response.json()["access_token"]
    print("   ✓ Login successful")
else:
    print(f"   ✗ Login failed: {response.text}")
    exit(1)

headers = {"Authorization": f"Bearer {token}"}

# Get jobs
print("\n2. Getting jobs...")
response = requests.get(f"{BASE_URL}/api/jobs", headers=headers)
jobs = response.json()
print(f"   Found {len(jobs)} jobs")

# Find jobs with collection "33"
collection_33_jobs = [j for j in jobs if j.get("name") == "33" or "33" in str(j.get("name", ""))]
print(f"   Found {len(collection_33_jobs)} jobs with collection '33'")

if collection_33_jobs:
    job = collection_33_jobs[0]
    job_id = job["id"]
    print(f"\n3. Testing job: {job_id} ({job.get('name')})")

    # Get job details
    response = requests.get(f"{BASE_URL}/api/jobs/{job_id}", headers=headers)
    if response.status_code != 200:
        print(f"   Failed to get job details: {response.text}")
    else:
        job_details = response.json()
        files = job_details.get("files", [])
        print(f"   Job has {len(files)} files")

        if files:
            # Test first file
            file_info = files[0]
            doc_id = file_info.get("doc_id", "0")

            print(f"\n4. Testing document view for doc_id: {doc_id}")

            # Try to get document info
            response = requests.get(f"{BASE_URL}/api/documents/{job_id}/{doc_id}/info", headers=headers)
            print(f"   Document info status: {response.status_code}")
            if response.status_code == 200:
                print("   ✓ Document info retrieved successfully!")
            else:
                print(f"   ✗ Failed: {response.text}")

            # Try to get document content
            response = requests.get(f"{BASE_URL}/api/documents/{job_id}/{doc_id}", headers=headers)
            print(f"   Document content status: {response.status_code}")
            if response.status_code == 200:
                print("   ✓ Document content retrieved successfully!")
                print(f"   Content preview: {response.text[:100]}...")
            else:
                print(f"   ✗ Failed: {response.text}")

# Also search for documents
print("\n5. Testing search...")
search_data = {"query": "test", "collection_name": "33", "limit": 5}
response = requests.post(f"{BASE_URL}/api/search", json=search_data, headers=headers)
if response.status_code == 200:
    results = response.json()
    print(f"   ✓ Search successful, found {len(results.get('results', []))} results")
else:
    print(f"   ✗ Search failed: {response.text}")

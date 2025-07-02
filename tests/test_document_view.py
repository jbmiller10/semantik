#!/usr/bin/env python3
"""Test script to verify document viewing works after fix"""

import json

import requests

# Base URL
BASE_URL = "http://localhost:8080"

# Step 1: Register a test user
print("1. Creating test user...")
register_data = {
    "username": "testuser",
    "password": "testpass123",
    "email": "test@example.com",
    "full_name": "Test User",
}
response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
if response.status_code == 200:
    print("   ✓ User created successfully")
else:
    print(f"   ✗ Failed to create user: {response.text}")

# Step 2: Login
print("\n2. Logging in...")
login_data = {"username": "testuser", "password": "testpass123"}
response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
if response.status_code == 200:
    token = response.json()["access_token"]
    print("   ✓ Login successful")
else:
    print(f"   ✗ Login failed: {response.text}")
    exit(1)

headers = {"Authorization": f"Bearer {token}"}

# Step 3: Check if any jobs exist
print("\n3. Checking for existing jobs...")
response = requests.get(f"{BASE_URL}/api/jobs", headers=headers)
jobs = response.json()
print(f"   Found {len(jobs)} jobs")

if len(jobs) > 0:
    # Test with first job
    job = jobs[0]
    job_id = job["id"]
    print(f"\n4. Testing document view for job: {job_id}")

    # Get job details
    response = requests.get(f"{BASE_URL}/api/jobs/{job_id}", headers=headers)
    job_details = response.json()

    # Check if job has files
    if job_details.get("files"):
        file_info = job_details["files"][0]
        doc_id = file_info.get("doc_id", "0")

        print(f"   Testing document view for doc_id: {doc_id}")

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
        else:
            print(f"   ✗ Failed: {response.text}")
    else:
        print("   No files found in job")
else:
    print("\n   No jobs found. The fix should work for newly created jobs with proper user_id.")

print("\n5. Summary:")
print("   - Database schema updated to include user_id in jobs table")
print("   - Job creation now stores the current user's ID")
print("   - Document viewing should work for new jobs created after this fix")

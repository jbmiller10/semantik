#!/usr/bin/env python3
"""
Test script for authentication system
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8080"

def test_auth():
    print("Testing Authentication System...")
    print("=" * 50)
    
    # Test 1: Register a new user
    print("\n1. Testing user registration...")
    register_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123",
        "full_name": "Test User"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
        if response.status_code == 200:
            print("✓ Registration successful!")
            user = response.json()
            print(f"  User created: {user['username']} ({user['email']})")
        else:
            print(f"✗ Registration failed: {response.status_code}")
            print(f"  Error: {response.json()}")
    except Exception as e:
        print(f"✗ Registration error: {e}")
    
    # Test 2: Login
    print("\n2. Testing login...")
    login_data = {
        "username": "testuser",
        "password": "password123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        if response.status_code == 200:
            print("✓ Login successful!")
            tokens = response.json()
            access_token = tokens['access_token']
            print(f"  Access token received: {access_token[:20]}...")
        else:
            print(f"✗ Login failed: {response.status_code}")
            print(f"  Error: {response.json()}")
            return
    except Exception as e:
        print(f"✗ Login error: {e}")
        return
    
    # Test 3: Access protected endpoint without token
    print("\n3. Testing protected endpoint without token...")
    try:
        response = requests.get(f"{BASE_URL}/api/jobs")
        if response.status_code == 401:
            print("✓ Correctly rejected request without token")
        else:
            print(f"✗ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Access protected endpoint with token
    print("\n4. Testing protected endpoint with token...")
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/api/jobs", headers=headers)
        if response.status_code == 200:
            print("✓ Successfully accessed protected endpoint!")
            jobs = response.json()
            print(f"  Jobs found: {len(jobs)}")
        else:
            print(f"✗ Failed to access protected endpoint: {response.status_code}")
            print(f"  Error: {response.json()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 5: Get current user info
    print("\n5. Testing current user endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
        if response.status_code == 200:
            print("✓ Successfully retrieved user info!")
            user = response.json()
            print(f"  Current user: {user['username']} ({user['email']})")
        else:
            print(f"✗ Failed to get user info: {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Authentication tests completed!")

if __name__ == "__main__":
    test_auth()
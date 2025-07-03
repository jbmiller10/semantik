#!/usr/bin/env python3
"""Test authentication endpoints"""

import json

import requests

BASE_URL = "http://localhost:8080"


def test_auth():
    # Test registration
    register_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User",
    }

    print("Testing registration...")
    response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
    print(f"Register status: {response.status_code}")
    if response.status_code != 200:
        print(f"Register error: {response.text}")
    else:
        print(f"Register response: {json.dumps(response.json(), indent=2)}")

    # Test login
    login_data = {"username": "testuser", "password": "testpassword123"}

    print("\nTesting login...")
    response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
    print(f"Login status: {response.status_code}")
    if response.status_code != 200:
        print(f"Login error: {response.text}")
        return

    login_response = response.json()
    print(f"Login response: {json.dumps(login_response, indent=2)}")

    access_token = login_response.get("access_token")
    refresh_token = login_response.get("refresh_token")

    # Test /me endpoint
    print("\nTesting /me endpoint...")
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
    print(f"Me status: {response.status_code}")
    if response.status_code != 200:
        print(f"Me error: {response.text}")
    else:
        print(f"Me response: {json.dumps(response.json(), indent=2)}")

    # Test logout
    print("\nTesting logout...")
    logout_data = {"refresh_token": refresh_token}
    response = requests.post(f"{BASE_URL}/api/auth/logout", headers=headers, json=logout_data)
    print(f"Logout status: {response.status_code}")
    if response.status_code != 200:
        print(f"Logout error: {response.text}")
    else:
        print(f"Logout response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    test_auth()

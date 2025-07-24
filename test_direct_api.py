#!/usr/bin/env python3
"""Direct API test to verify race condition prevention."""

import httpx
import asyncio
import time
import json


async def test_api_directly():
    async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
        # 1. Login
        print("1. Logging in...")
        login_response = await client.post(
            "/api/auth/login", json={"username": "testuser", "password": "testpassword123"}
        )

        if login_response.status_code != 200:
            print(f"Login failed: {login_response.status_code}")
            return

        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Create collection
        print("2. Creating collection...")
        collection_name = f"API Test {int(time.time())}"
        create_response = await client.post(
            "/api/v2/collections",
            json={"name": collection_name, "description": "Testing race condition via API"},
            headers=headers,
        )

        if create_response.status_code != 201:
            print(f"Collection creation failed: {create_response.status_code}")
            print(create_response.json())
            return

        collection_data = create_response.json()
        collection_id = collection_data["id"]
        print(f"✓ Collection created: {collection_id}")
        print(f"  Response data: {json.dumps(collection_data, indent=2)}")

        # 3. Immediately try to add source (should fail due to active INDEX operation)
        print("\n3. Attempting to add source immediately...")
        source_response = await client.post(
            f"/api/v2/collections/{collection_id}/sources", json={"source_path": "/mnt/docs"}, headers=headers
        )

        print(f"  Source addition response: {source_response.status_code}")
        source_data = source_response.json()
        print(f"  Response: {json.dumps(source_data, indent=2)}")

        if source_response.status_code == 409:
            print("\n✓ TEST PASSED: Race condition prevented!")
            print("  The system correctly rejected the source addition")
            print("  because the INDEX operation is still active.")
        elif source_response.status_code == 202:
            print("\n✗ TEST FAILED: Race condition NOT prevented!")
            print("  The APPEND operation was accepted while INDEX is active.")
        else:
            print(f"\n? Unexpected status code: {source_response.status_code}")

        # 4. Check operations
        print("\n4. Checking operations...")
        ops_response = await client.get(f"/api/v2/collections/{collection_id}/operations", headers=headers)

        if ops_response.status_code == 200:
            operations = ops_response.json()
            print("Operations:")
            for op in operations.get("data", []):
                print(f"  - {op['type']} - Status: {op['status']} - Created: {op['created_at']}")


if __name__ == "__main__":
    asyncio.run(test_api_directly())

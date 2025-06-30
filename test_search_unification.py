#!/usr/bin/env python3
"""
Test script to verify that UI search returns identical results to REST API
"""

import asyncio
import httpx
import json
import sys


async def test_search_unification():
    """Test that WebUI and REST API return identical search results"""

    # Test parameters
    test_query = "test query"
    test_k = 5
    test_collection = "work_docs"

    print("Testing Search Unification")
    print("=" * 50)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Call REST API directly
        print(f"\n1. Calling REST API /search endpoint...")
        try:
            rest_response = await client.post(
                "http://localhost:8000/search",
                json={"query": test_query, "k": test_k, "collection": test_collection, "search_type": "semantic"},
            )
            rest_response.raise_for_status()
            rest_data = rest_response.json()
            print(f"   ✓ REST API returned {len(rest_data.get('results', []))} results")
            print(f"   Response keys: {list(rest_data.keys())}")
        except Exception as e:
            print(f"   ✗ REST API failed: {e}")
            return False

        # 2. Call WebUI API (requires auth)
        print(f"\n2. Calling WebUI /api/search endpoint...")

        # First, we need to authenticate to get a token
        # For testing, we'll use default test credentials
        auth_response = await client.post(
            "http://localhost:8080/api/auth/login",
            json={"username": "admin", "password": "admin"},  # Replace with actual test credentials
        )

        if auth_response.status_code != 200:
            print("   ✗ Failed to authenticate. Please ensure test user exists.")
            print("   You may need to create a test user first.")
            return False

        auth_data = auth_response.json()
        access_token = auth_data.get("access_token")

        try:
            webui_response = await client.post(
                "http://localhost:8080/api/search",
                json={"query": test_query, "k": test_k, "job_id": None},  # This will use "work_docs" collection
                headers={"Authorization": f"Bearer {access_token}"},
            )
            webui_response.raise_for_status()
            webui_data = webui_response.json()
            print(f"   ✓ WebUI API returned {len(webui_data.get('results', []))} results")
            print(f"   Response keys: {list(webui_data.keys())}")
        except Exception as e:
            print(f"   ✗ WebUI API failed: {e}")
            return False

        # 3. Compare results
        print(f"\n3. Comparing results...")

        # Check if both responses have the same structure
        if set(rest_data.keys()) != set(webui_data.keys()):
            print(f"   ✗ Response structures differ!")
            print(f"   REST keys: {set(rest_data.keys())}")
            print(f"   WebUI keys: {set(webui_data.keys())}")
            return False

        # Check if results match
        rest_results = rest_data.get("results", [])
        webui_results = webui_data.get("results", [])

        if len(rest_results) != len(webui_results):
            print(f"   ✗ Different number of results!")
            print(f"   REST: {len(rest_results)}, WebUI: {len(webui_results)}")
            return False

        # Compare each result
        for i, (rest_result, webui_result) in enumerate(zip(rest_results, webui_results)):
            if rest_result != webui_result:
                print(f"   ✗ Result {i} differs!")
                print(f"   REST: {json.dumps(rest_result, indent=2)}")
                print(f"   WebUI: {json.dumps(webui_result, indent=2)}")
                return False

        print(f"   ✓ All {len(rest_results)} results match exactly!")

        # 4. Test hybrid search
        print(f"\n4. Testing hybrid search...")

        # REST API hybrid search
        try:
            rest_hybrid = await client.get(
                "http://localhost:8000/hybrid_search",
                params={
                    "q": test_query,
                    "k": test_k,
                    "collection": test_collection,
                    "mode": "filter",
                    "keyword_mode": "any",
                },
            )
            rest_hybrid.raise_for_status()
            rest_hybrid_data = rest_hybrid.json()
            print(f"   ✓ REST API hybrid search returned {len(rest_hybrid_data.get('results', []))} results")
        except Exception as e:
            print(f"   ✗ REST API hybrid search failed: {e}")
            return False

        # WebUI hybrid search
        try:
            webui_hybrid = await client.post(
                "http://localhost:8080/api/hybrid_search",
                json={"query": test_query, "k": test_k, "job_id": None, "mode": "filter", "keyword_mode": "any"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            webui_hybrid.raise_for_status()
            webui_hybrid_data = webui_hybrid.json()
            print(f"   ✓ WebUI API hybrid search returned {len(webui_hybrid_data.get('results', []))} results")
        except Exception as e:
            print(f"   ✗ WebUI API hybrid search failed: {e}")
            return False

        # Compare hybrid results
        if webui_hybrid_data == rest_hybrid_data:
            print(f"   ✓ Hybrid search results match exactly!")
        else:
            print(f"   ✗ Hybrid search results differ!")
            return False

    print("\n" + "=" * 50)
    print("✓ All tests passed! Search unification successful.")
    return True


async def main():
    """Main test runner"""
    print("\nSearch Unification Test")
    print("This test verifies that WebUI proxies correctly to REST API")
    print("\nPrerequisites:")
    print("1. REST API running on http://localhost:8000")
    print("2. WebUI running on http://localhost:8080")
    print("3. Test user 'admin' with password 'admin' exists")
    print("4. Collection 'work_docs' exists with some data")

    print("\nStarting tests...")

    success = await test_search_unification()

    if not success:
        print("\n✗ Tests failed!")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

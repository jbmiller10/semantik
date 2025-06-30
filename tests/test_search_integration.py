#!/usr/bin/env python3
"""
Test the updated production search API with real Qwen3 embeddings
"""

import httpx
import os


def test_search_api():
    """Test the search API with both mock and real embeddings"""

    base_url = "http://localhost:8000"

    print("Testing Search API Integration")
    print("=" * 60)

    # Test 1: Check health and embedding status
    print("\n1. Checking API health and embedding status...")
    try:
        response = httpx.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy")
            print(f"  Collection: {data['collection']}")
            print(f"  Points: {data['points_count']}")
            print(f"  Embedding mode: {data.get('embedding_mode', 'unknown')}")
            if data.get("embedding_model"):
                print(f"  Model: {data['embedding_model']}")
                print(f"  Quantization: {data.get('quantization', 'N/A')}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Could not connect to API: {e}")
        return

    # Test 2: Check embedding configuration
    print("\n2. Checking embedding configuration...")
    try:
        response = httpx.get(f"{base_url}/embedding/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Embedding info retrieved")
            print(f"  Mode: {data['mode']}")
            print(f"  Available: {data['available']}")
            if data.get("current_model"):
                print(f"  Current model: {data['current_model']}")
                print(f"  Device: {data.get('device', 'N/A')}")
                if "model_details" in data:
                    details = data["model_details"]
                    print(f"  Embedding dimension: {details.get('embedding_dim', 'N/A')}")
                    print(f"  Model size: {details.get('model_size_mb', 'N/A')} MB")
        else:
            print(f"✗ Failed to get embedding info: {response.status_code}")
    except Exception as e:
        print(f"✗ Error getting embedding info: {e}")

    # Test 3: Perform a search
    print("\n3. Testing search functionality...")
    test_queries = [
        "machine learning algorithms",
        "how to implement transformer models",
        "python programming best practices",
    ]

    for query in test_queries[:1]:  # Test just one query
        try:
            response = httpx.get(f"{base_url}/search", params={"q": query, "k": 5}, timeout=30.0)

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Search successful for: '{query}'")
                print(f"  Results found: {data['num_results']}")

                # Show top 3 results
                for i, result in enumerate(data["results"][:3]):
                    print(f"  Result {i+1}:")
                    print(f"    Path: {result['path']}")
                    print(f"    Score: {result['score']:.4f}")
            else:
                print(f"✗ Search failed for '{query}': {response.status_code}")
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"✗ Search error for '{query}': {e}")

    print("\n" + "=" * 60)
    print("Integration test complete!")

    # Check if we're using real embeddings
    use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"
    if use_mock:
        print("\nNote: Currently using MOCK embeddings.")
        print("To use real Qwen3 embeddings, set USE_MOCK_EMBEDDINGS=false")
    else:
        print("\nNote: Configured to use REAL Qwen3 embeddings.")
        print("If the model failed to load, it will fall back to mock embeddings.")


if __name__ == "__main__":
    test_search_api()

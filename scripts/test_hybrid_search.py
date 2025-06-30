#!/usr/bin/env python3
"""Test hybrid search functionality (vector + keyword search)"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import json
import numpy as np
from typing import List, Dict
from vecpipe.config import settings

COLLECTION_NAME = "test_hybrid"
VECTOR_SIZE = 384  # Using smaller vector for testing


def create_test_collection():
    """Create a test collection with text index"""
    base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    try:
        # Delete if exists
        httpx.delete(f"{base_url}/collections/{COLLECTION_NAME}")
    except:
        pass

    # Create collection
    collection_config = {"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}, "on_disk_payload": True}

    response = httpx.put(f"{base_url}/collections/{COLLECTION_NAME}", json=collection_config, timeout=30.0)
    response.raise_for_status()

    # Create text index
    index_response = httpx.put(
        f"{base_url}/collections/{COLLECTION_NAME}/index",
        json={"field_name": "text", "field_schema": "text"},
        timeout=30.0,
    )
    index_response.raise_for_status()

    print(f"✓ Created collection '{COLLECTION_NAME}' with text index")
    return True


def insert_test_data():
    """Insert test documents with various keywords"""
    base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    # Test documents with specific keywords and acronyms
    test_docs = [
        {
            "id": "1",
            "text": "The ERROR_CODE_404 indicates that the requested resource was not found in the system.",
            "keywords": ["ERROR_CODE_404", "error", "404"],
        },
        {
            "id": "2",
            "text": "Use the function calculate_total() to compute the sum of all values in the array.",
            "keywords": ["calculate_total", "function", "compute"],
        },
        {
            "id": "3",
            "text": "The API endpoint /api/v2/users returns a JSON response with user information.",
            "keywords": ["API", "endpoint", "/api/v2/users", "JSON"],
        },
        {
            "id": "4",
            "text": "Machine learning models can identify patterns in large datasets automatically.",
            "keywords": ["machine learning", "models", "patterns"],
        },
        {
            "id": "5",
            "text": "The SQL query SELECT * FROM orders WHERE status='ERROR_CODE_500' finds failed transactions.",
            "keywords": ["SQL", "ERROR_CODE_500", "SELECT", "orders"],
        },
    ]

    # Generate mock embeddings (random vectors normalized)
    points = []
    for doc in test_docs:
        # Create a deterministic "embedding" based on text content
        np.random.seed(hash(doc["text"]) % 2**32)
        vector = np.random.randn(VECTOR_SIZE).astype(float)
        vector = vector / np.linalg.norm(vector)  # Normalize

        point = {
            "id": doc["id"],
            "vector": vector.tolist(),
            "payload": {"text": doc["text"], "keywords": doc["keywords"]},
        }
        points.append(point)

    # Upload points
    response = httpx.put(f"{base_url}/collections/{COLLECTION_NAME}/points", json={"points": points}, timeout=30.0)
    response.raise_for_status()

    print(f"✓ Inserted {len(points)} test documents")
    return test_docs


def test_hybrid_search(query: str, test_docs: List[Dict]):
    """Test hybrid search with both vector and keyword components"""
    base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    print(f"\n🔍 Testing hybrid search for: '{query}'")

    # Generate a random query vector (in real scenario, this would be from embedding model)
    np.random.seed(42)
    query_vector = np.random.randn(VECTOR_SIZE).astype(float)
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Perform hybrid search
    search_request = {
        "vector": query_vector.tolist(),
        "q": query,  # Text query for keyword search
        "limit": 5,
        "with_payload": True,
    }

    response = httpx.post(f"{base_url}/collections/{COLLECTION_NAME}/points/search", json=search_request, timeout=30.0)
    response.raise_for_status()

    results = response.json()["result"]

    print(f"\nResults (found {len(results)} matches):")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.4f}")
        print(f"   Text: {result['payload']['text']}")

        # Check if query keywords appear in result
        query_words = query.lower().split()
        text_lower = result["payload"]["text"].lower()
        matching_keywords = [w for w in query_words if w in text_lower]

        if matching_keywords:
            print(f"   ✓ Keyword matches: {', '.join(matching_keywords)}")
        else:
            print(f"   ℹ️  No exact keyword matches (semantic match)")


def main():
    """Run hybrid search tests"""
    print("Testing Hybrid Search (Vector + Keyword)")
    print("=" * 50)

    # Setup
    if not create_test_collection():
        print("✗ Failed to create collection")
        sys.exit(1)

    test_docs = insert_test_data()

    # Test queries that benefit from hybrid search
    test_queries = [
        "ERROR_CODE_404",  # Exact keyword - should match document 1
        "calculate_total function",  # Function name - should match document 2
        "API endpoint users",  # Technical terms - should match document 3
        "ERROR_CODE_500 SQL",  # Multiple keywords - should match document 5
        "machine learning patterns data",  # Semantic query - should match document 4
        "system errors and failures",  # Mixed semantic/keyword query
    ]

    for query in test_queries:
        test_hybrid_search(query, test_docs)
        print("\n" + "-" * 50)

    # Cleanup
    try:
        httpx.delete(f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}")
        print(f"\n✓ Cleaned up test collection")
    except:
        pass


if __name__ == "__main__":
    main()

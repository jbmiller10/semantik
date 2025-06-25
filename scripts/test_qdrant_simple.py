#!/usr/bin/env python3
"""Simple test of Qdrant connection using HTTP API"""

import httpx
import json

QDRANT_HOST = "192.168.1.173"
QDRANT_PORT = 6333
COLLECTION_NAME = "work_docs"
VECTOR_SIZE = 1024

def test_connection():
    """Test basic connection to Qdrant"""
    base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    
    try:
        # Test basic connectivity
        response = httpx.get(f"{base_url}/collections")
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"  Status: {data['status']}")
        print(f"  Collections: {[c['name'] for c in data['result']['collections']]}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        return False

def create_collection():
    """Create the work_docs collection via HTTP API"""
    base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    
    try:
        # Check if collection exists
        response = httpx.get(f"{base_url}/collections/{COLLECTION_NAME}")
        if response.status_code == 200:
            print(f"  Collection '{COLLECTION_NAME}' exists, deleting...")
            delete_response = httpx.delete(f"{base_url}/collections/{COLLECTION_NAME}")
            delete_response.raise_for_status()
            print(f"  Deleted existing collection")
        
        # Create collection
        collection_config = {
            "vectors": {
                "size": VECTOR_SIZE,
                "distance": "Cosine"
            },
            "hnsw_config": {
                "m": 32,
                "ef_construct": 200,
                "full_scan_threshold": 10000
            },
            "optimizers_config": {
                "indexing_threshold": 100000,
                "memmap_threshold": 0
            },
            "on_disk_payload": True
        }
        
        response = httpx.put(
            f"{base_url}/collections/{COLLECTION_NAME}",
            json=collection_config,
            timeout=30.0
        )
        response.raise_for_status()
        
        print(f"✓ Created collection '{COLLECTION_NAME}' with:")
        print(f"  - Vector size: {VECTOR_SIZE}")
        print(f"  - Distance metric: Cosine")
        print(f"  - HNSW index: m=32, ef_construct=200")
        print(f"  - Disk-based storage enabled")
        
        # Verify collection
        verify_response = httpx.get(f"{base_url}/collections/{COLLECTION_NAME}")
        verify_response.raise_for_status()
        info = verify_response.json()['result']
        print(f"✓ Collection verified: {info['points_count']} points")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create collection: {e}")
        return False

if __name__ == "__main__":
    print("Testing Qdrant connection and collection setup...")
    if test_connection():
        if create_collection():
            print("\n✓ Qdrant setup complete!")
        else:
            print("\n✗ Collection creation failed")
    else:
        print("\n✗ Connection test failed")
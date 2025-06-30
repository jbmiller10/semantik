#!/usr/bin/env python3
"""Simple test of Qdrant connection using official qdrant-client"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from vecpipe.config import settings

VECTOR_SIZE = 1024


def test_connection():
    """Test basic connection to Qdrant"""

    try:
        # Initialize client
        client = QdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

        # Test basic connectivity
        collections = client.get_collections()

        print(f"✓ Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        print(f"  Collections: {[c.name for c in collections.collections]}")

        return True
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        return False


def create_collection():
    """Create the work_docs collection using official client"""

    try:
        # Initialize client
        client = QdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

        # Check if collection exists
        try:
            client.get_collection(settings.DEFAULT_COLLECTION)
            print(f"  Collection '{settings.DEFAULT_COLLECTION}' exists, deleting...")
            client.delete_collection(settings.DEFAULT_COLLECTION)
            print(f"  Deleted existing collection")
        except:
            pass  # Collection doesn't exist

        # Create collection
        client.create_collection(
            collection_name=settings.DEFAULT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            hnsw_config={"m": 32, "ef_construct": 200, "full_scan_threshold": 10000},
            optimizers_config={"indexing_threshold": 100000, "memmap_threshold": 0},
            on_disk_payload=True,
        )

        print(f"✓ Created collection '{settings.DEFAULT_COLLECTION}' with:")
        print(f"  - Vector size: {VECTOR_SIZE}")
        print(f"  - Distance metric: Cosine")
        print(f"  - HNSW index: m=32, ef_construct=200")
        print(f"  - Disk-based storage enabled")

        # Verify collection
        info = client.get_collection(settings.DEFAULT_COLLECTION)
        print(f"✓ Collection verified: {info.points_count} points")

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

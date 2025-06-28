#!/usr/bin/env python3
"""Test Qdrant connection and create collection"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient, models
from vecpipe.config import settings

VECTOR_SIZE = 1024

def test_connection():
    """Test connection to Qdrant server"""
    try:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        
        # Test connection
        info = client.get_collections()
        print(f"✓ Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        print(f"  Existing collections: {[c.name for c in info.collections]}")
        
        return client
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        sys.exit(1)

def create_collection(client):
    """Create or recreate the work_docs collection"""
    try:
        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        
        if settings.DEFAULT_COLLECTION in collections:
            print(f"  Collection '{settings.DEFAULT_COLLECTION}' already exists, recreating...")
            client.delete_collection(settings.DEFAULT_COLLECTION)
        
        # Create collection with optimized settings
        client.create_collection(
            collection_name=settings.DEFAULT_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(
                m=32,
                ef_construct=200,
                full_scan_threshold=10000
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=100000,
                memmap_threshold=0  # Use disk storage for vectors
            ),
            on_disk_payload=True  # Store payload on disk
        )
        
        print(f"✓ Created collection '{settings.DEFAULT_COLLECTION}' with:")
        print(f"  - Vector size: {VECTOR_SIZE}")
        print(f"  - Distance metric: Cosine")
        print(f"  - HNSW index: m=32, ef_construct=200")
        print(f"  - Disk-based storage enabled")
        
        # Verify collection
        info = client.get_collection(settings.DEFAULT_COLLECTION)
        print(f"✓ Collection verified: {info.points_count} points")
        
    except Exception as e:
        print(f"✗ Failed to create collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Testing Qdrant connection and collection setup...")
    client = test_connection()
    create_collection(client)
    print("\n✓ Qdrant setup complete!")
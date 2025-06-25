#!/usr/bin/env python3
"""Test Qdrant connection and create collection"""

from qdrant_client import QdrantClient, models
import sys

QDRANT_HOST = "192.168.1.173"
QDRANT_PORT = 6333
COLLECTION_NAME = "work_docs"
VECTOR_SIZE = 1024

def test_connection():
    """Test connection to Qdrant server"""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Test connection
        info = client.get_collections()
        print(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
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
        
        if COLLECTION_NAME in collections:
            print(f"  Collection '{COLLECTION_NAME}' already exists, recreating...")
            client.delete_collection(COLLECTION_NAME)
        
        # Create collection with optimized settings
        client.create_collection(
            collection_name=COLLECTION_NAME,
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
        
        print(f"✓ Created collection '{COLLECTION_NAME}' with:")
        print(f"  - Vector size: {VECTOR_SIZE}")
        print(f"  - Distance metric: Cosine")
        print(f"  - HNSW index: m=32, ef_construct=200")
        print(f"  - Disk-based storage enabled")
        
        # Verify collection
        info = client.get_collection(COLLECTION_NAME)
        print(f"✓ Collection verified: {info.points_count} points")
        
    except Exception as e:
        print(f"✗ Failed to create collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Testing Qdrant connection and collection setup...")
    client = test_connection()
    create_collection(client)
    print("\n✓ Qdrant setup complete!")
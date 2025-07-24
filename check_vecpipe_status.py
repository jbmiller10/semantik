#!/usr/bin/env python3
"""
Check vecpipe service status and test embedding generation.
"""

import asyncio
import httpx
import json


async def check_vecpipe_health():
    """Check if vecpipe service is running and healthy."""
    vecpipe_url = "http://vecpipe:8000"

    print("Checking vecpipe service status...")
    print(f"Base URL: {vecpipe_url}")
    print("-" * 50)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check if vecpipe is reachable
            try:
                response = await client.get(f"{vecpipe_url}/")
                print(f"✓ Vecpipe is reachable: {response.status_code}")
                if response.status_code == 200:
                    print(f"  Response: {response.text[:200]}...")
            except Exception as e:
                print(f"✗ Cannot reach vecpipe: {e}")
                return False

            # Check docs endpoint
            try:
                response = await client.get(f"{vecpipe_url}/docs")
                print(f"✓ API docs accessible: {response.status_code}")
            except Exception as e:
                print(f"✗ API docs not accessible: {e}")

            return True

    except Exception as e:
        print(f"✗ Failed to connect to vecpipe: {e}")
        return False


async def test_embedding_generation():
    """Test embedding generation with a sample text."""
    vecpipe_url = "http://vecpipe:8000"

    print("\nTesting embedding generation...")
    print("-" * 50)

    # Test data
    test_request = {
        "texts": ["This is a test document for embedding generation."],
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "instruction": None,
        "batch_size": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"Sending test request to {vecpipe_url}/embed")
            print(f"Request data: {json.dumps(test_request, indent=2)}")

            response = await client.post(f"{vecpipe_url}/embed", json=test_request)

            print(f"\nResponse status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                if "embeddings" in result and result["embeddings"]:
                    embedding = result["embeddings"][0]
                    print(f"✓ Embedding generated successfully!")
                    print(f"  Embedding dimension: {len(embedding)}")
                    print(f"  First 10 values: {embedding[:10]}")
                else:
                    print(f"✗ No embeddings in response: {result}")
            else:
                print(f"✗ Failed to generate embeddings:")
                print(f"  Status: {response.status_code}")
                print(f"  Response: {response.text}")

    except Exception as e:
        print(f"✗ Error during embedding test: {e}")
        import traceback

        traceback.print_exc()


async def test_qdrant_connection():
    """Test Qdrant connection via vecpipe."""
    vecpipe_url = "http://vecpipe:8000"

    print("\nTesting Qdrant connection...")
    print("-" * 50)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try to list collections via vecpipe (if such endpoint exists)
            # Note: This assumes vecpipe has a collections endpoint
            response = await client.get(f"{vecpipe_url}/collections")

            if response.status_code == 200:
                print(f"✓ Can access Qdrant via vecpipe")
                collections = response.json()
                print(f"  Collections found: {len(collections) if isinstance(collections, list) else 'Unknown'}")
            else:
                print(f"  Note: Collections endpoint returned {response.status_code}")

    except Exception as e:
        print(f"  Note: Could not test Qdrant connection: {e}")


async def check_required_services():
    """Check if all required services are running."""
    services = {
        "PostgreSQL": "postgresql://semantik:semantik@postgres:5432/semantik",
        "Redis": "redis://redis:6379/0",
        "Qdrant": "http://qdrant:6333",
        "Vecpipe": "http://vecpipe:8000",
    }

    print("\nChecking required services...")
    print("-" * 50)

    for service_name, url in services.items():
        try:
            if service_name == "PostgreSQL":
                # For PostgreSQL, we'll just check if we can import the connection manager
                from shared.database import pg_connection_manager

                print(f"✓ {service_name}: Connection module available")
            elif service_name == "Redis":
                # For Redis, check if we can import redis
                import redis

                print(f"✓ {service_name}: Redis module available")
            else:
                # For HTTP services, try a simple GET request
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(url if url.startswith("http") else f"http://{url}")
                    print(f"✓ {service_name}: Reachable (status: {response.status_code})")
        except Exception as e:
            print(f"✗ {service_name}: Not accessible - {str(e)[:100]}")


async def main():
    """Run all checks."""
    print("Vecpipe Service Status Check")
    print("=" * 50)

    # Check all required services
    await check_required_services()

    # Check vecpipe specifically
    if await check_vecpipe_health():
        # If vecpipe is healthy, test embedding generation
        await test_embedding_generation()
        await test_qdrant_connection()
    else:
        print("\n⚠️  Vecpipe service is not accessible. This is required for embedding generation.")
        print("Make sure the vecpipe container is running: docker ps | grep vecpipe")


if __name__ == "__main__":
    asyncio.run(main())

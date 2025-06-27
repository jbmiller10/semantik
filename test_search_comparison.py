#!/usr/bin/env python3
"""
Quick test to compare original vs optimized search API
"""

import httpx
import time
import json

def test_original_api():
    """Test the original search API (mock embeddings)"""
    print("Testing Original Search API (Mock Embeddings)")
    print("-" * 50)
    
    try:
        response = httpx.get(
            "http://localhost:8000/search",
            params={"q": "machine learning transformers", "k": 5}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Search successful")
            print(f"  Query: {data['query']}")
            print(f"  Results: {data['num_results']}")
            print(f"  Note: Using mock embeddings (not real)")
        else:
            print(f"✗ Search failed: {response.status_code}")
    except Exception as e:
        print(f"✗ API not running or error: {e}")

def test_optimized_api():
    """Test the optimized search API (real Qwen3 embeddings)"""
    print("\nTesting Optimized Search API (Qwen3 Embeddings)")
    print("-" * 50)
    
    # Test 1: Basic search
    print("\n1. Basic Search:")
    try:
        start = time.time()
        response = httpx.post(
            "http://localhost:8000/search",
            json={
                "query": "machine learning transformers",
                "k": 5,
                "search_type": "semantic"
            },
            timeout=30.0
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Search successful in {elapsed:.2f}s")
            print(f"  Query: {data['query']}")
            print(f"  Results: {data['num_results']}")
            print(f"  Model: {data['model_used']}")
            print(f"  Embedding time: {data.get('embedding_time_ms', 'N/A')}ms")
            print(f"  Search time: {data.get('search_time_ms', 'N/A')}ms")
        else:
            print(f"✗ Search failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ API error: {e}")
    
    # Test 2: Question search with instructions
    print("\n2. Question Search with Instructions:")
    try:
        response = httpx.post(
            "http://localhost:8000/search",
            json={
                "query": "What are the benefits of transformer models?",
                "k": 5,
                "search_type": "question",
                "model_name": "Qwen/Qwen3-Embedding-0.6B",
                "quantization": "float16"
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Question search successful")
            print(f"  Search type: {data['search_type']}")
            print(f"  Model: {data['model_used']}")
        else:
            print(f"✗ Question search failed: {response.status_code}")
    except Exception as e:
        print(f"✗ API error: {e}")
    
    # Test 3: Batch search
    print("\n3. Batch Search:")
    try:
        response = httpx.post(
            "http://localhost:8000/search/batch",
            json={
                "queries": [
                    "What is BERT?",
                    "How does GPT work?",
                    "Explain attention mechanism"
                ],
                "k": 3,
                "search_type": "question"
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Batch search successful")
            print(f"  Total time: {data['total_time_ms']:.2f}ms")
            print(f"  Queries processed: {len(data['responses'])}")
            for i, resp in enumerate(data['responses']):
                print(f"    Query {i+1}: {resp['num_results']} results")
        else:
            print(f"✗ Batch search failed: {response.status_code}")
    except Exception as e:
        print(f"✗ API error: {e}")
    
    # Test 4: Model info
    print("\n4. Available Models:")
    try:
        response = httpx.get("http://localhost:8000/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {len(data['models'])} models")
            # Show Qwen3 models
            qwen_models = [m for m in data['models'] if m['is_qwen3']]
            for model in qwen_models[:3]:
                print(f"  - {model['name']}")
                print(f"    Dimension: {model['dimension']}")
                print(f"    Recommended: {model['recommended_quantization']}")
        else:
            print(f"✗ Failed to get models: {response.status_code}")
    except Exception as e:
        print(f"✗ API error: {e}")

def main():
    print("=" * 60)
    print("Search API Comparison Test")
    print("=" * 60)
    
    # Test original API
    test_original_api()
    
    # Test optimized API
    test_optimized_api()
    
    print("\n" + "=" * 60)
    print("Key Improvements in Optimized API:")
    print("- Real Qwen3 embeddings (not mock)")
    print("- Task-specific instructions for better relevance")
    print("- Support for multiple models and quantization")
    print("- Batch search capability")
    print("- Performance metrics in response")
    print("=" * 60)

if __name__ == "__main__":
    main()
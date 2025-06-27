#!/usr/bin/env python3
"""
Test script for hybrid search functionality
Tests both the search API and direct hybrid search
"""

import sys
import requests
import json
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
SEARCH_API_URL = "http://localhost:8000"
WEBUI_API_URL = "http://localhost:8080"

def test_search_api_hybrid():
    """Test hybrid search via search API"""
    print("\n=== Testing Search API Hybrid Search ===")
    
    # Test queries
    test_queries = [
        {
            "query": "machine learning algorithms",
            "mode": "filter",
            "keyword_mode": "any"
        },
        {
            "query": "docker container deployment",
            "mode": "rerank",
            "keyword_mode": "all"
        },
        {
            "query": "python data analysis",
            "mode": "filter",
            "keyword_mode": "any",
            "score_threshold": 0.5
        }
    ]
    
    for test in test_queries:
        print(f"\nTest Query: {test['query']}")
        print(f"Mode: {test['mode']}, Keyword Mode: {test['keyword_mode']}")
        
        try:
            # Make request
            params = {
                "q": test["query"],
                "k": 5,
                "mode": test["mode"],
                "keyword_mode": test["keyword_mode"]
            }
            
            if "score_threshold" in test:
                params["score_threshold"] = test["score_threshold"]
            
            response = requests.get(f"{SEARCH_API_URL}/hybrid_search", params=params)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Keywords extracted: {data['keywords_extracted']}")
                print(f"Found {data['num_results']} results")
                
                for i, result in enumerate(data['results'][:3]):  # Show top 3
                    print(f"\nResult {i+1}:")
                    print(f"  Path: {result['path']}")
                    print(f"  Score: {result['score']:.4f}")
                    print(f"  Matched Keywords: {result['matched_keywords']}")
                    if result.get('keyword_score') is not None:
                        print(f"  Keyword Score: {result['keyword_score']:.4f}")
                    if result.get('combined_score') is not None:
                        print(f"  Combined Score: {result['combined_score']:.4f}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Failed to test query: {e}")

def test_keyword_search():
    """Test keyword-only search"""
    print("\n=== Testing Keyword-Only Search ===")
    
    test_queries = [
        {
            "query": "python docker kubernetes",
            "mode": "any"
        },
        {
            "query": "machine learning model",
            "mode": "all"
        }
    ]
    
    for test in test_queries:
        print(f"\nKeyword Query: {test['query']}")
        print(f"Mode: {test['mode']}")
        
        try:
            params = {
                "q": test["query"],
                "k": 5,
                "mode": test["mode"]
            }
            
            response = requests.get(f"{SEARCH_API_URL}/keyword_search", params=params)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Keywords extracted: {data['keywords_extracted']}")
                print(f"Found {data['num_results']} results")
                
                for i, result in enumerate(data['results'][:3]):  # Show top 3
                    print(f"\nResult {i+1}:")
                    print(f"  Path: {result['path']}")
                    print(f"  Matched Keywords: {result['matched_keywords']}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Failed to test keyword query: {e}")

def test_direct_hybrid_search():
    """Test hybrid search directly using the hybrid_search module"""
    print("\n=== Testing Direct Hybrid Search ===")
    
    try:
        # Add parent directory to path
        sys.path.append('/root/document-embedding-project')
        from vecpipe.hybrid_search import HybridSearchEngine
        from webui.embedding_service import EmbeddingService
        
        # Initialize services
        hybrid_engine = HybridSearchEngine("192.168.1.173", 6333, "work_docs")
        embedding_service = EmbeddingService(mock_mode=True)
        
        # Test query
        query = "data processing pipeline"
        print(f"\nDirect test query: {query}")
        
        # Generate mock embedding
        query_vector = [0.1] * 1024  # Mock 1024-dim vector
        
        # Test filter mode
        print("\nFilter mode results:")
        results = hybrid_engine.hybrid_search(
            query_vector=query_vector,
            query_text=query,
            limit=3,
            hybrid_mode="filter"
        )
        
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result['payload'].get('path', 'N/A')}")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Matched: {result.get('matched_keywords', [])}")
        
        # Test rerank mode
        print("\nRerank mode results:")
        results = hybrid_engine.hybrid_search(
            query_vector=query_vector,
            query_text=query,
            limit=3,
            hybrid_mode="rerank"
        )
        
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result['payload'].get('path', 'N/A')}")
            print(f"  Vector Score: {result['score']:.4f}")
            print(f"  Keyword Score: {result.get('keyword_score', 0):.4f}")
            print(f"  Combined Score: {result.get('combined_score', 0):.4f}")
        
        hybrid_engine.close()
        
    except Exception as e:
        print(f"Direct test failed: {e}")
        import traceback
        traceback.print_exc()

def check_api_health():
    """Check if APIs are running"""
    print("=== Checking API Health ===")
    
    apis = [
        ("Search API", f"{SEARCH_API_URL}/"),
        ("WebUI API", f"{WEBUI_API_URL}/api/health")
    ]
    
    for name, url in apis:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {name} is running at {url}")
            else:
                print(f"✗ {name} returned status {response.status_code}")
        except Exception as e:
            print(f"✗ {name} is not accessible: {e}")
    
    print()

def main():
    """Run all tests"""
    print("Hybrid Search Test Suite")
    print("========================")
    
    # Check API health
    check_api_health()
    
    # Run tests
    test_search_api_hybrid()
    test_keyword_search()
    test_direct_hybrid_search()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
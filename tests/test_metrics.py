#!/usr/bin/env python3
"""
Simple test script to check the /api/metrics endpoint
"""
import json
from datetime import datetime

import requests


def test_metrics_endpoint():
    """Test the metrics endpoint and print raw response"""
    base_url = "http://localhost:8080"
    metrics_url = f"{base_url}/api/metrics"

    print(f"Testing metrics endpoint at: {metrics_url}")
    print(f"Time: {datetime.now()}")
    print("-" * 60)

    try:
        # Make request to metrics endpoint
        response = requests.get(metrics_url)

        # Print response details
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print("-" * 60)

        # Print raw response
        print("Raw Response:")
        print(response.text)
        print("-" * 60)

        # Try to parse as JSON
        if response.status_code == 200:
            try:
                data = response.json()
                print("Parsed JSON:")
                print(json.dumps(data, indent=2))

                # Check specific fields
                print("-" * 60)
                print("Metrics Analysis:")
                print(f"Total jobs: {data.get('total_jobs', 'NOT FOUND')}")
                print(f"Running jobs: {data.get('running_jobs', 'NOT FOUND')}")
                print(f"Completed jobs: {data.get('completed_jobs', 'NOT FOUND')}")
                print(f"Failed jobs: {data.get('failed_jobs', 'NOT FOUND')}")
                print(f"Total documents: {data.get('total_documents', 'NOT FOUND')}")
                print(f"Total embeddings: {data.get('total_embeddings', 'NOT FOUND')}")

                # Check if any values are non-zero
                non_zero_metrics = [k for k, v in data.items() if isinstance(v, (int, float)) and v > 0]
                if non_zero_metrics:
                    print(f"\nNon-zero metrics: {non_zero_metrics}")
                else:
                    print("\nWARNING: All metrics appear to be zero!")

            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
        else:
            print(f"Non-200 status code: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the server. Is it running on port 8080?")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


def test_with_curl():
    """Also test with curl for comparison"""
    import subprocess

    print("\n" + "=" * 60)
    print("Testing with curl:")
    print("-" * 60)

    try:
        result = subprocess.run(
            ["curl", "-s", "-v", "http://localhost:8080/api/metrics"], capture_output=True, text=True
        )
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR (verbose output):")
        print(result.stderr)
    except FileNotFoundError:
        print("curl not found, skipping curl test")
    except Exception as e:
        print(f"ERROR running curl: {e}")


if __name__ == "__main__":
    test_metrics_endpoint()
    test_with_curl()

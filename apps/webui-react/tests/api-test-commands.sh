#!/bin/bash

# API Test Commands for VecPipe WebUI
# This script contains curl commands to test all critical API endpoints

# Configuration
API_BASE="${API_BASE:-http://localhost:8000}"
AUTH_TOKEN="${AUTH_TOKEN:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function to print test headers
print_test() {
    echo -e "\n${YELLOW}=== $1 ===${NC}"
}

# Helper function to print results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ Success${NC}"
    else
        echo -e "${RED}❌ Failed (HTTP $1)${NC}"
    fi
}

# Helper function for authenticated requests
auth_header() {
    if [ -n "$AUTH_TOKEN" ]; then
        echo "-H 'Authorization: Bearer $AUTH_TOKEN'"
    else
        echo ""
    fi
}

echo "API Test Commands for VecPipe WebUI"
echo "Base URL: $API_BASE"
echo "-----------------------------------"

# Test 1: Health Check
print_test "Health Check"
curl -s -o /dev/null -w "%{http_code}" "$API_BASE/api/health" | {
    read status
    print_result $status
}

# Test 2: List Jobs
print_test "List Jobs"
curl -s -X GET "$API_BASE/api/jobs" \
    -H "Content-Type: application/json" \
    $(auth_header) | jq '.' || echo "Failed to parse JSON"

# Test 3: Create Job
print_test "Create Embedding Job"
JOB_RESPONSE=$(curl -s -X POST "$API_BASE/api/jobs" \
    -H "Content-Type: application/json" \
    $(auth_header) \
    -d '{
        "job_name": "Test Job",
        "directory": "/tmp",
        "glob_patterns": ["*.txt", "*.pdf"],
        "recursive": true,
        "max_workers": 2,
        "embedding_model": "BAAI/bge-small-en-v1.5"
    }')

echo "$JOB_RESPONSE" | jq '.' || echo "Failed to parse JSON"
JOB_ID=$(echo "$JOB_RESPONSE" | jq -r '.job_id // empty')

if [ -n "$JOB_ID" ]; then
    echo "Created job with ID: $JOB_ID"
fi

# Test 4: Get Job Status
if [ -n "$JOB_ID" ]; then
    print_test "Get Job Status"
    curl -s -X GET "$API_BASE/api/jobs/$JOB_ID" \
        -H "Content-Type: application/json" \
        $(auth_header) | jq '.' || echo "Failed to parse JSON"
fi

# Test 5: Directory Scan
print_test "Directory Scan"
curl -s -X POST "$API_BASE/api/scan-directory" \
    -H "Content-Type: application/json" \
    $(auth_header) \
    -d '{
        "path": "/tmp",
        "recursive": true
    }' | jq '.' || echo "Failed to parse JSON"

# Test 6: Vector Search
print_test "Vector Search"
curl -s -X POST "$API_BASE/api/search" \
    -H "Content-Type: application/json" \
    $(auth_header) \
    -d '{
        "query": "test search query",
        "top_k": 10,
        "mode": "vector"
    }' | jq '.' || echo "Failed to parse JSON"

# Test 7: Hybrid Search (if available)
print_test "Hybrid Search Test"
curl -s -X POST "$API_BASE/api/search" \
    -H "Content-Type: application/json" \
    $(auth_header) \
    -d '{
        "query": "test hybrid search",
        "top_k": 10,
        "mode": "hybrid",
        "alpha": 0.5
    }' | jq '.' || echo "Failed to parse JSON"

# Test 8: Settings
print_test "Get Settings"
curl -s -X GET "$API_BASE/api/settings" \
    -H "Content-Type: application/json" \
    $(auth_header) | jq '.' || echo "Failed to parse JSON"

# Test 9: Document Preview
print_test "Document Preview (requires valid file path)"
# Note: Replace with actual file path
curl -s -X GET "$API_BASE/api/documents/preview?path=/path/to/document.pdf" \
    $(auth_header) \
    -o /tmp/preview_test.pdf -w "%{http_code}" | {
    read status
    print_result $status
    if [ $status -eq 200 ]; then
        echo "Preview saved to /tmp/preview_test.pdf"
    fi
}

# Test 10: WebSocket Connection (using wscat if available)
print_test "WebSocket Tests"
if command -v wscat &> /dev/null; then
    echo "Testing WebSocket connections with wscat..."
    
    # Test job progress WebSocket
    echo "Job Progress WebSocket test (will timeout in 5s):"
    timeout 5 wscat -c "ws://localhost:8000/ws/test-job-id" || true
    
    # Test directory scan WebSocket
    echo "Directory Scan WebSocket test (will timeout in 5s):"
    timeout 5 wscat -c "ws://localhost:8000/ws/scan/test-scan-id" || true
else
    echo "wscat not installed. Install with: npm install -g wscat"
fi

# Test 11: Performance Test - Multiple Searches
print_test "Performance Test - 5 Concurrent Searches"
for i in {1..5}; do
    (
        time curl -s -X POST "$API_BASE/api/search" \
            -H "Content-Type: application/json" \
            $(auth_header) \
            -d "{
                \"query\": \"performance test query $i\",
                \"top_k\": 5
            }" > /dev/null
    ) 2>&1 | grep real &
done
wait

# Test 12: Error Handling
print_test "Error Handling Tests"

# Invalid job ID
echo "Testing invalid job ID:"
curl -s -X GET "$API_BASE/api/jobs/invalid-job-id" \
    -H "Content-Type: application/json" \
    $(auth_header) | jq '.' || echo "Failed to parse JSON"

# Invalid search parameters
echo "Testing invalid search parameters:"
curl -s -X POST "$API_BASE/api/search" \
    -H "Content-Type: application/json" \
    $(auth_header) \
    -d '{
        "query": "",
        "top_k": -1
    }' | jq '.' || echo "Failed to parse JSON"

# Test 13: File Upload (if supported)
print_test "File Upload Test"
echo "test content" > /tmp/test_upload.txt
curl -s -X POST "$API_BASE/api/upload" \
    $(auth_header) \
    -F "file=@/tmp/test_upload.txt" | jq '.' || echo "Upload endpoint not available"
rm /tmp/test_upload.txt

# Test 14: Cancel Job
if [ -n "$JOB_ID" ]; then
    print_test "Cancel Job"
    curl -s -X POST "$API_BASE/api/jobs/$JOB_ID/cancel" \
        -H "Content-Type: application/json" \
        $(auth_header) | jq '.' || echo "Failed to parse JSON"
fi

# Summary
echo -e "\n${YELLOW}=== Test Summary ===${NC}"
echo "All tests completed. Check the output above for any failures."
echo "For WebSocket tests, consider using the HTML test files or wscat."

# Advanced tests section
echo -e "\n${YELLOW}=== Advanced API Tests ===${NC}"

# Test search with filters
print_test "Search with Filters"
curl -s -X POST "$API_BASE/api/search" \
    -H "Content-Type: application/json" \
    $(auth_header) \
    -d '{
        "query": "test with filters",
        "top_k": 10,
        "filters": {
            "file_types": [".pdf", ".txt"],
            "created_after": "2024-01-01"
        }
    }' | jq '.' || echo "Filters not supported"

# Test batch operations (if available)
print_test "Batch Job Creation"
curl -s -X POST "$API_BASE/api/jobs/batch" \
    -H "Content-Type: application/json" \
    $(auth_header) \
    -d '{
        "jobs": [
            {
                "job_name": "Batch Job 1",
                "directory": "/tmp/dir1",
                "glob_patterns": ["*.txt"]
            },
            {
                "job_name": "Batch Job 2",
                "directory": "/tmp/dir2",
                "glob_patterns": ["*.pdf"]
            }
        ]
    }' | jq '.' || echo "Batch operations not supported"

# Test metrics endpoint
print_test "System Metrics"
curl -s -X GET "$API_BASE/api/metrics" \
    -H "Content-Type: application/json" \
    $(auth_header) | jq '.' || echo "Metrics endpoint not available"

echo -e "\n${GREEN}All tests completed!${NC}"
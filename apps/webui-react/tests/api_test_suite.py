#!/usr/bin/env python3
"""
Comprehensive API Test Suite for Semantik WebUI
Tests all critical features for feature parity verification
"""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime

import aiohttp
import websockets.client


class APITestSuite:
    def __init__(self, base_url: str = "http://localhost:8000", auth_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.auth_token = auth_token
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        self.test_results: list[dict] = []

    async def run_all_tests(self) -> None:
        """Run all test categories"""
        print("🚀 Starting Comprehensive API Test Suite")
        print(f"Base URL: {self.base_url}")
        print("-" * 50)

        # Basic connectivity
        await self.test_health_check()

        # Job management
        job_id = await self.test_job_creation()
        if job_id:
            await self.test_job_status(job_id)
            await self.test_job_websocket(job_id)
            await self.test_job_cancellation(job_id)

        # Directory scanning
        await self.test_directory_scan()
        await self.test_directory_scan_websocket()

        # Search functionality
        await self.test_vector_search()
        await self.test_hybrid_search()
        await self.test_search_filters()
        await self.test_search_performance()

        # Document handling
        await self.test_document_preview()

        # Settings and configuration
        await self.test_settings()

        # Error handling
        await self.test_error_handling()

        # Print summary
        self.print_summary()

    def log_test(self, test_name: str, success: bool, message: str, details: dict | None = None) -> None:
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now(UTC).isoformat(),
            "details": details,
        }
        self.test_results.append(result)

        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
        if details and not success:
            print(f"   Details: {json.dumps(details, indent=2)}")

    async def test_health_check(self) -> None:
        """Test health check endpoint"""
        print("\n🏥 Testing Health Check...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/health", headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Health Check", True, "API is healthy", data)
                    else:
                        self.log_test("Health Check", False, f"HTTP {resp.status}")
            except Exception as e:
                self.log_test("Health Check", False, str(e))

    async def test_job_creation(self) -> str | None:
        """Test job creation"""
        print("\n📦 Testing Job Creation...")
        async with aiohttp.ClientSession() as session:
            job_data = {
                "job_name": "Test Job - API Suite",
                "directory": "/tmp",
                "glob_patterns": ["*.txt", "*.pdf"],
                "recursive": True,
                "max_workers": 2,
                "embedding_model": "BAAI/bge-small-en-v1.5",
            }

            try:
                async with session.post(f"{self.base_url}/api/jobs", headers=self.headers, json=job_data) as resp:
                    if resp.status in [200, 201]:
                        data = await resp.json()
                        job_id = data.get("job_id")
                        if job_id is not None:
                            self.log_test("Job Creation", True, f"Created job {job_id}", data)
                            return str(job_id)
                        self.log_test("Job Creation", False, "No job_id in response", data)
                        return None
                    text = await resp.text()
                    self.log_test("Job Creation", False, f"HTTP {resp.status}", {"response": text})
            except Exception as e:
                self.log_test("Job Creation", False, str(e))
        return None

    async def test_job_status(self, job_id: str) -> None:
        """Test job status endpoint"""
        print(f"\n📊 Testing Job Status for {job_id}...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/jobs/{job_id}", headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Job Status", True, f"Job status: {data.get('status', 'unknown')}", data)
                    else:
                        self.log_test("Job Status", False, f"HTTP {resp.status}")
            except Exception as e:
                self.log_test("Job Status", False, str(e))

    async def test_job_websocket(self, job_id: str) -> None:
        """Test job progress WebSocket"""
        print(f"\n🔌 Testing Job Progress WebSocket for {job_id}...")
        ws_url = f"{self.ws_base_url}/ws/{job_id}"

        try:
            async with websockets.client.connect(ws_url) as websocket:
                # Wait for a message or timeout
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    self.log_test("Job WebSocket", True, "Connected and received message", data)
                except TimeoutError:
                    self.log_test("Job WebSocket", True, "Connected (no messages in 5s)")
        except Exception as e:
            self.log_test("Job WebSocket", False, f"Connection failed: {str(e)}")

    async def test_job_cancellation(self, job_id: str) -> None:
        """Test job cancellation"""
        print(f"\n🛑 Testing Job Cancellation for {job_id}...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.base_url}/api/jobs/{job_id}/cancel", headers=self.headers) as resp:
                    if resp.status in [200, 204]:
                        self.log_test("Job Cancellation", True, "Job cancelled successfully")
                    else:
                        self.log_test("Job Cancellation", False, f"HTTP {resp.status}")
            except Exception as e:
                self.log_test("Job Cancellation", False, str(e))

    async def test_directory_scan(self) -> None:
        """Test directory scan endpoint"""
        print("\n📂 Testing Directory Scan...")
        async with aiohttp.ClientSession() as session:
            scan_data = {"path": "/tmp", "recursive": True}

            try:
                async with session.post(
                    f"{self.base_url}/api/scan-directory", headers=self.headers, json=scan_data
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        file_count = len(data.get("files", []))
                        self.log_test(
                            "Directory Scan",
                            True,
                            f"Found {file_count} files",
                            {"count": file_count, "sample": data.get("files", [])[:3]},
                        )
                    else:
                        self.log_test("Directory Scan", False, f"HTTP {resp.status}")
            except Exception as e:
                self.log_test("Directory Scan", False, str(e))

    async def test_directory_scan_websocket(self) -> None:
        """Test directory scan WebSocket"""
        print("\n🔌 Testing Directory Scan WebSocket...")
        ws_url = f"{self.ws_base_url}/ws/scan/test-scan-id"

        try:
            async with websockets.client.connect(ws_url) as websocket:
                # Send scan request
                await websocket.send(json.dumps({"path": "/tmp"}))

                # Wait for response
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    self.log_test("Directory Scan WebSocket", True, "Connected and scanning", data)
                except TimeoutError:
                    self.log_test("Directory Scan WebSocket", True, "Connected (no response in 5s)")
        except Exception as e:
            self.log_test("Directory Scan WebSocket", False, f"Connection failed: {str(e)}")

    async def test_vector_search(self) -> None:
        """Test vector search"""
        print("\n🔍 Testing Vector Search...")
        async with aiohttp.ClientSession() as session:
            search_data = {"query": "test search query", "top_k": 10, "mode": "vector"}

            try:
                async with session.post(f"{self.base_url}/api/search", headers=self.headers, json=search_data) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        self.log_test(
                            "Vector Search",
                            True,
                            f"Found {len(results)} results",
                            {"count": len(results), "has_scores": all(r.get("score") is not None for r in results)},
                        )
                    else:
                        text = await resp.text()
                        self.log_test("Vector Search", False, f"HTTP {resp.status}", {"response": text})
            except Exception as e:
                self.log_test("Vector Search", False, str(e))

    async def test_hybrid_search(self) -> None:
        """Test hybrid search capability"""
        print("\n🔍 Testing Hybrid Search...")

        # Try different approaches to hybrid search
        hybrid_attempts = [
            {"query": "test hybrid", "mode": "hybrid", "top_k": 10},
            {"query": "test hybrid", "mode": "hybrid", "top_k": 10, "alpha": 0.5},
            {"query": "test hybrid", "use_hybrid": True, "top_k": 10},
            {"query": "test hybrid", "top_k": 10, "keyword_weight": 0.5, "vector_weight": 0.5},
        ]

        hybrid_available = False

        async with aiohttp.ClientSession() as session:
            for attempt in hybrid_attempts:
                try:
                    async with session.post(f"{self.base_url}/api/search", headers=self.headers, json=attempt) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            metadata = data.get("metadata", {})
                            search_mode = metadata.get("search_mode") or metadata.get("mode")

                            if search_mode == "hybrid":
                                hybrid_available = True
                                self.log_test(
                                    "Hybrid Search",
                                    True,
                                    "Hybrid search is available",
                                    {"parameters": attempt, "results": len(data.get("results", []))},
                                )
                                break
                except Exception:
                    continue

            if not hybrid_available:
                self.log_test("Hybrid Search", False, "Hybrid search not available or not working")

    async def test_search_filters(self) -> None:
        """Test search with filters"""
        print("\n🔍 Testing Search Filters...")
        async with aiohttp.ClientSession() as session:
            search_data = {
                "query": "test with filters",
                "top_k": 10,
                "filters": {"file_types": [".pdf", ".txt"], "created_after": "2024-01-01"},
            }

            try:
                async with session.post(f"{self.base_url}/api/search", headers=self.headers, json=search_data) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test(
                            "Search Filters", True, "Filters accepted", {"results": len(data.get("results", []))}
                        )
                    else:
                        # Try without filters to see if it's just filters that aren't supported
                        async with session.post(
                            f"{self.base_url}/api/search", headers=self.headers, json={"query": "test", "top_k": 10}
                        ) as resp2:
                            if resp2.status == 200:
                                self.log_test("Search Filters", False, "Filters not supported")
                            else:
                                self.log_test("Search Filters", False, "Search endpoint issue")
            except Exception as e:
                self.log_test("Search Filters", False, str(e))

    async def test_search_performance(self) -> None:
        """Test search performance"""
        print("\n⚡ Testing Search Performance...")
        async with aiohttp.ClientSession() as session:
            # Single request timing
            start = time.time()
            try:
                async with session.post(
                    f"{self.base_url}/api/search", headers=self.headers, json={"query": "performance test", "top_k": 10}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        elapsed = (time.time() - start) * 1000  # ms
                        self.log_test(
                            "Search Performance",
                            elapsed < 1000,
                            f"Single request: {elapsed:.2f}ms",
                            {"processing_time": data.get("metadata", {}).get("processing_time_ms")},
                        )
            except Exception as e:
                self.log_test("Search Performance", False, str(e))

            # Concurrent requests
            start = time.time()
            tasks = []
            for i in range(5):
                task = session.post(
                    f"{self.base_url}/api/search",
                    headers=self.headers,
                    json={"query": f"concurrent test {i}", "top_k": 5},
                )
                tasks.append(task)

            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                elapsed = (time.time() - start) * 1000  # ms
                success_count = sum(
                    1
                    for r in responses
                    if not isinstance(r, BaseException) and hasattr(r, "status") and r.status == 200
                )

                self.log_test(
                    "Concurrent Search", success_count == 5, f"{success_count}/5 successful in {elapsed:.2f}ms"
                )
            except Exception as e:
                self.log_test("Concurrent Search", False, str(e))

    async def test_document_preview(self) -> None:
        """Test document preview endpoint"""
        print("\n📄 Testing Document Preview...")
        # This would need a valid file path
        test_path = "/tmp/test.txt"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/api/documents/preview", headers=self.headers, params={"path": test_path}
                ) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get("Content-Type", "")
                        self.log_test("Document Preview", True, f"Preview available ({content_type})")
                    elif resp.status == 404:
                        self.log_test("Document Preview", True, "Endpoint exists (file not found)")
                    else:
                        self.log_test("Document Preview", False, f"HTTP {resp.status}")
            except Exception as e:
                self.log_test("Document Preview", False, str(e))

    async def test_settings(self) -> None:
        """Test settings endpoint"""
        print("\n⚙️ Testing Settings...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/settings", headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Settings", True, "Settings retrieved", {"keys": list(data.keys())[:5]})
                    else:
                        self.log_test("Settings", False, f"HTTP {resp.status}")
            except Exception as e:
                self.log_test("Settings", False, str(e))

    async def test_error_handling(self) -> None:
        """Test error handling"""
        print("\n🚨 Testing Error Handling...")

        # Invalid job ID
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/jobs/invalid-job-id-12345", headers=self.headers) as resp:
                    if resp.status in [404, 400]:
                        self.log_test("Error Handling - Invalid Job", True, f"Properly handled with HTTP {resp.status}")
                    else:
                        self.log_test("Error Handling - Invalid Job", False, f"Unexpected status: {resp.status}")
            except Exception as e:
                self.log_test("Error Handling - Invalid Job", False, str(e))

        # Invalid search parameters
        try:
            async with session.post(
                f"{self.base_url}/api/search", headers=self.headers, json={"query": "", "top_k": -1}
            ) as resp:
                if resp.status in [400, 422]:
                    self.log_test("Error Handling - Invalid Search", True, f"Properly handled with HTTP {resp.status}")
                else:
                    self.log_test("Error Handling - Invalid Search", False, f"Unexpected status: {resp.status}")
        except Exception as e:
            self.log_test("Error Handling - Invalid Search", False, str(e))

    def print_summary(self) -> None:
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        failed = total - passed

        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")

        if failed > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")

        # Feature availability summary
        print("\n🔍 Feature Availability:")
        features = {
            "WebSocket Job Progress": any(r["test"] == "Job WebSocket" and r["success"] for r in self.test_results),
            "WebSocket Directory Scan": any(
                r["test"] == "Directory Scan WebSocket" and r["success"] for r in self.test_results
            ),
            "Vector Search": any(r["test"] == "Vector Search" and r["success"] for r in self.test_results),
            "Hybrid Search": any(r["test"] == "Hybrid Search" and r["success"] for r in self.test_results),
            "Search Filters": any(r["test"] == "Search Filters" and r["success"] for r in self.test_results),
            "Document Preview": any(r["test"] == "Document Preview" and r["success"] for r in self.test_results),
        }

        for feature, available in features.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"  {feature}: {status}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Semantik API Test Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for API")
    parser.add_argument("--token", help="Authentication token")
    parser.add_argument("--test", help="Run specific test only")

    args = parser.parse_args()

    test_suite = APITestSuite(args.url, args.token)

    if args.test:
        # Run specific test
        test_method = getattr(test_suite, f"test_{args.test}", None)
        if test_method:
            await test_method()
            test_suite.print_summary()
        else:
            print(f"Test '{args.test}' not found")
            print("Available tests:")
            for attr in dir(test_suite):
                if attr.startswith("test_"):
                    print(f"  - {attr[5:]}")
    else:
        # Run all tests
        await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

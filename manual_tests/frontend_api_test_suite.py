#!/usr/bin/env python3
"""
Manual comprehensive API test suite for Semantik WebUI.

Relocated from apps/webui-react/tests/api_test_suite.py on 2025-10-16. Execute
directly to validate API parity and websocket flows against a running stack.
"""

from __future__ import annotations

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
        print("🚀 Starting Comprehensive API Test Suite")
        print(f"Base URL: {self.base_url}")
        print("-" * 50)

        await self.test_health_check()
        job_id = await self.test_job_creation()
        if job_id:
            await self.test_job_status(job_id)
            await self.test_job_websocket(job_id)
            await self.test_job_cancellation(job_id)
        await self.test_directory_scan()
        await self.test_directory_scan_websocket()
        await self.test_vector_search()
        await self.test_hybrid_search()
        await self.test_search_filters()
        await self.test_search_performance()
        await self.test_document_preview()
        await self.test_settings()
        await self.test_error_handling()
        self.print_summary()

    def log_test(self, test_name: str, success: bool, message: str, details: dict | None = None) -> None:
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
        print("\n🏥 Testing Health Check...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/health", headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Health Check", True, "API is healthy", data)
                    else:
                        self.log_test("Health Check", False, f"HTTP {resp.status}")
            except Exception as exc:
                self.log_test("Health Check", False, str(exc))

    async def test_job_creation(self) -> str | None:
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
                    if resp.status in (200, 201):
                        data = await resp.json()
                        job_id = data.get("job_id")
                        if job_id is not None:
                            self.log_test("Job Creation", True, f"Created job {job_id}", data)
                            return str(job_id)
                        self.log_test("Job Creation", False, "No job_id in response", data)
                        return None
                    text = await resp.text()
                    self.log_test("Job Creation", False, f"HTTP {resp.status}", {"response": text})
            except Exception as exc:
                self.log_test("Job Creation", False, str(exc))
        return None

    async def test_job_status(self, job_id: str) -> None:
        print(f"\n📊 Testing Job Status for {job_id}...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/jobs/{job_id}", headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Job Status", True, f"Job status: {data.get('status', 'unknown')}", data)
                    else:
                        self.log_test("Job Status", False, f"HTTP {resp.status}")
            except Exception as exc:
                self.log_test("Job Status", False, str(exc))

    async def test_job_websocket(self, job_id: str) -> None:
        print(f"\n🔌 Testing Job Progress WebSocket for {job_id}...")
        ws_url = f"{self.ws_base_url}/ws/{job_id}"
        try:
            async with websockets.client.connect(ws_url) as websocket:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    self.log_test("Job WebSocket", True, "Connected and received message", data)
                except TimeoutError:
                    self.log_test("Job WebSocket", True, "Connected (no messages in 5s)")
        except Exception as exc:
            self.log_test("Job WebSocket", False, f"Connection failed: {exc}")

    async def test_job_cancellation(self, job_id: str) -> None:
        print(f"\n🛑 Testing Job Cancellation for {job_id}...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.base_url}/api/jobs/{job_id}/cancel", headers=self.headers) as resp:
                    if resp.status in (200, 204):
                        self.log_test("Job Cancellation", True, "Job cancelled successfully")
                    else:
                        self.log_test("Job Cancellation", False, f"HTTP {resp.status}")
            except Exception as exc:
                self.log_test("Job Cancellation", False, str(exc))

    async def test_directory_scan(self) -> None:
        print("\n📂 Testing Directory Scan...")
        async with aiohttp.ClientSession() as session:
            scan_data = {"path": "/tmp", "recursive": True}
            try:
                async with session.post(f"{self.base_url}/api/scan-directory", headers=self.headers, json=scan_data) as resp:
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
            except Exception as exc:
                self.log_test("Directory Scan", False, str(exc))

    async def test_directory_scan_websocket(self) -> None:
        print("\n🔌 Testing Directory Scan WebSocket...")
        ws_url = f"{self.ws_base_url}/ws/scan/test-scan-id"
        try:
            async with websockets.client.connect(ws_url) as websocket:
                try:
                    await websocket.send(json.dumps({"action": "start", "path": "/tmp"}))
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    self.log_test("Directory Scan WebSocket", True, "Received message", data)
                except TimeoutError:
                    self.log_test("Directory Scan WebSocket", True, "Connected (no messages in 5s)")
        except Exception as exc:
            self.log_test("Directory Scan WebSocket", False, f"Connection failed: {exc}")

    async def test_vector_search(self) -> None:
        print("\n🔍 Testing Vector Search...")
        async with aiohttp.ClientSession() as session:
            payload = {"query": "test", "collection": "test_collection", "top_k": 5}
            try:
                async with session.post(f"{self.base_url}/api/search/vector", headers=self.headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Vector Search", True, "Vector search succeeded", data)
                    else:
                        self.log_test("Vector Search", False, f"HTTP {resp.status}")
            except Exception as exc:
                self.log_test("Vector Search", False, str(exc))

    async def test_hybrid_search(self) -> None:
        print("\n🧪 Testing Hybrid Search...")
        async with aiohttp.ClientSession() as session:
            payload = {"query": "test", "collection": "test_collection", "top_k": 5, "hybrid": True}
            try:
                async with session.post(f"{self.base_url}/api/search/hybrid", headers=self.headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Hybrid Search", True, "Hybrid search succeeded", data)
                    else:
                        self.log_test("Hybrid Search", False, f"HTTP {resp.status}")
            except Exception as exc:
                self.log_test("Hybrid Search", False, str(exc))

    async def test_search_filters(self) -> None:
        print("\n🎯 Testing Search Filters...")
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": "test",
                "collection": "test_collection",
                "top_k": 5,
                "filters": {"document_type": ["pdf"]},
            }
            try:
                async with session.post(f"{self.base_url}/api/search/vector", headers=self.headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Search Filters", True, "Filter search succeeded", data)
                    else:
                        self.log_test("Search Filters", False, f"HTTP {resp.status}")
            except Exception as exc:
                self.log_test("Search Filters", False, str(exc))

    async def test_search_performance(self) -> None:
        print("\n⏱️ Testing Search Performance...")
        async with aiohttp.ClientSession() as session:
            payload = {"query": "performance test", "collection": "test_collection", "top_k": 20}
            try:
                start = time.perf_counter()
                async with session.post(f"{self.base_url}/api/search/vector", headers=self.headers, json=payload) as resp:
                    elapsed = time.perf_counter() - start
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test(
                            "Search Performance",
                            True,
                            f"Completed in {elapsed:.2f}s",
                            {"elapsed_seconds": elapsed, "result_count": len(data.get("results", []))},
                        )
                    else:
                        self.log_test("Search Performance", False, f"HTTP {resp.status}")
            except Exception as exc:
                self.log_test("Search Performance", False, str(exc))

    async def test_document_preview(self) -> None:
        print("\n🗂️ Testing Document Preview...")
        async with aiohttp.ClientSession() as session:
            payload = {"document_id": "test-doc-id", "collection": "test_collection"}
            try:
                async with session.post(f"{self.base_url}/api/documents/preview", headers=self.headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Document Preview", True, "Preview retrieved", data)
                    else:
                        self.log_test("Document Preview", False, f"HTTP {resp.status}")
            except Exception as exc:
                self.log_test("Document Preview", False, str(exc))

    async def test_settings(self) -> None:
        print("\n⚙️ Testing Settings...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/settings", headers=self.headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.log_test("Settings", True, "Fetched settings", data)
                    else:
                        self.log_test("Settings", False, f"HTTP {resp.status}")
            except Exception as exc:
                self.log_test("Settings", False, str(exc))

    async def test_error_handling(self) -> None:
        print("\n🚨 Testing Error Handling...")
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/nonexistent", headers=self.headers) as resp:
                    if resp.status == 404:
                        self.log_test("Error Handling", True, "Received 404 for nonexistent endpoint")
                    else:
                        self.log_test("Error Handling", False, f"Expected 404, got {resp.status}")
            except Exception as exc:
                self.log_test("Error Handling", False, str(exc))

    def print_summary(self) -> None:
        print("\n📊 Test Summary")
        success_count = sum(1 for result in self.test_results if result["success"])
        print(f"Passed: {success_count}/{len(self.test_results)}")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}: {result['message']}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Semantik WebUI API test suite manually.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for API requests.")
    parser.add_argument("--auth-token", default=None, help="Optional bearer token for authenticated requests.")
    args = parser.parse_args()

    suite = APITestSuite(base_url=args.base_url, auth_token=args.auth_token)
    await suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

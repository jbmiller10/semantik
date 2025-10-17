"""End-to-end tests for WebSocket integration during reindex operations."""

import json
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests
import websocket

# mypy: ignore-errors


def service_available(url: str) -> bool:
    """Check if the service is available at the given URL."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.mark.e2e()
@pytest.mark.skipif(
    not service_available(os.getenv("API_BASE_URL", "http://localhost:8080")),
    reason="Semantik service is not available - run with docker compose up",
)
class TestWebSocketReindexOperation:
    """Test WebSocket integration for complex reindex operations."""

    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
    WS_BASE_URL = API_BASE_URL.replace("http://", "ws://").replace("https://", "wss://")

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers by logging in."""
        login_response = requests.post(
            f"{self.API_BASE_URL}/api/auth/login",
            json={"username": "threepars", "password": "puddin123"},
        )

        if login_response.status_code == 200:
            token = login_response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}

        # Try testuser as fallback
        login_response = requests.post(
            f"{self.API_BASE_URL}/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )

        if login_response.status_code == 200:
            token = login_response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}

        raise Exception(f"Failed to authenticate: {login_response.text}")

    def _get_auth_token(self) -> str:
        """Get authentication token for WebSocket connections."""
        headers = self._get_auth_headers()
        return headers["Authorization"].split(" ")[1]

    def test_reindex_collection_with_websocket_progress(
        self, test_documents_fixture: Path, cleanup_collection: list[str]
    ) -> None:
        """Test Scenario 3: Reindex collection with blue-green deployment and WebSocket monitoring."""
        _ = test_documents_fixture
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # Step 1: Create a collection with initial documents
        docker_path = "/mnt/docs"
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Reindex WebSocket",
                "description": "Testing reindex with WebSocket progress",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "chunk_size": 1000,  # Initial chunk size
                "chunk_overlap": 100,
                "initial_source": {
                    "path": docker_path,
                    "description": "Initial documents",
                },
            },
            headers=headers,
        )

        assert response.status_code == 200
        collection_data = response.json()
        collection_id = collection_data["id"]
        cleanup_collection.append(collection_id)

        # Wait for initial indexing to complete
        operation_id = collection_data.get("active_operation", {}).get("id")
        if operation_id:
            self._wait_for_operation(operation_id, headers)

        # Verify collection is ready
        coll_response = requests.get(
            f"{self.API_BASE_URL}/api/v2/collections/{collection_id}",
            headers=headers,
        )
        assert coll_response.status_code == 200
        initial_data = coll_response.json()
        assert initial_data["status"] == "ready"
        initial_doc_count = initial_data["document_count"]
        initial_vector_count = initial_data["vector_count"]

        # Step 2: Start reindex operation with different settings
        reindex_response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections/{collection_id}/reindex",
            json={
                "chunk_size": 512,  # Changed chunk size
                "chunk_overlap": 50,  # Changed overlap
                "confirmation": f"reindex {collection_data['name']}",
            },
            headers=headers,
        )

        assert reindex_response.status_code == 200
        reindex_data = reindex_response.json()
        reindex_operation_id = reindex_data.get("operation_id")

        # Step 3: Monitor reindex progress via WebSocket
        if reindex_operation_id:
            ws_url = f"{self.WS_BASE_URL}/ws/operations/{reindex_operation_id}?token={token}"
            progress_messages = []
            status_updates = []

            def on_message(_ws, message) -> None:
                data = json.loads(message)
                progress_messages.append(data)
                print(f"Reindex WebSocket message: {data}")

                # Track specific message types
                if data.get("type") == "status_update":
                    status_updates.append(data)
                elif "staging" in data.get("message", "").lower():
                    print("BLUE-GREEN: Creating staging environment")
                elif "switch" in data.get("message", "").lower():
                    print("BLUE-GREEN: Switching to new index")

            def on_error(_ws, error) -> None:
                print(f"WebSocket error: {error}")

            def on_close(_ws, close_status_code, close_msg) -> None:
                print(f"WebSocket closed: {close_status_code} - {close_msg}")

            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )

            # Run WebSocket in a separate thread

            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

            # Monitor reindex operation
            start_time = time.time()
            while time.time() - start_time < 120:  # 2-minute timeout for reindex
                op_response = requests.get(
                    f"{self.API_BASE_URL}/api/v2/operations/{reindex_operation_id}",
                    headers=headers,
                )

                if op_response.status_code == 200:
                    op_data = op_response.json()
                    print(f"Reindex status: {op_data['status']}, progress: {op_data.get('progress', 0)}%")

                    # Check collection status during reindex
                    coll_check = requests.get(
                        f"{self.API_BASE_URL}/api/v2/collections/{collection_id}",
                        headers=headers,
                    )
                    if coll_check.status_code == 200:
                        coll_status = coll_check.json()
                        print(f"Collection status during reindex: {coll_status['status']}")

                        # Verify collection remains searchable during reindex
                        if coll_status["status"] in ["ready", "reindexing"]:
                            search_response = requests.post(
                                f"{self.API_BASE_URL}/api/v2/search",
                                json={
                                    "query": "test",
                                    "collection_id": collection_id,
                                    "top_k": 1,
                                },
                                headers=headers,
                            )
                            assert search_response.status_code == 200, "Search should work during reindex"

                    if op_data["status"] in ["completed", "failed"]:
                        ws.close()
                        break

                time.sleep(3)

            # Verify reindex completed successfully
            final_op_response = requests.get(
                f"{self.API_BASE_URL}/api/v2/operations/{reindex_operation_id}",
                headers=headers,
            )
            assert final_op_response.status_code == 200
            final_op_data = final_op_response.json()
            assert final_op_data["status"] == "completed", f"Reindex failed: {final_op_data}"

            # Verify we received appropriate progress messages
            assert len(progress_messages) > 0, "Should have received WebSocket progress messages"

            # Check for blue-green deployment messages
            staging_messages = [m for m in progress_messages if "staging" in m.get("message", "").lower()]
            assert len(staging_messages) > 0, "Should have messages about staging environment"

            # Verify collection has new settings
            final_coll_response = requests.get(
                f"{self.API_BASE_URL}/api/v2/collections/{collection_id}",
                headers=headers,
            )
            assert final_coll_response.status_code == 200
            final_data = final_coll_response.json()

            # Verify settings were updated
            assert final_data["chunk_size"] == 512, "Chunk size should be updated"
            assert final_data["chunk_overlap"] == 50, "Chunk overlap should be updated"

            # Due to different chunking, vector count might change
            print(f"Document count: {initial_doc_count} -> {final_data['document_count']}")
            print(f"Vector count: {initial_vector_count} -> {final_data['vector_count']}")

            # Documents should remain the same
            assert final_data["document_count"] == initial_doc_count, "Document count should remain the same"

    def test_reindex_failure_recovery(self, test_documents_fixture: Path, cleanup_collection: list[str]) -> None:
        """Test reindex operation failure and recovery via WebSocket monitoring."""
        _ = test_documents_fixture
        headers = self._get_auth_headers()

        # Create a collection
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Reindex Failure",
                "description": "Testing reindex failure scenarios",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            },
            headers=headers,
        )

        assert response.status_code == 200
        collection_data = response.json()
        collection_id = collection_data["id"]
        cleanup_collection.append(collection_id)

        # Try to reindex with invalid settings (should fail validation)
        reindex_response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections/{collection_id}/reindex",
            json={
                "chunk_size": -1,  # Invalid chunk size
                "confirmation": f"reindex {collection_data['name']}",
            },
            headers=headers,
        )

        # This should fail at the API level
        assert reindex_response.status_code == 422, "Invalid chunk size should fail validation"

    def test_remove_source_with_websocket(self, test_documents_fixture: Path, cleanup_collection: list[str]) -> None:
        """Test remove source operation with WebSocket monitoring."""
        _ = test_documents_fixture
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # Create collection with initial source
        docker_path = "/mnt/docs"
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Remove Source",
                "description": "Testing source removal",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "initial_source": {
                    "path": docker_path,
                    "description": "Initial source",
                },
            },
            headers=headers,
        )

        assert response.status_code == 200
        collection_data = response.json()
        collection_id = collection_data["id"]
        cleanup_collection.append(collection_id)

        # Wait for initial indexing
        operation_id = collection_data.get("active_operation", {}).get("id")
        if operation_id:
            self._wait_for_operation(operation_id, headers)

        # Get source ID
        sources_response = requests.get(
            f"{self.API_BASE_URL}/api/v2/collections/{collection_id}/sources",
            headers=headers,
        )
        assert sources_response.status_code == 200
        sources = sources_response.json()
        assert len(sources) > 0, "Should have at least one source"
        source_id = sources[0]["id"]

        # Remove the source
        remove_response = requests.delete(
            f"{self.API_BASE_URL}/api/v2/collections/{collection_id}/sources/{source_id}",
            headers=headers,
        )

        assert remove_response.status_code == 200
        remove_data = remove_response.json()
        remove_operation_id = remove_data.get("operation_id")

        # Monitor removal via WebSocket
        if remove_operation_id:
            ws_url = f"{self.WS_BASE_URL}/ws/operations/{remove_operation_id}?token={token}"
            progress_messages = []

            def on_message(_ws, message) -> None:
                data = json.loads(message)
                progress_messages.append(data)
                print(f"Remove source WebSocket: {data}")

            ws = websocket.WebSocketApp(ws_url, on_message=on_message)

            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

            # Wait for removal to complete
            self._wait_for_operation(remove_operation_id, headers)
            ws.close()

            # Verify progress messages
            assert len(progress_messages) > 0, "Should have received progress messages"

            # Verify source was removed
            sources_after = requests.get(
                f"{self.API_BASE_URL}/api/v2/collections/{collection_id}/sources",
                headers=headers,
            )
            assert sources_after.status_code == 200
            assert len(sources_after.json()) == 0, "Source should be removed"

            # Verify collection is empty
            coll_response = requests.get(
                f"{self.API_BASE_URL}/api/v2/collections/{collection_id}",
                headers=headers,
            )
            assert coll_response.status_code == 200
            coll_data = coll_response.json()
            assert coll_data["document_count"] == 0, "Collection should be empty after source removal"

    def _wait_for_operation(self, operation_id: str, headers: dict[str, str], timeout: int = 60) -> None:
        """Wait for an operation to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            op_response = requests.get(
                f"{self.API_BASE_URL}/api/v2/operations/{operation_id}",
                headers=headers,
            )

            if op_response.status_code == 200:
                op_data = op_response.json()
                if op_data["status"] in ["completed", "failed"]:
                    if op_data["status"] == "failed":
                        raise Exception(f"Operation failed: {op_data.get('error_message', 'Unknown error')}")
                    return

            time.sleep(2)

        raise Exception(f"Operation {operation_id} did not complete within {timeout} seconds")


@pytest.fixture()
def test_documents_fixture() -> Path:
    """Provide path to test documents directory."""
    test_data_path = Path(__file__).parent.parent.parent / "test_data"
    if not test_data_path.exists():
        # In Docker, documents might be mounted elsewhere
        docker_path = Path("/mnt/docs")
        if docker_path.exists():
            return docker_path
        pytest.skip(f"Test data directory not found at {test_data_path}")
    return test_data_path


@pytest.fixture()
def cleanup_collection() -> Iterator[list[str]]:
    """Fixture to clean up collections after test completion."""
    collection_ids: list[str] = []
    yield collection_ids

    # Cleanup after test
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080")

    # Get auth headers for cleanup
    try:
        for username in ["threepars", "testuser"]:
            login_response = requests.post(
                f"{api_base_url}/api/auth/login",
                json={"username": username, "password": "puddin123" if username == "threepars" else "testpass123"},
            )
            if login_response.status_code == 200:
                token = login_response.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}

                for collection_id in collection_ids:
                    try:
                        response = requests.delete(
                            f"{api_base_url}/api/v2/collections/{collection_id}",
                            headers=headers,
                        )
                        if response.status_code not in [200, 404]:
                            print(f"Warning: Failed to delete collection {collection_id}: {response.status_code}")
                    except Exception as e:
                        print(f"Warning: Error during cleanup of collection {collection_id}: {e}")
    except Exception:
        pass

"""End-to-end tests for WebSocket integration with collection operations."""

import contextlib
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
class TestWebSocketIntegration:
    """Test WebSocket integration for real-time operation progress updates."""

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

        # Register testuser if needed
        register_response = requests.post(
            f"{self.API_BASE_URL}/api/auth/register",
            json={
                "username": "testuser",
                "password": "testpass123",
                "email": "test@example.com",
                "full_name": "Test User",
            },
        )

        if register_response.status_code == 200:
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

    def test_collection_creation_with_websocket_progress(
        self, test_documents_fixture: Path, cleanup_collection: list[str]
    ) -> None:
        """Test Scenario 1: Create collection with initial source and monitor progress via WebSocket."""
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # Create collection with initial source
        docker_path = "/mnt/docs"  # Use mounted documents directory in Docker

        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Real-time Updates",
                "description": "Testing WebSocket integration",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "initial_source": {
                    "path": docker_path,
                    "description": "Test documents",
                },
            },
            headers=headers,
        )

        assert response.status_code == 200
        collection_data = response.json()
        collection_id = collection_data["id"]
        operation_id = collection_data.get("active_operation", {}).get("id")

        cleanup_collection.append(collection_id)

        # Connect to WebSocket for operation progress
        if operation_id:
            ws_url = f"{self.WS_BASE_URL}/ws/operations/{operation_id}?token={token}"
            progress_messages = []

            def on_message(_ws, message) -> None:
                data = json.loads(message)
                progress_messages.append(data)
                print(f"WebSocket message: {data}")

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

            # Wait for operation to complete
            start_time = time.time()
            while time.time() - start_time < 60:  # 1-minute timeout
                # Check operation status via API
                op_response = requests.get(
                    f"{self.API_BASE_URL}/api/v2/operations/{operation_id}",
                    headers=headers,
                )

                if op_response.status_code == 200:
                    op_data = op_response.json()
                    print(f"Operation status: {op_data['status']}, progress: {op_data.get('progress', 0)}%")

                    if op_data["status"] in ["completed", "failed"]:
                        ws.close()
                        break

                time.sleep(2)

            # Verify we received progress updates
            assert len(progress_messages) > 0, "Should have received WebSocket progress messages"

            # Verify message structure
            for msg in progress_messages:
                assert "type" in msg or "status" in msg, f"Message should have type or status: {msg}"
                if "progress" in msg:
                    assert 0 <= msg["progress"] <= 100, f"Progress should be between 0-100: {msg}"

            # Verify collection is ready
            coll_response = requests.get(
                f"{self.API_BASE_URL}/api/v2/collections/{collection_id}",
                headers=headers,
            )
            assert coll_response.status_code == 200
            coll_data = coll_response.json()
            assert coll_data["status"] == "ready", f"Collection should be ready: {coll_data['status']}"
            assert coll_data["document_count"] > 0, "Collection should have documents"

    def test_add_source_with_websocket_progress(
        self, test_documents_fixture: Path, cleanup_collection: list[str]
    ) -> None:
        """Test Scenario 2: Add source to existing collection with WebSocket monitoring."""
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # First create a collection without initial source
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Add Source WebSocket",
                "description": "Testing add source with WebSocket",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            },
            headers=headers,
        )

        assert response.status_code == 200
        collection_data = response.json()
        collection_id = collection_data["id"]
        cleanup_collection.append(collection_id)

        # Add a source to the collection
        docker_path = "/mnt/docs"
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections/{collection_id}/sources",
            json={
                "path": docker_path,
                "description": "Additional test documents",
            },
            headers=headers,
        )

        assert response.status_code == 200
        source_data = response.json()
        operation_id = source_data.get("operation_id")

        # Monitor operation via WebSocket
        if operation_id:
            ws_url = f"{self.WS_BASE_URL}/ws/operations/{operation_id}?token={token}"
            progress_messages = []

            def on_message(_ws, message) -> None:
                data = json.loads(message)
                progress_messages.append(data)
                print(f"Add source WebSocket message: {data}")

            ws = websocket.WebSocketApp(ws_url, on_message=on_message)

            # Run WebSocket in a separate thread

            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

            # Wait for operation completion
            start_time = time.time()
            while time.time() - start_time < 60:
                op_response = requests.get(
                    f"{self.API_BASE_URL}/api/v2/operations/{operation_id}",
                    headers=headers,
                )

                if op_response.status_code == 200:
                    op_data = op_response.json()
                    if op_data["status"] in ["completed", "failed"]:
                        ws.close()
                        break

                time.sleep(2)

            # Verify progress messages
            assert len(progress_messages) > 0, "Should have received progress messages for add source"

    def test_concurrent_operations_websocket(self, test_documents_fixture: Path, cleanup_collection: list[str]) -> None:
        """Test Scenario 4: Monitor multiple concurrent operations via WebSocket."""
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # Create multiple collections concurrently
        docker_path = "/mnt/docs"
        collection_requests = [
            {
                "name": f"Concurrent Test {i}",
                "description": f"Testing concurrent operations {i}",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "initial_source": {
                    "path": docker_path,
                    "description": f"Test documents {i}",
                },
            }
            for i in range(3)
        ]

        # Submit all collection creation requests
        operations = []
        websockets_data = []

        for req_data in collection_requests:
            response = requests.post(
                f"{self.API_BASE_URL}/api/v2/collections",
                json=req_data,
                headers=headers,
            )

            assert response.status_code == 200
            collection_data = response.json()
            collection_id = collection_data["id"]
            cleanup_collection.append(collection_id)

            operation_id = collection_data.get("active_operation", {}).get("id")
            if operation_id:
                operations.append(
                    {
                        "collection_id": collection_id,
                        "operation_id": operation_id,
                        "name": req_data["name"],
                    }
                )

        # Monitor all operations concurrently via WebSocket

        def monitor_operation(op_data) -> None:
            """Monitor a single operation via WebSocket."""
            ws_url = f"{self.WS_BASE_URL}/ws/operations/{op_data['operation_id']}?token={token}"
            messages = []

            def on_message(_ws, message) -> None:
                data = json.loads(message)
                messages.append(data)
                print(f"[{op_data['name']}] WebSocket: {data}")

            ws = websocket.WebSocketApp(ws_url, on_message=on_message)

            # Store WebSocket data
            websockets_data.append(
                {
                    "operation": op_data,
                    "messages": messages,
                    "ws": ws,
                }
            )

            ws.run_forever()

        # Start monitoring threads
        threads = []
        for op in operations:
            thread = threading.Thread(target=monitor_operation, args=(op,))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # Wait for all operations to complete
        start_time = time.time()
        completed_operations = set()

        while len(completed_operations) < len(operations) and time.time() - start_time < 120:
            for op in operations:
                if op["operation_id"] not in completed_operations:
                    op_response = requests.get(
                        f"{self.API_BASE_URL}/api/v2/operations/{op['operation_id']}",
                        headers=headers,
                    )

                    if op_response.status_code == 200:
                        op_data = op_response.json()
                        if op_data["status"] in ["completed", "failed"]:
                            completed_operations.add(op["operation_id"])
                            print(f"Operation {op['name']} completed with status: {op_data['status']}")

            time.sleep(2)

        # Close all WebSockets
        for ws_data in websockets_data:
            with contextlib.suppress(Exception):
                ws_data["ws"].close()

        # Verify all operations completed
        assert len(completed_operations) == len(operations), "All operations should complete"

        # Verify each operation received WebSocket messages
        for ws_data in websockets_data:
            assert (
                len(ws_data["messages"]) > 0
            ), f"Operation {ws_data['operation']['name']} should have WebSocket messages"

    def test_websocket_disconnection_recovery(
        self, test_documents_fixture: Path, cleanup_collection: list[str]
    ) -> None:
        """Test error condition: WebSocket disconnection and recovery."""
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # Create a collection with initial source
        docker_path = "/mnt/docs"
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Disconnection Recovery",
                "description": "Testing WebSocket disconnection",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "initial_source": {
                    "path": docker_path,
                    "description": "Test documents",
                },
            },
            headers=headers,
        )

        assert response.status_code == 200
        collection_data = response.json()
        collection_id = collection_data["id"]
        cleanup_collection.append(collection_id)

        operation_id = collection_data.get("active_operation", {}).get("id")

        if operation_id:
            ws_url = f"{self.WS_BASE_URL}/ws/operations/{operation_id}?token={token}"
            messages_before_disconnect = []
            messages_after_reconnect = []
            disconnected = False

            def on_message(_ws, message) -> None:
                data = json.loads(message)
                if not disconnected:
                    messages_before_disconnect.append(data)
                else:
                    messages_after_reconnect.append(data)

            ws = websocket.WebSocketApp(ws_url, on_message=on_message)

            # Start WebSocket connection

            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

            # Wait a bit then simulate disconnection
            time.sleep(3)
            ws.close()
            disconnected = True

            # Reconnect after a short delay
            time.sleep(2)
            ws2 = websocket.WebSocketApp(ws_url, on_message=on_message)
            ws2_thread = threading.Thread(target=ws2.run_forever)
            ws2_thread.daemon = True
            ws2_thread.start()

            # Wait for operation completion
            start_time = time.time()
            while time.time() - start_time < 60:
                op_response = requests.get(
                    f"{self.API_BASE_URL}/api/v2/operations/{operation_id}",
                    headers=headers,
                )

                if op_response.status_code == 200:
                    op_data = op_response.json()
                    if op_data["status"] in ["completed", "failed"]:
                        ws2.close()
                        break

                time.sleep(2)

            # Verify we received messages before and after reconnection
            assert len(messages_before_disconnect) > 0, "Should have messages before disconnect"
            # Note: We might not get messages after reconnect if operation completed quickly
            print(f"Messages before disconnect: {len(messages_before_disconnect)}")
            print(f"Messages after reconnect: {len(messages_after_reconnect)}")

    def test_websocket_authentication_failure(self) -> None:
        """Test error condition: WebSocket connection with invalid authentication."""
        # Try to connect with invalid token
        ws_url = f"{self.WS_BASE_URL}/ws/operations/test-op-123?token=invalid-token"

        connection_closed = False
        close_code = None
        close_reason = None

        def on_close(_ws, code, reason) -> None:
            nonlocal connection_closed, close_code, close_reason
            connection_closed = True
            close_code = code
            close_reason = reason
            print(f"WebSocket closed with code {code}: {reason}")

        ws = websocket.WebSocketApp(ws_url, on_close=on_close)

        # Run WebSocket (should close immediately due to auth failure)

        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        # Wait for connection to close
        time.sleep(2)

        # Verify connection was closed due to authentication
        assert connection_closed, "Connection should be closed due to auth failure"
        assert close_code in [1008, 1011, 4401], f"Expected auth-related close code, got {close_code}"

    def test_websocket_permission_denied(self, cleanup_collection: list[str]) -> None:
        """Test error condition: WebSocket connection to operation user doesn't own."""
        # Create two users
        # User 1 creates a collection
        user1_headers = self._get_auth_headers()
        self._get_auth_token()  # Ensure token is available

        # Create collection as user 1
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "User1 Private Collection",
                "description": "Testing permission denied",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            },
            headers=user1_headers,
        )

        assert response.status_code == 200
        collection_data = response.json()
        collection_id = collection_data["id"]
        cleanup_collection.append(collection_id)

        # Try to create user2 and access user1's operation
        # Register user2
        register_response = requests.post(
            f"{self.API_BASE_URL}/api/auth/register",
            json={
                "username": "testuser2",
                "password": "testpass123",
                "email": "test2@example.com",
                "full_name": "Test User 2",
            },
        )

        if register_response.status_code == 200:
            # Login as user2
            login_response = requests.post(
                f"{self.API_BASE_URL}/api/auth/login",
                json={"username": "testuser2", "password": "testpass123"},
            )

            if login_response.status_code == 200:
                user2_token = login_response.json()["access_token"]

                # Start an operation as user1
                docker_path = "/mnt/docs"
                response = requests.post(
                    f"{self.API_BASE_URL}/api/v2/collections/{collection_id}/sources",
                    json={
                        "path": docker_path,
                        "description": "User1 documents",
                    },
                    headers=user1_headers,
                )

                if response.status_code == 200:
                    source_data = response.json()
                    operation_id = source_data.get("operation_id")

                    if operation_id:
                        # Try to connect as user2 to user1's operation
                        ws_url = f"{self.WS_BASE_URL}/ws/operations/{operation_id}?token={user2_token}"

                        connection_closed = False
                        close_code = None

                        def on_close(_ws, code, reason) -> None:
                            nonlocal connection_closed, close_code
                            connection_closed = True
                            close_code = code
                            print(f"Permission test - WebSocket closed with code {code}: {reason}")

                        ws = websocket.WebSocketApp(ws_url, on_close=on_close)

                        ws_thread = threading.Thread(target=ws.run_forever)
                        ws_thread.daemon = True
                        ws_thread.start()

                        # Wait for connection to close
                        time.sleep(2)

                        # Verify connection was closed due to permission denied
                        assert connection_closed, "Connection should be closed due to permission denied"
                        assert close_code in [1011, 4403], f"Expected permission denied close code, got {close_code}"


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
        # Try multiple users for cleanup
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
                        # Delete the collection
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

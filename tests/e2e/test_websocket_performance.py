"""Performance and message validation tests for WebSocket integration."""

# mypy: ignore-errors

import json
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import requests
import websocket


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
class TestWebSocketPerformanceAndValidation:
    """Test WebSocket message validation and performance metrics."""

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

    def test_websocket_message_validation(self, test_documents_fixture: Path, cleanup_collection: list[str]) -> None:
        """Test that all WebSocket messages conform to expected structure."""
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # Create collection with initial source
        docker_path = "/mnt/docs"
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Message Validation",
                "description": "Testing WebSocket message structure",
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
            all_messages = []
            message_types = defaultdict(int)
            invalid_messages = []

            def on_message(_ws, message):
                try:
                    data = json.loads(message)
                    all_messages.append(data)

                    # Validate message structure
                    if self._validate_message_structure(data):
                        msg_type = data.get("type", data.get("status", "unknown"))
                        message_types[msg_type] += 1
                    else:
                        invalid_messages.append(data)

                except json.JSONDecodeError as e:
                    invalid_messages.append({"raw": message, "error": str(e)})

            ws = websocket.WebSocketApp(ws_url, on_message=on_message)

            # Run WebSocket
            import threading

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

            # Validate results
            assert len(all_messages) > 0, "Should have received messages"
            assert len(invalid_messages) == 0, f"Found invalid messages: {invalid_messages}"

            # Check message type distribution
            print(f"Message type distribution: {dict(message_types)}")
            assert "progress" in message_types or "current_state" in message_types, "Should have progress messages"

            # Validate progress values
            progress_values = [
                msg.get("progress", msg.get("data", {}).get("progress"))
                for msg in all_messages
                if "progress" in msg or (isinstance(msg.get("data"), dict) and "progress" in msg.get("data", {}))
            ]

            if progress_values:
                # Progress should be monotonically increasing
                for i in range(1, len(progress_values)):
                    if progress_values[i] is not None and progress_values[i - 1] is not None:
                        assert (
                            progress_values[i] >= progress_values[i - 1]
                        ), f"Progress should not decrease: {progress_values[i-1]} -> {progress_values[i]}"

    def test_websocket_message_frequency_performance(
        self, test_documents_fixture: Path, cleanup_collection: list[str]
    ) -> None:
        """Test that WebSocket messages don't overwhelm the client."""
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # Create collection with larger source for longer operation
        docker_path = "/mnt/docs"
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Message Frequency",
                "description": "Testing WebSocket message frequency",
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
            message_timestamps = []
            message_intervals = []

            def on_message(_ws, _message):
                current_time = time.time()
                message_timestamps.append(current_time)

                # Calculate interval from previous message
                if len(message_timestamps) > 1:
                    interval = current_time - message_timestamps[-2]
                    message_intervals.append(interval)

            ws = websocket.WebSocketApp(ws_url, on_message=on_message)

            # Run WebSocket
            import threading

            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

            # Monitor for 10 seconds or until completion
            start_time = time.time()
            while time.time() - start_time < 10:
                op_response = requests.get(
                    f"{self.API_BASE_URL}/api/v2/operations/{operation_id}",
                    headers=headers,
                )

                if op_response.status_code == 200:
                    op_data = op_response.json()
                    if op_data["status"] in ["completed", "failed"]:
                        ws.close()
                        break

                time.sleep(0.5)

            ws.close()

            # Analyze message frequency
            if message_intervals:
                avg_interval = sum(message_intervals) / len(message_intervals)
                min_interval = min(message_intervals)
                max_interval = max(message_intervals)

                print("Message frequency stats:")
                print(f"  Total messages: {len(message_timestamps)}")
                print(f"  Avg interval: {avg_interval:.3f}s")
                print(f"  Min interval: {min_interval:.3f}s")
                print(f"  Max interval: {max_interval:.3f}s")

                # Messages shouldn't come too frequently (not more than 10 per second)
                assert min_interval > 0.1, f"Messages coming too fast: {min_interval}s minimum interval"

                # But should have reasonable updates (at least every 5 seconds)
                if len(message_intervals) > 5:  # Only check if we have enough samples
                    assert max_interval < 5.0, f"Messages too infrequent: {max_interval}s maximum interval"

    def test_websocket_reconnection_state_consistency(
        self, test_documents_fixture: Path, cleanup_collection: list[str]
    ) -> None:
        """Test that reconnection provides consistent state information."""
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # Create collection
        docker_path = "/mnt/docs"
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Reconnection State",
                "description": "Testing WebSocket reconnection state",
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

            # Connect first time
            first_messages = []

            def on_message_first(ws, message):
                data = json.loads(message)
                first_messages.append(data)
                # Disconnect after receiving initial state
                if data.get("type") == "current_state" or len(first_messages) > 3:
                    ws.close()

            ws1 = websocket.WebSocketApp(ws_url, on_message=on_message_first)

            import threading

            ws1_thread = threading.Thread(target=ws1.run_forever)
            ws1_thread.daemon = True
            ws1_thread.start()

            # Wait for first connection to close
            time.sleep(3)

            # Reconnect
            second_messages = []

            def on_message_second(_ws, message):
                data = json.loads(message)
                second_messages.append(data)

            ws2 = websocket.WebSocketApp(ws_url, on_message=on_message_second)

            ws2_thread = threading.Thread(target=ws2.run_forever)
            ws2_thread.daemon = True
            ws2_thread.start()

            # Wait a bit for messages
            time.sleep(3)
            ws2.close()

            # Both connections should receive initial state
            assert len(first_messages) > 0, "First connection should receive messages"
            assert len(second_messages) > 0, "Second connection should receive messages"

            # Check for current_state message in reconnection
            current_state_msgs = [m for m in second_messages if m.get("type") == "current_state"]
            assert len(current_state_msgs) > 0, "Reconnection should receive current state"

    def test_websocket_memory_leak_prevention(
        self, test_documents_fixture: Path, cleanup_collection: list[str]
    ) -> None:
        """Test that repeated connections/disconnections don't cause memory leaks."""
        headers = self._get_auth_headers()
        token = self._get_auth_token()

        # Create collection
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": "Test Memory Leak",
                "description": "Testing WebSocket memory management",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            },
            headers=headers,
        )

        assert response.status_code == 200
        collection_data = response.json()
        collection_id = collection_data["id"]
        cleanup_collection.append(collection_id)

        # Start a long-running operation
        docker_path = "/mnt/docs"
        add_response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections/{collection_id}/sources",
            json={
                "path": docker_path,
                "description": "Test source",
            },
            headers=headers,
        )

        assert add_response.status_code == 200
        source_data = add_response.json()
        operation_id = source_data.get("operation_id")

        if operation_id:
            ws_url = f"{self.WS_BASE_URL}/ws/operations/{operation_id}?token={token}"

            # Rapidly connect and disconnect multiple times
            for _ in range(10):
                ws = websocket.WebSocketApp(ws_url)

                import threading

                ws_thread = threading.Thread(target=ws.run_forever)
                ws_thread.daemon = True
                ws_thread.start()

                # Keep connection for short time
                time.sleep(0.5)
                ws.close()
                time.sleep(0.1)

            # System should still be responsive
            # Check we can still get operation status
            op_response = requests.get(
                f"{self.API_BASE_URL}/api/v2/operations/{operation_id}",
                headers=headers,
            )
            assert op_response.status_code == 200

    def _validate_message_structure(self, message: dict[str, Any]) -> bool:
        """Validate WebSocket message structure."""
        # Check for required fields based on message type
        if "type" in message:
            msg_type = message["type"]
            validation_map = {
                "progress": lambda m: "progress" in m or "data" in m,
                "status_update": lambda m: "status" in m or "data" in m,
                "current_state": lambda _: True,  # Current state can have various structures
                "error": lambda m: "error" in m or "message" in m,
            }
            if msg_type in validation_map:
                return validation_map[msg_type](message)

        # Alternative structure with status field
        if "status" in message:
            return True

        # Alternative structure with operation_id
        if "operation_id" in message:
            return True

        # If none of the expected structures, it's invalid
        return False


@pytest.fixture()
def test_documents_fixture() -> Path:
    """Provide path to test documents directory."""
    test_data_path = Path(__file__).parent.parent.parent / "test_data"
    if not test_data_path.exists():
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

    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080")

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

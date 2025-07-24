"""End-to-end test to capture exact current behavior for regression testing during refactoring."""

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests


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
class TestCurrentSystemBehavior:
    """Capture exact current behavior for regression testing."""

    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers by logging in or registering."""
        # Try to login with default test credentials
        login_response = requests.post(
            f"{self.API_BASE_URL}/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )

        if login_response.status_code == 200:
            token = login_response.json()["access_token"]
            return {"Authorization": f"Bearer {token}"}

        # If login fails, try to register
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
            # Now login with the newly created user
            login_response = requests.post(
                f"{self.API_BASE_URL}/api/auth/login",
                json={"username": "testuser", "password": "testpass123"},
            )
            if login_response.status_code == 200:
                token = login_response.json()["access_token"]
                return {"Authorization": f"Bearer {token}"}

        raise Exception(f"Failed to authenticate: Login: {login_response.text}, Register: {register_response.text}")

    def test_complete_embedding_pipeline(self, test_documents_fixture: Path, cleanup_operation: list[str]) -> None:
        """Test document ingestion through search."""
        # Get authentication headers
        headers = self._get_auth_headers()

        # 1. Create collection
        import uuid
        collection_name = f"E2E Test Collection {uuid.uuid4().hex[:8]}"
        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections",
            json={
                "name": collection_name,
                "description": "End-to-end test for refactoring validation",
            },
            headers=headers,
        )
        if response.status_code != 201:
            print(f"Failed to create collection: {response.status_code} - {response.text}")
        assert response.status_code == 201

        collection_data = response.json()
        print(f"Collection response data: {collection_data}")
        collection_id = collection_data["id"]

        # Store collection_id for cleanup
        cleanup_operation.append(collection_id)

        # Get the initial operation ID if present
        initial_operation_id = collection_data.get("initial_operation_id")

        # Wait for initial INDEX operation to complete
        if initial_operation_id:
            print(f"Waiting for initial INDEX operation {initial_operation_id} to complete...")
            start_time = time.time()
            while time.time() - start_time < 60:  # 1-minute timeout for initial operation
                status_response = requests.get(
                    f"{self.API_BASE_URL}/api/v2/operations/{initial_operation_id}",
                    headers=headers
                )
                if status_response.status_code == 200:
                    op_status = status_response.json()
                    if op_status["status"] in ["completed", "failed"]:
                        print(f"Initial operation {op_status['status']} after {time.time() - start_time:.1f}s")
                        if op_status["status"] == "failed":
                            pytest.fail(f"Initial INDEX operation failed: {op_status.get('error_message', 'Unknown error')}")
                        break
                time.sleep(2)
        else:
            # If no initial_operation_id, wait a bit for any background operations to complete
            print("No initial_operation_id received, waiting 5 seconds for background operations...")
            time.sleep(5)

        # 2. Add source to collection
        # Note: When running in Docker, we use a path accessible inside the container
        # For local development, use test_documents_fixture
        docker_path = "/mnt/docs"  # Use the mounted documents directory in Docker

        response = requests.post(
            f"{self.API_BASE_URL}/api/v2/collections/{collection_id}/sources",
            json={
                "source_path": docker_path,
            },
            headers=headers,
        )
        if response.status_code != 202:
            print(f"Failed to add source: {response.status_code} - {response.text}")
        assert response.status_code == 202

        operation_data = response.json()
        operation_id = operation_data["id"]

        # 3. Wait for completion (with a timeout)
        start_time = time.time()
        while time.time() - start_time < 300:  # 5-minute timeout
            # Get operation status
            status_response = requests.get(f"{self.API_BASE_URL}/api/v2/operations/{operation_id}", headers=headers)
            operation_status = status_response.json()
            print(f"Operation status at {time.time() - start_time:.1f}s: {operation_status['status']}")

            if operation_status["status"] == "completed":
                break
            if operation_status["status"] == "failed":
                pytest.fail(f"Operation failed with error: {operation_status.get('error_message', 'Unknown error')}")

            time.sleep(5)  # Check every 5 seconds
        else:
            pytest.fail(f"Operation did not complete within timeout. Final status: {operation_status}")

        # 4. Verify embeddings by performing a search
        search_response = requests.post(
            f"{self.API_BASE_URL}/api/v2/search/single",
            json={"query": "communication process", "k": 1, "collection_id": collection_id},
            headers=headers,
        )
        assert search_response.status_code == 200
        results = search_response.json()["results"]
        assert len(results) > 0
        # Assert that we got a result with valid file information
        first_result = results[0]
        assert "file_path" in first_result or "path" in first_result.get("payload", {})
        assert first_result.get("file_path") or first_result.get("payload", {}).get("path")  # Non-empty path
        assert "score" in first_result  # Verify we have a relevance score
        assert first_result["score"] > 0  # Score should be positive


@pytest.fixture()
def test_documents_fixture() -> Path:
    """Provide path to test documents directory."""
    test_data_path = Path(__file__).parent.parent.parent / "test_data"
    if not test_data_path.exists():
        pytest.skip(f"Test data directory not found at {test_data_path}")
    return test_data_path


@pytest.fixture()
def cleanup_operation() -> Iterator[list[str]]:
    """Fixture to clean up collections after test completion."""
    collection_ids: list[str] = []
    yield collection_ids

    # Cleanup after test
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080")

    # Get auth headers for cleanup
    try:
        login_response = requests.post(
            f"{api_base_url}/api/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        if login_response.status_code == 200:
            token = login_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
        else:
            headers = {}
    except Exception:
        headers = {}

    for collection_id in collection_ids:
        try:
            # Delete the collection
            response = requests.delete(f"{api_base_url}/api/v2/collections/{collection_id}", headers=headers)
            if response.status_code not in (200, 204):
                print(f"Warning: Failed to delete collection {collection_id}: {response.status_code}")

            # Also try to delete the Qdrant collection
            # Note: This assumes the Qdrant API is accessible
            # In the future, this could be done through the webui API
        except Exception as e:
            print(f"Warning: Error during cleanup of collection {collection_id}: {e}")

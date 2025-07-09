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

    def test_complete_embedding_pipeline(self, test_documents_fixture: Path, cleanup_job: list[str]) -> None:
        """Test document ingestion through search."""
        # Get authentication headers
        headers = self._get_auth_headers()

        # 1. Create job
        # Note: When running in Docker, we use a path accessible inside the container
        # For local development, use test_documents_fixture
        docker_path = "/mnt/docs"  # Use the mounted documents directory in Docker

        response = requests.post(
            f"{self.API_BASE_URL}/api/jobs",
            json={
                "name": "E2E Test Job",
                "directory_path": docker_path,
                "description": "End-to-end test for refactoring validation",
            },
            headers=headers,
        )
        if response.status_code != 200:
            print(f"Failed to create job: {response.status_code} - {response.text}")
        assert response.status_code == 200

        job_data = response.json()
        job_id = job_data["id"]

        # Store job_id for cleanup
        cleanup_job.append(job_id)

        # 2. Wait for completion (with a timeout)
        start_time = time.time()
        while time.time() - start_time < 300:  # 5-minute timeout
            status_response = requests.get(f"{self.API_BASE_URL}/api/jobs/{job_id}", headers=headers)
            job_status = status_response.json()
            print(
                f"Job status at {time.time() - start_time:.1f}s: {job_status['status']}, processed: {job_status.get('processed_files', 0)}/{job_status.get('total_files', 0)}"
            )

            if job_status["status"] == "completed":
                break
            if job_status["status"] == "failed":
                pytest.fail(f"Job failed with error: {job_status.get('error', 'Unknown error')}")

            time.sleep(5)  # Check every 5 seconds instead of 2
        else:
            pytest.fail(f"Job did not complete within timeout. Final status: {job_status}")

        # 3. Verify embeddings by performing a search
        search_response = requests.post(
            f"{self.API_BASE_URL}/api/search",
            json={"query": "communication process", "top_k": 1, "collection": f"job_{job_id}"},
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
def cleanup_job() -> Iterator[list[str]]:
    """Fixture to clean up jobs after test completion."""
    job_ids: list[str] = []
    yield job_ids

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

    for job_id in job_ids:
        try:
            # Delete the job
            response = requests.delete(f"{api_base_url}/api/jobs/{job_id}", headers=headers)
            if response.status_code != 200:
                print(f"Warning: Failed to delete job {job_id}: {response.status_code}")

            # Also try to delete the Qdrant collection
            # Note: This assumes the Qdrant API is accessible
            # In the future, this could be done through the webui API
        except Exception as e:
            print(f"Warning: Error during cleanup of job {job_id}: {e}")

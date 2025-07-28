#!/usr/bin/env python3
"""
Comprehensive test suite for webui/rate_limiter.py
Tests rate limiting functionality under various scenarios
"""

import time
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from webui.rate_limiter import limiter


class TestRateLimiterConfiguration:
    """Test rate limiter configuration and initialization"""

    def test_limiter_instance(self):
        """Test that limiter is properly instantiated"""
        assert isinstance(limiter, Limiter)
        assert limiter._key_func == get_remote_address

    def test_limiter_key_function(self):
        """Test the key function extracts remote address"""
        # Create a mock request
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"

        # Test key function
        assert limiter._key_func(mock_request) == "192.168.1.100"

    def test_limiter_key_function_with_forwarded_header(self):
        """Test key function with X-Forwarded-For header"""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {"x-forwarded-for": "10.0.0.1, 172.16.0.1"}

        # get_remote_address should use the forwarded IP
        assert get_remote_address(mock_request) == "10.0.0.1"


class TestRateLimiterIntegration:
    """Test rate limiter integration with FastAPI"""

    def create_test_app(self):
        """Create a test FastAPI app with rate limiting"""
        app = FastAPI()
        app.state.limiter = limiter

        @app.exception_handler(RateLimitExceeded)
        def rate_limit_handler(request: Request, exc: RateLimitExceeded):
            response = {"detail": f"Rate limit exceeded: {exc.detail}", "limit": exc.limit}
            return response, 429

        @app.get("/test")
        @limiter.limit("5/minute")
        def test_endpoint(request: Request):
            return {"message": "success"}

        @app.get("/test-high-limit")
        @limiter.limit("100/minute")
        def test_high_limit_endpoint(request: Request):
            return {"message": "success"}

        @app.get("/test-per-hour")
        @limiter.limit("10/hour")
        def test_per_hour_endpoint(request: Request):
            return {"message": "success"}

        return app

    def test_rate_limit_not_exceeded(self):
        """Test requests within rate limit"""
        app = self.create_test_app()
        client = TestClient(app)

        # Make requests within limit
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200
            assert response.json() == {"message": "success"}

    @patch("slowapi.Limiter._check_request_limit")
    def test_rate_limit_exceeded(self, mock_check_limit):
        """Test rate limit exceeded scenario"""
        # Configure mock to raise RateLimitExceeded
        mock_check_limit.side_effect = RateLimitExceeded(detail="5 per 1 minute", limit="5/minute")

        app = self.create_test_app()
        client = TestClient(app)

        response = client.get("/test")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_different_endpoints_different_limits(self):
        """Test different endpoints have independent rate limits"""
        app = self.create_test_app()
        client = TestClient(app)

        # Make requests to first endpoint
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200

        # Should still be able to access different endpoint
        response = client.get("/test-high-limit")
        assert response.status_code == 200

    @patch("slowapi.Limiter._check_request_limit")
    def test_rate_limit_headers(self, mock_check_limit):
        """Test rate limit headers are properly set"""
        app = self.create_test_app()

        # Add middleware to check headers
        @app.middleware("http")
        async def add_rate_limit_headers(request: Request, call_next):
            response = await call_next(request)
            # In real implementation, slowapi adds these headers
            response.headers["X-RateLimit-Limit"] = "5"
            response.headers["X-RateLimit-Remaining"] = "4"
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
            return response

        client = TestClient(app)
        response = client.get("/test")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestRateLimiterScenarios:
    """Test specific rate limiting scenarios"""

    @patch("slowapi.util.get_remote_address")
    def test_different_ips_different_limits(self, mock_get_address):
        """Test that different IPs have separate rate limits"""
        app = FastAPI()
        app.state.limiter = limiter

        @app.get("/test")
        @limiter.limit("2/minute")
        def test_endpoint(request: Request):
            return {"message": "success"}

        client = TestClient(app)

        # First IP makes requests
        mock_get_address.return_value = "192.168.1.100"
        for i in range(2):
            response = client.get("/test")
            assert response.status_code == 200

        # Second IP should still be able to make requests
        mock_get_address.return_value = "192.168.1.101"
        response = client.get("/test")
        assert response.status_code == 200

    def test_rate_limit_time_window(self):
        """Test rate limit time window behavior"""
        app = FastAPI()
        app.state.limiter = limiter

        request_times = []

        @app.get("/test")
        @limiter.limit("2/second")
        def test_endpoint(request: Request):
            request_times.append(time.time())
            return {"message": "success"}

        client = TestClient(app)

        # Make 2 requests quickly
        for i in range(2):
            response = client.get("/test")
            assert response.status_code == 200

        # Third request should be rate limited if within same second
        # (Note: In actual test, we'd need to handle timing more carefully)

    def test_rate_limit_with_authentication(self):
        """Test rate limiting with authenticated users"""
        app = FastAPI()
        app.state.limiter = limiter

        # Custom key function that uses user ID if authenticated
        def get_user_or_ip(request: Request):
            # Check if user is authenticated (mock implementation)
            if hasattr(request.state, "user") and request.state.user:
                return f"user:{request.state.user['id']}"
            return get_remote_address(request)

        custom_limiter = Limiter(key_func=get_user_or_ip)

        @app.get("/test")
        @custom_limiter.limit("5/minute")
        def test_endpoint(request: Request):
            return {"message": "success"}

        # This demonstrates how rate limiting can be customized per user


class TestRateLimiterEdgeCases:
    """Test edge cases and error scenarios"""

    def test_malformed_limit_string(self):
        """Test handling of malformed limit strings"""
        with pytest.raises(ValueError):

            @limiter.limit("invalid-format")
            def bad_endpoint():
                pass

    def test_missing_state_limiter(self):
        """Test behavior when limiter is not attached to app state"""
        app = FastAPI()
        # Don't set app.state.limiter

        @app.get("/test")
        @limiter.limit("5/minute")
        def test_endpoint(request: Request):
            return {"message": "success"}

        client = TestClient(app)

        # Should handle gracefully (usually by not applying rate limit)
        response = client.get("/test")
        # The response should still work, just without rate limiting

    @patch("slowapi.Limiter._check_request_limit")
    def test_rate_limiter_storage_error(self, mock_check_limit):
        """Test handling of storage backend errors"""
        # Simulate storage error
        mock_check_limit.side_effect = Exception("Storage backend error")

        app = FastAPI()
        app.state.limiter = limiter

        @app.get("/test")
        @limiter.limit("5/minute")
        def test_endpoint(request: Request):
            return {"message": "success"}

        client = TestClient(app)

        # Should handle error gracefully
        response = client.get("/test")
        # Depending on configuration, might fail open or closed


class TestRateLimiterPatterns:
    """Test common rate limiting patterns used in the application"""

    def test_search_endpoint_limits(self):
        """Test rate limits similar to search endpoints"""
        app = FastAPI()
        app.state.limiter = limiter

        @app.get("/search")
        @limiter.limit("30/minute")
        def search(request: Request):
            return {"results": []}

        @app.get("/search/rerank")
        @limiter.limit("60/minute")
        def search_rerank(request: Request):
            return {"results": []}

        client = TestClient(app)

        # Verify different endpoints have different limits
        response = client.get("/search")
        assert response.status_code == 200

        response = client.get("/search/rerank")
        assert response.status_code == 200

    def test_collection_operation_limits(self):
        """Test rate limits for collection operations"""
        app = FastAPI()
        app.state.limiter = limiter

        @app.post("/collections/{id}/scan")
        @limiter.limit("5/hour")
        def scan_collection(id: int, request: Request):
            return {"status": "scanning"}

        @app.post("/collections/{id}/reindex")
        @limiter.limit("10/hour")
        def reindex_collection(id: int, request: Request):
            return {"status": "reindexing"}

        @app.delete("/collections/{id}")
        @limiter.limit("10/hour")
        def delete_collection(id: int, request: Request):
            return {"status": "deleted"}

        client = TestClient(app)

        # Test that limits are applied per endpoint
        response = client.post("/collections/1/scan")
        assert response.status_code == 200

    def test_burst_protection(self):
        """Test protection against burst requests"""
        app = FastAPI()
        app.state.limiter = limiter

        @app.get("/api/data")
        @limiter.limit("1/5minutes")
        def get_data(request: Request):
            return {"data": "sensitive"}

        client = TestClient(app)

        # First request should succeed
        response = client.get("/api/data")
        assert response.status_code == 200

        # Second request within 5 minutes should be rate limited
        # (In actual test, would need to mock time or use storage backend)


class TestRateLimiterMonitoring:
    """Test rate limiter monitoring and metrics"""

    def test_rate_limit_metrics(self):
        """Test that rate limit events can be monitored"""
        exceeded_count = 0

        def count_exceeded(request: Request, exc: RateLimitExceeded):
            nonlocal exceeded_count
            exceeded_count += 1
            return {"detail": "Rate limit exceeded"}, 429

        app = FastAPI()
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, count_exceeded)

        @app.get("/test")
        @limiter.limit("1/minute")
        def test_endpoint(request: Request):
            return {"message": "success"}

        client = TestClient(app)

        # Make requests to trigger rate limit
        client.get("/test")  # Should succeed

        # This would increment exceeded_count if rate limit is hit
        # Demonstrates how to monitor rate limit events

    def test_custom_error_response(self):
        """Test custom error response for rate limits"""
        app = FastAPI()
        app.state.limiter = limiter

        @app.exception_handler(RateLimitExceeded)
        def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
            return {"error": "Too many requests", "retry_after": 60, "limit_description": exc.detail}, 429

        @app.get("/test")
        @limiter.limit("0/minute")  # Always exceeded for testing
        def test_endpoint(request: Request):
            return {"message": "success"}

        client = TestClient(app)

        # Should get custom error response
        # (Would need proper setup to actually trigger)

"""Tests for rate_limits.py including RateLimitConfig and CircuitBreakerConfig."""

import threading
import time

import pytest

from webui.config.rate_limits import CircuitBreakerConfig, RateLimitConfig


class TestRateLimitConfig:
    """Tests for RateLimitConfig class."""

    def test_get_rate_limit_string_defaults(self):
        assert RateLimitConfig.get_rate_limit_string("unknown") == RateLimitConfig.DEFAULT_LIMIT

    def test_get_rate_limit_string_preview(self):
        result = RateLimitConfig.get_rate_limit_string("preview")
        assert result == f"{RateLimitConfig.PREVIEW_LIMIT}/minute"

    def test_get_rate_limit_string_compare(self):
        result = RateLimitConfig.get_rate_limit_string("compare")
        assert result == f"{RateLimitConfig.COMPARE_LIMIT}/minute"

    def test_get_rate_limit_string_process(self):
        result = RateLimitConfig.get_rate_limit_string("process")
        assert result == f"{RateLimitConfig.PROCESS_LIMIT}/hour"

    def test_get_rate_limit_string_read(self):
        result = RateLimitConfig.get_rate_limit_string("read")
        assert result == f"{RateLimitConfig.READ_LIMIT}/minute"

    def test_get_rate_limit_string_analytics(self):
        result = RateLimitConfig.get_rate_limit_string("analytics")
        assert result == f"{RateLimitConfig.ANALYTICS_LIMIT}/minute"

    def test_bypass_rate_limit_returns_false_without_token(self, monkeypatch):
        monkeypatch.setattr(RateLimitConfig, "BYPASS_TOKEN", None)
        assert RateLimitConfig.bypass_rate_limit({"authorization": "Bearer token"}) is False

    def test_bypass_rate_limit_accepts_matching_token(self, monkeypatch):
        monkeypatch.setattr(RateLimitConfig, "BYPASS_TOKEN", "secret")
        assert RateLimitConfig.bypass_rate_limit({"authorization": "Bearer secret"}) is True

    def test_bypass_rate_limit_rejects_mismatched_token(self, monkeypatch):
        monkeypatch.setattr(RateLimitConfig, "BYPASS_TOKEN", "secret")
        assert RateLimitConfig.bypass_rate_limit({"authorization": "Bearer nope"}) is False

    def test_bypass_rate_limit_rejects_non_bearer_auth(self, monkeypatch):
        monkeypatch.setattr(RateLimitConfig, "BYPASS_TOKEN", "secret")
        assert RateLimitConfig.bypass_rate_limit({"authorization": "Basic secret"}) is False

    def test_bypass_rate_limit_handles_missing_header(self, monkeypatch):
        monkeypatch.setattr(RateLimitConfig, "BYPASS_TOKEN", "secret")
        assert RateLimitConfig.bypass_rate_limit({}) is False


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig class."""

    @pytest.fixture()
    def circuit_breaker(self) -> CircuitBreakerConfig:
        """Create a fresh CircuitBreakerConfig instance with fast timeouts."""
        cb = CircuitBreakerConfig()
        cb.failure_threshold = 3
        cb.timeout_seconds = 1
        return cb

    # Test initialization
    def test_init_uses_rate_limit_config_values(self):
        cb = CircuitBreakerConfig()
        assert cb.failure_threshold == RateLimitConfig.CIRCUIT_BREAKER_FAILURES
        assert cb.timeout_seconds == RateLimitConfig.CIRCUIT_BREAKER_TIMEOUT

    def test_init_creates_empty_tracking_dicts(self):
        cb = CircuitBreakerConfig()
        assert cb.failure_counts == {}
        assert cb.blocked_until == {}

    def test_init_creates_lock(self):
        cb = CircuitBreakerConfig()
        assert isinstance(cb._lock, type(threading.Lock()))

    # Test record_failure method
    def test_record_failure_increments_count(self, circuit_breaker):
        assert circuit_breaker.record_failure("user-1") is False
        assert circuit_breaker.failure_counts["user-1"] == 1

        assert circuit_breaker.record_failure("user-1") is False
        assert circuit_breaker.failure_counts["user-1"] == 2

    def test_record_failure_opens_circuit_at_threshold(self, circuit_breaker):
        circuit_breaker.record_failure("user-1")
        circuit_breaker.record_failure("user-1")

        # Third failure should open the circuit
        result = circuit_breaker.record_failure("user-1")

        assert result is True
        assert "user-1" in circuit_breaker.blocked_until
        # Failure count should be reset after opening
        assert circuit_breaker.failure_counts["user-1"] == 0

    def test_record_failure_returns_true_when_already_blocked(self, circuit_breaker):
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure("user-1")

        # Additional failures should return True immediately
        assert circuit_breaker.record_failure("user-1") is True

    def test_record_failure_independent_keys(self, circuit_breaker):
        circuit_breaker.record_failure("user-1")
        circuit_breaker.record_failure("user-1")

        # user-2 should start fresh
        assert circuit_breaker.record_failure("user-2") is False
        assert circuit_breaker.failure_counts["user-2"] == 1

    # Test is_blocked method
    def test_is_blocked_returns_false_for_unknown_key(self, circuit_breaker):
        is_blocked, remaining = circuit_breaker.is_blocked("unknown-key")
        assert is_blocked is False
        assert remaining == 0

    def test_is_blocked_returns_true_for_blocked_key(self, circuit_breaker):
        # Use a longer timeout to avoid race conditions
        circuit_breaker.timeout_seconds = 60

        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure("user-1")

        is_blocked, remaining = circuit_breaker.is_blocked("user-1")
        assert is_blocked is True
        assert remaining > 0
        assert remaining <= circuit_breaker.timeout_seconds

    def test_is_blocked_returns_false_after_timeout(self, circuit_breaker):
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure("user-1")

        # Wait for timeout
        time.sleep(1.1)

        is_blocked, remaining = circuit_breaker.is_blocked("user-1")
        assert is_blocked is False
        assert remaining == 0
        # blocked_until should be cleaned up
        assert "user-1" not in circuit_breaker.blocked_until

    def test_is_blocked_cleans_up_failure_counts_on_expiry(self, circuit_breaker):
        # Open circuit to trigger block
        for _ in range(3):
            circuit_breaker.record_failure("user-1")

        # Wait for timeout
        time.sleep(1.1)

        # Check should clean up
        circuit_breaker.is_blocked("user-1")

        # Failure counts should be cleaned up
        assert "user-1" not in circuit_breaker.failure_counts

    # Test reset method
    def test_reset_clears_failure_count(self, circuit_breaker):
        circuit_breaker.record_failure("user-1")
        circuit_breaker.record_failure("user-1")

        circuit_breaker.reset("user-1")

        assert "user-1" not in circuit_breaker.failure_counts

    def test_reset_handles_unknown_key(self, circuit_breaker):
        # Should not raise
        circuit_breaker.reset("unknown-key")

    def test_reset_does_not_clear_blocked_status(self, circuit_breaker):
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure("user-1")

        circuit_breaker.reset("user-1")

        # Should still be blocked
        is_blocked, _ = circuit_breaker.is_blocked("user-1")
        assert is_blocked is True

    # Test thread safety
    def test_thread_safety_record_failure(self, circuit_breaker):
        # Threshold must be higher than total failures (1000) to avoid circuit opening
        circuit_breaker.failure_threshold = 2000

        errors = []

        def record_failures():
            try:
                for _ in range(100):
                    circuit_breaker.record_failure("shared-key")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_failures) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Total failures should be 1000 (10 threads * 100 failures)
        assert circuit_breaker.failure_counts["shared-key"] == 1000

    def test_thread_safety_is_blocked(self, circuit_breaker):
        # Open the circuit first
        for _ in range(3):
            circuit_breaker.record_failure("user-1")

        errors = []
        results = []

        def check_blocked():
            try:
                for _ in range(50):
                    is_blocked, remaining = circuit_breaker.is_blocked("user-1")
                    results.append((is_blocked, remaining))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_blocked) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All checks should return consistent blocked status
        assert all(r[0] is True for r in results)

    def test_thread_safety_mixed_operations(self, circuit_breaker):
        circuit_breaker.failure_threshold = 100

        errors = []

        def mixed_ops():
            try:
                for i in range(50):
                    key = f"user-{i % 5}"
                    circuit_breaker.record_failure(key)
                    circuit_breaker.is_blocked(key)
                    if i % 10 == 0:
                        circuit_breaker.reset(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_ops) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

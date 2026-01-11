from __future__ import annotations

import pytest

from webui.tasks import error_classifier
from webui.tasks.error_classifier import ErrorCategory, classify_error, get_retry_delay, is_retryable


def test_classify_error_string_patterns() -> None:
    assert classify_error("Connection timed out") == ErrorCategory.TRANSIENT
    assert classify_error("File not found: doc.pdf") == ErrorCategory.PERMANENT
    assert classify_error("some unexpected failure") == ErrorCategory.UNKNOWN


@pytest.mark.skipif(not error_classifier.HTTPX_AVAILABLE, reason="httpx not available")
def test_classify_error_httpx_status_and_request_errors() -> None:
    httpx = error_classifier.httpx
    assert httpx is not None

    request = httpx.Request("GET", "http://example.com")
    transient_response = httpx.Response(503, request=request)
    permanent_response = httpx.Response(404, request=request)

    transient_exc = httpx.HTTPStatusError("boom", request=request, response=transient_response)
    permanent_exc = httpx.HTTPStatusError("boom", request=request, response=permanent_response)

    assert classify_error(transient_exc) == ErrorCategory.TRANSIENT
    assert classify_error(permanent_exc) == ErrorCategory.PERMANENT

    request_error = httpx.RequestError("network down", request=request)
    assert classify_error(request_error) == ErrorCategory.TRANSIENT


def test_is_retryable_behaviour() -> None:
    assert is_retryable("file not found", max_retries=3, current_retry=0) is False
    assert is_retryable("connection reset", max_retries=3, current_retry=0) is True

    # Unknown errors retry only up to half the max retries (rounded down) + 1
    assert is_retryable("mysterious", max_retries=4, current_retry=2) is True
    assert is_retryable("mysterious", max_retries=4, current_retry=3) is False


def test_get_retry_delay_caps_at_max() -> None:
    assert get_retry_delay(0, base_delay=2.0, max_delay=30.0) == 2.0
    assert get_retry_delay(1, base_delay=2.0, max_delay=30.0) == 4.0
    assert get_retry_delay(10, base_delay=2.0, max_delay=30.0) == 30.0

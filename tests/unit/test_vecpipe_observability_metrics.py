"""Unit tests for VecPipe observability metrics.

Tests that the new Prometheus metrics are properly registered and configured.
"""

from prometheus_client import Counter, Histogram

from vecpipe.search.metrics import (
    collection_metadata_fetch_latency,
    get_or_create_metric,
    gpu_free_probe_latency,
    payload_fetch_latency,
    qdrant_ad_hoc_client_total,
    rerank_content_fetch_latency,
)


class TestMetricRegistration:
    """Test that metrics are registered with correct names and types."""

    def test_gpu_free_probe_latency_registered(self) -> None:
        """Test that gpu_free_probe_latency is registered as a Histogram."""
        assert gpu_free_probe_latency is not None
        assert isinstance(gpu_free_probe_latency, Histogram)
        assert gpu_free_probe_latency._name == "semantik_gpu_free_probe_seconds"

    def test_qdrant_ad_hoc_client_total_registered(self) -> None:
        """Test that qdrant_ad_hoc_client_total is registered as a Counter with labels."""
        assert qdrant_ad_hoc_client_total is not None
        assert isinstance(qdrant_ad_hoc_client_total, Counter)
        # Note: Counter internal _name doesn't include _total suffix (added by prometheus-client)
        assert qdrant_ad_hoc_client_total._name == "semantik_qdrant_ad_hoc_client"
        # Verify it has the location label
        assert "location" in qdrant_ad_hoc_client_total._labelnames

    def test_payload_fetch_latency_registered(self) -> None:
        """Test that payload_fetch_latency is registered as a Histogram."""
        assert payload_fetch_latency is not None
        assert isinstance(payload_fetch_latency, Histogram)
        assert payload_fetch_latency._name == "semantik_payload_fetch_seconds"

    def test_rerank_content_fetch_latency_registered(self) -> None:
        """Test that rerank_content_fetch_latency is registered as a Histogram."""
        assert rerank_content_fetch_latency is not None
        assert isinstance(rerank_content_fetch_latency, Histogram)
        assert rerank_content_fetch_latency._name == "semantik_rerank_content_fetch_seconds"

    def test_collection_metadata_fetch_latency_registered(self) -> None:
        """Test that collection_metadata_fetch_latency is registered as a Histogram."""
        assert collection_metadata_fetch_latency is not None
        assert isinstance(collection_metadata_fetch_latency, Histogram)
        assert collection_metadata_fetch_latency._name == "semantik_collection_metadata_fetch_seconds"


class TestQdrantAdHocClientLabels:
    """Test that the ad-hoc client counter has proper label support."""

    def test_location_labels_work(self) -> None:
        """Test that location labels can be applied without error."""
        # These should not raise any exceptions
        labeled = qdrant_ad_hoc_client_total.labels(location="sparse_config")
        assert labeled is not None

        labeled = qdrant_ad_hoc_client_total.labels(location="sparse_search")
        assert labeled is not None

        labeled = qdrant_ad_hoc_client_total.labels(location="payload_fetch")
        assert labeled is not None

        labeled = qdrant_ad_hoc_client_total.labels(location="collection_info")
        assert labeled is not None

        labeled = qdrant_ad_hoc_client_total.labels(location="metadata_fetch")
        assert labeled is not None

        labeled = qdrant_ad_hoc_client_total.labels(location="search_utils")
        assert labeled is not None


class TestGetOrCreateMetricIdempotence:
    """Test that get_or_create_metric is idempotent."""

    def test_returns_same_instance_for_histogram(self) -> None:
        """Test that calling get_or_create_metric twice returns the same Histogram."""
        metric1 = get_or_create_metric(
            Histogram,
            "semantik_gpu_free_probe_seconds",
            "Time to probe GPU free memory",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
        )
        metric2 = get_or_create_metric(
            Histogram,
            "semantik_gpu_free_probe_seconds",
            "Time to probe GPU free memory",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
        )
        # Should be the same object (idempotent)
        assert metric1 is metric2

    def test_counter_already_registered(self) -> None:
        """Test that the counter is properly registered and usable."""
        # The counter was already created at module load time
        # Verify it's a valid counter that can be used with labels
        assert qdrant_ad_hoc_client_total is not None
        assert isinstance(qdrant_ad_hoc_client_total, Counter)

        # Should be able to use labels without error
        labeled = qdrant_ad_hoc_client_total.labels(location="test")
        assert labeled is not None


class TestHistogramBuckets:
    """Test that histogram buckets are appropriately configured."""

    def test_gpu_probe_has_sub_millisecond_buckets(self) -> None:
        """GPU probe latency should have sub-millisecond resolution buckets."""
        # The GPU probe operation should be fast (< 100ms typically)
        buckets = gpu_free_probe_latency._upper_bounds
        # Should have buckets starting at 0.0001s (0.1ms)
        assert min(buckets[:-1]) <= 0.0001  # Ignore +Inf bucket

    def test_payload_fetch_has_second_scale_buckets(self) -> None:
        """Payload fetch latency should have second-scale buckets for network ops."""
        buckets = payload_fetch_latency._upper_bounds
        # Should have buckets up to at least 1 second
        assert max(buckets[:-1]) >= 1.0  # Ignore +Inf bucket

    def test_collection_metadata_fetch_has_appropriate_buckets(self) -> None:
        """Collection metadata fetch should have sub-second to second buckets."""
        buckets = collection_metadata_fetch_latency._upper_bounds
        # Should have buckets in the 10ms to 2s range
        assert min(buckets[:-1]) <= 0.01  # 10ms
        assert max(buckets[:-1]) >= 1.0  # 1s

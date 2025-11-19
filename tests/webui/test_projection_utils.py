"""Focused tests for helper utilities in packages.webui.tasks.projection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import packages.webui.tasks.projection as projection_module


@pytest.mark.parametrize(
    ("value", "expected_iso"),
    [
        (datetime(2024, 1, 2, tzinfo=UTC), "2024-01-02T00:00:00+00:00"),
        (datetime(2024, 1, 3, 15, tzinfo=None), "2024-01-03T15:00:00+00:00"),
        (1_700_000_000, datetime.fromtimestamp(1_700_000_000, tz=UTC).isoformat()),
        (
            1_700_000_000_000_000,
            datetime.fromtimestamp(1_700_000_000_000_000 / 1_000_000, tz=UTC).isoformat(),
        ),
        ("2024-03-04T05:06:07Z", "2024-03-04T05:06:07+00:00"),
    ],
)
def test_parse_timestamp_variants(value: Any, expected_iso: str) -> None:
    """Various timestamp representations should normalize to UTC datetimes."""

    parsed = projection_module._parse_timestamp(value)
    assert parsed is not None
    assert parsed.isoformat() == expected_iso


def test_parse_timestamp_invalid_values() -> None:
    """Invalid timestamp payloads should return ``None``."""

    assert projection_module._parse_timestamp("not-a-date") is None
    assert projection_module._parse_timestamp(object()) is None


def test_bucket_age_boundaries() -> None:
    """_bucket_age should map deltas into the documented buckets."""

    now = datetime(2024, 5, 1, tzinfo=UTC)
    expectations = [
        (now + timedelta(seconds=1), "future"),
        (now - timedelta(hours=12), "≤1d"),
        (now - timedelta(days=3), "≤7d"),
        (now - timedelta(days=25), "≤30d"),
        (now - timedelta(days=75), "≤90d"),
        (now - timedelta(days=150), "≤180d"),
        (now - timedelta(days=360), "≤1y"),
        (now - timedelta(days=500), ">1y"),
    ]

    for timestamp, label in expectations:
        assert projection_module._bucket_age(timestamp, now) == label


def test_extract_source_dir_prefers_parent_directory() -> None:
    payload = {"source_path": "/docs/reports/summary.pdf"}
    assert projection_module._extract_source_dir(payload) == "reports"

    payload = {"metadata": {"source_path": "/archive/2024"}}
    assert projection_module._extract_source_dir(payload) == "2024"


def test_extract_source_dir_handles_missing_values() -> None:
    payload: dict[str, Any] = {"source_path": None}
    assert projection_module._extract_source_dir(payload) == projection_module.UNKNOWN_CATEGORY_LABEL


def test_extract_filetype_prefers_mime_and_path_fallback() -> None:
    payload = {"mime_type": "Application/JSON"}
    assert projection_module._extract_filetype(payload) == "application/json"

    payload = {"path": "/tmp/MyDoc.PDF"}
    assert projection_module._extract_filetype(payload) == "pdf"

    payload = {}
    assert projection_module._extract_filetype(payload) == projection_module.UNKNOWN_CATEGORY_LABEL


def test_extract_age_bucket_looks_through_metadata() -> None:
    now = datetime(2024, 5, 1, tzinfo=UTC)
    payload = {"metadata": {"created_at": "2024-04-29T10:00:00Z"}}

    assert projection_module._extract_age_bucket(payload, now) == "≤7d"

    payload = {}
    assert projection_module._extract_age_bucket(payload, now) == projection_module.UNKNOWN_CATEGORY_LABEL


def test_derive_category_label_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2024, 5, 1, tzinfo=UTC)
    payload = {
        "doc_id": "doc-123",
        "source_path": "/root/folder/file.txt",
        "mime_type": "text/plain",
        "ingested_at": now.isoformat(),
    }

    label, doc_id = projection_module._derive_category_label(payload, "document_id", now)
    assert label == "doc-123"
    assert doc_id == "doc-123"

    label, doc_id = projection_module._derive_category_label(payload, "source_dir", now)
    assert label == "folder"
    assert doc_id is None

    label, doc_id = projection_module._derive_category_label(payload, "filetype", now)
    assert label == "text/plain"
    assert doc_id is None

    label, doc_id = projection_module._derive_category_label(payload, "age_bucket", now)
    assert label == "≤1d"
    assert doc_id is None

    # Unexpected color_by values should fall back to document semantics
    label, doc_id = projection_module._derive_category_label(payload, "unknown-mode", now)
    assert label == "doc-123"
    assert doc_id == "doc-123"


def test_derive_category_label_handles_missing_payload() -> None:
    label, doc_id = projection_module._derive_category_label(None, "document_id", datetime.now(UTC))
    assert label == projection_module.UNKNOWN_CATEGORY_LABEL
    assert doc_id is None


def test_ensure_float32_preserves_dtype_when_possible() -> None:
    base = np.array([1.0, 2.0], dtype=np.float32)
    converted = projection_module._ensure_float32(base)
    assert converted.dtype == np.float32
    assert converted is base

    base64 = np.array([1.0, 2.0], dtype=np.float64)
    converted = projection_module._ensure_float32(base64)
    assert converted.dtype == np.float32


def test_compute_pca_projection_downsamples_large_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(projection_module, "PCA_SVD_SAMPLE_LIMIT", 2)
    original_svd = np.linalg.svd
    calls: list[tuple[int, int]] = []

    def recording_svd(matrix: np.ndarray, *args: Any, **kwargs: Any):
        calls.append(matrix.shape)
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", recording_svd)

    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)
    result = projection_module._compute_pca_projection(vectors)

    assert calls  # Ensure the downsampled SVD path ran
    assert calls[0][0] == projection_module.PCA_SVD_SAMPLE_LIMIT
    assert result["projection"].shape == (6, 2)
    assert result["projection"].dtype == np.float32


@pytest.mark.parametrize(
    "vectors",
    [np.ones((1, 3), dtype=np.float32), np.ones((3, 1), dtype=np.float32)],
)
def test_compute_pca_projection_validates_input(vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        projection_module._compute_pca_projection(vectors)


def test_compute_umap_projection_invokes_reducer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, Any] = {}

    class DummyReducer:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
            return np.column_stack((vectors[:, 0], vectors[:, 0] + 1.0))

    monkeypatch.setattr(projection_module, "umap", SimpleNamespace(UMAP=DummyReducer))

    vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    result = projection_module._compute_umap_projection(
        vectors,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
    )

    assert captured_kwargs["n_neighbors"] == 15
    assert result["projection"].shape == (2, 2)
    assert result["projection"].dtype == np.float32


def test_compute_umap_projection_requires_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(projection_module, "umap", None)
    with pytest.raises(RuntimeError):
        projection_module._compute_umap_projection(
            np.ones((2, 3), dtype=np.float32),
            n_neighbors=5,
            min_dist=0.1,
            metric="euclidean",
        )


def test_compute_tsne_projection_with_learning_rate_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordingTSNE:
        def __init__(
            self,
            *,
            n_components: int,
            perplexity: float,
            metric: str,
            init: str,
            random_state: int,
            learning_rate: float,
            n_iter: int,
            square_distances: bool,
        ) -> None:
            self.kwargs = {
                "n_components": n_components,
                "perplexity": perplexity,
                "learning_rate": learning_rate,
                "n_iter": n_iter,
                "square_distances": square_distances,
            }
            self.kl_divergence_ = 0.42
            self.n_iter_ = n_iter

        def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
            return np.column_stack((vectors[:, 0], vectors[:, 1]))

    monkeypatch.setattr(projection_module, "TSNE", RecordingTSNE)

    vectors = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32)
    result = projection_module._compute_tsne_projection(
        vectors,
        perplexity=50.0,
        learning_rate=5.0,
        n_iter=100,
        metric="euclidean",
        init="random",
    )

    assert result["projection"].shape == (3, 2)
    assert result["projection"].dtype == np.float32
    # Perplexity must be clamped to n_samples - 1 == 2
    assert result["perplexity"] == 2.0
    assert result["learning_rate"] == 10.0  # Raised to minimum
    assert result["n_iter"] == 250  # Raised to minimum iterations
    assert result["kl_divergence"] == pytest.approx(0.42)


def test_compute_tsne_projection_with_legacy_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    class LegacyTSNE:
        def __init__(
            self,
            *,
            n_components: int,
            perplexity: float,
            metric: str,
            init: str,
            random_state: int,
            learning_rate_init: float,
            max_iter: int,
        ) -> None:
            self.kwargs = {
                "learning_rate_init": learning_rate_init,
                "max_iter": max_iter,
            }
            self.kl_divergence_ = float("nan")
            self.n_iter = max_iter

        def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
            # Return deterministic layout to ensure dtype conversion still applies
            return np.column_stack((vectors[:, 0], vectors[:, 0] * 0.5))

    monkeypatch.setattr(projection_module, "TSNE", LegacyTSNE)

    vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = projection_module._compute_tsne_projection(
        vectors,
        perplexity=1.0,
        learning_rate=15.0,
        n_iter=500,
        metric="euclidean",
        init="unknown",
    )

    assert result["projection"].shape == (2, 2)
    assert result["projection"].dtype == np.float32
    assert result["learning_rate"] == 15.0
    # With only two points, the perplexity should settle at 1.0
    assert result["perplexity"] == 1.0
    assert result["n_iter"] == 500


@pytest.mark.asyncio()
async def test_operation_updates_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class DummyUpdater:
        def __init__(self, operation_id: str) -> None:
            self.operation_id = operation_id

        async def __aenter__(self) -> "DummyUpdater":
            events.append(f"enter:{self.operation_id}")
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            events.append(f"exit:{self.operation_id}")

    monkeypatch.setattr(projection_module, "CeleryTaskWithOperationUpdates", DummyUpdater)

    async with projection_module._operation_updates("op-123") as updater:
        assert isinstance(updater, DummyUpdater)
        assert updater.operation_id == "op-123"

    assert events == ["enter:op-123", "exit:op-123"]


@pytest.mark.asyncio()
async def test_operation_updates_noop_without_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    class FailingUpdater:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - ensures no instantiation
            nonlocal called
            called = True
            raise AssertionError("Should not be constructed")

    monkeypatch.setattr(projection_module, "CeleryTaskWithOperationUpdates", FailingUpdater)

    async with projection_module._operation_updates(None) as updater:
        assert updater is None

    assert called is False

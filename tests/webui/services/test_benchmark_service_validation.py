"""
Tests for BenchmarkService validation rules.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.database.exceptions import ValidationError
from shared.database.models import MappingStatus
from webui.services.benchmark_service import BenchmarkService


class TestBenchmarkServiceValidation:
    @pytest.mark.asyncio()
    async def test_rejects_top_k_values_below_max_metrics_k(self) -> None:
        db_session = AsyncMock()

        benchmark_repo = AsyncMock()
        benchmark_dataset_repo = AsyncMock()
        collection_repo = AsyncMock()
        operation_repo = AsyncMock()
        search_service = AsyncMock()

        mapping = MagicMock()
        mapping.id = 123
        mapping.dataset_id = "dataset-1"
        mapping.collection_id = "collection-1"
        mapping.mapping_status = MappingStatus.RESOLVED.value
        benchmark_dataset_repo.get_mapping.return_value = mapping
        benchmark_dataset_repo.get_by_uuid_for_user.return_value = MagicMock()

        collection_repo.get_by_uuid.return_value = MagicMock()

        service = BenchmarkService(
            db_session=db_session,
            benchmark_repo=benchmark_repo,
            benchmark_dataset_repo=benchmark_dataset_repo,
            collection_repo=collection_repo,
            operation_repo=operation_repo,
            search_service=search_service,
        )

        with pytest.raises(ValidationError, match=r"top_k_values must be >= max\(k_values_for_metrics\)"):
            await service.create_benchmark(
                user_id=1,
                mapping_id=123,
                name="Test",
                description=None,
                config_matrix={
                    "primary_k": 10,
                    "k_values_for_metrics": [10, 20],
                    "search_modes": ["dense"],
                    "use_reranker": [False],
                    "top_k_values": [5],
                    "rrf_k_values": [60],
                    "score_thresholds": [None],
                },
                top_k=10,
            )

        benchmark_repo.create.assert_not_called()

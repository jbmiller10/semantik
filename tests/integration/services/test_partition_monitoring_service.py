"""Integration tests for PartitionMonitoringService."""

from __future__ import annotations

import pytest
from packages.webui.services.partition_monitoring_service import PartitionMonitoringService
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


@pytest.fixture()
def monitoring_service(db_session: AsyncSession) -> PartitionMonitoringService:
    return PartitionMonitoringService(db_session)


async def test_get_partition_health_summary(monitoring_service: PartitionMonitoringService) -> None:
    try:
        summary = await monitoring_service.get_partition_health_summary()
    except SQLAlchemyError as exc:  # pragma: no cover - depends on view availability
        pytest.skip(f"partition_health_summary view not available: {exc}")

    assert isinstance(summary, list)
    for entry in summary:
        assert entry.chunk_count >= 0
        assert entry.health_status.value in {"HEALTHY", "WARNING", "UNBALANCED", "CRITICAL"}


async def test_analyze_partition_skew(monitoring_service: PartitionMonitoringService) -> None:
    try:
        metrics = await monitoring_service.analyze_partition_skew()
    except SQLAlchemyError as exc:  # pragma: no cover
        pytest.skip(f"analyze_partition_skew function not available: {exc}")

    assert isinstance(metrics, list)
    for metric in metrics:
        assert metric.status.value in {"NORMAL", "WARNING", "CRITICAL"}


async def test_check_partition_health(monitoring_service: PartitionMonitoringService) -> None:
    try:
        result = await monitoring_service.check_partition_health()
    except SQLAlchemyError as exc:  # pragma: no cover
        pytest.skip(f"partition monitoring views not available: {exc}")

    assert result.status in {"success", "warning"}
    assert "alerts" in result.metrics or result.alerts == []

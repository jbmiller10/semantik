"""Integration tests for PartitionMonitoringService using synthetic database views."""

from __future__ import annotations

import pytest
from packages.webui.services.partition_monitoring_service import (
    MonitoringResult,
    PartitionHealthStatus,
    PartitionMonitoringService,
    SkewStatus,
)
from sqlalchemy import text


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestPartitionMonitoringServiceIntegration:
    """Ensure monitoring queries parse real database results."""

    @pytest.fixture()
    async def partition_monitoring_schema(self, db_session):
        await db_session.execute(text("DROP VIEW IF EXISTS partition_health_summary"))
        await db_session.execute(
            text(
                """
                CREATE VIEW partition_health_summary AS
                SELECT * FROM (
                    VALUES
                        (0, 120::bigint, 300::bigint, 40.0::numeric, 38.0::numeric, 'HEALTHY'::text, 0.10::numeric, 0.08::numeric, NULL::text),
                        (1, 150::bigint, 300::bigint, 50.0::numeric, 52.0::numeric, 'WARNING'::text, 0.32::numeric, 0.35::numeric, 'Monitor growth'::text),
                        (2, 30::bigint, 300::bigint, 10.0::numeric, 10.0::numeric, 'UNBALANCED'::text, 0.55::numeric, 0.50::numeric, 'Consider rebalancing'::text)
                ) AS s(partition_num, chunk_count, total_chunks, chunk_percentage, size_percentage, health_status, chunk_skew, size_skew, recommendation)
                """
            )
        )
        await db_session.execute(text("DROP FUNCTION IF EXISTS analyze_partition_skew()"))
        await db_session.execute(
            text(
                """
                CREATE FUNCTION analyze_partition_skew()
                RETURNS TABLE(metric text, value numeric, status text, details text)
                LANGUAGE sql
                AS $$
                    SELECT 'chunk_distribution', 0.35::numeric, 'WARNING', 'Moderate imbalance'
                    UNION ALL
                    SELECT 'size_distribution', 0.12::numeric, 'NORMAL', 'Within tolerance'
                $$
                """
            )
        )
        await db_session.commit()
        try:
            yield
        finally:
            await db_session.execute(text("DROP VIEW IF EXISTS partition_health_summary"))
            await db_session.execute(text("DROP FUNCTION IF EXISTS analyze_partition_skew()"))
            await db_session.commit()

    @pytest.fixture()
    def service(self, db_session):
        return PartitionMonitoringService(db_session)

    async def test_get_partition_health_summary(self, service, partition_monitoring_schema):
        summary = await service.get_partition_health_summary()
        assert len(summary) == 3
        assert summary[0].health_status is PartitionHealthStatus.HEALTHY
        assert summary[1].health_status is PartitionHealthStatus.WARNING
        assert summary[2].recommendation == 'Consider rebalancing'

    async def test_analyze_partition_skew(self, service, partition_monitoring_schema):
        metrics = await service.analyze_partition_skew()
        assert len(metrics) == 2
        assert metrics[0].metric == 'chunk_distribution'
        assert metrics[0].status is SkewStatus.WARNING
        assert metrics[1].status is SkewStatus.NORMAL

    async def test_check_partition_health_compiles_alerts(self, service, partition_monitoring_schema):
        result = await service.check_partition_health()
        assert isinstance(result, MonitoringResult)
        assert result.status == 'success'
        assert result.metrics["unbalanced_count"] == 1
        assert result.metrics["warning_count"] == 1
        assert result.metrics["healthy_count"] == 1
        assert any(alert["partition"] == 2 for alert in result.alerts)

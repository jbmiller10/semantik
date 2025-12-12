# Partition Monitoring and Maintenance Guide

This guide describes the operational improvements implemented for monitoring and maintaining the partitioned chunk table system in Semantik.

> **Access control**
>
> Partition monitoring endpoints require an authenticated admin user or a request signed with a valid `X-Internal-Api-Key` header. Ensure the proper credentials are configured before calling these APIs in scripts or dashboards.

## Overview

The chunking system uses 100 partitions (configurable via `CHUNK_PARTITION_COUNT`) to distribute chunks across multiple tables for better performance. To ensure optimal performance and detect imbalances early, we've implemented comprehensive monitoring and alerting capabilities.

## Components

### 1. Automated Materialized View Refresh

The `collection_chunking_stats` materialized view is now automatically refreshed every hour via a Celery task:

- **Task**: `webui.tasks.refresh_collection_chunking_stats`
- **Schedule**: Hourly
- **Purpose**: Keeps collection statistics up-to-date without manual intervention

### 2. Monitoring Views

Five database views provide real-time insights into partition health:

#### `partition_size_distribution`
- Shows physical size of each partition
- Calculates percentage of total size
- Provides row count estimates

#### `partition_chunk_distribution`
- Displays chunk count per partition
- Shows document and collection distribution
- Calculates average token counts
- Tracks age of chunks in each partition

#### `partition_hot_spots`
- Identifies partitions with high recent activity
- Tracks chunks created in last hour/day/week
- Helps identify skewed write patterns

#### `partition_stats`
- Aggregates chunk and size distribution data
- Shows total size in MB per partition
- Calculates average chunk size in KB
- Tracks last created timestamp per partition

#### `partition_health_summary`
- Consolidated view of partition health
- Calculates skew factors for chunks and size
- Provides health status (HEALTHY/WARNING/UNBALANCED)
- Includes actionable recommendations

### 3. Analysis Function

`analyze_partition_skew()` - A PostgreSQL function that provides:
- Overall skew factor calculation
- Distribution variance metrics
- Hot partition detection
- Detailed status reporting

### 4. Partition Maintenance Script

Located at `/scripts/partition_maintenance.py`, this script provides command-line access to all monitoring features:

```bash
# Check overall partition health
./scripts/partition_maintenance.py health

# View chunk distribution across partitions
./scripts/partition_maintenance.py distribution

# Check partition sizes
./scripts/partition_maintenance.py sizes

# Identify hot partitions
./scripts/partition_maintenance.py hot

# Analyze partition skew
./scripts/partition_maintenance.py skew

# Manually refresh materialized view
./scripts/partition_maintenance.py refresh

# View collection statistics
./scripts/partition_maintenance.py collections

# Generate full maintenance report
./scripts/partition_maintenance.py full
```

Options:
- `--db-url`: Override database URL from environment
- `--format`: Choose output format (table/json)

### 5. Automated Health Monitoring

A Celery task runs every 6 hours to check partition health:

- **Task**: `webui.tasks.monitor_partition_health`
- **Schedule**: Every 6 hours
- **Actions**:
  - Checks for unbalanced partitions
  - Detects warning signs of imbalance
  - Logs alerts at appropriate levels
  - Returns structured monitoring data

## Alert Thresholds

Alert thresholds are based on **skew factor**, which measures the deviation from expected partition distribution. With 100 partitions, each partition should ideally contain 1% of the data.

### Critical / Unbalanced Alerts
- **Skew factor >= 0.5** (50% or more deviation from expected)
- Any partition marked as UNBALANCED
- Triggers rebalancing recommendations

### Warning Alerts
- **Skew factor >= 0.3** (30% or more deviation from expected)
- Partitions showing early signs of imbalance
- Recommendation: Monitor partition growth

### Rebalancing Threshold
- **Skew factor >= 0.4** (40% deviation) triggers rebalancing recommendations

## Usage Examples

### Daily Operations Check
```bash
# Run full health report
./scripts/partition_maintenance.py full

# Check for hot partitions during peak hours
./scripts/partition_maintenance.py hot
```

### Investigating Performance Issues
```bash
# Check if partitions are unbalanced
./scripts/partition_maintenance.py health

# Analyze skew metrics
./scripts/partition_maintenance.py skew

# View detailed distribution
./scripts/partition_maintenance.py distribution
```

### Manual Maintenance
```bash
# Refresh statistics if needed
./scripts/partition_maintenance.py refresh

# Export data for analysis
./scripts/partition_maintenance.py health --format json > partition_health.json
```

## Integration Points

### Extending Alerts

The `monitor_partition_health` task includes commented examples for integrating with external alerting systems:

```python
# In production, send alert here (email, Slack, etc.)
# Example: send_alert_to_slack(alert)
```

To add Slack alerts:
1. Install slack-sdk: `uv add slack-sdk`
2. Add Slack webhook URL to environment
3. Implement `send_alert_to_slack()` function
4. Uncomment the alert sending code

### Custom Monitoring

The monitoring views can be queried directly for custom dashboards:

```sql
-- Get current partition health
SELECT * FROM partition_health_summary;

-- Check for hot partitions
SELECT * FROM partition_hot_spots
WHERE hour_percentage > 12.5;

-- Analyze skew
SELECT * FROM analyze_partition_skew();
```

## API Endpoints

The partition monitoring API is available under `/api/v2/partitions`. All endpoints require admin authentication or a valid `X-Internal-Api-Key` header.

### GET `/api/v2/partitions/health`

Returns comprehensive partition health status including alerts and metrics.

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-10-20T10:15:00Z",
  "alerts": [
    {
      "level": "WARNING",
      "message": "2 partitions showing early signs of imbalance",
      "action": "Monitor closely and plan preventive maintenance"
    }
  ],
  "metrics": {
    "total_partitions": 100,
    "unbalanced_count": 0,
    "warning_count": 2,
    "healthy_count": 98,
    "skew_metrics": { ... }
  },
  "error": null
}
```

### GET `/api/v2/partitions/statistics`

Returns detailed statistics for partitions.

**Query Parameters:**
- `partition_num` (optional): Specific partition number (0-99). Omit for aggregate statistics.

**Response (aggregate):**
```json
{
  "partition_num": null,
  "statistics": {
    "partition_count": 100,
    "total_chunks": 1500000,
    "total_size_mb": 2048.5,
    "avg_chunks_per_partition": 15000.0,
    "chunk_count_stddev": 150.5
  }
}
```

**Response (single partition):**
```json
{
  "partition_num": 42,
  "statistics": {
    "partition_num": 42,
    "chunk_count": 15200,
    "total_size_mb": 20.5,
    "avg_chunk_size_kb": 1.38,
    "created_at": "2025-10-19T14:30:00Z"
  }
}
```

### GET `/api/v2/partitions/recommendations`

Returns rebalancing recommendations based on current partition distribution.

**Response:**
```json
{
  "count": 2,
  "recommendations": [
    {
      "partition": 15,
      "reason": "High chunk skew",
      "current_skew": 0.45,
      "action": "Consider data redistribution",
      "priority": "MEDIUM"
    }
  ]
}
```

### GET `/api/v2/partitions/health-summary`

Returns a simplified view of partition health suitable for dashboards.

**Response:**
```json
{
  "total_partitions": 100,
  "healthy_count": 95,
  "warning_count": 4,
  "unbalanced_count": 1,
  "health_percentage": 95.0,
  "partitions": [
    {
      "partition_num": 0,
      "health_status": "HEALTHY",
      "chunk_percentage": 1.02,
      "chunk_skew": 0.02
    }
  ]
}
```

## Service Layer

The `PartitionMonitoringService` class provides programmatic access to partition monitoring features.

### Initialization

```python
from webui.services.partition_monitoring_service import PartitionMonitoringService
from sqlalchemy.ext.asyncio import AsyncSession

async def monitor_partitions(session: AsyncSession):
    service = PartitionMonitoringService(session)
    health = await service.check_partition_health()
```

### Methods

#### `get_partition_health_summary() -> list[PartitionHealth]`
Returns health summary for all partitions, ordered by chunk skew (descending).

#### `analyze_partition_skew() -> list[SkewMetric]`
Analyzes partition skew metrics by calling the `analyze_partition_skew()` database function.

#### `check_partition_health() -> MonitoringResult`
Performs comprehensive partition health check, combining health summary and skew analysis. Returns alerts for any detected issues.

#### `get_partition_statistics(partition_num: int | None = None) -> dict[str, Any]`
Returns detailed statistics for a specific partition or aggregate statistics for all partitions.

#### `get_rebalancing_recommendations() -> list[dict[str, Any]]`
Returns recommendations for partitions that exceed the rebalancing threshold (skew factor >= 0.4).

### Class Constants

```python
SKEW_WARNING_THRESHOLD = 0.3   # 30% deviation triggers WARNING
SKEW_CRITICAL_THRESHOLD = 0.5  # 50% deviation triggers CRITICAL/UNBALANCED
REBALANCE_THRESHOLD = 0.4      # 40% deviation triggers rebalance recommendation
```

## Data Structures

### PartitionHealth

Health information for a single partition.

```python
@dataclass
class PartitionHealth:
    partition_num: int          # Partition number (0-99)
    chunk_count: int            # Number of chunks in partition
    total_chunks: int           # Total chunks across all partitions
    chunk_percentage: float     # Percentage of total chunks
    size_percentage: float      # Percentage of total size
    health_status: PartitionHealthStatus  # HEALTHY, WARNING, or UNBALANCED
    chunk_skew: float           # Deviation from expected chunk distribution
    size_skew: float            # Deviation from expected size distribution
    recommendation: str | None  # Action recommendation if unhealthy
```

### SkewMetric

Result from skew analysis function.

```python
@dataclass
class SkewMetric:
    metric: str      # Metric name (chunk_distribution, size_distribution, etc.)
    value: float     # Numeric value of the metric
    status: SkewStatus  # NORMAL, WARNING, or CRITICAL
    details: str     # Human-readable description
```

### MonitoringResult

Result of comprehensive partition health check.

```python
@dataclass
class MonitoringResult:
    status: str                   # "success" or "failed"
    timestamp: str                # ISO 8601 timestamp
    alerts: list[dict[str, Any]]  # List of alert objects
    metrics: dict[str, Any]       # Aggregated metrics
    error: str | None             # Error message if status is "failed"
```

### Enums

```python
class PartitionHealthStatus(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    UNBALANCED = "UNBALANCED"
    CRITICAL = "CRITICAL"

class SkewStatus(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
```

## Best Practices

1. **Regular Monitoring**: Check partition health at least daily
2. **Proactive Rebalancing**: Address WARNING status before it becomes UNBALANCED
3. **Peak Hour Analysis**: Monitor hot partitions during high-traffic periods
4. **Capacity Planning**: Use size distribution data for storage planning
5. **Alert Response**: Have a runbook for responding to partition imbalance alerts

## Troubleshooting

### High Skew Factor
- Review document hashing algorithm
- Check for patterns in document IDs causing clustering
- Consider adjusting partition count (requires migration)

### Hot Partitions
- Analyze application write patterns
- Check for time-based clustering
- Review collection distribution

### Growing Imbalance
- Monitor trend over time
- Plan maintenance window for rebalancing
- Consider partition key changes

## Future Enhancements

Potential improvements to consider:
- Automatic partition rebalancing
- Predictive imbalance detection
- Integration with Grafana/Prometheus
- Dynamic partition count adjustment
- Historical trend analysis

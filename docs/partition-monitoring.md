# Partition Monitoring

Monitoring and maintenance for the partitioned chunk table system (100 partitions by default).

**Auth required:** Admin user or `X-Internal-Api-Key` header.

## Components

### Automated Stats Refresh

Celery task `refresh_collection_chunking_stats` runs hourly to keep the `collection_chunking_stats` materialized view current.

### Monitoring Views

- `partition_size_distribution` - Physical size per partition
- `partition_chunk_distribution` - Chunk counts, avg tokens, age
- `partition_hot_spots` - Recent activity (hour/day/week)
- `partition_stats` - Aggregated chunk + size data
- `partition_health_summary` - Health status and skew factors

### Analysis Function

`analyze_partition_skew()` - Postgres function for skew calculation, variance, and hot partition detection.

### Maintenance Script

`/scripts/partition_maintenance.py`:

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

Options: `--db-url`, `--format` (table/json)

### Automated Health Monitoring

Celery task `monitor_partition_health` runs every 6 hours to check balance and log alerts.

## Thresholds

Skew factor measures deviation from expected (1% per partition with 100 partitions).

- **WARNING:** skew >= 0.3 (30%)
- **UNBALANCED/CRITICAL:** skew >= 0.5 (50%)
- **Rebalancing recommended:** skew >= 0.4 (40%)

## Usage

Daily check: `./scripts/partition_maintenance.py full`
Performance issues: `./scripts/partition_maintenance.py health && skew`
Export: `./scripts/partition_maintenance.py health --format json`

## Custom Monitoring

Query views directly:
```sql
SELECT * FROM partition_health_summary;
SELECT * FROM partition_hot_spots WHERE hour_percentage > 12.5;
SELECT * FROM analyze_partition_skew();
```

For Slack alerts, implement `send_alert_to_slack()` in `monitor_partition_health` task.

## API Endpoints

All under `/api/v2/partitions` (admin or internal key required):

- `GET /health` - Health status + alerts
- `GET /statistics?partition_num=42` - Partition stats (omit param for aggregate)
- `GET /recommendations` - Rebalancing suggestions
- `GET /health-summary` - Dashboard view

## Service Layer

`PartitionMonitoringService` methods:
- `get_partition_health_summary()` - All partition health
- `analyze_partition_skew()` - Skew metrics
- `check_partition_health()` - Full health check with alerts
- `get_partition_statistics(partition_num)` - Stats for one/all partitions
- `get_rebalancing_recommendations()` - Rebalance suggestions

Thresholds: WARNING=0.3, CRITICAL=0.5, REBALANCE=0.4

## Troubleshooting

**High skew:** Review hashing algorithm, check for ID clustering
**Hot partitions:** Analyze write patterns, check time-based clustering
**Growing imbalance:** Monitor trends, plan rebalancing

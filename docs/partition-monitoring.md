# Partition Monitoring and Maintenance Guide

This guide describes the operational improvements implemented for monitoring and maintaining the partitioned chunk table system in Semantik.

## Overview

The chunking system uses 16 partitions (configurable via `CHUNK_PARTITION_COUNT`) to distribute chunks across multiple tables for better performance. To ensure optimal performance and detect imbalances early, we've implemented comprehensive monitoring and alerting capabilities.

## Components

### 1. Automated Materialized View Refresh

The `collection_chunking_stats` materialized view is now automatically refreshed every hour via a Celery task:

- **Task**: `webui.tasks.refresh_collection_chunking_stats`
- **Schedule**: Hourly
- **Purpose**: Keeps collection statistics up-to-date without manual intervention

### 2. Monitoring Views

Four new database views provide real-time insights into partition health:

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

### Critical Alerts
- Partition has >16.25% of total chunks (2.6x expected)
- Overall skew factor >20%
- Any partition marked as UNBALANCED

### Warnings
- Partition has >11.25% of chunks (1.8x expected)
- Overall skew factor >10%
- More than 2 partitions showing WARNING status

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
1. Install slack-sdk: `poetry add slack-sdk`
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
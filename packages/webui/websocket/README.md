# Horizontally Scalable WebSocket Architecture for Semantik

## Executive Summary

This document outlines a production-ready, horizontally scalable WebSocket architecture for Semantik that supports 10,000+ concurrent connections with sub-100ms latency. The design uses Redis Pub/Sub with sticky sessions as the primary approach, with optional Server-Sent Events (SSE) fallback for restricted networks.

## Architecture Overview

### Core Components

1. **WebSocket Gateway Layer** - Handles connection management and routing
2. **Redis Pub/Sub** - Message distribution backbone
3. **Connection Registry** - Tracks active connections across instances
4. **HAProxy Load Balancer** - Sticky session routing
5. **Health Monitor** - Connection health and metrics
6. **SSE Fallback** - Alternative for WebSocket-restricted environments

### Design Principles

- **Stateless Instances**: No shared memory between instances
- **Event-Driven**: Asynchronous message processing
- **Resilient**: Automatic failover and recovery
- **Observable**: Comprehensive metrics and tracing
- **Secure**: Authentication and rate limiting per connection

## Technical Architecture

### 1. Connection Flow

```
Client → HAProxy (sticky) → WebUI Instance → Redis Pub/Sub ← Worker
                               ↓
                        Connection Registry
```

### 2. Message Flow Types

#### Direct Channel Messages (User-Specific)
- Pattern: `user:{user_id}:operation:{operation_id}`
- Use: Operation-specific updates
- TTL: 24 hours

#### Broadcast Messages (Collection-Wide)
- Pattern: `collection:{collection_id}:*`
- Use: Collection state changes
- TTL: 1 hour

#### System Messages
- Pattern: `system:*`
- Use: Maintenance, alerts
- TTL: 5 minutes

### 3. Scaling Strategy

#### Horizontal Scaling
- Add instances behind load balancer
- Redis handles cross-instance communication
- Connection registry prevents orphaned connections

#### Vertical Scaling
- Increase connection limits per instance
- Tune Redis connection pool
- Optimize message batching

## Implementation Components

### Core WebSocket Manager

Located at: `/packages/webui/websocket/manager.py`

Features:
- Connection pooling with limits
- Automatic reconnection logic
- Message deduplication
- Backpressure handling
- Graceful degradation without Redis

### Connection Registry

Located at: `/packages/webui/websocket/registry.py`

Features:
- Instance registration with heartbeat
- Connection tracking across instances
- Automatic cleanup on instance failure
- Connection migration support

### Message Router

Located at: `/packages/webui/websocket/router.py`

Features:
- Pattern-based routing
- Message prioritization
- Rate limiting per channel
- Dead letter queue for failed messages

### Health Monitor

Located at: `/packages/webui/websocket/health.py`

Features:
- Connection health checks
- Latency monitoring
- Resource usage tracking
- Automatic connection pruning

## Configuration

### Redis Configuration

```python
REDIS_CONFIG = {
    "max_connections": 100,
    "connection_pool_class": "redis.BlockingConnectionPool",
    "health_check_interval": 30,
    "socket_keepalive": True,
    "socket_keepalive_options": {
        1: 1,  # TCP_KEEPIDLE
        2: 2,  # TCP_KEEPINTVL
        3: 3,  # TCP_KEEPCNT
    }
}
```

### WebSocket Limits

```python
WEBSOCKET_CONFIG = {
    "max_connections_per_user": 10,
    "max_total_connections": 1000,
    "heartbeat_interval": 30,
    "message_rate_limit": 100,  # per second
    "max_message_size": 1048576,  # 1MB
}
```

### HAProxy Configuration

```
global
    maxconn 50000
    tune.ssl.default-dh-param 2048

defaults
    timeout connect 5s
    timeout client 30s
    timeout server 30s
    timeout tunnel 1h  # For WebSocket

frontend websocket_frontend
    bind *:443 ssl crt /etc/ssl/certs/semantik.pem
    
    # WebSocket detection
    acl is_websocket hdr(Upgrade) -i WebSocket
    acl is_websocket hdr_beg(Host) -i ws
    
    # Sticky sessions based on user cookie
    cookie SERVERID insert indirect nocache
    
    use_backend websocket_backend if is_websocket
    default_backend http_backend

backend websocket_backend
    balance leastconn
    
    # Sticky sessions
    cookie SERVERID insert indirect nocache
    stick-table type string len 32 size 100k expire 1h
    stick on cookie(session_id)
    
    # Health checks
    option httpchk GET /ws/health
    
    # Backend servers with WebSocket support
    server webui1 webui1:8080 check cookie s1 maxconn 1000
    server webui2 webui2:8080 check cookie s2 maxconn 1000
    server webui3 webui3:8080 check cookie s3 maxconn 1000
```

## Performance Characteristics

### Latency Targets

| Operation | Target | Max |
|-----------|--------|-----|
| Connection establishment | 50ms | 200ms |
| Message delivery (same instance) | 5ms | 20ms |
| Message delivery (cross-instance) | 20ms | 100ms |
| Reconnection | 100ms | 500ms |

### Throughput Capacity

| Metric | Per Instance | Total (10 instances) |
|--------|--------------|----------------------|
| Concurrent connections | 1,000 | 10,000 |
| Messages/second | 10,000 | 100,000 |
| Bandwidth | 100 Mbps | 1 Gbps |

### Resource Requirements

| Component | CPU | Memory | Network |
|-----------|-----|--------|---------|
| WebUI Instance | 2 cores | 4GB | 100 Mbps |
| Redis | 4 cores | 8GB | 1 Gbps |
| HAProxy | 2 cores | 2GB | 1 Gbps |

## Monitoring & Observability

### Key Metrics

1. **Connection Metrics**
   - Total active connections
   - Connections per user
   - Connection churn rate
   - Failed connection attempts

2. **Message Metrics**
   - Messages sent/received per second
   - Message latency (p50, p95, p99)
   - Failed message deliveries
   - Message queue depth

3. **System Metrics**
   - CPU/Memory usage per instance
   - Redis memory usage
   - Network bandwidth utilization
   - Error rates by type

### Monitoring Stack

```yaml
monitoring:
  - prometheus:
      scrape_interval: 15s
      targets:
        - webui:9092/metrics
        - redis_exporter:9121/metrics
        
  - grafana:
      dashboards:
        - websocket_overview
        - connection_details
        - message_flow
        - error_analysis
        
  - alertmanager:
      alerts:
        - high_connection_count
        - message_delivery_failure
        - instance_unhealthy
        - redis_memory_high
```

## Failure Scenarios & Recovery

### Instance Failure

1. HAProxy detects unhealthy instance
2. Removes from rotation
3. Existing connections timeout and reconnect
4. Sticky sessions route to new healthy instance
5. Connection registry cleanup after heartbeat timeout

### Redis Failure

1. WebSocket manager detects Redis unavailable
2. Switches to degraded mode (in-memory only)
3. Attempts reconnection with exponential backoff
4. Resumes normal operation when Redis available

### Network Partition

1. Heartbeat failures detected
2. Split-brain prevention via Redis SET NX
3. Minority partition enters read-only mode
4. Automatic recovery when partition heals

## Security Considerations

### Authentication

- JWT tokens validated on connection
- Token refresh handled transparently
- Connection terminated on auth failure

### Rate Limiting

- Per-user connection limits
- Message rate limiting per channel
- Global rate limits for system protection

### Input Validation

- Message size limits
- JSON schema validation
- SQL injection prevention in routing

## Migration Strategy

### Phase 1: Foundation (Week 1)
- Implement new WebSocket manager
- Add Redis Pub/Sub support
- Create connection registry

### Phase 2: Scaling (Week 2)
- Add HAProxy configuration
- Implement health monitoring
- Add metrics collection

### Phase 3: Resilience (Week 3)
- Add SSE fallback
- Implement reconnection logic
- Add circuit breakers

### Phase 4: Production (Week 4)
- Performance testing
- Load testing (10k connections)
- Documentation and training

## Testing Strategy

### Unit Tests
- Connection management
- Message routing
- Registry operations

### Integration Tests
- Multi-instance messaging
- Failover scenarios
- Redis failure handling

### Load Tests
- 10,000 concurrent connections
- 100,000 messages/second
- Instance failure during load

### Chaos Engineering
- Random instance kills
- Network latency injection
- Redis restarts

## Alternative Approaches Considered

### 1. Socket.io
- Pros: Built-in scaling, fallbacks
- Cons: Additional complexity, client library dependency
- Decision: Native WebSocket for simplicity

### 2. Redis Streams
- Pros: Message persistence, replay capability
- Cons: Higher complexity, storage overhead
- Decision: Keep for operation history, use Pub/Sub for real-time

### 3. RabbitMQ
- Pros: Advanced routing, guaranteed delivery
- Cons: Additional infrastructure, higher latency
- Decision: Redis sufficient for our needs

### 4. GraphQL Subscriptions
- Pros: Type safety, query flexibility
- Cons: Complete frontend rewrite needed
- Decision: Future consideration

## Conclusion

This architecture provides a robust, scalable WebSocket solution that:
- Scales horizontally to 10,000+ connections
- Maintains <100ms latency
- Handles failures gracefully
- Provides comprehensive observability
- Supports future growth

The implementation prioritizes reliability and performance while maintaining code simplicity and operational excellence.
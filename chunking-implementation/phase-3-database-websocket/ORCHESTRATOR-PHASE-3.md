# Phase 3: Database & WebSocket Scaling - Orchestrator Guide

**Phase Duration**: 3 days  
**Phase Priority**: HIGH  
**Agent Requirements**: 3 agents working in parallel  
**Can Start**: After Phase 0 completes (parallel with Phase 1 & 2)  

## Phase Overview

This phase implements proper database partitioning and scalable WebSocket architecture. All work can be done in parallel.

## Critical Context

🚨 **PRE-RELEASE APPLICATION** - Drop old tables and create optimal structure from scratch. No migration needed.

## Parallel Execution

All three tickets can be executed simultaneously:
- `TICKET-DB-001.md` - 100 Direct Partitions (Agent A - PostgreSQL Expert)
- `TICKET-DB-002.md` - Partition Monitoring (Agent B - Monitoring Expert)  
- `TICKET-WS-001.md` - WebSocket Scaling (Agent C - WebSocket Expert)

## Critical Decision Reminder

**IMPORTANT**: We are using 100 DIRECT LIST partitions, NOT virtual partition mapping:
```sql
-- CORRECT ✅
PARTITION BY LIST (hashtext(collection_id::text) % 100);

-- WRONG ❌ 
-- No virtual partition tables
-- No mapping layers
-- No 1M virtual partitions
```

## Success Criteria

### Database Validation
```bash
# Check partition count
psql -c "SELECT COUNT(*) FROM pg_partitions WHERE tablename='chunks';"
# Expected: 100

# Check distribution
psql -c "SELECT partition_name, COUNT(*) FROM chunks GROUP BY partition_name;"
# Should be roughly even
```

### WebSocket Scaling Test
```bash
# Start 3 instances
docker-compose up --scale webui=3

# Connect 1000 WebSockets
python tests/load/websocket_test.py --connections=1000

# Verify message routing works across instances
```

## Resource Allocation

| Ticket | Agent Type | Time | Complexity |
|--------|-----------|------|------------|
| DB-001 | PostgreSQL Expert | 2 days | Low |
| DB-002 | Monitoring Expert | 1 day | Low |
| WS-001 | WebSocket Expert | 3 days | Medium |

## Phase Handoff

After completion:
- Database ready for production scale
- WebSockets horizontally scalable
- Monitoring in place for operations
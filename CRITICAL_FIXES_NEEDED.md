# Critical Fixes Needed - Quick Reference

## 🔴 MUST FIX IMMEDIATELY (Blocking Issues)

### 1. Partition Strategy - PICK ONE:
```sql
-- FINAL DECISION: 100 LIST partitions with hash distribution
CREATE TABLE chunks_part_00 PARTITION OF chunks
FOR VALUES IN (0) TABLESPACE chunk_data;
-- ... repeat for 0-99
-- Hash function: partition_id = hash(collection_id) % 100
```
**Update in**: CHUNKING_REDESIGN_BLUEPRINT.md line 48-52

### 2. Remove Duplicate Tickets:
- DELETE lines 4395-4397 in CHUNKING_FIX_TICKETS_FOR_AI_AGENTS.md
- DELETE lines 5385-5488 in CHUNKING_FIX_TICKETS_FOR_AI_AGENTS.md  
- KEEP lines 2946-3773 as the canonical TICKET-STREAM-001A

### 3. Fix Timeline:
```markdown
Old: 7-8 weeks
New: 8-9 weeks with buffers
- Phase 0: 1 week (was 3 days)
- Phase 2: 2 weeks (was 1 week)  
- Add 20% buffer to all estimates
```

## 🟡 SHOULD FIX (Quality Issues)

### 4. Standardize Ticket Format:
```markdown
### TICKET-XXX-NNNA: [Clear Title]
**Time**: X days
**Dependencies**: [List] or None
**Files to Modify**: [Exact paths]
**Memory/Performance Constraints**: [Specific limits]

**Acceptance Criteria**:
- [ ] Specific measurable outcome 1
- [ ] Specific measurable outcome 2

**Pre-release Note**: Can break compatibility/drop tables/change APIs
```

### 5. Add Concrete Specifications:
```python
# Add to blueprint
TECHNICAL_CONSTRAINTS = {
    "max_memory_per_doc": "100MB",
    "stream_buffer_size": "64KB", 
    "chunk_token_range": (1000, 4000),
    "redis_ttl_seconds": 3600,
    "rate_limit_per_minute": 100,
    "partition_count": 100,
    "websocket_max_connections": 10000
}
```

## 🟢 NICE TO HAVE (Improvements)

### 6. Feature Flags:
```python
# Add to implementation tickets
FEATURE_FLAGS = {
    "use_streaming": False,
    "use_new_partitions": False,
    "enable_websocket_scaling": False
}
```

### 7. Monitoring Metrics:
```yaml
# Add to each phase completion criteria
required_metrics:
  - chunk_processing_time_p99 < 30s
  - memory_per_document_p99 < 100MB
  - partition_skew_ratio < 10:1
```

## Quick Command to Fix Duplicates:

```bash
# To quickly identify all duplicate ticket IDs
grep -n "^### TICKET-" CHUNKING_FIX_TICKETS_FOR_AI_AGENTS.md | \
  awk -F: '{print $2}' | sort | uniq -d
```

## Files Needing Updates:

1. **CHUNKING_REDESIGN_BLUEPRINT.md**
   - Line 48-52: Clarify "100 LIST partitions, NOT virtual"
   - Add TECHNICAL_CONSTRAINTS section

2. **CHUNKING_FIX_TICKETS_FOR_AI_AGENTS.md**  
   - Remove duplicate STREAM tickets
   - Standardize all ticket formats

3. **CHUNKING_ORCHESTRATION_GUIDE.md**
   - Update timeline to 8-9 weeks
   - Add specific parallel execution map

4. **CHUNKING_IMPLEMENTATION_ROADMAP.md**
   - Adjust all week estimates by +20%
   - Add buffer time explicitly
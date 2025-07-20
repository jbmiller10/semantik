# Ticket #005: Fix Minor Issues and Polish

**Priority**: MINOR
**Type**: Bug Fix / Enhancement
**Component**: Backend
**Affects**: System reliability and monitoring

## Summary
Several minor issues were discovered during testing that should be addressed for system polish and reliability. While not blocking core functionality, these issues affect monitoring, debugging, and overall system quality.

## Issues to Address

### 1. Metrics Server Port Conflict
**Symptom**: "Failed to start metrics server: [Errno 98] Address already in use"
**Location**: Worker startup logs
**Impact**: Metrics collection may not work properly

**Fix**:
```python
# Add port availability check before binding
def start_metrics_server(port: int = 9092):
    max_retries = 5
    for i in range(max_retries):
        try:
            # Try to start server
            server.bind(('', port + i))
            logger.info(f"Metrics server started on port {port + i}")
            return
        except OSError as e:
            if e.errno == 98:  # Address in use
                logger.warning(f"Port {port + i} in use, trying next port")
            else:
                raise
    logger.error("Could not find available port for metrics server")
```

### 2. Audit Log Escape Character Error
**Symptom**: "Failed to create audit log: bad escape \U at position 2"
**Location**: Collection creation task
**Impact**: Audit trail incomplete

**Fix**:
```python
# Properly escape strings before JSON storage
import json

def create_audit_log(action: str, details: dict):
    try:
        # Ensure proper JSON encoding
        details_json = json.dumps(details, ensure_ascii=False)
        audit_log = CollectionAuditLog(
            action=action,
            details=details_json,
            # ... other fields
        )
        db.session.add(audit_log)
        db.session.commit()
    except Exception as e:
        logger.error(f"Failed to create audit log: {str(e)}")
        # Don't fail the main operation for audit log issues
```

### 3. Database Lock for Operation Metrics
**Symptom**: "Failed to record operation metrics: database is locked"
**Location**: Operation completion
**Impact**: Metrics not recorded

**Fix**:
```python
# Use proper database transaction handling
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

# For SQLite, use WAL mode and proper connection handling
engine = create_engine(
    "sqlite:///webui.db",
    connect_args={
        "check_same_thread": False,
        "timeout": 30,  # 30 second timeout
        "isolation_level": "DEFERRED"
    },
    poolclass=NullPool  # Don't pool connections for SQLite
)

# Enable WAL mode for better concurrency
with engine.connect() as conn:
    conn.execute("PRAGMA journal_mode=WAL")
```

### 4. Active Operations Tab Empty on Initial Creation
**Symptom**: Collection creation doesn't show in Active Operations
**Location**: Active Operations view
**Impact**: Users don't see operations in progress

**Fix**:
- Ensure operations are created in the database before the task starts
- Add proper polling or WebSocket updates to the Active Operations view
- Consider showing recently completed operations with a time limit

### 5. Setup Wizard Terminal Input Issues
**Symptom**: "Warning: Input is not a terminal (fd=0)"
**Location**: Initial setup
**Impact**: Can't use interactive setup

**Fix**:
```bash
# Detect non-interactive environment and skip wizard
if [ -t 0 ]; then
    # Interactive terminal, run wizard
    python setup_wizard.py
else
    # Non-interactive, use defaults or environment variables
    echo "Non-interactive environment detected, using default configuration"
    cp .env.example .env
fi
```

## Implementation Priority
1. Database locking issue (affects data integrity)
2. Audit log fix (important for compliance/debugging)
3. Active Operations visibility (user experience)
4. Metrics server port (monitoring)
5. Setup wizard (only affects initial setup)

## Testing Requirements
1. Verify metrics server starts reliably
2. Confirm audit logs are created for all operations
3. Test concurrent database operations don't lock
4. Verify Active Operations shows current tasks
5. Test setup in both interactive and non-interactive modes

## Acceptance Criteria
- [ ] No port conflict errors in logs
- [ ] All operations create audit logs successfully
- [ ] No database locking errors under normal load
- [ ] Active Operations tab shows operations in real-time
- [ ] Setup wizard handles non-interactive mode gracefully
- [ ] Clean logs without error spam

## Related Code Locations
- Metrics server: `packages/webui/main.py`
- Audit logging: `packages/webui/services/collection_service.py`
- Database configuration: `packages/shared/database/__init__.py`
- Active Operations: `apps/webui-react/src/components/operations/`
- Setup wizard: `scripts/setup_wizard.py` or `Makefile`

## Notes
These issues represent the "last 10%" of polish that makes the difference between a prototype and a production-ready system. While individually minor, addressing them collectively will significantly improve system reliability and user experience.
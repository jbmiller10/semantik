# Phase 0: Security Fixes - Orchestrator Guide

**Phase Duration**: 3 days  
**Phase Priority**: BLOCKER - Must complete before ANY other work  
**Agent Requirements**: 2-3 parallel agents with security expertise  

## Phase Overview

This phase addresses critical security vulnerabilities that put the entire system at risk. These are non-negotiable blockers that must be fixed before any other development work can proceed.

## Critical Context

🚨 **PRE-RELEASE APPLICATION** - We can completely replace security implementations without maintaining backwards compatibility. Delete vulnerable code and implement correct solutions from scratch.

## Ticket Execution Order

### Parallel Execution Group 1 (Morning)
Execute these tickets simultaneously:
- `TICKET-SEC-001.md` - Path Traversal Fix (Agent A)
- `TICKET-SEC-002.md` - Rate Limiting Implementation (Agent B)

### Sequential Execution (Afternoon)
After Group 1 completes:
- `TICKET-SEC-003.md` - Redis TTL Implementation (Agent A or B)

## Dependencies

- **External**: None - This phase can start immediately
- **Internal**: SEC-003 should wait for SEC-001/002 to ensure no conflicts

## Success Criteria

Before marking this phase complete, verify:

### Security Validation
```bash
# Run security scanner
npm run security:scan
# Expected: 0 vulnerabilities

# Test path traversal
python tests/security/test_path_traversal.py
# Expected: All tests pass

# Verify rate limiting
curl -X POST http://localhost:8000/api/v2/chunking/preview
# Expected: 429 after limit exceeded
```

### Memory Validation
```bash
# Check Redis memory before/after operations
redis-cli INFO memory
# Expected: Memory stable after 1 hour
```

### Checklist
- [ ] Path traversal vulnerability completely eliminated
- [ ] Rate limiting active on all chunking endpoints
- [ ] Redis TTL set on all operation keys
- [ ] Security tests passing (100% coverage)
- [ ] No performance regression (validation <10ms)

## Resource Allocation

| Ticket | Agent Type | Estimated Time | Complexity |
|--------|------------|----------------|------------|
| SEC-001 | Security Expert | 4 hours | Medium |
| SEC-002 | API/Rate Limit Expert | 4 hours | Low |
| SEC-003 | Redis Expert | 2 hours | Low |

## Risk Mitigation

### Common Issues to Avoid:
1. **Partial fixes** - Must handle ALL attack vectors, not just common ones
2. **Performance impact** - Security validation should not exceed 10ms
3. **Breaking legitimate use** - Ensure valid paths aren't blocked

### Escalation Triggers:
- If any ticket blocked >2 hours
- If security scanner still reports vulnerabilities after fixes
- If performance degrades >50% after implementation

## Communication Template

### Phase Start Message:
```
Starting Phase 0: Critical Security Fixes
- 3 tickets to resolve security vulnerabilities
- Parallel execution of SEC-001 and SEC-002
- Expected completion: [time + 1 day]
```

### Phase Completion Message:
```
Phase 0 Complete: Security Fixes
✅ Path traversal vulnerability: FIXED
✅ Rate limiting: ACTIVE (10/min preview, 5/min compare)
✅ Redis TTL: SET (1hr active, 5min completed)
✅ Security scan: CLEAN
Ready to proceed with Phase 1
```

## Files to Modify

Key files that will be changed in this phase:
- `packages/webui/services/chunking_security.py`
- `packages/webui/api/v2/chunking.py`
- `packages/webui/websocket_manager.py`
- `packages/webui/config/rate_limits.py` (new file)
- `tests/security/test_path_traversal.py` (new file)

## Notes for Agents

- **DELETE vulnerable code** - Don't try to patch it
- **Implement from scratch** - We're pre-release, make it right
- **Test exhaustively** - Security must be bulletproof
- **Document patterns** - Help future agents understand attack vectors

## Phase Handoff

After this phase completes:
- Phase 1 (Foundation) can begin immediately
- Phase 3 (Database/WebSocket) can also start in parallel
- Document any security patterns discovered for future reference
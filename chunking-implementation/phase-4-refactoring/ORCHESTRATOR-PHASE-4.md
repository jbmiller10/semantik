# Phase 4: Service Refactoring - Orchestrator Guide

**Phase Duration**: 3 days  
**Phase Priority**: MEDIUM  
**Agent Requirements**: 1-2 agents with refactoring expertise  
**Can Start**: After Phase 2 (Streaming) completes  

## Phase Overview

Break apart the monolithic 2100+ line service into focused, manageable services with clear responsibilities.

## Critical Context

🚨 **PRE-RELEASE APPLICATION** - Complete restructure allowed. Delete the monolithic service and create clean, focused services from scratch.

## Execution

Single main ticket with optional parallel work:
- `TICKET-REFACTOR-001.md` - Break Apart Monolithic Service

## Target Architecture

```
Before: 1 service with 2100+ lines
After:  5 services, each <500 lines

ChunkingOrchestrator (300 lines) - Coordinates everything
ValidationService (200 lines)     - Input validation only
ProcessingService (400 lines)     - Core chunking logic
CacheService (250 lines)         - Redis operations
MetricsService (200 lines)       - Monitoring & analytics
```

## Success Criteria

- [ ] No service exceeds 500 lines
- [ ] Each service has single responsibility
- [ ] All tests still pass
- [ ] Clean dependency injection
- [ ] Old monolithic service deleted

## Notes

- Focus on separation of concerns
- Keep services in same deployable unit (modular monolith)
- Don't create network boundaries between services
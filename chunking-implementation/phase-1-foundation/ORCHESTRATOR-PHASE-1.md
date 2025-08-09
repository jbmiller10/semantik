# Phase 1: Foundation Refactoring - Orchestrator Guide

**Phase Duration**: 5 days  
**Phase Priority**: HIGH  
**Agent Requirements**: 2 agents with DDD/Clean Architecture expertise  
**Can Start**: After Phase 0 completes  

## Phase Overview

This phase establishes proper architectural boundaries using Domain-Driven Design (DDD) principles. We're extracting business logic from the 2100+ line monolithic service and creating clean, testable layers.

## Critical Context

🚨 **PRE-RELEASE APPLICATION** - We can completely restructure the codebase without maintaining the old architecture. Delete the monolithic service and build proper layers from scratch.

## Ticket Execution Order

### Parallel Execution
Both tickets can be worked on simultaneously by different agents:
- `TICKET-ARCH-001.md` - Extract Domain Layer (Agent A - DDD Expert)
- `TICKET-ARCH-002.md` - Create Application Layer (Agent B - Clean Architecture Expert)

## Dependencies

- **External**: Phase 0 must be complete
- **Internal**: Both tickets can reference each other's interfaces but don't need full implementation

## Success Criteria

### Architecture Validation
```python
# Domain layer purity check
import ast
import os

def check_domain_purity():
    domain_path = "packages/shared/chunking/domain"
    for root, dirs, files in os.walk(domain_path):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file)) as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            # Should not import infrastructure
                            assert 'sqlalchemy' not in node.names
                            assert 'redis' not in node.names
                            assert 'fastapi' not in node.names
    return True
```

### Checklist
- [ ] Domain layer has zero infrastructure dependencies
- [ ] All business rules moved to value objects
- [ ] Use cases clearly defined (one per file)
- [ ] Old monolithic service DELETED
- [ ] All tests still passing
- [ ] New structure follows DDD principles

## New Architecture Structure

```
packages/shared/chunking/
├── domain/                 # ARCH-001 creates this
│   ├── entities/
│   │   ├── chunking_operation.py
│   │   └── chunk_collection.py
│   ├── value_objects/
│   │   ├── chunk_config.py
│   │   └── operation_status.py
│   ├── services/
│   │   └── chunking_strategies/
│   │       ├── base.py
│   │       ├── character.py
│   │       ├── recursive.py
│   │       ├── semantic.py
│   │       ├── markdown.py
│   │       ├── hierarchical.py
│   │       └── hybrid.py
│   └── exceptions.py
│
├── application/            # ARCH-002 creates this
│   ├── use_cases/
│   │   ├── preview_chunking.py
│   │   ├── process_document.py
│   │   ├── compare_strategies.py
│   │   └── get_operation_status.py
│   ├── interfaces/
│   │   ├── repositories.py
│   │   └── services.py
│   └── dto/
│       └── requests.py
│
└── infrastructure/         # Phase 2 will populate this
    ├── repositories/
    ├── services/
    └── adapters/
```

## Resource Allocation

| Ticket | Agent Type | Estimated Time | Complexity |
|--------|------------|----------------|------------|
| ARCH-001 | DDD Expert | 2 days | Medium |
| ARCH-002 | Clean Architecture Expert | 2 days | Medium |

## Communication Between Agents

Since both agents work in parallel, they should:
1. Define interfaces first (morning of Day 1)
2. Share interface definitions via comments in code
3. Mock dependencies for testing
4. Integrate on Day 3

### Interface Agreement Template
```python
# domain/services/chunking_strategies/base.py
from abc import ABC, abstractmethod

class ChunkingStrategy(ABC):
    """Interface agreed between ARCH-001 and ARCH-002"""
    
    @abstractmethod
    async def chunk(self, content: str, config: ChunkConfig) -> List[Chunk]:
        """Break content into chunks based on strategy"""
        pass
```

## Files to Delete

**IMPORTANT**: These files should be DELETED, not refactored:
- `packages/webui/services/chunking_service.py` (2100+ lines)
- `packages/webui/services/chunking_validator.py` (mixed concerns)
- Any file with mixed domain/infrastructure logic

## Validation Steps

1. **Check domain purity**:
   ```bash
   python scripts/check_domain_purity.py
   # Should report: "Domain layer is pure ✓"
   ```

2. **Run architecture tests**:
   ```bash
   pytest tests/architecture/ -v
   # All architecture rule tests should pass
   ```

3. **Verify old code deleted**:
   ```bash
   ls packages/webui/services/chunking_service.py
   # Should return: "No such file or directory"
   ```

## Phase Handoff

After this phase completes:
- Phase 2 (Streaming) can begin immediately
- Infrastructure layer will connect domain to databases
- API layer will use application use cases
- All business logic will be testable in isolation

## Risk Mitigation

### Common Issues:
1. **Infrastructure leakage** - Watch for database imports in domain
2. **Anemic domain** - Ensure business logic is IN the domain, not around it
3. **Over-abstraction** - Keep it simple, don't create unnecessary layers

### Escalation Triggers:
- If agents disagree on interface design
- If tests fail after refactoring
- If performance degrades significantly

## Success Message Template

```
Phase 1 Complete: Foundation Refactoring
✅ Domain layer extracted (0 infrastructure deps)
✅ Application layer created (5 use cases)
✅ Monolithic service deleted (removed 2100 lines)
✅ All tests passing
✅ Architecture rules enforced
Ready for Phase 2: Streaming Implementation
```

## Notes for Agents

- **Start with interfaces** - Define contracts before implementation
- **Delete, don't wrap** - Remove old code entirely
- **Test in isolation** - Domain should be testable without any infrastructure
- **Keep it simple** - Don't over-engineer the solution
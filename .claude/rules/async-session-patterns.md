---
paths:
  - "packages/shared/database/**/*.py"
  - "packages/webui/**/*.py"
---

# Async Session Patterns

## Standard Session Usage

```python
from shared.database.database import get_async_session

async with get_async_session() as session:
    repo = CollectionRepository(session)
    collection = await repo.create(name="test", owner_id=user.id)
    await session.commit()  # Don't forget!
```

## Common Mistakes

### 1. Forgetting to Commit

```python
# WRONG - changes not persisted
async with get_async_session() as session:
    await repo.create(...)
    # Missing: await session.commit()

# CORRECT
async with get_async_session() as session:
    await repo.create(...)
    await session.commit()
```

### 2. Using Session After Context Exits

```python
# WRONG - session is closed
async with get_async_session() as session:
    collection = await repo.get(id)
await collection.documents  # Error: session closed!

# CORRECT - eager load or stay in context
async with get_async_session() as session:
    collection = await repo.get_with_documents(id)
    docs = collection.documents  # Loaded while session open
```

### 3. Async in Celery Workers

Celery workers are sync. Use `resolve_awaitable_sync()`:

```python
from webui.tasks.utils import resolve_awaitable_sync

def celery_task():
    # This runs async code in sync context safely
    result = resolve_awaitable_sync(async_db_operation())
```

Note: `pg_connection_manager.reset()` is called automatically when the event loop changes (asyncpg connections are loop-bound).

## Factory Fixtures in Tests

Always pass `owner_id`/`user_id` explicitly:

```python
# CORRECT
collection = await collection_factory(owner_id=test_user_db.id)
operation = await operation_factory(user_id=test_user_db.id)

# WRONG - don't rely on defaults
collection = await collection_factory()  # May fail or use wrong user
```

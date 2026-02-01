---
paths:
  - "packages/webui/services/**/*.py"
  - "packages/webui/api/**/*.py"
---

# Celery Task Dispatch Patterns

## Transaction Before Dispatch

ALWAYS commit the database transaction BEFORE dispatching a Celery task.

```python
# CORRECT
async with session.begin():
    operation = await operation_repo.create(...)
await session.commit()  # Data exists in DB first!
celery_app.send_task("process_operation", args=[str(operation.uuid)])

# WRONG - Race condition!
celery_app.send_task("process_operation", args=[str(operation.uuid)])
await session.commit()  # Task may query before data exists
```

## Why This Matters

Celery tasks run in separate worker processes with their own database connections. If you dispatch before commit, the worker may query the database before the transaction completes, finding no data or stale data.

## Operation Lifecycle Pattern

```python
# 1. Create operation record
operation = await operation_repo.create(
    collection_id=collection.id,
    user_id=user.id,
    operation_type=OperationType.INDEX,
    status=OperationStatus.PENDING,
)

# 2. Commit transaction (data now visible to workers)
await session.commit()

# 3. Dispatch Celery task
celery_app.send_task(
    "webui.tasks.ingestion.process_index_operation",
    args=[str(operation.uuid)],
)

# 4. Return operation UUID to client
return {"operation_id": str(operation.uuid)}
```

## Progress Updates

Tasks report progress via Redis pub/sub using `CeleryTaskWithOperationUpdates`:

```python
async with CeleryTaskWithOperationUpdates(operation_id) as updater:
    updater.set_user_id(user_id)
    await updater.send_update("processing", {"progress": 50})
```

Clients subscribe to `operation:{uuid}` channel for real-time updates.

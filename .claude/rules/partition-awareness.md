---
paths:
  - "packages/shared/database/**/*.py"
  - "packages/webui/services/**/*.py"
  - "packages/webui/tasks/**/*.py"
---

# Chunk Partition Awareness

The `chunks` table uses 100 LIST partitions on `partition_key` (computed as `abs(hashtext(collection_id)) % 100`).

## CRITICAL: Always include collection_id

```python
# CORRECT - partition pruning enabled
select(Chunk).where(
    Chunk.collection_id == collection_id,
    Chunk.document_id == doc_id
)

# WRONG - scans ALL 100 partitions (extremely slow!)
select(Chunk).where(Chunk.document_id == doc_id)
```

## Why This Matters

Without `collection_id` in the WHERE clause, PostgreSQL cannot determine which partition contains the data. It must scan all 100 partitions, turning a fast indexed lookup into a full table scan.

## Repository Pattern

Use `PartitionAwareMixin` for chunk operations. All `ChunkRepository` methods require `collection_id` as the first parameter:

```python
# ChunkRepository always takes collection_id first
chunks = await chunk_repo.get_by_document(collection_id, document_id)
await chunk_repo.delete_by_collection(collection_id)
```

## Bulk Operations

Group bulk operations by `collection_id` to maximize partition pruning efficiency:

```python
# Group chunks by collection before processing
for collection_id, chunk_group in itertools.groupby(chunks, key=lambda c: c.collection_id):
    await process_chunks(collection_id, list(chunk_group))
```

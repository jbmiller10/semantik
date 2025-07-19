# High Priority TODO Tickets

## TICKET-TODO-001: Implement Document Embedding Generation for Collection Append

**Priority:** CRITICAL
**Location:** `packages/webui/tasks.py:1959`
**Blocked By:** None
**Blocks:** Full collection functionality

### Description
When documents are appended to a collection via the scan operation, they are registered in the database but no embeddings are generated. This makes the documents unsearchable.

### Acceptance Criteria
- [ ] Generate embeddings for all newly registered documents
- [ ] Store embeddings in Qdrant with proper metadata (document_id, chunk_index, etc.)
- [ ] Update document status to "processed" after successful embedding
- [ ] Handle errors gracefully with retry logic
- [ ] Send progress updates via WebSocket during processing

### Implementation Notes
- Use the shared embedding service from `packages.shared.services.embeddings`
- Process documents in batches to manage memory
- Store document IDs in Qdrant payload for future deletion support

---

## TICKET-TODO-002: Implement Vector Deletion from Qdrant

**Priority:** HIGH  
**Location:** `packages/webui/tasks.py:2634, 2649`
**Blocked By:** Document IDs must be stored in Qdrant payloads
**Blocks:** Complete document removal functionality

### Description
Document removal currently only updates the database but doesn't remove vectors from Qdrant, leading to orphaned vectors in search results.

### Acceptance Criteria
- [ ] Delete all vectors associated with removed documents
- [ ] Implement batch deletion for performance
- [ ] Ensure transactional consistency between database and Qdrant
- [ ] Add error handling and rollback capability
- [ ] Update collection vector counts after deletion

### Implementation Notes
- First verify that document IDs are stored in Qdrant payloads during embedding creation
- Use Qdrant's filter API to find vectors by document_id
- Consider implementing a cleanup job for orphaned vectors

---

## TICKET-TODO-003: Add Document ID Storage in Qdrant Payloads

**Priority:** HIGH (Prerequisite)
**Location:** Implied requirement for TODO-002
**Blocked By:** None
**Blocks:** TICKET-TODO-002

### Description
To support vector deletion, we need to store document IDs in Qdrant payloads when creating embeddings.

### Acceptance Criteria
- [ ] Modify embedding storage to include document_id in payload
- [ ] Add chunk_index to payload for multi-chunk documents
- [ ] Ensure backward compatibility or provide migration
- [ ] Update search results to include document metadata

### Implementation Notes
- This may require updating existing vectors with a migration script
- Consider what other metadata should be stored (source_path, created_at, etc.)

---

## Quick Implementation Order

1. **TICKET-TODO-003** - Add document IDs to Qdrant (prerequisite)
2. **TICKET-TODO-001** - Implement embedding generation (critical feature)
3. **TICKET-TODO-002** - Implement vector deletion (complete the lifecycle)

## Estimated Effort

- TICKET-TODO-003: 1-2 days (includes testing)
- TICKET-TODO-001: 3-4 days (main feature implementation)
- TICKET-TODO-002: 2-3 days (depends on TODO-003)

**Total: 6-9 days of development work**
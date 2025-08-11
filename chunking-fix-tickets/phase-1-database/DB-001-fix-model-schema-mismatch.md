# DB-001: Fix SQLAlchemy Model-Database Schema Mismatch

## Ticket Information
- **Priority**: BLOCKER
- **Estimated Time**: 2 hours
- **Dependencies**: None
- **Risk Level**: HIGH - ORM operations will fail in production
- **Affected Files**:
  - `packages/shared/database/models.py`
  - `packages/shared/database/repositories/chunk_repository.py`
  - All files importing Chunk model

## Context

The current SQLAlchemy `Chunk` model is out of sync with the actual database schema. The database was migrated from 16 HASH partitions to 100 LIST partitions, but the models still reference the old structure.

### Current State (INCORRECT)
- Model expects: 16 HASH partitions on `collection_id`
- Model primary key: `(id, collection_id)` where id is String
- No `partition_key` column in model
- Model uses declarative partitioning that doesn't match DB

### Actual Database State
- Database has: 100 LIST partitions on `partition_key`
- Database primary key: `(id, collection_id, partition_key)` where id is BigInteger
- `partition_key` column exists and is computed via trigger
- Database uses LIST partitioning with values 0-99

## Requirements

1. Update the `Chunk` model in `models.py` to exactly match the database schema
2. Remove all references to the old 16-partition structure
3. Add the `partition_key` column to the model
4. Update the primary key constraint to match database
5. Update all relationships that reference the Chunk model
6. Ensure the model works with the existing 100-partition structure

## Technical Details

### Model Changes Required

```python
# Current (WRONG)
class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        PrimaryKeyConstraint('id', 'collection_id'),
        # ... partition info for 16 partitions
    )
    id = Column(String, primary_key=True)
    collection_id = Column(String, primary_key=True)
    # Missing: partition_key

# Required (CORRECT)
class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        PrimaryKeyConstraint('id', 'collection_id', 'partition_key'),
        Index('idx_chunk_collection_partition', 'collection_id', 'partition_key'),
        Index('idx_chunk_document', 'document_id'),
        Index('idx_chunk_created', 'created_at'),
        {'schema': 'public'}  # No partition inheritance at model level
    )
    
    id = Column(BigInteger, primary_key=True, server_default=text("nextval('chunks_id_seq')"))
    collection_id = Column(String(255), primary_key=True, nullable=False)
    partition_key = Column(Integer, primary_key=True, nullable=False,
                          server_default=text("mod(hashtext(collection_id::text), 100)"))
    document_id = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON)
    embedding_vector = Column(JSON)  # Or appropriate vector type
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Fix relationships
    collection = relationship(
        "Collection",
        primaryjoin="and_(Chunk.collection_id==Collection.id)",
        foreign_keys=[collection_id],
        back_populates="chunks"
    )
    
    document = relationship(
        "Document",
        primaryjoin="and_(Chunk.document_id==Document.id)",
        foreign_keys=[document_id],
        back_populates="chunks"
    )
```

### Repository Updates Required

Update `ChunkRepository` to handle the composite primary key:

```python
class ChunkRepository:
    async def get_chunk(self, chunk_id: int, collection_id: str, partition_key: int):
        # Must provide all three key components
        return await self.session.get(
            Chunk,
            {"id": chunk_id, "collection_id": collection_id, "partition_key": partition_key}
        )
    
    async def create_chunk(self, chunk_data: dict):
        # partition_key will be auto-computed by database
        # Don't set it manually
        chunk = Chunk(**chunk_data)
        self.session.add(chunk)
        await self.session.flush()
        return chunk
```

## Acceptance Criteria

1. **Model Matches Database**
   - [ ] Chunk model has all three primary key columns
   - [ ] `id` is BigInteger, not String
   - [ ] `partition_key` column exists with correct default
   - [ ] All column types match database exactly

2. **ORM Operations Work**
   - [ ] Can create new chunks via ORM
   - [ ] Can query chunks by collection_id
   - [ ] Can update existing chunks
   - [ ] Can delete chunks
   - [ ] Bulk operations work correctly

3. **Relationships Function**
   - [ ] Collection -> Chunks relationship works
   - [ ] Document -> Chunks relationship works
   - [ ] No foreign key errors

4. **No Partition Logic in Model**
   - [ ] Model doesn't try to manage partitions
   - [ ] Model works with transparent partitioning
   - [ ] No partition-specific table inheritance

## Testing Requirements

1. **Unit Tests**
   ```python
   async def test_chunk_model_matches_database():
       # Create chunk with only required fields
       chunk = Chunk(
           collection_id="test-collection",
           document_id="test-doc",
           chunk_index=0,
           content="test content"
       )
       session.add(chunk)
       await session.commit()
       
       # Verify partition_key was auto-computed
       assert chunk.partition_key is not None
       assert 0 <= chunk.partition_key < 100
       
       # Verify can query back
       result = await session.get(
           Chunk,
           {"id": chunk.id, "collection_id": chunk.collection_id, 
            "partition_key": chunk.partition_key}
       )
       assert result is not None
   ```

2. **Integration Tests**
   - Test with actual PostgreSQL database
   - Verify partition pruning works in queries
   - Test bulk insert performance
   - Verify indexes are used properly

3. **Migration Safety**
   - Ensure existing data still accessible
   - No data corruption
   - Rollback plan if issues found

## Rollback Plan

If issues are discovered:
1. Keep backup of original models.py
2. Can temporarily use raw SQL queries if ORM fails
3. Document any manual interventions needed

## Success Metrics

- Zero ORM-related errors in logs
- All existing chunk queries continue working
- Chunk creation/update operations succeed
- Performance remains same or improves
- All tests pass

## Notes for LLM Agent

- DO NOT change the database schema, only update the model to match it
- The database already has the correct structure with 100 partitions
- The partition_key is computed automatically by database trigger/generated column
- Don't try to manually set partition_key values in application code
- Ensure backward compatibility with existing repository methods where possible
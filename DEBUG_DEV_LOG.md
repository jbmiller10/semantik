# Debug Development Log - PostgreSQL Enum Issues

## Problem Statement
Collection creation fails with PostgreSQL error: `invalid input value for enum collection_status: "PENDING"`

## Timeline of Debugging Attempts

### Initial Investigation
1. **Identified the error**: PostgreSQL expects lowercase enum values ("pending") but receives uppercase ("PENDING")
2. **Root cause**: Mismatch between Python enum names (uppercase) and PostgreSQL enum values (lowercase)

### First Fix Attempt - Adding .value to Enum Usage
**Files Modified:**
- `packages/shared/database/repositories/operation_repository.py`
- `packages/shared/database/repositories/collection_repository.py`

**Changes Made:**
- Changed `status=CollectionStatus.PENDING` to `status=CollectionStatus.PENDING.value`
- Changed `status=OperationStatus.PENDING` to `status=OperationStatus.PENDING.value`
- Updated all enum comparisons to use `.value`

**Result**: ❌ Still failing with same error

### Second Investigation - Frontend API Calls
**Hypothesis**: Frontend might be sending uppercase enum values in API calls
**Findings**: 
- Frontend uses `status: 'processing,pending'` (lowercase) in ActiveOperationsTab.tsx
- API endpoint converts these to enum objects correctly
- Not the source of the issue

### Third Investigation - SQLAlchemy Model Definition
**Key Discovery**: 
```python
# In models.py
status = Column(Enum(CollectionStatus, name='collection_status'), nullable=False, default=CollectionStatus.PENDING.value, index=True)
```

**The Issue**: 
- Column is defined as `Enum` type in SQLAlchemy
- When we pass string values to Enum columns, SQLAlchemy converts them incorrectly
- SQLAlchemy expects the actual enum object, not the string value

### Final Fix - Pass Enum Objects Instead of Values
**Files Modified:**
- `packages/shared/database/repositories/collection_repository.py`
  - `status=CollectionStatus.PENDING.value` → `status=CollectionStatus.PENDING`
  - `collection.status = status.value` → `collection.status = status`
  
- `packages/shared/database/repositories/operation_repository.py`  
  - `status=OperationStatus.PENDING.value` → `status=OperationStatus.PENDING`
  - `operation.status = status.value` → `operation.status = status`
  - `operation.status = OperationStatus.CANCELLED.value` → `operation.status = OperationStatus.CANCELLED`

**Rationale**: SQLAlchemy's Enum column type handles the conversion to database values internally

## Key Learnings
1. SQLAlchemy Enum columns expect enum objects, not string values
2. Using `.value` is only needed when comparing or filtering in queries
3. The model's default value in the Column definition is evaluated at model definition time, not runtime

## Final Root Cause
The issue was NOT in the repository code, but in the SQLAlchemy model definitions themselves!

In `models.py`, the Column definitions had:
```python
status = Column(Enum(CollectionStatus, name='collection_status'), nullable=False, default=CollectionStatus.PENDING.value, index=True)
```

The `default` parameter was using `.value` which returns the string "pending", but SQLAlchemy's Enum column type expects the actual enum object. This mismatch caused SQLAlchemy to use the enum's name ("PENDING") instead of its value ("pending") when inserting.

**Fixed by changing all enum defaults in models.py:**
- `default=CollectionStatus.PENDING.value` → `default=CollectionStatus.PENDING`
- `default=DocumentStatus.PENDING.value` → `default=DocumentStatus.PENDING`
- `default=OperationStatus.PENDING.value` → `default=OperationStatus.PENDING`

## Final Final Root Cause
The issue was deeper than just the defaults. SQLAlchemy's Enum type was using the enum's NAME instead of its VALUE when inserting into PostgreSQL. This requires configuring the Enum column with `values_callable` to explicitly tell SQLAlchemy to use the enum values:

```python
# Before:
status = Column(Enum(CollectionStatus, name='collection_status'), ...)

# After:
status = Column(Enum(CollectionStatus, name='collection_status', values_callable=lambda obj: [e.value for e in obj]), ...)
```

This ensures SQLAlchemy uses "pending" instead of "PENDING" when converting the Python enum to SQL.

## Test Status
- [x] Fixed enum handling in repositories (turned out to be unnecessary)
- [x] Fixed enum defaults in model definitions (partial fix)
- [x] Added values_callable to all Enum columns (the complete fix)
- [x] Restarted webui with all fixes
- [x] Tested collection creation - SUCCESS!
- [x] Verified no PostgreSQL enum errors - No errors after the fix!

## Final Resolution
Collection creation now works successfully. The "Final Test Collection" was created without any PostgreSQL enum errors.

## Environment Details
- PostgreSQL 16 (Alpine)
- SQLAlchemy with asyncpg driver
- Python enums with string values
- Running locally with Docker services
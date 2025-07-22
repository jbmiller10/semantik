# Comprehensive Enum Analysis: CollectionStatus and OperationStatus

## Overview
This document comprehensively analyzes all uses of CollectionStatus and OperationStatus enums to identify the root cause of enum mismatch errors.

## Error Summary
- **CollectionStatus Error**: `'processing' is not among the defined enum values. Enum name: collection_status. Possible values: PENDING, READY, PROCESSING, ..., DEGRADED`
- **OperationStatus Error**: `invalid input value for enum operation_status: "PROCESSING"`

## Analysis Results

### 1. Enum Definitions (models.py)

```python
# In packages/shared/database/models.py

class CollectionStatus(str, enum.Enum):
    """Status of a collection."""
    PENDING = "pending"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DEGRADED = "degraded"

class OperationStatus(str, enum.Enum):
    """Status of an operation."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DocumentStatus(str, enum.Enum):
    """Status of document processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
```

### 2. SQLAlchemy Column Definitions

```python
# CollectionStatus - FIXED (uses native_enum=True, create_constraint=False)
status = Column(Enum(CollectionStatus, name='collection_status', native_enum=True, create_constraint=False), nullable=False, default=CollectionStatus.PENDING, index=True)

# OperationStatus - FIXED (uses native_enum=True, create_constraint=False)
status = Column(Enum(OperationStatus, name='operation_status', native_enum=True, create_constraint=False), nullable=False, default=OperationStatus.PENDING, index=True)

# DocumentStatus - STILL BROKEN (uses values_callable)
status = Column(Enum(DocumentStatus, name='document_status', values_callable=lambda obj: [e.value for e in obj]), nullable=False, default=DocumentStatus.PENDING, index=True)
```

### 3. Database Migration Files

From `alembic/versions/91784cc819aa_add_operations_and_supporting_tables_.py`:

```python
# PostgreSQL enum types were created with lowercase values:
collection_status_values = ["pending", "ready", "processing", "error", "degraded"]
operation_status_values = ["pending", "processing", "completed", "failed", "cancelled"]

# Created as:
CREATE TYPE collection_status AS ENUM ('pending', 'ready', 'processing', 'error', 'degraded');
CREATE TYPE operation_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');
```

### 4. Repository Usage Issues

In `packages/shared/database/repositories/operation_repository.py`:

```python
# list_for_user method has a bug:
if status_list is not None and len(status_list) > 0:
    # This converts enum to string values
    status_values = [s.value if hasattr(s, 'value') else s for s in status_list]
    # But then uses .in_() which expects enum values, not strings!
    query = query.where(Operation.status.in_(status_values))
```

This is causing the "PROCESSING" uppercase error because:
1. API receives lowercase strings ("processing,pending")
2. Converts to OperationStatus enums
3. Repository converts back to strings
4. SQLAlchemy tries to use the strings but expects enums

### 5. API Endpoints

In `packages/webui/api/v2/operations.py`:
```python
# Correctly converts string to enum:
status_list.append(OperationStatus(s))  # s = "processing"
```

### 6. Frontend TypeScript Definitions

```typescript
// Correctly uses lowercase values:
export type CollectionStatus = 
  | 'pending'
  | 'ready'
  | 'processing'
  | 'error'
  | 'degraded';

export type OperationStatus = 
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled';
```

### 7. Frontend API Call

In `apps/webui-react/src/components/ActiveOperationsTab.tsx`:
```typescript
// Sends lowercase status values
const response = await operationsV2Api.list({ 
  status: 'processing,pending',
  limit: 100 
});
```

## Root Cause Analysis

The enum errors are caused by multiple issues:

1. **Inconsistent SQLAlchemy Configuration**: 
   - CollectionStatus and OperationStatus were "fixed" with `native_enum=True, create_constraint=False`
   - But DocumentStatus still uses the old `values_callable` approach
   - The "fix" didn't actually work because the enums still have issues

2. **Repository Query Bug**:
   - The `list_for_user` method in OperationRepository converts enum values to strings
   - But then uses `.in_()` which expects enum instances, not strings
   - This causes SQLAlchemy to send uppercase enum names instead of lowercase values

3. **Enum Value Confusion**:
   - Python enums have names (PROCESSING) and values ("processing")
   - PostgreSQL enums store lowercase values
   - SQLAlchemy sometimes uses names, sometimes values, depending on configuration

## The Real Problem

The error message `'processing' is not among the defined enum values. Enum name: collection_status. Possible values: PENDING, READY, PROCESSING, ..., DEGRADED` is misleading. It's showing enum NAMES not VALUES.

What's happening:
1. Database has lowercase values: "processing"
2. SQLAlchemy is comparing against enum names: PROCESSING
3. The comparison fails

## Recommendations

### Immediate Fix

1. **Fix the repository query bug**:
```python
# In operation_repository.py, line 336:
# Don't convert to values, keep as enums
query = query.where(Operation.status.in_(status_list))
```

2. **Fix DocumentStatus to match others**:
```python
status = Column(Enum(DocumentStatus, name='document_status', native_enum=True, create_constraint=False), nullable=False, default=DocumentStatus.PENDING, index=True)
```

### Long-term Solution

1. **Standardize all enum usage**:
   - Always use `native_enum=True, create_constraint=False` for PostgreSQL
   - Never use `values_callable`
   - Always pass enum instances to SQLAlchemy, not strings

2. **Add tests** to verify enum behavior across all layers

3. **Consider removing the string conversion logic** in repositories entirely
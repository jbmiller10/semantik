# collection_service.py Test Coverage

Tests: `tests/webui/services/test_collection_service.py`

## Covered Methods

| Method | Tests |
|--------|-------|
| create_collection | success, defaults, empty name, whitespace, exists, db error |
| add_source | success, invalid status, active op, not found, access denied |
| reindex_collection | success, no config updates, invalid status, active op |
| delete_collection | success, not owner, active ops, qdrant errors |
| remove_source | success, invalid status, active ops |
| list_for_user | success, pagination |
| update | success, not owner, not found, name exists |
| list_documents | success, pagination, access denied |
| list_operations | success, pagination, not found |

Edge cases: None/empty configs, whitespace validation, missing Qdrant collections, concurrent op prevention

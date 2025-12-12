# collections.py API Test Coverage

Tests: `tests/webui/api/v2/test_collections.py` (1338 lines)

## Covered Endpoints

| Endpoint | Tests |
|----------|-------|
| POST /collections | create, duplicate, validation, service error |
| GET /collections | list, pagination, include_public, empty |
| GET /collections/{id} | get, access control |
| PUT /collections/{id} | full update, partial, 404, 403, 409 |
| DELETE /collections/{id} | delete, 404, 403, in_progress |
| POST /sources | add source, 404, 403, invalid state |
| DELETE /sources | remove source, 404, 403, invalid state |
| POST /reindex | reindex, config updates, 404, 403, 409 |
| GET /operations | list, filter status/type, pagination |
| GET /documents | list, filter status, pagination |

Edge cases: special chars in names, empty lists, concurrent ops, pagination bounds, null handling
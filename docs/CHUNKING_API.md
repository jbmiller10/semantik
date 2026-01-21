# Chunking API

Endpoints for managing chunking strategies, previews, collection processing, and analytics.

**Auth**: JWT bearer token required
**Base URL**: `http://localhost:8080/api/v2/chunking`

## Design

- Strategy-based (6 strategies)
- Async processing with WebSocket progress
- Preview caching (15min)
- Rate limiting
- Quality metrics built-in

## Endpoints

### Strategies

**GET /strategies** - List all strategies with metadata
**GET /strategies/{id}** - Strategy details
**POST /strategies/recommend?file_types=[...]** - Get recommendation

### Plugins

Subclass `ChunkingStrategy`, implement `chunk()`, register via:
```python
ChunkingStrategyFactory.register_strategy("my_plugin", MyPluginStrategy)
register_strategy_definition(api_id="my_plugin", ...)
```

Configs stored in DB (`chunking_config_profiles`). Legacy JSON file no longer used.

### Preview

**POST /preview** (10/min) - Generate preview, cached 15min
**POST /compare** (5/min) - Compare multiple strategies
**GET /preview/{id}** - Get cached
**DELETE /preview/{id}** - Clear cache

### Collections

**POST /collections/{id}/chunk** - Start async operation (returns `operation_id`)
**PATCH /collections/{id}/chunking-strategy** - Update strategy
**GET /collections/{id}/chunks?page=1&page_size=20** - List chunks
**GET /collections/{id}/chunking-stats** - Stats

### Analytics

**GET /metrics?period_days=30** - Global metrics
**GET /metrics/by-strategy** - Per-strategy metrics
**POST /analyze** - Analyze document, recommend strategy
**GET /quality-scores?collection_id=...** - Quality analysis

### Configs

**POST /configs** - Save custom config
**GET /configs?strategy=...&is_default=...** - List configs

### Progress

**GET /operations/{id}/progress** - Poll progress

## WebSocket

Connect to `/ws/operations/{operation_id}?token={jwt}`. Message format: `{type, data: {status, progress, message}}`. See `WEBSOCKET_API.md` for schema.

## Errors

Standard format: `{error, error_type, correlation_id, details, suggestions}`

**Types**: `CHUNKING_VALIDATION_ERROR` (400), `CHUNKING_MEMORY_ERROR` (413), `CHUNKING_TIMEOUT_ERROR` (504), `CHUNKING_CONFIGURATION_ERROR` (400)

## Rate Limits

- Preview: 10/min
- Compare: 5/min
- Collection chunking: 5/5min
- Config save: 10/min

## Best Practices

**Chunk sizes**: 100-300 (Q&A), 300-600 (general), 600-1000 (summarization)
**Overlap**: 0% (fast), 10-20% (balanced), 30-50% (max context)
**Testing**: Preview before full processing, compare strategies, monitor quality metrics

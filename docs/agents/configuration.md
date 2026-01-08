# Agent Configuration

This document covers configuration options for the Semantik Agent Plugin System.

## Environment Variables

### API Keys

```bash
# Claude Agent SDK (required for claude-agent plugin)
ANTHROPIC_API_KEY=sk-ant-...
```

### Feature Flags

```bash
# Enable agent plugins (default: true)
SEMANTIK_ENABLE_AGENT_PLUGINS=true
```

## Agent Plugin Configuration

Agent plugins can be configured through the REST API or when creating sessions.

### Configuration Schema

All agent plugins accept the following standard configuration:

```json
{
  "model": "claude-sonnet-4-20250514",
  "system_prompt": "You are a helpful assistant...",
  "temperature": 0.7,
  "max_tokens": 4096,
  "allowed_tools": ["semantic_search", "retrieve_document"],
  "timeout_seconds": 120
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | string | Plugin default | Model to use for inference |
| `system_prompt` | string | Plugin default | Base system prompt |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | 4096 | Maximum output tokens |
| `allowed_tools` | string[] | All enabled | Tool whitelist |
| `timeout_seconds` | float | 120 | Execution timeout |

### Model Selection

The Claude agent plugin supports these models:

| Model | Context | Output | Description |
|-------|---------|--------|-------------|
| `claude-sonnet-4-20250514` | 200K | 16K | Default, balanced |
| `claude-opus-4-20250514` | 200K | 16K | Most capable |
| `claude-haiku-3-5-20241022` | 200K | 8K | Fastest, most cost-effective |

## Session Configuration

Sessions can be configured at creation time:

```bash
curl -X POST http://localhost:8000/api/v2/agents/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "claude-agent",
    "config": {
      "model": "claude-sonnet-4-20250514",
      "temperature": 0.5
    },
    "collection_id": "abc123"
  }'
```

### Session Options

| Option | Type | Description |
|--------|------|-------------|
| `agent_id` | string | Required. Agent plugin ID |
| `config` | object | Agent configuration (see above) |
| `collection_id` | string | Associate with a collection |
| `title` | string | Session title for display |

## Tool Configuration

### Built-in Tools

Built-in tools are registered automatically on startup:

| Tool | Category | Description |
|------|----------|-------------|
| `semantic_search` | search | Search documents by semantic similarity |
| `retrieve_document` | search | Retrieve a document by ID |
| `retrieve_chunks` | search | Get chunks for a document |
| `list_collections` | search | List available collections |

### Enabling/Disabling Tools

Tools can be enabled or disabled via the ToolRegistry:

```python
from shared.agents.tools.registry import get_tool_registry

registry = get_tool_registry()

# Disable a tool
registry.set_enabled("semantic_search", False)

# Re-enable
registry.set_enabled("semantic_search", True)
```

### Tool Whitelisting per Session

Limit which tools are available in an execution:

```json
{
  "prompt": "Search for information",
  "tools": ["semantic_search"]
}
```

## Rate Limiting

Agent API endpoints have rate limiting configured:

| Endpoint | Limit |
|----------|-------|
| `/api/v2/agents` | 100/minute |
| `/api/v2/agents/{id}/execute` | 20/minute |
| `/api/v2/agents/sessions` | 50/minute |

## Logging

Agent operations are logged at various levels:

```bash
# Set log level for agent operations
LOG_LEVEL=INFO

# Debug logging for agent execution
LOG_LEVEL=DEBUG
```

Log messages include:

- Session creation/resumption
- Plugin initialization
- Tool execution
- Execution completion/errors

## Metrics

Prometheus metrics are exposed for monitoring:

```bash
# Metrics endpoint
curl http://localhost:9090/metrics | grep semantik_agent
```

Key metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `semantik_agent_executions_total` | Counter | Total executions |
| `semantik_agent_execution_duration_seconds` | Histogram | Execution latency |
| `semantik_agent_tokens_total` | Counter | Token usage |
| `semantik_agent_tool_calls_total` | Counter | Tool invocations |
| `semantik_agent_sessions_active` | Gauge | Active sessions |
| `semantik_agent_errors_total` | Counter | Error counts |

## Best Practices

1. **Model Selection**: Use `claude-haiku-3-5-20241022` for cost-sensitive operations like query expansion
2. **Temperature**: Lower values (0.3-0.5) for factual tasks, higher (0.7-1.0) for creative tasks
3. **Timeouts**: Set appropriate timeouts based on expected task complexity
4. **Tool Whitelisting**: Only enable tools needed for the specific use case
5. **Session Management**: Archive old sessions to keep database performant

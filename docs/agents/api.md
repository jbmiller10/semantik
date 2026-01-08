# Agent API Reference

This document describes the REST API and WebSocket protocol for the Agent Plugin System.

## Authentication

All endpoints require JWT authentication:

```bash
Authorization: Bearer <token>
```

## REST API

Base URL: `/api/v2/agents`

### List Agents

Returns all available agent plugins.

```
GET /api/v2/agents
```

**Response:**

```json
{
  "agents": [
    {
      "id": "claude-agent",
      "version": "1.0.0",
      "manifest": {
        "name": "Claude Agent",
        "description": "Claude-powered conversational agent"
      },
      "capabilities": {
        "supports_streaming": true,
        "supports_tools": true,
        "supports_sessions": true,
        "supported_models": ["claude-sonnet-4-20250514"]
      },
      "use_cases": ["assistant", "tool_use", "agentic_search"]
    }
  ]
}
```

### Get Agent Details

```
GET /api/v2/agents/{agent_id}
```

**Response:**

```json
{
  "id": "claude-agent",
  "version": "1.0.0",
  "manifest": {...},
  "capabilities": {...},
  "use_cases": [...],
  "config_schema": {
    "type": "object",
    "properties": {
      "model": {"type": "string"},
      "temperature": {"type": "number", "minimum": 0, "maximum": 2}
    }
  }
}
```

### Get Agent Capabilities

```
GET /api/v2/agents/{agent_id}/capabilities
```

**Response:**

```json
{
  "supports_streaming": true,
  "supports_tools": true,
  "supports_parallel_tools": true,
  "supports_sessions": true,
  "supports_session_fork": false,
  "supports_interruption": true,
  "supports_extended_thinking": false,
  "max_context_tokens": 200000,
  "max_output_tokens": 16000,
  "supported_models": ["claude-sonnet-4-20250514"],
  "default_model": "claude-sonnet-4-20250514"
}
```

### Execute Agent

Execute an agent with a prompt. Creates a new session if no `session_id` provided.

```
POST /api/v2/agents/{agent_id}/execute
```

**Request:**

```json
{
  "prompt": "Search for documents about machine learning",
  "session_id": null,
  "config": {
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.7
  },
  "tools": ["semantic_search", "retrieve_document"],
  "system_prompt": null,
  "max_tokens": 4096,
  "stream": true
}
```

**Response (Streaming):**

Server-Sent Events format:

```
data: {"type": "message", "message": {"id": "...", "role": "user", "content": "Search for..."}}

data: {"type": "message", "message": {"id": "...", "role": "assistant", "content": "", "is_partial": true}}

data: {"type": "message", "message": {"id": "...", "role": "assistant", "content": "I found...", "is_partial": false}}

data: {"type": "complete", "session_id": "abc123"}
```

### List Sessions

```
GET /api/v2/agents/sessions
```

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `status` | string | Filter by status (active, archived, deleted) |
| `collection_id` | string | Filter by collection |
| `limit` | int | Max results (default: 50) |
| `offset` | int | Pagination offset |

**Response:**

```json
{
  "sessions": [
    {
      "id": "abc123",
      "agent_plugin_id": "claude-agent",
      "status": "active",
      "title": "ML Research Session",
      "message_count": 5,
      "input_tokens": 1500,
      "output_tokens": 800,
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity_at": "2024-01-15T11:00:00Z"
    }
  ],
  "total": 42
}
```

### Create Session

```
POST /api/v2/agents/sessions
```

**Request:**

```json
{
  "agent_id": "claude-agent",
  "config": {
    "model": "claude-sonnet-4-20250514"
  },
  "collection_id": "collection-uuid",
  "title": "My Session"
}
```

**Response:**

```json
{
  "id": "abc123",
  "agent_plugin_id": "claude-agent",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Get Session

```
GET /api/v2/agents/sessions/{session_id}
```

**Response:**

```json
{
  "id": "abc123",
  "agent_plugin_id": "claude-agent",
  "collection_id": "collection-uuid",
  "status": "active",
  "title": "My Session",
  "message_count": 10,
  "input_tokens": 3500,
  "output_tokens": 2100,
  "cost_usd": 0.0156,
  "fork_count": 0,
  "parent_session_id": null,
  "created_at": "2024-01-15T10:30:00Z",
  "last_activity_at": "2024-01-15T12:00:00Z"
}
```

### Get Session Messages

```
GET /api/v2/agents/sessions/{session_id}/messages
```

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `limit` | int | Max messages (default: 100) |
| `offset` | int | Pagination offset |
| `after_sequence` | int | Only messages after this sequence |

**Response:**

```json
{
  "messages": [
    {
      "id": "msg-1",
      "sequence": 0,
      "role": "user",
      "type": "text",
      "content": "Search for documents about ML",
      "created_at": "2024-01-15T10:30:00Z"
    },
    {
      "id": "msg-2",
      "sequence": 1,
      "role": "assistant",
      "type": "tool_use",
      "content": "",
      "tool_name": "semantic_search",
      "tool_call_id": "call-1",
      "tool_input": {"query": "machine learning"},
      "created_at": "2024-01-15T10:30:01Z"
    },
    {
      "id": "msg-3",
      "sequence": 2,
      "role": "assistant",
      "type": "text",
      "content": "I found 5 relevant documents...",
      "model": "claude-sonnet-4-20250514",
      "input_tokens": 500,
      "output_tokens": 200,
      "created_at": "2024-01-15T10:30:03Z"
    }
  ]
}
```

### Fork Session

Create a branch of a session for exploration.

```
POST /api/v2/agents/sessions/{session_id}/fork
```

**Response:**

```json
{
  "session_id": "new-session-id",
  "parent_session_id": "abc123"
}
```

### Delete Session

```
DELETE /api/v2/agents/sessions/{session_id}
```

**Response:** `204 No Content`

### List Tools

```
GET /api/v2/agents/tools
```

**Response:**

```json
{
  "tools": [
    {
      "name": "semantic_search",
      "description": "Search documents by semantic similarity",
      "category": "search",
      "parameters": [
        {
          "name": "query",
          "type": "string",
          "description": "Search query",
          "required": true
        },
        {
          "name": "limit",
          "type": "integer",
          "description": "Max results",
          "required": false,
          "default": 10
        }
      ]
    }
  ]
}
```

## WebSocket API

Endpoint: `/ws/agents/{session_id}`

### Authentication

Pass JWT token via:

1. `Sec-WebSocket-Protocol` header: `Bearer, <token>`
2. Query parameter: `?token=<token>`

### Client Messages

#### Execute

```json
{
  "type": "execute",
  "prompt": "Hello, search for...",
  "tools": ["semantic_search"],
  "model": "claude-sonnet-4-20250514",
  "temperature": 0.7,
  "max_tokens": 4096
}
```

#### Interrupt

```json
{
  "type": "interrupt"
}
```

#### Ping

```json
{
  "type": "ping"
}
```

### Server Messages

#### Message

```json
{
  "type": "message",
  "message": {
    "id": "msg-uuid",
    "role": "assistant",
    "type": "text",
    "content": "I found...",
    "is_partial": false,
    "sequence_number": 2,
    "model": "claude-sonnet-4-20250514",
    "usage": {
      "input_tokens": 100,
      "output_tokens": 50
    }
  }
}
```

#### Complete

```json
{
  "type": "complete"
}
```

#### Error

```json
{
  "type": "error",
  "error": "Execution failed",
  "error_code": "AgentExecutionError"
}
```

#### Pong

```json
{
  "type": "pong"
}
```

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid request body",
  "errors": [
    {"field": "prompt", "message": "Field required"}
  ]
}
```

### 401 Unauthorized

```json
{
  "detail": "Not authenticated"
}
```

### 404 Not Found

```json
{
  "detail": "Session not found: abc123"
}
```

### 429 Too Many Requests

```json
{
  "detail": "Rate limit exceeded. Try again in 30 seconds."
}
```

### 500 Internal Server Error

```json
{
  "detail": "Agent execution failed",
  "error_code": "AgentExecutionError"
}
```

## Message Types

| Type | Description |
|------|-------------|
| `text` | Plain text content |
| `thinking` | Extended thinking/reasoning |
| `tool_use` | Agent wants to use a tool |
| `tool_output` | Result from tool execution |
| `partial` | Streaming partial content |
| `final` | Final response |
| `error` | Error message |
| `metadata` | Usage stats, costs, etc. |

## Message Roles

| Role | Description |
|------|-------------|
| `user` | User message |
| `assistant` | Agent response |
| `system` | System message |
| `tool_call` | Tool invocation |
| `tool_result` | Tool result |
| `error` | Error message |

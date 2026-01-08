# Agent Plugin System

The Agent Plugin System enables LLM-powered capabilities in Semantik, including conversational assistants, agentic search, query enhancement, and result synthesis.

## Overview

Agents are plugins that integrate Large Language Model (LLM) capabilities into Semantik. They follow the same plugin architecture as other Semantik plugins (embedding, chunking, etc.) but provide specialized functionality for:

- **Conversational Assistants** - Interactive chat interfaces for end-users
- **Agentic Search** - Multi-step retrieval with reasoning and tool use
- **HyDE (Hypothetical Document Embeddings)** - Query enhancement via generated documents
- **Query Expansion** - Generate alternative queries for improved recall
- **Summarization** - Compress and synthesize retrieved content
- **Answer Synthesis** - RAG-style answer generation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Semantik System                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   React WebUI   │    │   REST API      │    │   WebSocket     │  │
│  │                 │    │  /api/v2/agents │    │   /ws/agents    │  │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘  │
│           │                      │                      │            │
│           └──────────────────────┼──────────────────────┘            │
│                                  │                                   │
│                                  ▼                                   │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                         AgentService                           │   │
│  │  - Plugin instance management                                  │   │
│  │  - Session orchestration                                       │   │
│  │  - Tool injection                                              │   │
│  │  - Metrics collection                                          │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                  │                                   │
│           ┌──────────────────────┼──────────────────────┐            │
│           │                      │                      │            │
│           ▼                      ▼                      ▼            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│  │  PluginRegistry │    │  ToolRegistry   │    │ SessionRepository│  │
│  │  (agent type)   │    │                 │    │  (PostgreSQL)   │   │
│  └────────┬────────┘    └────────┬────────┘    └─────────────────┘   │
│           │                      │                                   │
│           ▼                      ▼                                   │
│  ┌─────────────────┐    ┌─────────────────┐                          │
│  │  AgentPlugin    │    │  AgentTool      │                          │
│  │  (ABC)          │    │  Definitions    │                          │
│  └────────┬────────┘    └─────────────────┘                          │
│           │                                                          │
│           ▼                                                          │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                     AgentAdapter (ABC)                         │   │
│  │  - SDK-agnostic interface                                      │   │
│  │  - Streaming execution                                         │   │
│  │  - Session management                                          │   │
│  └───────────────────────────────────────────────────────────────┘   │
│           │                                          │               │
│           ▼                                          ▼               │
│  ┌─────────────────┐                        ┌─────────────────┐     │
│  │ ClaudeAdapter   │                        │ OpenAIAdapter   │     │
│  │ (claude-agent-  │                        │ (Future)        │     │
│  │  sdk)           │                        │                 │     │
│  └─────────────────┘                        └─────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### AgentPlugin

The base class for all agent plugins. Extends `SemanticPlugin` with agent-specific methods:

```python
from shared.plugins.types.agent import AgentPlugin
from shared.agents.types import AgentCapabilities, AgentContext, AgentMessage, AgentUseCase

class MyAgentPlugin(AgentPlugin):
    PLUGIN_ID = "my-agent"
    PLUGIN_VERSION = "1.0.0"

    @classmethod
    def get_capabilities(cls) -> AgentCapabilities:
        return AgentCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supported_models=["model-name"],
        )

    @classmethod
    def supported_use_cases(cls) -> list[AgentUseCase]:
        return [AgentUseCase.ASSISTANT, AgentUseCase.TOOL_USE]

    async def execute(self, prompt: str, **kwargs) -> AsyncIterator[AgentMessage]:
        # Implementation
        yield message
```

### AgentAdapter

SDK-agnostic interface for LLM execution. The `ClaudeAgentAdapter` is the primary implementation using the Claude Agent SDK.

### ToolRegistry

Central registry for agent-accessible tools. Built-in tools include:

- `semantic_search` - Search documents by semantic similarity
- `retrieve_document` - Retrieve a specific document by ID
- `retrieve_chunks` - Get chunks for a document
- `list_collections` - List available collections

### AgentService

Service layer that orchestrates:

- Plugin instance caching
- Session creation/resumption
- Tool resolution
- Message persistence
- Metrics collection

## Use Cases

| Use Case | Description |
|----------|-------------|
| `ASSISTANT` | Conversational interface |
| `AGENTIC_SEARCH` | Multi-step retrieval with reasoning |
| `HYDE` | Hypothetical document generation |
| `QUERY_EXPANSION` | Generate alternative queries |
| `SUMMARIZATION` | Content compression |
| `ANSWER_SYNTHESIS` | RAG answer generation |
| `TOOL_USE` | General tool-using agent |
| `REASONING` | Chain-of-thought, planning |

## Quick Start

### List Available Agents

```bash
curl -X GET http://localhost:8000/api/v2/agents \
  -H "Authorization: Bearer $TOKEN"
```

### Create a Session and Execute

```bash
curl -X POST http://localhost:8000/api/v2/agents/claude-agent/execute \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Search for documents about machine learning",
    "tools": ["semantic_search"]
  }'
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/agents/SESSION_ID');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'execute',
    prompt: 'Hello, search for...',
    tools: ['semantic_search']
  }));
};
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'message') {
    console.log(data.message.content);
  }
};
```

## Documentation

- [Configuration](./configuration.md) - Environment variables and agent config
- [API Reference](./api.md) - REST API and WebSocket protocol
- [Tool Development](./tools.md) - Creating custom tools

## Metrics

Agent operations are instrumented with Prometheus metrics:

- `semantik_agent_executions_total` - Total executions by agent and status
- `semantik_agent_execution_duration_seconds` - Execution duration histogram
- `semantik_agent_tokens_total` - Token usage by agent and direction
- `semantik_agent_tool_calls_total` - Tool calls by name and status
- `semantik_agent_sessions_active` - Active session count

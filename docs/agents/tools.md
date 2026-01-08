# Agent Tool Development Guide

This guide covers creating custom tools for the Semantik Agent Plugin System.

## Overview

Tools are functions that agents can call during execution. They extend agent capabilities beyond text generation to include search, retrieval, and other actions.

## Built-in Tools

The system includes these built-in tools:

| Tool | Description |
|------|-------------|
| `semantic_search` | Search documents by semantic similarity |
| `retrieve_document` | Retrieve a document by ID |
| `retrieve_chunks` | Get chunks for a document |
| `list_collections` | List available collections |

## Creating a Custom Tool

### 1. Extend AgentTool

```python
from typing import Any
from shared.agents.tools.base import AgentTool, ToolDefinition, ToolParameter
from shared.agents.types import AgentContext


class MyCustomTool(AgentTool):
    """A custom tool for agents."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "my_custom_tool"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Does something useful for agents",
            category="custom",
            parameters=[
                ToolParameter(
                    name="input_text",
                    type="string",
                    description="The text to process",
                    required=True,
                ),
                ToolParameter(
                    name="option",
                    type="string",
                    description="Processing option",
                    required=False,
                    default="default",
                    enum=["default", "advanced"],
                ),
            ],
            timeout_seconds=30.0,
        )

    async def execute(
        self,
        args: dict[str, Any],
        context: AgentContext | None = None,
    ) -> dict[str, Any]:
        """Execute the tool.

        Args:
            args: Validated arguments matching the definition.
            context: Optional runtime context with collection/session info.

        Returns:
            Result dictionary to return to the agent.
        """
        input_text = args["input_text"]
        option = args.get("option", "default")

        # Implement your logic here
        result = await self._process(input_text, option)

        return {
            "success": True,
            "result": result,
        }

    async def _process(self, text: str, option: str) -> str:
        # Your implementation
        return f"Processed: {text} with {option}"
```

### 2. Register the Tool

```python
from shared.agents.tools.registry import get_tool_registry

# Register at module load or application startup
registry = get_tool_registry()
registry.register(
    MyCustomTool(),
    source="plugin",  # or "custom"
    enabled=True,
    metadata={"author": "your-name"},
)
```

## ToolDefinition

The `ToolDefinition` dataclass describes the tool to the agent:

```python
@dataclass(frozen=True)
class ToolDefinition:
    name: str              # Unique tool name
    description: str       # What the tool does (shown to agent)
    category: str          # Grouping (e.g., "search", "utility")
    parameters: list[ToolParameter]  # Input parameters
    returns: str | None    # Description of return value
    timeout_seconds: float # Execution timeout (default: 30.0)
    examples: list[dict]   # Usage examples for the agent
```

## ToolParameter

Each parameter is defined with:

```python
@dataclass(frozen=True)
class ToolParameter:
    name: str           # Parameter name
    type: str           # Type: "string", "integer", "number", "boolean", "array", "object"
    description: str    # Parameter description
    required: bool      # Is this parameter required?
    default: Any        # Default value if not required
    enum: list | None   # Allowed values (for strings)
    items: dict | None  # For arrays: {"type": "string"}
```

## Using Context

The `AgentContext` provides runtime information:

```python
async def execute(
    self,
    args: dict[str, Any],
    context: AgentContext | None = None,
) -> dict[str, Any]:
    # Access context if available
    if context:
        collection_id = context.collection_id
        user_id = context.user_id
        request_id = context.request_id

    # Use context for authorization, scoping, etc.
    ...
```

Context fields:

| Field | Description |
|-------|-------------|
| `request_id` | Unique request identifier |
| `user_id` | Current user ID |
| `collection_id` | Associated collection |
| `collection_name` | Collection name |
| `session_id` | Current session |
| `original_query` | Original user query |
| `retrieved_chunks` | Previously retrieved chunks |

## Validation

Tool arguments are validated automatically against the definition. You can add custom validation:

```python
def validate_args(self, args: dict[str, Any]) -> tuple[bool, list[str]]:
    """Custom argument validation.

    Args:
        args: Arguments to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    is_valid, errors = super().validate_args(args)

    # Add custom validation
    if "input_text" in args and len(args["input_text"]) > 10000:
        errors.append("input_text exceeds maximum length of 10000")
        is_valid = False

    return is_valid, errors
```

## Error Handling

Raise `ToolExecutionError` for recoverable errors:

```python
from shared.agents.exceptions import ToolExecutionError

async def execute(self, args: dict[str, Any], context: AgentContext | None = None):
    try:
        result = await self._do_work(args)
        return {"result": result}
    except SomeError as e:
        raise ToolExecutionError(
            f"Processing failed: {e}",
            tool_name=self.name,
            cause=str(e),
        )
```

## Examples in Definition

Provide examples to help agents use the tool correctly:

```python
@property
def definition(self) -> ToolDefinition:
    return ToolDefinition(
        name=self.name,
        description="Search for documents",
        category="search",
        parameters=[...],
        examples=[
            {
                "description": "Search for ML documents",
                "args": {"query": "machine learning algorithms"},
                "result": {"documents": [...]}
            },
            {
                "description": "Search with filters",
                "args": {"query": "neural networks", "limit": 5},
                "result": {"documents": [...]}
            }
        ]
    )
```

## Tool Categories

Organize tools by category for discoverability:

| Category | Purpose |
|----------|---------|
| `search` | Document/content search |
| `retrieval` | Fetch specific items |
| `utility` | General utilities |
| `analysis` | Data analysis |
| `external` | External API calls |

## Best Practices

1. **Clear Descriptions**: Write descriptions that help the agent understand when to use the tool
2. **Minimal Parameters**: Keep required parameters minimal; use defaults for optional ones
3. **Structured Returns**: Return structured data (dicts) rather than raw text
4. **Error Context**: Include context in error messages to help agents recover
5. **Timeouts**: Set appropriate timeouts based on expected execution time
6. **Idempotency**: Make tools idempotent where possible
7. **Context Awareness**: Use context for authorization and scoping

## Testing Tools

```python
import pytest
from shared.agents.types import AgentContext

@pytest.fixture
def tool():
    return MyCustomTool()

@pytest.fixture
def context():
    return AgentContext(
        request_id="test-123",
        user_id="user-1",
        collection_id="collection-1",
    )

async def test_tool_executes_successfully(tool, context):
    result = await tool.execute(
        {"input_text": "hello"},
        context=context,
    )
    assert result["success"] is True
    assert "result" in result

async def test_tool_validates_required_args(tool):
    is_valid, errors = tool.validate_args({})
    assert not is_valid
    assert "input_text" in errors[0]
```

## Registering via Entry Points

For plugin distribution, register tools via entry points:

```toml
# pyproject.toml
[project.entry-points."semantik.agent_tools"]
my_tool = "my_package.tools:MyCustomTool"
```

Then load in your plugin's initialization:

```python
from importlib.metadata import entry_points

def load_custom_tools():
    registry = get_tool_registry()
    eps = entry_points(group="semantik.agent_tools")
    for ep in eps:
        tool_class = ep.load()
        registry.register(tool_class(), source="plugin")
```

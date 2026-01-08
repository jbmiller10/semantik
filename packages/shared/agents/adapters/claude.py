"""Claude Agent SDK adapter.

This module implements the ClaudeAgentAdapter, which wraps the Claude Agent SDK
and translates between SDK-specific message formats and the unified AgentMessage type.

The adapter supports:
- Streaming responses via async generator
- Tool use via MCP integration
- Session management with resume/fork capabilities
- Interruption via asyncio Event
- Extended thinking/reasoning

Example usage:

    adapter = ClaudeAgentAdapter({"model": "claude-sonnet-4-20250514"})
    await adapter.initialize()

    async for message in adapter.execute("Hello, Claude!"):
        print(message.content)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from shared.agents.base import AgentAdapter
from shared.agents.exceptions import (
    AgentExecutionError,
    AgentInitializationError,
    AgentInterruptedError,
)
from shared.agents.types import (
    AgentCapabilities,
    AgentMessage,
    MessageRole,
    MessageType,
    TokenUsage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from shared.agents.types import AgentContext

logger = logging.getLogger(__name__)


class ClaudeAgentAdapter(AgentAdapter):
    """Adapter for Claude Agent SDK.

    This adapter wraps the Claude Agent SDK and provides:
    - Message translation to unified AgentMessage format
    - Streaming support with interrupt capability
    - Session management (resume, fork)
    - Tool injection via MCP

    Supported models:
    - claude-sonnet-4-20250514 (default)
    - claude-opus-4-20250514
    - claude-haiku-3-5-20241022

    Attributes:
        SUPPORTED_MODELS: Map of model IDs to their capabilities.
        DEFAULT_MODEL: Default model to use if none specified.
    """

    SUPPORTED_MODELS: dict[str, dict[str, int]] = {
        "claude-sonnet-4-20250514": {"context": 200000, "output": 16000},
        "claude-opus-4-20250514": {"context": 200000, "output": 16000},
        "claude-haiku-3-5-20241022": {"context": 200000, "output": 8192},
    }

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the Claude adapter.

        Args:
            config: Adapter configuration. Supported fields:
                - model: Model to use (default: claude-sonnet-4-20250514)
                - system_prompt: Default system prompt
                - max_turns: Maximum agentic turns
                - permission_mode: SDK permission mode
        """
        super().__init__(config)
        self._sdk_available = False

    @classmethod
    def get_capabilities(cls) -> AgentCapabilities:
        """Return Claude-specific capabilities.

        Returns:
            AgentCapabilities with Claude's supported features.
        """
        return AgentCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_sessions=True,
            supports_session_fork=True,
            supports_interruption=True,
            supports_extended_thinking=True,
            supports_thinking_budget=True,
            supports_subagents=False,
            supports_handoffs=False,
            max_context_tokens=200000,
            max_output_tokens=16000,
            supported_models=tuple(cls.SUPPORTED_MODELS.keys()),
            default_model=cls.DEFAULT_MODEL,
        )

    async def initialize(self) -> None:
        """Initialize the adapter and verify SDK availability.

        Raises:
            AgentInitializationError: If Claude SDK is not installed.
        """
        if self._is_initialized:
            return

        try:
            # Lazy import to verify SDK is available
            from claude_agent_sdk import query  # noqa: F401

            self._sdk_available = True
        except ImportError as e:
            raise AgentInitializationError(
                "Claude Agent SDK not installed. Install with: pip install claude-agent-sdk",
                adapter="claude",
                cause=str(e),
            ) from e

        await super().initialize()
        logger.info("ClaudeAgentAdapter initialized successfully")

    async def cleanup(self) -> None:
        """Clean up adapter resources."""
        await super().cleanup()
        self._sdk_available = False
        logger.info("ClaudeAgentAdapter cleaned up")

    async def execute(
        self,
        prompt: str,
        *,
        context: AgentContext | None = None,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        stream: bool = True,
    ) -> AsyncIterator[AgentMessage]:
        """Execute the agent with the given prompt.

        Wraps the Claude SDK query() function and translates responses
        to AgentMessage format.

        Args:
            prompt: User message or task description.
            context: Runtime context (collection, retrieved content, etc.).
            system_prompt: Override the default system prompt.
            tools: Tool instances to make available.
            model: Model override.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            session_id: Resume an existing SDK session.
            stream: Whether to stream partial responses.

        Yields:
            AgentMessage objects representing the response stream.

        Raises:
            AgentInitializationError: If SDK not available.
            AgentExecutionError: If execution fails.
            AgentInterruptedError: If execution is interrupted.
        """
        # Auto-initialize if needed
        if not self._is_initialized:
            await self.initialize()

        # Clear interrupt event for new execution
        if self._interrupt_event:
            self._interrupt_event.clear()

        # Import SDK
        from claude_agent_sdk import query

        # Build options
        options = self._build_options(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            session_id=session_id,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Build prompt with context injection
        full_prompt = self._inject_context(prompt, context)

        # Execute
        sequence = 0
        try:
            async for sdk_msg in query(prompt=full_prompt, options=options):
                # Check for interruption
                if self._interrupt_event and self._interrupt_event.is_set():
                    logger.info("Execution interrupted by user")
                    raise AgentInterruptedError("Execution interrupted by user")

                # Translate SDK message to AgentMessage
                message = self._translate_message(sdk_msg, sequence, model)
                sequence += 1

                # Capture session ID from SDK
                if hasattr(sdk_msg, "session_id") and sdk_msg.session_id:
                    self._current_session_id = sdk_msg.session_id

                yield message

        except AgentInterruptedError:
            raise
        except Exception as e:
            logger.exception("Claude execution failed")
            raise AgentExecutionError(
                f"Claude execution failed: {e}",
                adapter="claude",
                cause=str(e),
            ) from e

    async def fork_session(self, session_id: str) -> str:
        """Fork a session to create a branch point.

        Args:
            session_id: ID of the session to fork.

        Returns:
            ID of the new forked session.

        Raises:
            AgentExecutionError: If forking fails.
        """
        if not self._is_initialized:
            await self.initialize()

        from claude_agent_sdk import ClaudeAgentOptions, query

        options = ClaudeAgentOptions(
            resume=session_id,
            fork_session=True,
        )

        try:
            async for msg in query(prompt="", options=options):
                if hasattr(msg, "session_id") and msg.session_id:
                    return msg.session_id

            raise AgentExecutionError(
                "Failed to fork session: no session ID returned",
                adapter="claude",
            )
        except Exception as e:
            raise AgentExecutionError(
                f"Failed to fork session: {e}",
                adapter="claude",
                cause=str(e),
            ) from e

    def _build_options(
        self,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        session_id: str | None = None,
        stream: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Build ClaudeAgentOptions from parameters.

        Args:
            model: Model to use.
            system_prompt: System prompt override.
            tools: Tool instances.
            session_id: Session to resume.
            stream: Enable streaming.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            ClaudeAgentOptions instance.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        # Determine model
        effective_model = model or self._config.get("model") or self.DEFAULT_MODEL

        # Build base options
        options_kwargs: dict[str, Any] = {
            "model": effective_model,
            "include_partial_messages": stream,
        }

        # Add system prompt
        if system_prompt:
            options_kwargs["system_prompt"] = system_prompt
        elif self._config.get("system_prompt"):
            options_kwargs["system_prompt"] = self._config["system_prompt"]

        # Add temperature if specified
        if temperature is not None:
            options_kwargs["temperature"] = temperature

        # Add max tokens if specified
        if max_tokens is not None:
            options_kwargs["max_tokens"] = max_tokens

        # Add session resume
        if session_id:
            options_kwargs["resume"] = session_id

        # Add max turns from config
        if self._config.get("max_turns"):
            options_kwargs["max_turns"] = self._config["max_turns"]

        # Add permission mode from config
        if self._config.get("permission_mode"):
            options_kwargs["permission_mode"] = self._config["permission_mode"]

        # Add tools if provided
        if tools:
            options_kwargs["mcp_servers"] = self._convert_tools_to_mcp(tools)

        return ClaudeAgentOptions(**options_kwargs)

    def _inject_context(self, prompt: str, context: AgentContext | None) -> str:
        """Inject context into the prompt.

        Adds retrieved chunks and collection information to the prompt
        for RAG-style use cases.

        Args:
            prompt: Original user prompt.
            context: Runtime context.

        Returns:
            Prompt with context injected.
        """
        if context is None:
            return prompt

        parts: list[str] = []

        # Add collection context
        if context.collection_name:
            parts.append(f"<collection>{context.collection_name}</collection>")

        # Add retrieved chunks
        if context.retrieved_chunks:
            chunks_xml = ["<retrieved_context>"]
            for i, chunk in enumerate(context.retrieved_chunks):
                content = chunk.get("content", "")
                score = chunk.get("score", "")
                source = chunk.get("source", "")
                chunks_xml.append(f'  <chunk index="{i}" score="{score}" source="{source}">')
                chunks_xml.append(f"    {content}")
                chunks_xml.append("  </chunk>")
            chunks_xml.append("</retrieved_context>")
            parts.append("\n".join(chunks_xml))

        # Add original query if different from prompt
        if context.original_query and context.original_query != prompt:
            parts.append(f"<original_query>{context.original_query}</original_query>")

        # Add user prompt
        parts.append(prompt)

        return "\n\n".join(parts)

    def _translate_message(
        self,
        sdk_msg: Any,
        sequence: int,
        model: str | None = None,
    ) -> AgentMessage:
        """Translate SDK message to AgentMessage.

        Args:
            sdk_msg: SDK message object.
            sequence: Sequence number in the stream.
            model: Model used for generation.

        Returns:
            AgentMessage instance.
        """
        # Determine message type and role
        msg_type = MessageType.TEXT
        role = MessageRole.ASSISTANT
        content = ""
        tool_name = None
        tool_call_id = None
        tool_input = None
        tool_output = None
        is_partial = False
        usage = None

        # Extract content based on SDK message structure
        if hasattr(sdk_msg, "subtype"):
            subtype = sdk_msg.subtype

            if subtype == "init":
                msg_type = MessageType.METADATA
                content = "Execution started"

            elif subtype == "success":
                msg_type = MessageType.FINAL
                content = self._extract_text_content(sdk_msg)
                usage = self._extract_usage(sdk_msg)

            elif subtype == "error":
                msg_type = MessageType.ERROR
                role = MessageRole.ERROR
                content = getattr(sdk_msg, "error_message", str(sdk_msg))

            else:
                # Handle content blocks
                content, msg_type, role = self._extract_content_block(sdk_msg)
                is_partial = getattr(sdk_msg, "is_partial", False)

        elif hasattr(sdk_msg, "content"):
            content = self._extract_text_content(sdk_msg)
            is_partial = getattr(sdk_msg, "is_partial", False)
            if is_partial:
                msg_type = MessageType.PARTIAL

        # Handle tool use
        if hasattr(sdk_msg, "tool_use") and sdk_msg.tool_use:
            tool_data = sdk_msg.tool_use
            msg_type = MessageType.TOOL_USE
            role = MessageRole.TOOL_CALL
            tool_name = getattr(tool_data, "name", None)
            tool_call_id = getattr(tool_data, "id", None)
            tool_input = getattr(tool_data, "input", None)

        # Handle tool result
        if hasattr(sdk_msg, "tool_result") and sdk_msg.tool_result:
            msg_type = MessageType.TOOL_OUTPUT
            role = MessageRole.TOOL_RESULT
            tool_output = sdk_msg.tool_result

        # Handle thinking/reasoning
        if hasattr(sdk_msg, "thinking") and sdk_msg.thinking:
            msg_type = MessageType.THINKING
            content = sdk_msg.thinking

        return AgentMessage(
            role=role,
            type=msg_type,
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_input=tool_input,
            tool_output=tool_output,
            model=model or self._config.get("model") or self.DEFAULT_MODEL,
            usage=usage,
            is_partial=is_partial,
            sequence_number=sequence,
        )

    def _extract_text_content(self, sdk_msg: Any) -> str:
        """Extract text content from SDK message.

        Args:
            sdk_msg: SDK message object.

        Returns:
            Text content string.
        """
        if hasattr(sdk_msg, "content"):
            content = sdk_msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Handle content blocks
                text_parts = []
                for block in content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                return "".join(text_parts)
        return ""

    def _extract_content_block(
        self,
        sdk_msg: Any,
    ) -> tuple[str, MessageType, MessageRole]:
        """Extract content, type, and role from content block.

        Args:
            sdk_msg: SDK message with content blocks.

        Returns:
            Tuple of (content, message_type, role).
        """
        content = self._extract_text_content(sdk_msg)
        msg_type = MessageType.TEXT
        role = MessageRole.ASSISTANT

        # Check for specific block types
        if hasattr(sdk_msg, "type"):
            block_type = sdk_msg.type
            if block_type == "thinking":
                msg_type = MessageType.THINKING
            elif block_type == "tool_use":
                msg_type = MessageType.TOOL_USE
                role = MessageRole.TOOL_CALL
            elif block_type == "tool_result":
                msg_type = MessageType.TOOL_OUTPUT
                role = MessageRole.TOOL_RESULT

        return content, msg_type, role

    def _extract_usage(self, sdk_msg: Any) -> TokenUsage | None:
        """Extract token usage from SDK message.

        Args:
            sdk_msg: SDK message with usage data.

        Returns:
            TokenUsage instance or None.
        """
        if not hasattr(sdk_msg, "usage"):
            return None

        usage = sdk_msg.usage
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            reasoning_tokens=getattr(usage, "reasoning_tokens", 0),
        )

    def _convert_tools_to_mcp(self, tools: list[Any]) -> dict[str, Any]:
        """Convert tool instances to MCP server format.

        Args:
            tools: List of AgentTool instances.

        Returns:
            MCP servers dict for SDK options.
        """
        # Build MCP tools from AgentTool instances
        mcp_tools: list[dict[str, Any]] = []

        for tool in tools:
            if hasattr(tool, "definition"):
                definition = tool.definition
                mcp_tools.append(
                    {
                        "name": definition.name,
                        "description": definition.description,
                        "input_schema": definition.to_json_schema() if hasattr(definition, "to_json_schema") else {},
                    }
                )
            elif hasattr(tool, "name") and hasattr(tool, "description"):
                mcp_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": getattr(tool, "input_schema", {}),
                    }
                )

        # Return as MCP server config
        return {"semantik": {"tools": mcp_tools}}

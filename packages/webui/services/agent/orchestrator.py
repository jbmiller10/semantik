"""Main orchestrator for the pipeline builder agent.

The AgentOrchestrator coordinates conversations between the user and the
LLM-powered agent, managing tool execution, state persistence, and
conversation recovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from shared.llm.base import BaseLLMService
from shared.llm.exceptions import LLMAuthenticationError, LLMContextLengthError, LLMRateLimitError, LLMTimeoutError
from webui.api.v2.agent_schemas import AgentStreamEvent, AgentStreamEventType
from webui.services.agent.exceptions import AgentBusyError, AgentError, ConversationNotActiveError, ToolExecutionError
from webui.services.agent.message_store import ConversationMessage, MessageStore
from webui.services.agent.models import AgentConversation, ConversationStatus, UncertaintySeverity
from webui.services.agent.repository import AgentConversationRepository
from webui.services.agent.tools import (
    ApplyPipelineTool,
    BaseTool,
    BuildPipelineTool,
    GetPipelineStateTool,
    GetPluginDetailsTool,
    GetTemplateDetailsTool,
    ListPluginsTool,
    ListTemplatesTool,
    SpawnPipelineValidatorTool,
    SpawnSourceAnalyzerTool,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.llm.factory import LLMServiceFactory

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A tool call extracted from LLM response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    name: str
    success: bool
    data: Any = None
    error: str | None = None


@dataclass
class AgentResponse:
    """Response from a single agent turn."""

    content: str
    tool_calls_made: list[dict[str, Any]]
    pipeline_updated: bool
    uncertainties_added: list[dict[str, Any]]


# Tool classes available to the orchestrator
ORCHESTRATOR_TOOL_CLASSES: list[type[BaseTool]] = [
    ListPluginsTool,
    GetPluginDetailsTool,
    ListTemplatesTool,
    GetTemplateDetailsTool,
    GetPipelineStateTool,
    BuildPipelineTool,
    ApplyPipelineTool,
    SpawnSourceAnalyzerTool,
    SpawnPipelineValidatorTool,
]


class AgentOrchestrator:
    """Orchestrates pipeline builder agent conversations.

    The orchestrator manages the conversation flow between users and the LLM,
    handling tool execution, state persistence, and error recovery.

    Key responsibilities:
    - Load/save message history from Redis
    - Build LLM context (system prompt + messages + tool schemas)
    - Call LLM and parse responses for tool calls
    - Execute tools and collect results
    - Update conversation state in PostgreSQL
    - Handle conversation recovery when Redis expires

    Usage:
        orchestrator = AgentOrchestrator(
            conversation=conv,
            session=db_session,
            llm_factory=factory,
            message_store=store,
        )
        response = await orchestrator.handle_message("Use semantic chunking")
    """

    SYSTEM_PROMPT: ClassVar[
        str
    ] = """You are Semantik's Pipeline Builder Agent. Your job is to help users configure document processing pipelines for their collections.

You have access to tools that let you:
1. List and inspect available plugins (embedding models, chunking strategies, extractors)
2. List and inspect pipeline templates (pre-configured pipelines for common use cases)
3. Build custom pipelines from templates or from scratch
4. Analyze source documents to recommend optimal configurations
5. Validate pipeline configurations before applying them

## Guidelines

- Start by understanding what the user wants to accomplish
- Recommend appropriate templates when they fit the use case
- Use source analysis to make data-driven recommendations
- Validate pipelines before applying them
- Surface any uncertainties or concerns clearly

## Tool Usage

When you need to take an action, output a tool call in this format:
```tool
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
```

You can call multiple tools by including multiple tool blocks.

When you're done responding (no more tools to call), just write your response normally.

## Available Tools

{tool_descriptions}

## Available Templates

These are the pre-configured pipeline templates you can use with build_pipeline(template_id="..."):

{available_templates}

## Important Notes

- Always validate pipelines before applying
- Flag any uncertainties as blocking, notable, or info
- Use spawn_source_analyzer for complex document analysis
- Use spawn_pipeline_validator to check configurations
"""

    # Maximum conversation turns before forcing completion.
    # Prevents infinite loops while allowing complex multi-step workflows.
    MAX_TURNS: ClassVar[int] = 20

    # Regex pattern for extracting tool calls from LLM responses.
    # Format: ```tool\n{"name": "tool_name", "arguments": {...}}\n```
    TOOL_CALL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"```tool\s*\n?(.*?)\n?```",
        re.DOTALL,
    )

    # Retry policy (best-effort, interactive)
    LLM_RATE_LIMIT_MAX_ATTEMPTS: ClassVar[int] = 3
    LLM_RATE_LIMIT_BASE_DELAY_SECONDS: ClassVar[float] = 1.0
    LLM_RATE_LIMIT_MAX_DELAY_SECONDS: ClassVar[float] = 10.0
    LLM_TIMEOUT_MAX_ATTEMPTS: ClassVar[int] = 2  # initial + 1 retry
    LLM_TIMEOUT_RETRY_DELAY_SECONDS: ClassVar[float] = 0.25

    def __init__(
        self,
        conversation: AgentConversation,
        session: AsyncSession,
        llm_factory: LLMServiceFactory,
        message_store: MessageStore,
    ):
        """Initialize the orchestrator.

        Args:
            conversation: The conversation to manage
            session: Database session for persistence
            llm_factory: Factory for creating LLM providers
            message_store: Redis-based message storage
        """
        self.conversation = conversation
        self.session = session
        self.llm_factory = llm_factory
        self.message_store = message_store
        self.repo = AgentConversationRepository(session)

        # Track state changes during a turn
        self._pipeline_updated = False
        self._uncertainties_added: list[dict[str, Any]] = []
        self._tool_calls_made: list[dict[str, Any]] = []

        # Initialize tools
        self.tools: dict[str, BaseTool] = {}
        context = self._build_tool_context()
        for tool_class in ORCHESTRATOR_TOOL_CLASSES:
            tool = tool_class(context)
            self.tools[tool.NAME] = tool

    async def _llm_generate_with_retries(
        self,
        provider: BaseLLMService,
        *,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Call provider.generate() with retry/backoff for rate limits and timeouts.

        Retry policy:
        - Rate limits: exponential backoff with jitter, max 3 attempts
        - Timeouts: retry once with doubled timeout (capped at MAX_BACKGROUND_TIMEOUT)
        - Other LLM errors (auth, context length): no retry, propagate immediately
        """
        rate_limit_attempt = 0
        timeout_attempt = 0
        effective_timeout = timeout

        while True:
            try:
                return await provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=effective_timeout,
                )
            except LLMRateLimitError as e:
                rate_limit_attempt += 1
                if rate_limit_attempt >= self.LLM_RATE_LIMIT_MAX_ATTEMPTS:
                    raise

                base = (
                    e.retry_after
                    if e.retry_after is not None
                    else (self.LLM_RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** (rate_limit_attempt - 1)))
                )
                delay = min(base, self.LLM_RATE_LIMIT_MAX_DELAY_SECONDS)
                delay += random.random() * 0.25

                logger.info(
                    "LLM rate limited (provider=%s, attempt=%s/%s). Sleeping %.2fs",
                    e.provider,
                    rate_limit_attempt,
                    self.LLM_RATE_LIMIT_MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)
            except LLMTimeoutError as e:
                timeout_attempt += 1
                if timeout_attempt >= self.LLM_TIMEOUT_MAX_ATTEMPTS:
                    raise

                base_timeout = effective_timeout if effective_timeout is not None else e.timeout
                effective_timeout = min(base_timeout * 2, BaseLLMService.MAX_BACKGROUND_TIMEOUT)

                logger.info(
                    "LLM timed out (provider=%s, retrying with timeout=%.1fs)",
                    e.provider,
                    effective_timeout,
                )
                await asyncio.sleep(self.LLM_TIMEOUT_RETRY_DELAY_SECONDS)

    def _build_tool_context(self) -> dict[str, Any]:
        """Build context dictionary for tools."""
        return {
            "session": self.session,
            "user_id": self.conversation.user_id,
            "conversation": self.conversation,
            "orchestrator": self,
            "llm_factory": self.llm_factory,
        }

    async def _is_paused(self) -> bool:
        """Check if the conversation is paused."""
        extra_data = self.conversation.extra_data or {}
        return bool(extra_data.get("is_paused", False))

    def _emit_status(self, phase: str, message: str, progress: dict[str, int] | None = None) -> AgentStreamEvent:
        """Create a status event for streaming.

        Args:
            phase: Current agent phase (analyzing, building, etc.)
            message: Human-readable status message
            progress: Optional progress dict with 'current' and 'total'

        Returns:
            AgentStreamEvent with status data
        """
        data: dict[str, Any] = {"phase": phase, "message": message}
        if progress:
            data["progress"] = progress
        return AgentStreamEvent(event=AgentStreamEventType.STATUS, data=data)

    def _emit_activity(self, message: str) -> AgentStreamEvent:
        """Create an activity log event for streaming.

        Args:
            message: Activity description

        Returns:
            AgentStreamEvent with activity data
        """
        from datetime import UTC, datetime

        return AgentStreamEvent(
            event=AgentStreamEventType.ACTIVITY,
            data={
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def _emit_question(
        self,
        question_id: str,
        message: str,
        options: list[dict[str, str]],
        allow_custom: bool = True,
    ) -> AgentStreamEvent:
        """Create a question event for streaming.

        Args:
            question_id: Unique identifier for this question
            message: The question text
            options: List of options with 'id', 'label', and optional 'description'
            allow_custom: Whether user can provide custom text response

        Returns:
            AgentStreamEvent with question data
        """
        return AgentStreamEvent(
            event=AgentStreamEventType.QUESTION,
            data={
                "id": question_id,
                "message": message,
                "options": options,
                "allowCustom": allow_custom,
            },
        )

    def _get_tool_descriptions(self) -> str:
        """Generate formatted tool descriptions for the system prompt."""
        descriptions = []
        for tool in self.tools.values():
            desc = f"### {tool.NAME}\n{tool.DESCRIPTION}\n"
            params = tool.PARAMETERS.get("properties", {})
            if params:
                desc += "Parameters:\n"
                for name, spec in params.items():
                    required = name in tool.PARAMETERS.get("required", [])
                    req_marker = " (required)" if required else ""
                    desc += f"  - {name}{req_marker}: {spec.get('description', spec.get('type', 'any'))}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def _get_template_summaries(self) -> str:
        """Generate formatted template summaries for the system prompt."""
        try:
            from shared.pipeline.templates import list_templates

            templates = list_templates()
            if not templates:
                return "No templates available."

            summaries = []
            for t in templates:
                suggested = ", ".join(t.suggested_for[:3])  # First 3 use cases
                summaries.append(f"- **{t.id}**: {t.name} - {t.description[:100]}... (for: {suggested})")

            return "\n".join(summaries)
        except Exception as e:
            logger.warning(f"Failed to load templates for system prompt: {e}")
            return "Templates unavailable - use list_templates tool to discover them."

    def _build_system_prompt(self) -> str:
        """Build the full system prompt with tool descriptions, templates, and source context."""
        base_prompt = self.SYSTEM_PROMPT.format(
            tool_descriptions=self._get_tool_descriptions(),
            available_templates=self._get_template_summaries(),
        )

        # Add source context if available
        if self.conversation.inline_source_config:
            source_type = self.conversation.inline_source_config.get("source_type", "unknown")
            source_config = self.conversation.inline_source_config.get("source_config", {})

            # Extract relevant source details for the LLM
            source_details = []
            if "path" in source_config:
                source_details.append(f"Path: {source_config['path']}")
            if "repo_url" in source_config:
                source_details.append(f"Repository: {source_config['repo_url']}")
            if "recursive" in source_config:
                source_details.append(f"Recursive: {source_config['recursive']}")

            source_info = "\n".join(source_details) if source_details else json.dumps(source_config)

            base_prompt += f"""

## Current Source Configuration

The user has already configured a data source for this pipeline:
- **Type**: {source_type}
- {source_info}
"""

        # Add state-aware instructions based on what's already been done
        has_analysis = bool(self.conversation.source_analysis)
        has_pipeline = bool(self.conversation.current_pipeline)

        if has_pipeline:
            # Pipeline already built - just help refine or apply it
            base_prompt += """
## Current State: Pipeline Ready

A pipeline has already been configured. You can:
- Use get_pipeline_state to review it
- Use build_pipeline to modify it if the user requests changes
- Use apply_pipeline to create the collection when the user is ready
- Do NOT re-analyze the source unless the user explicitly asks

Respond to the user's request without re-running analysis or rebuilding the pipeline unless asked."""

        elif has_analysis:
            # Source analyzed but no pipeline yet - build one
            base_prompt += """
## Current State: Source Analyzed

The source has been analyzed. Now you should:
- Review the analysis results already in the conversation
- Use build_pipeline to create a pipeline based on the analysis
- Do NOT call spawn_source_analyzer again - analysis is complete

Build the pipeline now based on the analysis."""

        elif self.conversation.inline_source_config:
            # Source configured but not analyzed - analyze it
            base_prompt += """
## Current State: Awaiting Analysis

Use spawn_source_analyzer to analyze this source and recommend an appropriate pipeline configuration. Do NOT ask the user about their source - you already have the information above."""

        return base_prompt

    async def handle_message(self, user_message: str) -> AgentResponse:
        """Process a user message and return the agent's response.

        This method implements the main conversation loop:
        1. Load messages from Redis (or recover from summary)
        2. Append user message
        3. Call LLM with tools
        4. Execute any tool calls
        5. Repeat until LLM gives final response
        6. Save messages to Redis
        7. Update conversation state in DB if needed
        8. Return response

        Args:
            user_message: The message from the user

        Returns:
            AgentResponse with content and state updates

        Raises:
            ConversationNotActiveError: If conversation is not active
            AgentError: For other agent-related errors
        """
        # Verify conversation is active
        if self.conversation.status != ConversationStatus.ACTIVE:
            raise ConversationNotActiveError(
                self.conversation.id,
                self.conversation.status.value,
            )

        # Check if paused
        if await self._is_paused():
            raise ConversationNotActiveError(
                self.conversation.id,
                "paused",
            )

        # Reset turn tracking
        self._pipeline_updated = False
        self._uncertainties_added = []
        self._tool_calls_made = []

        try:
            async with self.message_store.conversation_lock(self.conversation.id) as acquired:
                if not acquired:
                    raise AgentBusyError(self.conversation.id)

                # Load messages (or recover from summary)
                messages = await self._load_or_recover_messages()

                # Append user message
                user_msg = ConversationMessage.create("user", user_message)
                messages.append(user_msg)
                await self.message_store.append_message(self.conversation.id, user_msg)

                # Run the conversation loop
                final_response = await self._run_conversation_loop(messages)

                # Generate summary if needed
                await self._generate_summary_if_needed(messages)

                return AgentResponse(
                    content=final_response,
                    tool_calls_made=self._tool_calls_made,
                    pipeline_updated=self._pipeline_updated,
                    uncertainties_added=self._uncertainties_added,
                )

        except ConversationNotActiveError:
            raise
        except LLMAuthenticationError as e:
            logger.error(f"LLM authentication failed for conversation {self.conversation.id}: {e}")
            raise AgentError("LLM authentication failed. Check your API key in settings.") from e
        except LLMRateLimitError as e:
            logger.warning(f"LLM rate limit hit for conversation {self.conversation.id}: {e}")
            raise AgentError("Rate limit reached. Please wait and try again.") from e
        except LLMContextLengthError as e:
            logger.warning(f"LLM context too long for conversation {self.conversation.id}: {e}")
            raise AgentError("Message too long. Please shorten and try again.") from e
        except Exception as e:
            logger.exception(f"Error handling message for conversation {self.conversation.id}: {e}")
            # Provide more context based on exception type
            error_type = type(e).__name__
            if "redis" in error_type.lower() or "connection" in str(e).lower():
                raise AgentError(
                    f"Connection error ({error_type}): Unable to communicate with message store. "
                    "Please try again in a moment."
                ) from e
            if "database" in error_type.lower() or "asyncpg" in error_type.lower():
                raise AgentError(
                    f"Database error ({error_type}): Unable to persist conversation state. "
                    "Your message may not have been saved."
                ) from e
            if "timeout" in error_type.lower():
                raise AgentError(
                    f"Timeout error ({error_type}): The operation took too long. "
                    "Please try again with a simpler request."
                ) from e
            raise AgentError(f"An unexpected error occurred ({error_type}): {str(e)[:100]}") from e

    async def handle_message_streaming(self, user_message: str) -> AsyncGenerator[AgentStreamEvent, None]:
        """Process user message with streaming events.

        Yields SSE events as processing happens, providing real-time feedback
        to the client. Events include:
        - tool_call_start: Before each tool execution
        - tool_call_end: After each tool completes
        - subagent_start/end: For sub-agent lifecycle
        - uncertainty: When uncertainties are flagged
        - pipeline_update: When pipeline changes
        - content: Final response text
        - done: Completion with full response metadata
        - error: On failures

        Args:
            user_message: The message from the user

        Yields:
            AgentStreamEvent instances as processing happens

        Raises:
            ConversationNotActiveError: If conversation is not active
        """
        # Verify conversation is active
        if self.conversation.status != ConversationStatus.ACTIVE:
            yield AgentStreamEvent(
                event=AgentStreamEventType.ERROR,
                data={
                    "error": f"Conversation is not active (status: {self.conversation.status.value})",
                    "conversation_id": self.conversation.id,
                },
            )
            return

        # Check if paused
        if await self._is_paused():
            yield AgentStreamEvent(
                event=AgentStreamEventType.ERROR,
                data={
                    "error": "Agent is paused. Resume to continue.",
                    "type": "agent_paused",
                },
            )
            return

        # Reset turn tracking
        self._pipeline_updated = False
        self._uncertainties_added = []
        self._tool_calls_made = []

        try:
            async with self.message_store.conversation_lock(self.conversation.id) as acquired:
                if not acquired:
                    yield AgentStreamEvent(
                        event=AgentStreamEventType.ERROR,
                        data={
                            "error": f"Conversation {self.conversation.id} is busy processing another request",
                            "type": "conversation_busy",
                        },
                    )
                    return

                # Load messages (or recover from summary)
                messages = await self._load_or_recover_messages()

                # Append user message
                user_msg = ConversationMessage.create("user", user_message)
                messages.append(user_msg)
                await self.message_store.append_message(self.conversation.id, user_msg)

                # Run the conversation loop with streaming
                async for event in self._run_conversation_loop_streaming(messages):
                    yield event

                # Generate summary if needed
                await self._generate_summary_if_needed(messages)

        except ConversationNotActiveError as e:
            yield AgentStreamEvent(
                event=AgentStreamEventType.ERROR,
                data={"error": str(e), "type": "conversation_not_active"},
            )
        except AgentBusyError as e:
            yield AgentStreamEvent(
                event=AgentStreamEventType.ERROR,
                data={"error": str(e), "type": "conversation_busy"},
            )
        except Exception as e:
            logger.exception(f"Error handling message for conversation {self.conversation.id}: {e}")
            yield AgentStreamEvent(
                event=AgentStreamEventType.ERROR,
                data={"error": f"Failed to process message: {e}", "type": "agent_error"},
            )

    async def _run_conversation_loop_streaming(
        self, messages: list[ConversationMessage]
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """Run the LLM conversation loop with streaming events.

        Args:
            messages: Current message history

        Yields:
            AgentStreamEvent instances for each processing step
        """
        from shared.llm.types import LLMQualityTier

        # Create LLM provider
        provider = await self.llm_factory.create_provider_for_tier(
            self.conversation.user_id,
            LLMQualityTier.HIGH,
        )

        try:
            async with provider:
                # Emit initial status
                yield self._emit_status("analyzing", "Processing your request...")
                yield self._emit_activity("Starting conversation turn")

                for turn in range(self.MAX_TURNS):
                    logger.debug(f"Conversation turn {turn + 1}/{self.MAX_TURNS}")

                    # Build prompt from messages
                    prompt = self._build_prompt(messages)

                    # Call LLM
                    response = await self._llm_generate_with_retries(
                        provider,
                        prompt=prompt,
                        system_prompt=self._build_system_prompt(),
                        max_tokens=4096,
                        temperature=0.7,
                    )

                    response_content = response.content

                    # Parse for tool calls
                    tool_calls = self._parse_tool_calls(response_content)

                    if not tool_calls:
                        # No tool calls - this is the final response
                        clean_response = self.TOOL_CALL_PATTERN.sub("", response_content).strip()

                        # Save assistant response
                        assistant_msg = ConversationMessage.create("assistant", clean_response)
                        messages.append(assistant_msg)
                        await self.message_store.append_message(self.conversation.id, assistant_msg)

                        # Yield content event
                        yield AgentStreamEvent(
                            event=AgentStreamEventType.CONTENT,
                            data={"text": clean_response},
                        )

                        # Yield pipeline update if changed
                        if self._pipeline_updated:
                            yield AgentStreamEvent(
                                event=AgentStreamEventType.PIPELINE_UPDATE,
                                data={"pipeline": self.conversation.current_pipeline},
                            )

                        # Persist state changes BEFORE emitting DONE event
                        # so any persistence failures can be included in uncertainties
                        await self._persist_state_changes()

                        # Emit ready status and done event
                        yield self._emit_status("ready", "Pipeline configuration complete")
                        yield self._emit_activity("Agent finished processing")
                        yield AgentStreamEvent(
                            event=AgentStreamEventType.DONE,
                            data={
                                "pipeline_updated": self._pipeline_updated,
                                "uncertainties_added": self._uncertainties_added,
                                "tool_calls": [
                                    {"tool": tc["name"], "success": tc.get("success", True)}
                                    for tc in self._tool_calls_made
                                ],
                            },
                        )
                        return

                    # Execute tools with streaming events
                    tool_results = []
                    async for event_or_result in self._execute_tools_streaming(tool_calls):
                        if isinstance(event_or_result, AgentStreamEvent):
                            yield event_or_result
                        else:
                            tool_results.append(event_or_result)

                    # Extract text response (content outside tool blocks)
                    text_response = self.TOOL_CALL_PATTERN.sub("", response_content).strip()

                    # Save assistant response with tool calls
                    assistant_msg = ConversationMessage.create(
                        "assistant",
                        text_response or "[Tool calls executed]",
                        metadata={
                            "tool_calls": [
                                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
                            ]
                        },
                    )
                    messages.append(assistant_msg)
                    await self.message_store.append_message(self.conversation.id, assistant_msg)

                    # Save tool results
                    for result in tool_results:
                        result_content = json.dumps(result.data) if result.success else f"Error: {result.error}"
                        tool_msg = ConversationMessage.create(
                            "tool",
                            result_content,
                            metadata={"tool_call_id": result.tool_call_id, "tool_name": result.name},
                        )
                        messages.append(tool_msg)
                        await self.message_store.append_message(self.conversation.id, tool_msg)

                # Max turns reached
                logger.warning(f"Conversation {self.conversation.id} reached max turns")

                # Persist state changes BEFORE emitting DONE event
                await self._persist_state_changes()

                yield AgentStreamEvent(
                    event=AgentStreamEventType.CONTENT,
                    data={
                        "text": "I've reached the maximum number of steps for this turn. Please continue the conversation."
                    },
                )
                yield AgentStreamEvent(
                    event=AgentStreamEventType.DONE,
                    data={
                        "pipeline_updated": self._pipeline_updated,
                        "uncertainties_added": self._uncertainties_added,
                        "tool_calls": [
                            {"tool": tc["name"], "success": tc.get("success", True)} for tc in self._tool_calls_made
                        ],
                        "max_turns_reached": True,
                    },
                )

        finally:
            # Note: State persistence is now handled before DONE events are emitted
            # to ensure persistence failures are included in the streamed response.
            # This finally block is kept for potential future cleanup needs.
            pass

    async def _execute_tools_streaming(
        self, tool_calls: list[ToolCall]
    ) -> AsyncGenerator[AgentStreamEvent | ToolResult, None]:
        """Execute tools with streaming events.

        Yields AgentStreamEvent for start/end of each tool,
        and ToolResult for collection.

        Args:
            tool_calls: List of tool calls to execute

        Yields:
            AgentStreamEvent or ToolResult
        """
        for call in tool_calls:
            tool = self.tools.get(call.name)

            # Emit status for tool execution
            yield self._emit_status(
                "building" if call.name == "build_pipeline" else "analyzing",
                f"Executing {call.name}...",
            )
            yield self._emit_activity(f"Running tool: {call.name}")

            # Yield start event
            yield AgentStreamEvent(
                event=AgentStreamEventType.TOOL_CALL_START,
                data={"tool": call.name, "arguments": call.arguments, "call_id": call.id},
            )

            if not tool:
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=False,
                    error=f"Unknown tool: {call.name}",
                )
                yield AgentStreamEvent(
                    event=AgentStreamEventType.TOOL_CALL_END,
                    data={
                        "tool": call.name,
                        "success": False,
                        "error": f"Unknown tool: {call.name}",
                        "call_id": call.id,
                    },
                )
                yield result
                continue

            try:
                # Check if this is a spawn tool (subagent)
                is_spawn_tool = call.name.startswith("spawn_")
                if is_spawn_tool:
                    yield self._emit_status("analyzing", f"Spawning {call.name.replace('spawn_', '')} agent...")
                    yield AgentStreamEvent(
                        event=AgentStreamEventType.SUBAGENT_START,
                        data={
                            "name": call.name.replace("spawn_", ""),
                            "task": call.arguments.get("user_intent", "Analyzing..."),
                            "call_id": call.id,
                        },
                    )

                # Execute the tool
                data = await tool.execute(**call.arguments)

                # Track the tool call with success status
                self._tool_calls_made.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": data,
                        "success": True,
                    }
                )

                # Check if pipeline was updated
                if call.name == "build_pipeline" and isinstance(data, dict) and data.get("success"):
                    self._pipeline_updated = True

                # Check for spawn tool results (subagents)
                if is_spawn_tool and isinstance(data, dict):
                    subagent_success = data.get("success", True)
                    subagent_name = call.name.replace("spawn_", "")
                    yield AgentStreamEvent(
                        event=AgentStreamEventType.SUBAGENT_END,
                        data={
                            "name": subagent_name,
                            "success": subagent_success,
                            "result": data.get("summary", ""),
                            "error": data.get("error") if not subagent_success else None,
                        },
                    )

                    # Update status when sub-agent fails so UI reflects the error
                    if not subagent_success:
                        error_msg = data.get("error") or data.get("summary", "Unknown error")
                        # Truncate error message for status display
                        error_preview = error_msg[:80] + "..." if len(str(error_msg)) > 80 else error_msg
                        yield self._emit_status(
                            "analyzing",
                            f"Sub-agent {subagent_name} failed: {error_preview}",
                        )

                # Check if tool added uncertainties
                before_uncertainties = len(self._uncertainties_added)
                await self._check_for_uncertainties_streaming(call.name, data)

                # Yield uncertainties added by this tool call
                for uncertainty in self._uncertainties_added[before_uncertainties:]:
                    yield AgentStreamEvent(event=AgentStreamEventType.UNCERTAINTY, data=uncertainty)

                # Yield end event
                yield AgentStreamEvent(
                    event=AgentStreamEventType.TOOL_CALL_END,
                    data={
                        "tool": call.name,
                        "success": True,
                        "call_id": call.id,
                        # Include a summary of the result, not the full data
                        "result_summary": self._summarize_tool_result(data),
                    },
                )

                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=True,
                    data=data,
                )
                yield result
                logger.info(f"Tool {call.name} executed successfully")

            except ToolExecutionError as e:
                logger.warning(f"Tool {call.name} failed: {e}")
                # Track failed tool call for metrics
                self._tool_calls_made.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "error": str(e),
                        "success": False,
                    }
                )
                yield AgentStreamEvent(
                    event=AgentStreamEventType.TOOL_CALL_END,
                    data={
                        "tool": call.name,
                        "success": False,
                        "error": str(e),
                        "call_id": call.id,
                    },
                )
                yield ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=False,
                    error=str(e),
                )
            except Exception as e:
                logger.exception(f"Unexpected error in tool {call.name}: {e}")
                # Track failed tool call for metrics
                self._tool_calls_made.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "error": f"Internal error: {e}",
                        "success": False,
                    }
                )
                yield AgentStreamEvent(
                    event=AgentStreamEventType.TOOL_CALL_END,
                    data={
                        "tool": call.name,
                        "success": False,
                        "error": f"Internal error: {e}",
                        "call_id": call.id,
                    },
                )
                yield ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=False,
                    error=f"Internal error: {e}",
                )

    async def _check_for_uncertainties_streaming(self, tool_name: str, result: Any) -> None:
        """Check tool results for uncertainties (streaming version).

        Same as _check_for_uncertainties but used in streaming context.
        """
        await self._check_for_uncertainties(tool_name, result)

    def _summarize_tool_result(self, data: Any) -> str:
        """Create a brief summary of tool result for streaming.

        Args:
            data: Tool result data

        Returns:
            Brief summary string
        """
        if not isinstance(data, dict):
            return str(data)[:100]

        if "plugins" in data:
            count = len(data.get("plugins", []))
            return f"Found {count} plugins"
        if "templates" in data:
            count = len(data.get("templates", []))
            return f"Found {count} templates"
        if "pipeline" in data:
            return "Pipeline configuration retrieved"
        if "success" in data:
            return "Success" if data["success"] else f"Failed: {data.get('error', 'Unknown error')}"
        if "analysis" in data:
            return "Source analysis complete"
        if "validation" in data:
            return "Validation complete"

        return "Completed"

    async def _load_or_recover_messages(self) -> list[ConversationMessage]:
        """Load messages from Redis or recover from summary.

        Returns:
            List of conversation messages
        """
        # Try to load from Redis
        if await self.message_store.has_messages(self.conversation.id):
            messages = await self.message_store.get_messages(self.conversation.id)
            logger.debug(f"Loaded {len(messages)} messages from Redis")
            return messages

        # No messages in Redis - check if we have a summary
        if self.conversation.summary:
            logger.info(f"Recovering conversation {self.conversation.id} from summary")
            # Create a recovery message
            recovery_msg = ConversationMessage.create(
                "assistant",
                f"[Previous conversation summary: {self.conversation.summary}]\n\n"
                "I've recovered the context from our previous conversation. How can I help you continue?",
            )
            messages = [recovery_msg]
            await self.message_store.set_messages(self.conversation.id, messages)
            return messages

        # New conversation
        return []

    async def _run_conversation_loop(self, messages: list[ConversationMessage]) -> str:
        """Run the LLM conversation loop until completion.

        Args:
            messages: Current message history

        Returns:
            Final response content from the LLM
        """
        from shared.llm.types import LLMQualityTier

        # Create LLM provider
        provider = await self.llm_factory.create_provider_for_tier(
            self.conversation.user_id,
            LLMQualityTier.HIGH,
        )

        try:
            async with provider:
                for turn in range(self.MAX_TURNS):
                    logger.debug(f"Conversation turn {turn + 1}/{self.MAX_TURNS}")

                    # Build prompt from messages
                    prompt = self._build_prompt(messages)

                    # Call LLM
                    response = await self._llm_generate_with_retries(
                        provider,
                        prompt=prompt,
                        system_prompt=self._build_system_prompt(),
                        max_tokens=4096,
                        temperature=0.7,
                    )

                    response_content = response.content

                    # Parse for tool calls
                    tool_calls = self._parse_tool_calls(response_content)

                    if not tool_calls:
                        # No tool calls - this is the final response
                        # Clean up the response (remove any tool blocks if present)
                        clean_response = self.TOOL_CALL_PATTERN.sub("", response_content).strip()

                        # Save assistant response
                        assistant_msg = ConversationMessage.create("assistant", clean_response)
                        messages.append(assistant_msg)
                        await self.message_store.append_message(self.conversation.id, assistant_msg)

                        return clean_response

                    # Execute tools
                    tool_results = await self._execute_tools(tool_calls)

                    # Extract text response (content outside tool blocks)
                    text_response = self.TOOL_CALL_PATTERN.sub("", response_content).strip()

                    # Save assistant response with tool calls
                    assistant_msg = ConversationMessage.create(
                        "assistant",
                        text_response or "[Tool calls executed]",
                        metadata={
                            "tool_calls": [
                                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
                            ]
                        },
                    )
                    messages.append(assistant_msg)
                    await self.message_store.append_message(self.conversation.id, assistant_msg)

                    # Save tool results
                    for result in tool_results:
                        result_content = json.dumps(result.data) if result.success else f"Error: {result.error}"
                        tool_msg = ConversationMessage.create(
                            "tool",
                            result_content,
                            metadata={"tool_call_id": result.tool_call_id, "tool_name": result.name},
                        )
                        messages.append(tool_msg)
                        await self.message_store.append_message(self.conversation.id, tool_msg)

                # Max turns reached
                logger.warning(f"Conversation {self.conversation.id} reached max turns")
                return "I've reached the maximum number of steps for this turn. Please continue the conversation."

        finally:
            # Persist any state changes
            await self._persist_state_changes()

        # All code paths in the try block return, but the type checker
        # cannot prove this, so we need an explicit return here.
        raise AssertionError("Unreachable: all conversation loop paths return")

    def _build_prompt(self, messages: list[ConversationMessage]) -> str:
        """Build a prompt string from message history.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        for msg in messages:
            if msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                tool_name = msg.metadata.get("tool_name", "unknown") if msg.metadata else "unknown"
                prompt_parts.append(f"Tool ({tool_name}): {msg.content}")
            elif msg.role == "subagent":
                subagent_type = msg.metadata.get("subagent_type", "unknown") if msg.metadata else "unknown"
                prompt_parts.append(f"SubAgent ({subagent_type}): {msg.content}")

        return "\n\n".join(prompt_parts)

    def _parse_tool_calls(self, response: str) -> list[ToolCall]:
        """Parse tool calls from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of extracted tool calls
        """
        tool_calls = []

        for i, match in enumerate(self.TOOL_CALL_PATTERN.finditer(response)):
            try:
                content = match.group(1).strip()
                data = json.loads(content)

                tool_call = ToolCall(
                    id=f"call_{i}",
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                )

                if tool_call.name:
                    tool_calls.append(tool_call)
                    logger.debug(f"Parsed tool call: {tool_call.name}")

            except json.JSONDecodeError as e:
                # Log more context for debugging malformed tool calls from LLM
                raw_content = match.group(1).strip()[:200]  # Truncate for logging
                logger.warning(f"Failed to parse tool call at index {i}: {e}. Raw content (truncated): {raw_content!r}")
                continue

        return tool_calls

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute a list of tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool results
        """
        results = []

        for call in tool_calls:
            tool = self.tools.get(call.name)

            if not tool:
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        success=False,
                        error=f"Unknown tool: {call.name}",
                    )
                )
                continue

            try:
                # Execute the tool
                data = await tool.execute(**call.arguments)

                # Track the tool call with success status
                self._tool_calls_made.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": data,
                        "success": True,
                    }
                )

                # Check if pipeline was updated
                if call.name == "build_pipeline" and isinstance(data, dict) and data.get("success"):
                    self._pipeline_updated = True

                # Check if tool added uncertainties
                await self._check_for_uncertainties(call.name, data)

                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        success=True,
                        data=data,
                    )
                )
                logger.info(f"Tool {call.name} executed successfully")

            except ToolExecutionError as e:
                logger.warning(f"Tool {call.name} failed: {e}")
                # Track failed tool call for metrics
                self._tool_calls_made.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "error": str(e),
                        "success": False,
                    }
                )
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        success=False,
                        error=str(e),
                    )
                )
            except Exception as e:
                logger.exception(f"Unexpected error in tool {call.name}: {e}")
                # Track failed tool call for metrics
                self._tool_calls_made.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "error": f"Internal error: {e}",
                        "success": False,
                    }
                )
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        success=False,
                        error=f"Internal error: {e}",
                    )
                )

        return results

    async def _check_for_uncertainties(self, tool_name: str, result: Any) -> None:
        """Check tool results for uncertainties and persist them.

        Uses an independent database session to avoid cancellation when
        the SSE connection is closed (e.g., after long-running spawn tools).

        Args:
            tool_name: Name of the tool that was called
            result: Result from the tool execution
        """
        if not isinstance(result, dict):
            return

        # Check for uncertainties from tool results
        uncertainties = result.get("uncertainties", [])
        if not uncertainties:
            return

        logger.debug(f"Tool {tool_name} returned {len(uncertainties)} uncertainties")

        # If uncertainties already include IDs, assume they were persisted by the tool.
        # In that case, just surface them in the response without re-inserting.
        if all(isinstance(u, dict) and u.get("id") for u in uncertainties):
            for u in uncertainties:
                self._uncertainties_added.append(
                    {
                        "id": u.get("id"),
                        "severity": u.get("severity", "info"),
                        "message": u.get("message", "Unknown uncertainty"),
                        "resolved": bool(u.get("resolved", False)),
                        "context": u.get("context"),
                    }
                )
            return

        # Use independent session to avoid SSE connection lifecycle issues
        from asyncpg.exceptions import InterfaceError as AsyncpgInterfaceError

        from shared.database.postgres_database import pg_connection_manager

        try:
            async with pg_connection_manager.get_session() as independent_session:
                repo = AgentConversationRepository(independent_session)
                for u in uncertainties:
                    if isinstance(u, dict):
                        severity_str = u.get("severity", "info")
                        try:
                            severity = UncertaintySeverity(severity_str)
                        except ValueError:
                            severity = UncertaintySeverity.INFO

                        uncertainty = await repo.add_uncertainty(
                            conversation_id=self.conversation.id,
                            user_id=self.conversation.user_id,
                            severity=severity,
                            message=u.get("message", "Unknown uncertainty"),
                            context=u.get("context"),
                        )
                        self._uncertainties_added.append(
                            {
                                "id": uncertainty.id,
                                "severity": severity.value,
                                "message": uncertainty.message,
                                "resolved": False,
                                "context": uncertainty.context,
                            }
                        )
        except AsyncpgInterfaceError as e:
            # Connection was closed during long-running operation
            # Include uncertainty data in warning so it can be reviewed in logs
            uncertainty_summaries = [
                f"[{u.get('severity', 'info')}] {u.get('message', 'Unknown')[:80]}"
                for u in uncertainties
                if isinstance(u, dict)
            ]
            logger.warning(
                f"DB connection closed during uncertainty persistence from {tool_name}: {e}. "
                f"Unpersisted uncertainties ({len(uncertainties)}): {uncertainty_summaries}"
            )
            # Notify user that uncertainties weren't saved
            self._uncertainties_added.append(
                {
                    "id": "connection_warning",
                    "severity": "notable",
                    "message": (
                        f"Connection lost: {len(uncertainties)} analysis result(s) from "
                        f"{tool_name} could not be saved. They may need to be re-generated."
                    ),
                    "resolved": False,
                    "context": {"tool": tool_name, "count": len(uncertainties)},
                }
            )
        except Exception as e:
            logger.error(f"Failed to persist uncertainties from {tool_name}: {e}", exc_info=True)
            # Notify user that results weren't saved
            self._uncertainties_added.append(
                {
                    "id": "persistence_warning",
                    "severity": "notable",
                    "message": "Warning: Some analysis results could not be saved.",
                    "resolved": False,
                }
            )
            # Don't re-raise - we don't want to crash the conversation loop

    async def _persist_state_changes(self) -> None:
        """Persist any state changes to the database.

        Uses an independent database session to avoid cancellation when
        the SSE connection is closed.
        """
        # Only persist if there are changes
        if not self._pipeline_updated or not self.conversation.current_pipeline:
            return

        # Use independent session to avoid SSE connection lifecycle issues
        from asyncpg.exceptions import InterfaceError as AsyncpgInterfaceError

        from shared.database.postgres_database import pg_connection_manager

        try:
            async with pg_connection_manager.get_session() as independent_session:
                repo = AgentConversationRepository(independent_session)
                await repo.update_pipeline(
                    conversation_id=self.conversation.id,
                    user_id=self.conversation.user_id,
                    pipeline_config=self.conversation.current_pipeline,
                )
                # Commit happens automatically via the context manager
        except AsyncpgInterfaceError as e:
            # Connection was closed during long-running operation (e.g., SSE timeout)
            # This is expected when sub-agents take a long time and the connection
            # pool times out idle connections. Log at warning level, not error.
            logger.warning(
                f"DB connection closed during state persistence for conversation "
                f"{self.conversation.id}: {e}. Pipeline state may need to be rebuilt."
            )
            # Notify user that pipeline changes may not have been saved
            self._uncertainties_added.append(
                {
                    "id": "state_persistence_warning",
                    "severity": "notable",
                    "message": (
                        "Connection lost: Pipeline changes may not have been saved. "
                        "If you refresh and changes are missing, please re-apply them."
                    ),
                    "resolved": False,
                    "context": {"conversation_id": self.conversation.id},
                }
            )
        except Exception as e:
            logger.error(f"Failed to persist state changes: {e}", exc_info=True)
            # Don't re-raise - we don't want to crash the conversation loop

    async def _generate_summary_if_needed(self, messages: list[ConversationMessage]) -> None:
        """Generate a conversation summary if the message count is high.

        Uses an independent database session to avoid cancellation when
        the SSE connection is closed.

        Args:
            messages: Current message list
        """
        # Generate summary every 20 messages
        if len(messages) >= 20 and len(messages) % 10 == 0:
            try:
                from shared.database.postgres_database import pg_connection_manager
                from shared.llm.factory import LLMServiceFactory
                from shared.llm.types import LLMQualityTier

                # Create LLM factory with independent session
                async with pg_connection_manager.get_session() as independent_session:
                    llm_factory = LLMServiceFactory(independent_session)
                    provider = await llm_factory.create_provider_for_tier(
                        self.conversation.user_id,
                        LLMQualityTier.LOW,  # Use low tier for summaries
                    )

                    async with provider:
                        summary_prompt = (
                            "Summarize this conversation in 2-3 sentences, focusing on:\n"
                            "1. What the user wants to accomplish\n"
                            "2. The current pipeline configuration\n"
                            "3. Any outstanding issues or next steps\n\n"
                            f"Conversation:\n{self._build_prompt(messages[-20:])}"
                        )

                        response = await self._llm_generate_with_retries(
                            provider,
                            prompt=summary_prompt,
                            max_tokens=256,
                            temperature=0.3,
                        )

                        repo = AgentConversationRepository(independent_session)
                        await repo.update_summary(
                            conversation_id=self.conversation.id,
                            user_id=self.conversation.user_id,
                            summary=response.content,
                        )
                        logger.info(f"Generated summary for conversation {self.conversation.id}")

            except Exception as e:
                from asyncpg.exceptions import InterfaceError as AsyncpgInterfaceError

                if isinstance(e, AsyncpgInterfaceError):
                    # Connection closed during long-running operation - expected, don't escalate
                    logger.warning(
                        f"DB connection closed during summary generation for conversation "
                        f"{self.conversation.id}: {e}"
                    )
                    return

                logger.error(f"Failed to generate summary for conversation {self.conversation.id}: {e}", exc_info=True)
                # Best-effort: try to add uncertainty with independent session
                try:
                    from shared.database.postgres_database import pg_connection_manager

                    async with pg_connection_manager.get_session() as err_session:
                        repo = AgentConversationRepository(err_session)
                        await repo.add_uncertainty(
                            conversation_id=self.conversation.id,
                            user_id=self.conversation.user_id,
                            severity=UncertaintySeverity.INFO,
                            message="Summary generation failed. Context recovery may be incomplete.",
                            context={"error": str(e)},
                        )
                except Exception as inner_e:
                    logger.warning(f"Failed to add uncertainty after summary failure: {inner_e}")

    async def add_subagent_result(
        self,
        subagent_type: str,
        result: dict[str, Any],
        summary: str,
    ) -> None:
        """Add a sub-agent result to the conversation.

        Called by spawn tools after sub-agent execution completes.
        Uses an independent database session to avoid cancellation when
        the SSE connection is closed.

        Args:
            subagent_type: Type of sub-agent (source_analyzer, pipeline_validator)
            result: Structured result from the sub-agent
            summary: Human-readable summary
        """
        # Save as subagent message
        msg = ConversationMessage.create(
            "subagent",
            summary,
            metadata={
                "subagent_type": subagent_type,
                "result": result,
            },
        )
        await self.message_store.append_message(self.conversation.id, msg)

        # Update source analysis if from source analyzer
        if subagent_type == "source_analyzer" and result:
            from asyncpg.exceptions import InterfaceError as AsyncpgInterfaceError

            from shared.database.postgres_database import pg_connection_manager

            try:
                async with pg_connection_manager.get_session() as independent_session:
                    repo = AgentConversationRepository(independent_session)
                    await repo.update_source_analysis(
                        conversation_id=self.conversation.id,
                        user_id=self.conversation.user_id,
                        source_analysis=result,
                    )
                self.conversation.source_analysis = result
            except AsyncpgInterfaceError as e:
                # Connection closed during long-running sub-agent operation
                logger.warning(
                    f"DB connection closed during source analysis update: {e}. "
                    "Local state updated but may need to re-analyze."
                )
                self.conversation.source_analysis = result
            except Exception as e:
                logger.error(f"Failed to update source analysis: {e}", exc_info=True)
                # Still update local state even if DB persist fails
                self.conversation.source_analysis = result


__all__ = ["AgentOrchestrator", "AgentResponse", "ORCHESTRATOR_TOOL_CLASSES"]

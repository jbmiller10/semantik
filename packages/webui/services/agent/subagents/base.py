"""Base classes for sub-agents.

Sub-agents are specialized agents with their own context windows that
handle complex, multi-step tasks. Each sub-agent has:
- A focused system prompt
- A specialized toolset
- An independent LLM conversation loop
- Structured result output
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from shared.llm.base import BaseLLMService
    from webui.services.agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Uncertainty:
    """An uncertainty flagged by an agent.

    Uncertainties represent issues, concerns, or unknowns that the agent
    has identified during its analysis. They are categorized by severity
    to help the user understand what needs attention.

    Attributes:
        severity: How serious this uncertainty is
            - "blocking": Must be resolved before proceeding
            - "notable": Should be surfaced but can proceed
            - "info": Informational, for transparency
        message: Human-readable description of the uncertainty
        context: Additional data (file references, error details, etc.)
    """

    severity: Literal["blocking", "notable", "info"]
    message: str
    context: dict[str, Any] | None = None


@dataclass
class SubAgentResult:
    """Structured result from a sub-agent execution.

    Sub-agents return this structured result to the orchestrator,
    which can then decide how to present it to the user.

    Attributes:
        success: Whether the sub-agent completed successfully
        data: Structured data produced by the sub-agent
        uncertainties: Issues or concerns flagged during execution
        summary: Human-readable summary for the orchestrator to relay
    """

    success: bool
    data: dict[str, Any]
    uncertainties: list[Uncertainty] = field(default_factory=list)
    summary: str = ""


@dataclass
class Message:
    """A message in the sub-agent conversation.

    Attributes:
        role: Who sent this message (user, assistant, tool)
        content: The message content
        tool_calls: Tool calls requested by the assistant
        tool_results: Results from tool execution
    """

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None


@dataclass
class ToolCall:
    """A tool call requested by the LLM.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to call
        arguments: Arguments to pass to the tool
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool.

    Attributes:
        tool_call_id: ID of the tool call this is responding to
        name: Name of the tool that was called
        success: Whether the tool executed successfully
        data: Result data from the tool
        error: Error message if the tool failed
    """

    tool_call_id: str
    name: str
    success: bool
    data: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate consistency of success and error fields."""
        if self.success and self.error is not None:
            raise ValueError("Successful tool result cannot have error")


class SubAgent(ABC):
    """Base class for sub-agents.

    Sub-agents are specialized agents that handle complex, multi-step tasks
    with their own context window. They are spawned by the orchestrator
    and return structured results.

    To create a sub-agent, subclass this and implement:
    - AGENT_ID: Unique identifier for this sub-agent type
    - SYSTEM_PROMPT: The system prompt that defines the agent's behavior
    - TOOLS: List of tool classes available to this agent
    - _build_initial_message(): Create the initial user message with context
    - _extract_result(): Extract structured result from the final response

    Class Attributes:
        AGENT_ID: Unique identifier for this sub-agent type
        SYSTEM_PROMPT: System prompt defining the agent's behavior
        TOOLS: List of tool classes available to this agent
        MAX_TURNS: Maximum number of conversation turns before stopping
        TIMEOUT_SECONDS: Maximum execution time before timeout
    """

    AGENT_ID: ClassVar[str]
    SYSTEM_PROMPT: ClassVar[str]
    TOOLS: ClassVar[list[type[BaseTool]]]
    MAX_TURNS: ClassVar[int] = 20
    TIMEOUT_SECONDS: ClassVar[int] = 300  # 5 minutes

    # Pattern for parsing tool calls from LLM responses
    TOOL_CALL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"```tool\s*\n?(.*?)\n?```",
        re.DOTALL,
    )

    def __init__(
        self,
        llm_provider: BaseLLMService,
        context: dict[str, Any],
    ):
        """Initialize the sub-agent.

        Args:
            llm_provider: LLM provider for generating responses
            context: Context data passed from the orchestrator
        """
        self.llm = llm_provider
        self.context = context
        self.messages: list[Message] = []
        self.tools: dict[str, BaseTool] = {}

        # Initialize tool instances
        for tool_class in self.TOOLS:
            tool = tool_class(context)
            self.tools[tool.NAME] = tool

    async def run(self) -> SubAgentResult:
        """Execute the sub-agent loop until complete or max_turns.

        Returns:
            SubAgentResult with success status, data, and any uncertainties

        The agent will:
        1. Build the initial message with context
        2. Generate LLM responses
        3. Execute tool calls as needed
        4. Continue until the LLM indicates completion or max_turns is reached
        """
        try:
            return await asyncio.wait_for(
                self._run_loop(),
                timeout=self.TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(f"Sub-agent {self.AGENT_ID} timed out after {self.TIMEOUT_SECONDS}s")
            return SubAgentResult(
                success=False,
                data=self._get_partial_result(),
                summary=f"Timed out after {self.TIMEOUT_SECONDS}s",
            )
        except Exception as e:
            logger.exception(f"Sub-agent {self.AGENT_ID} failed with error: {e}")
            return SubAgentResult(
                success=False,
                data=self._get_partial_result(),
                summary=f"Failed with error: {e}",
            )

    async def _run_loop(self) -> SubAgentResult:
        """Internal agent loop implementation."""
        # Build initial message
        initial_message = self._build_initial_message()
        self.messages.append(initial_message)

        for turn in range(self.MAX_TURNS):
            logger.debug(f"Sub-agent {self.AGENT_ID} turn {turn + 1}/{self.MAX_TURNS}")

            # Generate response from LLM
            response = await self._generate_response()
            self.messages.append(response)

            # Check if agent is done
            if not response.tool_calls:
                # No tool calls means the agent is done
                return self._extract_result(response)

            # Execute tools and collect results
            tool_results = await self._execute_tools(response.tool_calls)
            self.messages.append(Message(role="tool", content="", tool_results=tool_results))

        # Reached max turns without completing
        logger.warning(f"Sub-agent {self.AGENT_ID} reached max turns ({self.MAX_TURNS})")
        return SubAgentResult(
            success=False,
            data=self._get_partial_result(),
            summary=f"Reached max turns ({self.MAX_TURNS}) without completing",
        )

    async def _generate_response(self) -> Message:
        """Generate a response from the LLM.

        Builds a prompt from the full message history, calls the LLM,
        and parses tool calls from the response.
        """
        # Build full prompt from message history
        prompt = self._build_prompt()

        # Build system prompt with tool descriptions
        system_prompt = self._build_full_system_prompt()

        # Generate response
        # Use longer timeout for sub-agents (default 60s is too short)
        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=4096,
            temperature=0.7,
            timeout=180,
        )

        response_content = response.content

        # Parse response for tool calls
        tool_calls = self._parse_tool_calls(response_content)

        # Clean response content (remove tool call blocks)
        clean_content = self.TOOL_CALL_PATTERN.sub("", response_content).strip()

        return Message(role="assistant", content=clean_content, tool_calls=tool_calls or None)

    def _build_prompt(self) -> str:
        """Build a prompt string from message history."""
        prompt_parts = []

        for msg in self.messages:
            if msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool" and msg.tool_results:
                for result in msg.tool_results:
                    status = "Success" if result.success else "Error"
                    content = str(result.data) if result.success else result.error
                    prompt_parts.append(f"Tool ({result.name}) [{status}]: {content}")

        return "\n\n".join(prompt_parts)

    def _build_full_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tool_descriptions = self._get_tool_descriptions()
        return f"""{self.SYSTEM_PROMPT}

## Tool Usage

When you need to take an action, output a tool call in this format:
```tool
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
```

You can call multiple tools by including multiple tool blocks.

When you're done responding (no more tools to call), just write your response normally.

## Available Tools

{tool_descriptions}"""

    def _get_tool_descriptions(self) -> str:
        """Generate formatted tool descriptions."""
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

    def _parse_tool_calls(self, response: str) -> list[ToolCall]:
        """Parse tool calls from LLM response."""
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
                    logger.debug(f"Sub-agent parsed tool call: {tool_call.name}")

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call: {e}")
                continue

        return tool_calls

    def _build_llm_messages(self) -> list[dict[str, Any]]:
        """Convert internal messages to LLM format."""
        llm_messages = []
        for msg in self.messages:
            if msg.role == "tool" and msg.tool_results:
                # Format tool results for LLM
                for result in msg.tool_results:
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.tool_call_id,
                            "content": str(result.data) if result.success else result.error,
                        }
                    )
            else:
                llm_messages.append({"role": msg.role, "content": msg.content})
        return llm_messages

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all available tools."""
        return [tool.get_schema() for tool in self.tools.values()]

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
                data = await tool.execute(**call.arguments)
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        success=True,
                        data=data,
                    )
                )
            except Exception as e:
                logger.error(f"Tool {call.name} failed: {e}", exc_info=True)
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        success=False,
                        error=str(e),
                    )
                )

        return results

    @abstractmethod
    def _build_initial_message(self) -> Message:
        """Build the initial user message with task context.

        This should create a message that provides the sub-agent with
        all the context it needs to complete its task.

        Returns:
            Initial user message
        """
        ...

    @abstractmethod
    def _extract_result(self, response: Message) -> SubAgentResult:
        """Extract structured result from the final response.

        This should parse the agent's final response and create a
        structured SubAgentResult.

        Args:
            response: The agent's final response message

        Returns:
            Structured sub-agent result
        """
        ...

    def _get_partial_result(self) -> dict[str, Any]:
        """Get partial result for timeout/failure cases.

        Subclasses can override this to provide any partial data
        that was gathered before the failure.

        Returns:
            Partial result data (empty dict by default)
        """
        return {}

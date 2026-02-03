"""Callbacks for assisted flow SDK client.

This module provides callback handlers for the Claude SDK client,
including the can_use_tool callback for handling AskUserQuestion
tool calls interactively.

The interaction flow for AskUserQuestion:
1. SDK streams ToolUseBlock with tool_use_id and questions in input
2. SSE generator emits 'question' event with questions and question_id
3. Frontend displays question UI
4. User submits answer via POST /{session_id}/answer endpoint
5. QuestionManager resolves the pending future
6. can_use_tool callback unblocks and returns with answers
7. SDK continues processing

The question_id is derived from a hash of the questions content, allowing
the SSE generator and callback to independently generate the same ID.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_agent_sdk.types import (
        PermissionResultAllow,
        PermissionResultDeny,
        ToolPermissionContext,
    )

logger = logging.getLogger(__name__)


def compute_question_id(questions: list[dict[str, Any]]) -> str:
    """Compute a deterministic question_id from questions content.

    This allows both the SSE generator and callback to independently
    compute the same ID for the same questions.
    """
    # Sort keys for deterministic serialization
    content = json.dumps(questions, sort_keys=True)
    hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"q_{hash_digest}"


@dataclass
class PendingQuestion:
    """A pending question waiting for user answer."""

    question_id: str
    questions: list[dict[str, Any]]
    future: asyncio.Future[dict[str, str]] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    created_at: float = field(default_factory=time.time)


class QuestionManager:
    """Manages pending questions for assisted flow sessions.

    Thread-safe management of questions awaiting user answers.
    Questions are keyed by question_id for lookup from the answer endpoint.
    """

    def __init__(self) -> None:
        self._pending: dict[str, PendingQuestion] = {}
        self._lock = asyncio.Lock()

    async def create_question(
        self,
        question_id: str,
        questions: list[dict[str, Any]],
    ) -> PendingQuestion:
        """Create a new pending question.

        Args:
            question_id: Unique ID for this question
            questions: The questions array from the tool input

        Returns:
            PendingQuestion with awaitable future
        """
        question = PendingQuestion(
            question_id=question_id,
            questions=questions,
        )

        async with self._lock:
            self._pending[question_id] = question

        logger.info(f"Created pending question {question_id}")
        return question

    async def submit_answer(
        self,
        question_id: str,
        answers: dict[str, str],
    ) -> bool:
        """Submit an answer to a pending question.

        Args:
            question_id: The question ID to answer
            answers: Dict mapping question text to selected answer label

        Returns:
            True if answer was submitted, False if question not found
        """
        async with self._lock:
            question = self._pending.get(question_id)

        if not question:
            logger.warning(f"Question {question_id} not found")
            return False

        if question.future.done():
            logger.warning(f"Question {question_id} already answered")
            return False

        question.future.set_result(answers)
        logger.info(f"Submitted answer for question {question_id}")
        return True

    async def get_pending(self, question_id: str) -> PendingQuestion | None:
        """Get a pending question by ID."""
        async with self._lock:
            return self._pending.get(question_id)

    async def remove_question(self, question_id: str) -> PendingQuestion | None:
        """Remove and return a question by ID."""
        async with self._lock:
            return self._pending.pop(question_id, None)

    async def cancel_question(self, question_id: str) -> bool:
        """Cancel a pending question."""
        question = await self.remove_question(question_id)

        if not question:
            return False

        if not question.future.done():
            question.future.cancel()

        logger.info(f"Cancelled question {question_id}")
        return True

    async def cleanup_stale(self, max_age_seconds: float = 600) -> int:
        """Remove questions older than max_age_seconds."""
        now = time.time()
        removed = 0

        async with self._lock:
            stale_ids = [
                qid for qid, q in self._pending.items() if (now - q.created_at) > max_age_seconds
            ]
            for qid in stale_ids:
                question = self._pending.pop(qid, None)
                if question and not question.future.done():
                    question.future.cancel()
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} stale questions")

        return removed


# Global question manager - one per process
_question_manager = QuestionManager()


def get_question_manager() -> QuestionManager:
    """Get the global question manager."""
    return _question_manager


async def can_use_tool(
    tool_name: str,
    input_data: dict[str, Any],
    context: ToolPermissionContext,  # noqa: ARG001 - Required by SDK callback signature
) -> PermissionResultAllow | PermissionResultDeny:
    """Callback to handle tool use requests from the SDK client.

    This callback:
    1. Auto-approves MCP tools (mcp__assisted-flow__*)
    2. Captures AskUserQuestion calls and waits for user answers
    3. Denies all other tools

    Args:
        tool_name: Name of the tool being used
        input_data: Tool input parameters
        context: SDK permission context

    Returns:
        PermissionResultAllow or PermissionResultDeny
    """
    from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

    # Auto-approve our MCP tools
    if tool_name.startswith("mcp__assisted-flow__"):
        logger.debug(f"Auto-approving MCP tool: {tool_name}")
        return PermissionResultAllow()

    # Handle AskUserQuestion specially
    if tool_name == "AskUserQuestion":
        questions = input_data.get("questions", [])
        if not questions:
            logger.warning("AskUserQuestion called with no questions")
            return PermissionResultAllow()  # Let it proceed, will get empty result

        # Compute deterministic question_id from questions content
        # The SSE generator will compute the same ID for the question event
        question_id = compute_question_id(questions)

        # Create pending question and wait for answer
        manager = get_question_manager()
        pending = await manager.create_question(
            question_id=question_id,
            questions=questions,
        )

        try:
            # Wait for user to submit answer (with timeout)
            answers = await asyncio.wait_for(pending.future, timeout=300)  # 5 minute timeout
            logger.info(f"Got answers for question {pending.question_id}: {answers}")

            # Clean up the pending question
            await manager.remove_question(pending.question_id)

            # Return Allow with updated input containing answers
            return PermissionResultAllow(updated_input={"answers": answers, "questions": questions})

        except TimeoutError:
            logger.warning(f"Question {pending.question_id} timed out")
            await manager.cancel_question(pending.question_id)
            return PermissionResultDeny(message="Question timed out waiting for user response")

        except asyncio.CancelledError:
            logger.info(f"Question {pending.question_id} was cancelled")
            return PermissionResultDeny(message="Question was cancelled")

    # Deny all other tools
    logger.warning(f"Denying unexpected tool: {tool_name}")
    return PermissionResultDeny(message=f"Tool '{tool_name}' is not allowed in assisted flow")

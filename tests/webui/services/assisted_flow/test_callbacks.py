"""Tests for assisted flow callbacks module.

Tests cover:
- compute_question_id: deterministic hashing, session scoping, prefix
- PendingQuestion: dataclass creation with asyncio Future
- QuestionManager: async-safe question lifecycle (create, submit, cancel, cleanup)
- create_can_use_tool: session-scoped tool permission callback
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Mock SDK types - these are used by the callback via lazy import
# ---------------------------------------------------------------------------
from claude_agent_sdk.types import (
    PermissionResultAllow as SDKPermissionResultAllow,
    PermissionResultDeny as SDKPermissionResultDeny,
)

# Use aliases so test assertions use the real SDK types
MockPermissionResultAllow = SDKPermissionResultAllow
MockPermissionResultDeny = SDKPermissionResultDeny


class MockToolPermissionContext:
    """Mock for claude_agent_sdk.types.ToolPermissionContext."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _patch_sdk_types():
    """No-op context manager since we use real SDK types."""
    yield


# ===========================================================================
# compute_question_id
# ===========================================================================


class TestComputeQuestionId:
    """Test compute_question_id function."""

    def test_deterministic_same_input_same_output(self) -> None:
        """Same questions and session always produce the same ID."""
        from webui.services.assisted_flow.callbacks import compute_question_id

        questions = [{"text": "Pick a color", "options": ["red", "blue"]}]
        id1 = compute_question_id(questions, session_id="s1")
        id2 = compute_question_id(questions, session_id="s1")

        assert id1 == id2

    def test_starts_with_q_prefix(self) -> None:
        """Question ID starts with 'q_' prefix."""
        from webui.services.assisted_flow.callbacks import compute_question_id

        questions = [{"text": "Pick a color"}]
        qid = compute_question_id(questions)

        assert qid.startswith("q_")

    def test_different_questions_different_ids(self) -> None:
        """Different question content produces different IDs."""
        from webui.services.assisted_flow.callbacks import compute_question_id

        q1 = [{"text": "Pick a color", "options": ["red"]}]
        q2 = [{"text": "Pick a shape", "options": ["circle"]}]

        id1 = compute_question_id(q1, session_id="s1")
        id2 = compute_question_id(q2, session_id="s1")

        assert id1 != id2

    def test_same_questions_different_session_different_ids(self) -> None:
        """Same questions in different sessions produce different IDs (session scoping)."""
        from webui.services.assisted_flow.callbacks import compute_question_id

        questions = [{"text": "Pick a color", "options": ["red", "blue"]}]
        id1 = compute_question_id(questions, session_id="session-aaa")
        id2 = compute_question_id(questions, session_id="session-bbb")

        assert id1 != id2

    def test_default_session_id_is_empty_string(self) -> None:
        """Default session_id is empty string - still produces valid ID."""
        from webui.services.assisted_flow.callbacks import compute_question_id

        questions = [{"text": "hello"}]
        qid = compute_question_id(questions)

        assert qid.startswith("q_")
        assert len(qid) == 2 + 16  # "q_" + 16 hex chars

    def test_key_order_does_not_matter(self) -> None:
        """JSON serialization with sort_keys makes key order irrelevant."""
        from webui.services.assisted_flow.callbacks import compute_question_id

        q1 = [{"text": "hello", "options": ["a", "b"]}]
        q2 = [{"options": ["a", "b"], "text": "hello"}]

        assert compute_question_id(q1) == compute_question_id(q2)

    def test_empty_questions_list(self) -> None:
        """Empty questions list produces a valid ID."""
        from webui.services.assisted_flow.callbacks import compute_question_id

        qid = compute_question_id([], session_id="s1")

        assert qid.startswith("q_")
        assert len(qid) == 18

    def test_multiple_questions(self) -> None:
        """Multiple questions are included in the hash."""
        from webui.services.assisted_flow.callbacks import compute_question_id

        q_single = [{"text": "Q1"}]
        q_double = [{"text": "Q1"}, {"text": "Q2"}]

        assert compute_question_id(q_single) != compute_question_id(q_double)


# ===========================================================================
# PendingQuestion
# ===========================================================================


class TestPendingQuestion:
    """Test PendingQuestion dataclass."""

    @pytest.mark.asyncio()
    async def test_creates_with_future(self) -> None:
        """PendingQuestion creates an asyncio Future on init."""
        from webui.services.assisted_flow.callbacks import PendingQuestion

        pq = PendingQuestion(
            question_id="q_abc",
            questions=[{"text": "hello"}],
        )

        assert pq.question_id == "q_abc"
        assert pq.questions == [{"text": "hello"}]
        assert isinstance(pq.future, asyncio.Future)
        assert not pq.future.done()

    @pytest.mark.asyncio()
    async def test_created_at_is_set(self) -> None:
        """PendingQuestion records creation timestamp."""
        from webui.services.assisted_flow.callbacks import PendingQuestion

        before = time.time()
        pq = PendingQuestion(question_id="q_1", questions=[])
        after = time.time()

        assert before <= pq.created_at <= after

    @pytest.mark.asyncio()
    async def test_future_can_be_resolved(self) -> None:
        """PendingQuestion future can receive a result."""
        from webui.services.assisted_flow.callbacks import PendingQuestion

        pq = PendingQuestion(question_id="q_1", questions=[])
        pq.future.set_result({"answer": "yes"})

        assert pq.future.done()
        assert pq.future.result() == {"answer": "yes"}


# ===========================================================================
# QuestionManager
# ===========================================================================


class TestQuestionManager:
    """Test QuestionManager class."""

    @pytest.mark.asyncio()
    async def test_create_question_stores_pending(self) -> None:
        """create_question stores and returns a PendingQuestion."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        questions = [{"text": "Pick one", "options": ["a", "b"]}]

        pending = await manager.create_question("q_test1", questions)

        assert pending.question_id == "q_test1"
        assert pending.questions == questions
        assert not pending.future.done()

        # Verify it is stored
        retrieved = await manager.get_pending("q_test1")
        assert retrieved is pending

    @pytest.mark.asyncio()
    async def test_submit_answer_resolves_future(self) -> None:
        """submit_answer sets the future result and returns True."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        pending = await manager.create_question("q_ans1", [{"text": "Q?"}])

        result = await manager.submit_answer("q_ans1", {"Q?": "yes"})

        assert result is True
        assert pending.future.done()
        assert pending.future.result() == {"Q?": "yes"}

    @pytest.mark.asyncio()
    async def test_submit_answer_unknown_id_returns_false(self) -> None:
        """submit_answer with non-existent question ID returns False."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()

        result = await manager.submit_answer("q_nonexistent", {"a": "b"})

        assert result is False

    @pytest.mark.asyncio()
    async def test_submit_answer_already_answered_returns_false(self) -> None:
        """submit_answer on an already-answered question returns False."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        pending = await manager.create_question("q_dup", [{"text": "Q?"}])

        # First answer
        assert await manager.submit_answer("q_dup", {"Q?": "first"}) is True

        # Second answer - already done
        assert await manager.submit_answer("q_dup", {"Q?": "second"}) is False

        # Original answer is preserved
        assert pending.future.result() == {"Q?": "first"}

    @pytest.mark.asyncio()
    async def test_submit_answer_invalid_state_returns_false(self) -> None:
        """submit_answer handles InvalidStateError gracefully."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        pending = await manager.create_question("q_race", [{"text": "Q?"}])

        # Cancel the future to force InvalidStateError on set_result
        pending.future.cancel()

        # Submit should handle the InvalidStateError and return False
        result = await manager.submit_answer("q_race", {"Q?": "answer"})

        # The future is done (cancelled), so the "already answered" branch fires first
        assert result is False

    @pytest.mark.asyncio()
    async def test_get_pending_returns_question(self) -> None:
        """get_pending returns the stored question."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        pending = await manager.create_question("q_get", [{"text": "hi"}])

        result = await manager.get_pending("q_get")

        assert result is pending

    @pytest.mark.asyncio()
    async def test_get_pending_returns_none_for_unknown(self) -> None:
        """get_pending returns None for non-existent question."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()

        result = await manager.get_pending("q_unknown")

        assert result is None

    @pytest.mark.asyncio()
    async def test_remove_question_pops_and_returns(self) -> None:
        """remove_question removes and returns the question."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        pending = await manager.create_question("q_rm", [{"text": "bye"}])

        removed = await manager.remove_question("q_rm")

        assert removed is pending
        # No longer retrievable
        assert await manager.get_pending("q_rm") is None

    @pytest.mark.asyncio()
    async def test_remove_question_unknown_returns_none(self) -> None:
        """remove_question returns None for non-existent question."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()

        removed = await manager.remove_question("q_nope")

        assert removed is None

    @pytest.mark.asyncio()
    async def test_cancel_question_cancels_future_and_returns_true(self) -> None:
        """cancel_question cancels the future and returns True."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        pending = await manager.create_question("q_cancel", [{"text": "Q"}])

        result = await manager.cancel_question("q_cancel")

        assert result is True
        assert pending.future.cancelled()
        # No longer stored
        assert await manager.get_pending("q_cancel") is None

    @pytest.mark.asyncio()
    async def test_cancel_question_unknown_id_returns_false(self) -> None:
        """cancel_question with non-existent ID returns False."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()

        result = await manager.cancel_question("q_ghost")

        assert result is False

    @pytest.mark.asyncio()
    async def test_cancel_question_already_done_does_not_raise(self) -> None:
        """cancel_question on already-resolved future doesn't raise."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        pending = await manager.create_question("q_done", [{"text": "Q"}])
        pending.future.set_result({"answer": "yes"})

        result = await manager.cancel_question("q_done")

        assert result is True
        # Future is done but not cancelled (already resolved)
        assert pending.future.done()
        assert not pending.future.cancelled()

    @pytest.mark.asyncio()
    async def test_cleanup_stale_removes_old_questions(self) -> None:
        """cleanup_stale removes questions older than max_age_seconds."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()

        # Create a question and backdate its created_at
        pending = await manager.create_question("q_old", [{"text": "old"}])
        pending.created_at = time.time() - 700  # 700 seconds ago

        # Create a fresh question
        await manager.create_question("q_new", [{"text": "new"}])

        # Cleanup with 600 second threshold
        removed = await manager.cleanup_stale(max_age_seconds=600)

        assert removed == 1
        assert await manager.get_pending("q_old") is None
        assert await manager.get_pending("q_new") is not None

    @pytest.mark.asyncio()
    async def test_cleanup_stale_cancels_futures(self) -> None:
        """cleanup_stale cancels futures of removed questions."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()

        pending = await manager.create_question("q_stale", [{"text": "stale"}])
        pending.created_at = time.time() - 1000

        await manager.cleanup_stale(max_age_seconds=600)

        assert pending.future.cancelled()

    @pytest.mark.asyncio()
    async def test_cleanup_stale_no_stale_returns_zero(self) -> None:
        """cleanup_stale returns 0 when nothing is stale."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        await manager.create_question("q_fresh", [{"text": "fresh"}])

        removed = await manager.cleanup_stale(max_age_seconds=600)

        assert removed == 0

    @pytest.mark.asyncio()
    async def test_cleanup_stale_empty_manager(self) -> None:
        """cleanup_stale on empty manager returns 0."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()

        removed = await manager.cleanup_stale()

        assert removed == 0

    @pytest.mark.asyncio()
    async def test_cleanup_stale_already_done_future(self) -> None:
        """cleanup_stale handles already-resolved futures without error."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()
        pending = await manager.create_question("q_resolved", [{"text": "done"}])
        pending.created_at = time.time() - 1000
        pending.future.set_result({"answer": "resolved"})

        removed = await manager.cleanup_stale(max_age_seconds=600)

        assert removed == 1
        # Future was already done, so cancel() is a no-op; it stays resolved
        assert not pending.future.cancelled()
        assert pending.future.done()

    @pytest.mark.asyncio()
    async def test_concurrent_create_and_submit(self) -> None:
        """QuestionManager is safe under concurrent access."""
        from webui.services.assisted_flow.callbacks import QuestionManager

        manager = QuestionManager()

        async def create_and_answer(idx: int) -> bool:
            qid = f"q_concurrent_{idx}"
            await manager.create_question(qid, [{"text": f"Q{idx}"}])
            return await manager.submit_answer(qid, {f"Q{idx}": f"A{idx}"})

        results = await asyncio.gather(*[create_and_answer(i) for i in range(10)])

        assert all(results)


# ===========================================================================
# get_question_manager (singleton)
# ===========================================================================


class TestGetQuestionManager:
    """Test get_question_manager returns a global singleton."""

    def test_returns_same_instance(self) -> None:
        """get_question_manager always returns the same instance."""
        from webui.services.assisted_flow.callbacks import get_question_manager

        m1 = get_question_manager()
        m2 = get_question_manager()

        assert m1 is m2

    def test_returns_question_manager_type(self) -> None:
        """get_question_manager returns a QuestionManager instance."""
        from webui.services.assisted_flow.callbacks import (
            QuestionManager,
            get_question_manager,
        )

        assert isinstance(get_question_manager(), QuestionManager)


# ===========================================================================
# create_can_use_tool
# ===========================================================================


class TestCreateCanUseTool:
    """Test create_can_use_tool factory and the generated callback."""

    @pytest.mark.asyncio()
    async def test_auto_approves_mcp_tools(self) -> None:
        """MCP tools (mcp__assisted-flow__*) are auto-approved."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s1")
            context = MockToolPermissionContext()

            result = await callback("mcp__assisted-flow__list_plugins", {}, context)

            assert isinstance(result, MockPermissionResultAllow)
            assert result.updated_input is None

    @pytest.mark.asyncio()
    async def test_auto_approves_various_mcp_tool_names(self) -> None:
        """All mcp__assisted-flow__ prefixed tools are approved."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s1")
            context = MockToolPermissionContext()

            tool_names = [
                "mcp__assisted-flow__build_pipeline",
                "mcp__assisted-flow__apply_pipeline",
                "mcp__assisted-flow__sample_files",
                "mcp__assisted-flow__preview_content",
                "mcp__assisted-flow__detect_patterns",
                "mcp__assisted-flow__validate_pipeline",
            ]

            for name in tool_names:
                result = await callback(name, {}, context)
                assert isinstance(result, MockPermissionResultAllow), f"Expected Allow for {name}"

    @pytest.mark.asyncio()
    async def test_denies_unknown_tools(self) -> None:
        """Unknown/unexpected tools are denied."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s1")
            context = MockToolPermissionContext()

            result = await callback("Bash", {"command": "rm -rf /"}, context)

            assert isinstance(result, MockPermissionResultDeny)
            assert "not allowed" in result.message

    @pytest.mark.asyncio()
    async def test_denies_tools_with_similar_prefix(self) -> None:
        """Tools with similar but not matching prefix are denied."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s1")
            context = MockToolPermissionContext()

            # Close but not matching prefix
            result = await callback("mcp__other-server__tool", {}, context)

            assert isinstance(result, MockPermissionResultDeny)

    @pytest.mark.asyncio()
    async def test_denies_ask_user_question_with_empty_questions(self) -> None:
        """AskUserQuestion with empty questions list is denied."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s1")
            context = MockToolPermissionContext()

            result = await callback("AskUserQuestion", {"questions": []}, context)

            assert isinstance(result, MockPermissionResultDeny)
            assert "no questions" in result.message

    @pytest.mark.asyncio()
    async def test_denies_ask_user_question_with_missing_questions_key(self) -> None:
        """AskUserQuestion with no 'questions' key in input is denied."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s1")
            context = MockToolPermissionContext()

            result = await callback("AskUserQuestion", {}, context)

            assert isinstance(result, MockPermissionResultDeny)
            assert "no questions" in result.message

    @pytest.mark.asyncio()
    async def test_ask_user_question_creates_pending_and_waits(self) -> None:
        """AskUserQuestion creates a pending question and waits for answer."""
        from webui.services.assisted_flow.callbacks import (
            create_can_use_tool,
            get_question_manager,
        )

        manager = get_question_manager()

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s_wait")
            context = MockToolPermissionContext()

            questions = [{"text": "Choose format", "options": ["json", "yaml"]}]
            answers = {"Choose format": "json"}

            async def answer_later():
                """Simulate user answering after a brief delay."""
                await asyncio.sleep(0.05)
                # Find the pending question
                from webui.services.assisted_flow.callbacks import compute_question_id

                qid = compute_question_id(questions, session_id="s_wait")
                await manager.submit_answer(qid, answers)

            # Start the callback and the answerer concurrently
            answer_task = asyncio.create_task(answer_later())
            result = await callback("AskUserQuestion", {"questions": questions}, context)
            await answer_task

            assert isinstance(result, MockPermissionResultAllow)
            assert result.updated_input == {
                "answers": answers,
                "questions": questions,
            }

    @pytest.mark.asyncio()
    async def test_ask_user_question_cleans_up_after_answer(self) -> None:
        """After receiving an answer, the pending question is removed."""
        from webui.services.assisted_flow.callbacks import (
            compute_question_id,
            create_can_use_tool,
            get_question_manager,
        )

        manager = get_question_manager()
        questions = [{"text": "Pick one", "options": ["a"]}]
        qid = compute_question_id(questions, session_id="s_cleanup")

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s_cleanup")
            context = MockToolPermissionContext()

            async def answer_soon():
                await asyncio.sleep(0.05)
                await manager.submit_answer(qid, {"Pick one": "a"})

            answer_task = asyncio.create_task(answer_soon())
            await callback("AskUserQuestion", {"questions": questions}, context)
            await answer_task

            # After the callback returns, the question should be removed
            assert await manager.get_pending(qid) is None

    @pytest.mark.asyncio()
    async def test_ask_user_question_timeout(self) -> None:
        """AskUserQuestion returns Deny on timeout."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            # Patch the timeout to a very short duration for testing
            callback = create_can_use_tool(session_id="s_timeout")
            context = MockToolPermissionContext()

            questions = [{"text": "Will timeout", "options": ["yes"]}]

            # Patch asyncio.wait_for to immediately raise TimeoutError
            with patch("webui.services.assisted_flow.callbacks.asyncio.wait_for", side_effect=TimeoutError):
                result = await callback("AskUserQuestion", {"questions": questions}, context)

            assert isinstance(result, MockPermissionResultDeny)
            assert "timed out" in result.message

    @pytest.mark.asyncio()
    async def test_ask_user_question_cancelled(self) -> None:
        """AskUserQuestion returns Deny when cancelled."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s_cancel")
            context = MockToolPermissionContext()

            questions = [{"text": "Will cancel", "options": ["yes"]}]

            # Patch asyncio.wait_for to raise CancelledError
            with patch(
                "webui.services.assisted_flow.callbacks.asyncio.wait_for",
                side_effect=asyncio.CancelledError,
            ):
                result = await callback("AskUserQuestion", {"questions": questions}, context)

            assert isinstance(result, MockPermissionResultDeny)
            assert "cancelled" in result.message

    @pytest.mark.asyncio()
    async def test_session_scoping_produces_different_question_ids(self) -> None:
        """Callbacks with different session_ids produce different question_ids."""
        from webui.services.assisted_flow.callbacks import compute_question_id

        questions = [{"text": "Same question"}]

        qid_a = compute_question_id(questions, session_id="session_A")
        qid_b = compute_question_id(questions, session_id="session_B")

        assert qid_a != qid_b

    @pytest.mark.asyncio()
    async def test_callback_uses_session_id_for_question_id(self) -> None:
        """The callback uses its session_id when computing question IDs."""
        from webui.services.assisted_flow.callbacks import (
            compute_question_id,
            create_can_use_tool,
            get_question_manager,
        )

        manager = get_question_manager()
        questions = [{"text": "Scoped?", "options": ["yes"]}]
        session_id = "s_scope_check"
        expected_qid = compute_question_id(questions, session_id=session_id)

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id=session_id)
            context = MockToolPermissionContext()

            async def answer_with_expected_qid():
                await asyncio.sleep(0.05)
                # The callback should have created a question with expected_qid
                pending = await manager.get_pending(expected_qid)
                assert pending is not None, f"Expected pending question with ID {expected_qid}"
                await manager.submit_answer(expected_qid, {"Scoped?": "yes"})

            answer_task = asyncio.create_task(answer_with_expected_qid())
            result = await callback("AskUserQuestion", {"questions": questions}, context)
            await answer_task

            assert isinstance(result, MockPermissionResultAllow)

    @pytest.mark.asyncio()
    async def test_denies_empty_tool_name(self) -> None:
        """Empty tool name is denied."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s1")
            context = MockToolPermissionContext()

            result = await callback("", {}, context)

            assert isinstance(result, MockPermissionResultDeny)

    @pytest.mark.asyncio()
    async def test_default_session_id_empty_string(self) -> None:
        """create_can_use_tool with no session_id defaults to empty string."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool()
            context = MockToolPermissionContext()

            # Should still work for MCP tools
            result = await callback("mcp__assisted-flow__test", {}, context)
            assert isinstance(result, MockPermissionResultAllow)

    @pytest.mark.asyncio()
    async def test_ask_user_question_timeout_cleans_up(self) -> None:
        """AskUserQuestion timeout cancels the pending question in the manager."""
        from webui.services.assisted_flow.callbacks import (
            compute_question_id,
            create_can_use_tool,
            get_question_manager,
        )

        manager = get_question_manager()
        questions = [{"text": "Timeout cleanup", "options": ["x"]}]
        qid = compute_question_id(questions, session_id="s_timeout_cleanup")

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s_timeout_cleanup")
            context = MockToolPermissionContext()

            with patch(
                "webui.services.assisted_flow.callbacks.asyncio.wait_for",
                side_effect=TimeoutError,
            ):
                await callback("AskUserQuestion", {"questions": questions}, context)

        # After timeout, the question should be cleaned up
        assert await manager.get_pending(qid) is None

    @pytest.mark.asyncio()
    async def test_denies_dangerous_tool_names(self) -> None:
        """Security-sensitive tools like Bash, Read, Write are denied."""
        from webui.services.assisted_flow.callbacks import create_can_use_tool

        with _patch_sdk_types():
            callback = create_can_use_tool(session_id="s_sec")
            context = MockToolPermissionContext()

            dangerous_tools = [
                "Bash",
                "Read",
                "Write",
                "Edit",
                "computer",
                "execute_command",
            ]

            for tool_name in dangerous_tools:
                result = await callback(tool_name, {}, context)
                assert isinstance(result, MockPermissionResultDeny), (
                    f"Tool '{tool_name}' should be denied but was allowed"
                )
                assert "not allowed" in result.message

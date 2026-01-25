"""Unit tests for spawn tools (SourceAnalyzer and PipelineValidator)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webui.services.agent.tools.spawn import (
    SpawnPipelineValidatorTool,
    SpawnSourceAnalyzerTool,
)


@dataclass
class MockUncertainty:
    """Mock uncertainty for testing."""

    severity: str
    message: str
    context: dict[str, Any] | None = None


@dataclass
class MockSubAgentResult:
    """Mock SubAgentResult for testing."""

    success: bool
    data: dict[str, Any]
    uncertainties: list[MockUncertainty]
    summary: str = ""


@pytest.fixture()
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture()
def mock_llm_provider():
    """Create a mock LLM provider with context manager support."""
    provider = AsyncMock()
    provider.__aenter__ = AsyncMock(return_value=provider)
    provider.__aexit__ = AsyncMock(return_value=None)
    return provider


@pytest.fixture()
def mock_llm_factory(mock_llm_provider):
    """Create a mock LLM factory."""
    factory = MagicMock()
    factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)
    return factory


@pytest.fixture()
def mock_conversation():
    """Create a mock conversation."""
    conv = MagicMock()
    conv.id = "test-conv-123"
    conv.inline_source_config = None
    conv.source_analysis = None
    conv.current_pipeline_validation = None
    return conv


@pytest.fixture()
def mock_connector():
    """Create a mock connector."""
    connector = MagicMock()
    connector.enumerate = AsyncMock(return_value=[])
    return connector


class TestSpawnSourceAnalyzerTool:
    """Tests for SpawnSourceAnalyzerTool."""

    def test_schema_is_valid(self):
        """Test that the tool schema is properly defined."""
        tool = SpawnSourceAnalyzerTool(context={})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "spawn_source_analyzer"
        assert "source_id" in schema["function"]["parameters"]["properties"]
        assert "user_intent" in schema["function"]["parameters"]["properties"]
        # source_id is optional
        assert schema["function"]["parameters"]["required"] == []

    @pytest.mark.asyncio()
    async def test_missing_session_returns_error(self):
        """Test that missing session returns error."""
        tool = SpawnSourceAnalyzerTool(context={"user_id": 1, "llm_factory": MagicMock()})
        result = await tool.execute(source_id=1)

        assert result["success"] is False
        assert "No database session" in result["error"]

    @pytest.mark.asyncio()
    async def test_missing_user_id_returns_error(self, mock_session):
        """Test that missing user_id returns error."""
        tool = SpawnSourceAnalyzerTool(context={"session": mock_session, "llm_factory": MagicMock()})
        result = await tool.execute(source_id=1)

        assert result["success"] is False
        assert "No user ID" in result["error"]

    @pytest.mark.asyncio()
    async def test_missing_llm_factory_returns_error(self, mock_session):
        """Test that missing LLM factory returns error."""
        tool = SpawnSourceAnalyzerTool(context={"session": mock_session, "user_id": 1})
        result = await tool.execute(source_id=1)

        assert result["success"] is False
        assert "No LLM factory" in result["error"]

    @pytest.mark.asyncio()
    async def test_source_not_found_returns_error(self, mock_session, mock_llm_factory):
        """Test that non-existent source returns error."""
        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=None)

        with patch(
            "shared.database.repositories.collection_source_repository.CollectionSourceRepository",
            return_value=mock_repo,
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                }
            )
            result = await tool.execute(source_id=999)

        assert result["success"] is False
        assert "Source not found" in result["error"]

    @pytest.mark.asyncio()
    async def test_uses_existing_source(
        self, mock_session, mock_llm_factory, mock_llm_provider, mock_conversation
    ):
        """Test using an existing source from the database."""
        mock_source = MagicMock()
        mock_source.source_type = "local"
        mock_source.source_config = {"path": "/test/path"}

        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=mock_source)

        mock_connector = MagicMock()

        mock_result = MockSubAgentResult(
            success=True,
            data={"total_files": 10, "by_extension": {".pdf": {"count": 5}}},
            uncertainties=[],
            summary="Analysis complete",
        )

        mock_analyzer = MagicMock()
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "shared.database.repositories.collection_source_repository.CollectionSourceRepository",
                return_value=mock_repo,
            ),
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=mock_connector,
            ),
            patch(
                "webui.services.agent.subagents.source_analyzer.SourceAnalyzer",
                return_value=mock_analyzer,
            ),
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                }
            )
            result = await tool.execute(source_id=1)

        assert result["success"] is True
        assert result["analysis"]["total_files"] == 10
        assert result["summary"] == "Analysis complete"

    @pytest.mark.asyncio()
    async def test_uses_inline_source_config(
        self, mock_session, mock_llm_factory, mock_llm_provider
    ):
        """Test using inline source config from conversation."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.inline_source_config = {
            "source_type": "local",
            "source_config": {"path": "/inline/path"},
        }
        mock_conversation.source_analysis = None

        mock_connector = MagicMock()

        mock_result = MockSubAgentResult(
            success=True,
            data={"total_files": 5},
            uncertainties=[],
            summary="Inline analysis complete",
        )

        mock_analyzer = MagicMock()
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=mock_connector,
            ),
            patch(
                "webui.services.agent.subagents.source_analyzer.SourceAnalyzer",
                return_value=mock_analyzer,
            ),
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                }
            )
            # No source_id - should use inline config
            result = await tool.execute()

        assert result["success"] is True
        assert result["analysis"]["total_files"] == 5

    @pytest.mark.asyncio()
    async def test_inline_config_merges_pending_secrets(
        self, mock_session, mock_llm_factory, mock_llm_provider
    ):
        """Test that pending secrets are merged into source config."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.inline_source_config = {
            "source_type": "imap",
            "source_config": {"host": "mail.example.com"},
            "_pending_secrets": {"password": "secret123"},
        }
        mock_conversation.source_analysis = None

        captured_config = {}

        def capture_connector(_source_type, config):
            captured_config.update(config)
            return MagicMock()

        mock_result = MockSubAgentResult(
            success=True,
            data={},
            uncertainties=[],
        )

        mock_analyzer = MagicMock()
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                side_effect=capture_connector,
            ),
            patch(
                "webui.services.agent.subagents.source_analyzer.SourceAnalyzer",
                return_value=mock_analyzer,
            ),
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                }
            )
            await tool.execute()

        # Config should have both original values and merged secrets
        assert captured_config["host"] == "mail.example.com"
        assert captured_config["password"] == "secret123"

    @pytest.mark.asyncio()
    async def test_inline_config_missing_source_type_returns_error(
        self, mock_session, mock_llm_factory
    ):
        """Test that inline config without source_type returns error."""
        mock_conversation = MagicMock()
        mock_conversation.inline_source_config = {
            "source_config": {"path": "/some/path"},
            # Missing source_type
        }

        tool = SpawnSourceAnalyzerTool(
            context={
                "session": mock_session,
                "user_id": 1,
                "llm_factory": mock_llm_factory,
                "conversation": mock_conversation,
            }
        )
        result = await tool.execute()

        assert result["success"] is False
        assert "missing source_type" in result["error"]

    @pytest.mark.asyncio()
    async def test_no_source_and_no_inline_config_returns_error(
        self, mock_session, mock_llm_factory, mock_conversation
    ):
        """Test that missing source_id and inline_config returns error."""
        tool = SpawnSourceAnalyzerTool(
            context={
                "session": mock_session,
                "user_id": 1,
                "llm_factory": mock_llm_factory,
                "conversation": mock_conversation,
            }
        )
        result = await tool.execute()

        assert result["success"] is False
        assert "No source_id provided and no inline_source_config" in result["error"]

    @pytest.mark.asyncio()
    async def test_connector_factory_error_returns_error(
        self, mock_session, mock_llm_factory
    ):
        """Test that connector creation failure returns error."""
        mock_source = MagicMock()
        mock_source.source_type = "invalid_type"
        mock_source.source_config = {}

        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=mock_source)

        with (
            patch(
                "shared.database.repositories.collection_source_repository.CollectionSourceRepository",
                return_value=mock_repo,
            ),
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                side_effect=ValueError("Unknown connector type"),
            ),
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                }
            )
            result = await tool.execute(source_id=1)

        assert result["success"] is False
        assert "Failed to create connector" in result["error"]

    @pytest.mark.asyncio()
    async def test_llm_provider_creation_failure_returns_error(
        self, mock_session, mock_conversation
    ):
        """Test that LLM provider creation failure returns error."""
        mock_source = MagicMock()
        mock_source.source_type = "local"
        mock_source.source_config = {}

        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=mock_source)

        mock_llm_factory = MagicMock()
        mock_llm_factory.create_provider_for_tier = AsyncMock(
            side_effect=Exception("LLM unavailable")
        )

        with (
            patch(
                "shared.database.repositories.collection_source_repository.CollectionSourceRepository",
                return_value=mock_repo,
            ),
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=MagicMock(),
            ),
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                }
            )
            result = await tool.execute(source_id=1)

        assert result["success"] is False
        assert "Failed to create LLM provider" in result["error"]

    @pytest.mark.asyncio()
    async def test_subagent_exception_returns_error(
        self, mock_session, mock_llm_factory, mock_llm_provider, mock_conversation
    ):
        """Test that subagent exceptions are caught and return error."""
        mock_source = MagicMock()
        mock_source.source_type = "local"
        mock_source.source_config = {}

        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=mock_source)

        mock_analyzer = MagicMock()
        mock_analyzer.run = AsyncMock(side_effect=Exception("Subagent crashed"))

        with (
            patch(
                "shared.database.repositories.collection_source_repository.CollectionSourceRepository",
                return_value=mock_repo,
            ),
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=MagicMock(),
            ),
            patch(
                "webui.services.agent.subagents.source_analyzer.SourceAnalyzer",
                return_value=mock_analyzer,
            ),
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                }
            )
            result = await tool.execute(source_id=1)

        assert result["success"] is False
        assert "Subagent crashed" in result["error"]

    @pytest.mark.asyncio()
    async def test_analysis_stored_in_conversation(
        self, mock_session, mock_llm_factory, mock_llm_provider
    ):
        """Test that successful analysis is stored in conversation."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.inline_source_config = {
            "source_type": "local",
            "source_config": {},
        }
        mock_conversation.source_analysis = None

        analysis_data = {"total_files": 100, "by_extension": {}}

        mock_result = MockSubAgentResult(
            success=True,
            data=analysis_data,
            uncertainties=[],
        )

        mock_analyzer = MagicMock()
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=MagicMock(),
            ),
            patch(
                "webui.services.agent.subagents.source_analyzer.SourceAnalyzer",
                return_value=mock_analyzer,
            ),
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                }
            )
            await tool.execute()

        # Verify analysis was stored in conversation
        assert mock_conversation.source_analysis == analysis_data

    @pytest.mark.asyncio()
    async def test_uncertainties_persisted_to_repository(
        self, mock_session, mock_llm_factory, mock_llm_provider
    ):
        """Test that uncertainties are persisted to the repository."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.inline_source_config = {
            "source_type": "local",
            "source_config": {},
        }

        uncertainties = [
            MockUncertainty(
                severity="warning",
                message="Some files may be scanned PDFs",
                context={"affected_files": 5},
            ),
            MockUncertainty(
                severity="info",
                message="Detected multiple languages",
            ),
        ]

        mock_result = MockSubAgentResult(
            success=True,
            data={},
            uncertainties=uncertainties,
        )

        mock_analyzer = MagicMock()
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        mock_conv_repo = MagicMock()
        mock_conv_repo.add_uncertainty = AsyncMock()

        with (
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=MagicMock(),
            ),
            patch(
                "webui.services.agent.subagents.source_analyzer.SourceAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "webui.services.agent.repository.AgentConversationRepository",
                return_value=mock_conv_repo,
            ),
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                }
            )
            result = await tool.execute()

        # Verify add_uncertainty was called for each uncertainty
        assert mock_conv_repo.add_uncertainty.call_count == 2
        assert result["uncertainties"][0]["severity"] == "warning"
        assert result["uncertainties"][1]["message"] == "Detected multiple languages"

    @pytest.mark.asyncio()
    async def test_user_intent_passed_to_subagent_context(
        self, mock_session, mock_llm_factory, mock_llm_provider
    ):
        """Test that user_intent is passed to subagent context."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.inline_source_config = {
            "source_type": "local",
            "source_config": {},
        }

        captured_context = {}

        def capture_context(_llm_provider, context):
            captured_context.update(context)
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(
                return_value=MockSubAgentResult(success=True, data={}, uncertainties=[])
            )
            return mock_agent

        with (
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=MagicMock(),
            ),
            patch(
                "webui.services.agent.subagents.source_analyzer.SourceAnalyzer",
                side_effect=capture_context,
            ),
        ):
            tool = SpawnSourceAnalyzerTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                }
            )
            await tool.execute(user_intent="I want to index my PDF documents")

        assert captured_context["user_intent"] == "I want to index my PDF documents"


class TestSpawnPipelineValidatorTool:
    """Tests for SpawnPipelineValidatorTool."""

    def test_schema_is_valid(self):
        """Test that the tool schema is properly defined."""
        tool = SpawnPipelineValidatorTool(context={})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "spawn_pipeline_validator"
        assert "pipeline" in schema["function"]["parameters"]["properties"]
        assert "sample_count" in schema["function"]["parameters"]["properties"]
        assert "pipeline" in schema["function"]["parameters"]["required"]

    @pytest.mark.asyncio()
    async def test_missing_session_returns_error(self):
        """Test that missing session returns error."""
        tool = SpawnPipelineValidatorTool(
            context={"user_id": 1, "llm_factory": MagicMock()}
        )
        result = await tool.execute(pipeline={"stages": []})

        assert result["success"] is False
        assert "No database session" in result["error"]

    @pytest.mark.asyncio()
    async def test_missing_user_id_returns_error(self, mock_session):
        """Test that missing user_id returns error."""
        tool = SpawnPipelineValidatorTool(
            context={"session": mock_session, "llm_factory": MagicMock()}
        )
        result = await tool.execute(pipeline={"stages": []})

        assert result["success"] is False
        assert "No user ID" in result["error"]

    @pytest.mark.asyncio()
    async def test_missing_llm_factory_returns_error(self, mock_session):
        """Test that missing LLM factory returns error."""
        tool = SpawnPipelineValidatorTool(
            context={"session": mock_session, "user_id": 1}
        )
        result = await tool.execute(pipeline={"stages": []})

        assert result["success"] is False
        assert "No LLM factory" in result["error"]

    @pytest.mark.asyncio()
    async def test_missing_source_analysis_returns_error(
        self, mock_session, mock_llm_factory, mock_conversation
    ):
        """Test that missing source_analysis returns error."""
        tool = SpawnPipelineValidatorTool(
            context={
                "session": mock_session,
                "user_id": 1,
                "llm_factory": mock_llm_factory,
                "conversation": mock_conversation,
            }
        )
        result = await tool.execute(pipeline={"stages": []})

        assert result["success"] is False
        assert "No source analysis available" in result["error"]

    @pytest.mark.asyncio()
    async def test_no_samples_returns_error(
        self, mock_session, mock_llm_factory
    ):
        """Test that empty samples returns error."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.source_analysis = {"by_extension": {}}  # Empty

        tool = SpawnPipelineValidatorTool(
            context={
                "session": mock_session,
                "user_id": 1,
                "llm_factory": mock_llm_factory,
                "conversation": mock_conversation,
            }
        )
        result = await tool.execute(pipeline={"stages": []})

        assert result["success"] is False
        assert "No sample files available" in result["error"]

    @pytest.mark.asyncio()
    async def test_select_samples_from_source_analysis(self):
        """Test _select_samples extracts sample URIs correctly."""
        tool = SpawnPipelineValidatorTool(context={})

        source_analysis = {
            "by_extension": {
                ".pdf": {"count": 10, "sample_uris": ["file1.pdf", "file2.pdf"]},
                ".docx": {"count": 5, "sample_uris": ["doc1.docx"]},
            }
        }

        samples = tool._select_samples(source_analysis, sample_count=10)

        assert len(samples) == 3
        assert {"uri": "file1.pdf"} in samples
        assert {"uri": "file2.pdf"} in samples
        assert {"uri": "doc1.docx"} in samples

    @pytest.mark.asyncio()
    async def test_select_samples_respects_limit(self):
        """Test _select_samples respects sample_count limit."""
        tool = SpawnPipelineValidatorTool(context={})

        source_analysis = {
            "by_extension": {
                ".pdf": {
                    "count": 100,
                    "sample_uris": [f"file{i}.pdf" for i in range(50)],
                },
            }
        }

        samples = tool._select_samples(source_analysis, sample_count=10)

        assert len(samples) == 10

    @pytest.mark.asyncio()
    async def test_uses_connector_from_context(
        self, mock_session, mock_llm_factory, mock_llm_provider, mock_connector
    ):
        """Test that connector from context is used when available."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.source_analysis = {
            "by_extension": {".pdf": {"sample_uris": ["test.pdf"]}}
        }
        mock_conversation.current_pipeline_validation = None

        mock_result = MockSubAgentResult(
            success=True,
            data={"passed": True},
            uncertainties=[],
        )

        mock_validator = MagicMock()
        mock_validator.run = AsyncMock(return_value=mock_result)

        with patch(
            "webui.services.agent.subagents.pipeline_validator.PipelineValidator",
            return_value=mock_validator,
        ):
            tool = SpawnPipelineValidatorTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                    "connector": mock_connector,  # Connector in context
                }
            )
            result = await tool.execute(pipeline={"stages": []})

        assert result["success"] is True

    @pytest.mark.asyncio()
    async def test_creates_connector_from_source_id(
        self, mock_session, mock_llm_factory, mock_llm_provider
    ):
        """Test that connector is created from source_id when not in context."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.source_analysis = {
            "source_id": 42,
            "by_extension": {".pdf": {"sample_uris": ["test.pdf"]}},
        }
        mock_conversation.current_pipeline_validation = None

        mock_source = MagicMock()
        mock_source.source_type = "local"
        mock_source.source_config = {"path": "/test"}

        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=mock_source)

        mock_result = MockSubAgentResult(
            success=True,
            data={"passed": True},
            uncertainties=[],
        )

        mock_validator = MagicMock()
        mock_validator.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "shared.database.repositories.collection_source_repository.CollectionSourceRepository",
                return_value=mock_repo,
            ),
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=MagicMock(),
            ),
            patch(
                "webui.services.agent.subagents.pipeline_validator.PipelineValidator",
                return_value=mock_validator,
            ),
        ):
            tool = SpawnPipelineValidatorTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                    # No connector in context
                }
            )
            result = await tool.execute(pipeline={"stages": []})

        assert result["success"] is True
        mock_repo.get_by_id.assert_called_once_with(42)

    @pytest.mark.asyncio()
    async def test_no_connector_available_returns_error(
        self, mock_session, mock_llm_factory
    ):
        """Test that missing connector returns error."""
        mock_conversation = MagicMock()
        mock_conversation.source_analysis = {
            "by_extension": {".pdf": {"sample_uris": ["test.pdf"]}},
            # No source_id
        }

        tool = SpawnPipelineValidatorTool(
            context={
                "session": mock_session,
                "user_id": 1,
                "llm_factory": mock_llm_factory,
                "conversation": mock_conversation,
                # No connector in context
            }
        )
        result = await tool.execute(pipeline={"stages": []})

        assert result["success"] is False
        assert "No connector available" in result["error"]

    @pytest.mark.asyncio()
    async def test_successful_validation(
        self, mock_session, mock_llm_factory, mock_llm_provider, mock_connector
    ):
        """Test successful pipeline validation."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.source_analysis = {
            "by_extension": {".pdf": {"sample_uris": ["test.pdf"]}}
        }
        mock_conversation.current_pipeline_validation = None

        validation_report = {
            "passed": True,
            "success_rate": 0.95,
            "failures": [],
        }

        mock_result = MockSubAgentResult(
            success=True,
            data=validation_report,
            uncertainties=[],
            summary="Validation passed with 95% success rate",
        )

        mock_validator = MagicMock()
        mock_validator.run = AsyncMock(return_value=mock_result)

        with patch(
            "webui.services.agent.subagents.pipeline_validator.PipelineValidator",
            return_value=mock_validator,
        ):
            tool = SpawnPipelineValidatorTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                    "connector": mock_connector,
                }
            )
            result = await tool.execute(pipeline={"stages": [{"type": "extract"}]})

        assert result["success"] is True
        assert result["report"]["passed"] is True
        assert result["summary"] == "Validation passed with 95% success rate"

    @pytest.mark.asyncio()
    async def test_validation_report_stored_in_conversation(
        self, mock_session, mock_llm_factory, mock_llm_provider, mock_connector
    ):
        """Test that validation report is stored in conversation."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.source_analysis = {
            "by_extension": {".pdf": {"sample_uris": ["test.pdf"]}}
        }
        mock_conversation.current_pipeline_validation = None

        validation_report = {"passed": True}

        mock_result = MockSubAgentResult(
            success=True,
            data=validation_report,
            uncertainties=[],
        )

        mock_validator = MagicMock()
        mock_validator.run = AsyncMock(return_value=mock_result)

        with patch(
            "webui.services.agent.subagents.pipeline_validator.PipelineValidator",
            return_value=mock_validator,
        ):
            tool = SpawnPipelineValidatorTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                    "connector": mock_connector,
                }
            )
            await tool.execute(pipeline={"stages": []})

        # Verify report was stored
        assert mock_conversation.current_pipeline_validation == validation_report

    @pytest.mark.asyncio()
    async def test_subagent_exception_returns_error(
        self, mock_session, mock_llm_factory, mock_llm_provider, mock_connector
    ):
        """Test that subagent exceptions are caught and return error."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.source_analysis = {
            "by_extension": {".pdf": {"sample_uris": ["test.pdf"]}}
        }

        mock_validator = MagicMock()
        mock_validator.run = AsyncMock(side_effect=Exception("Validation crashed"))

        with patch(
            "webui.services.agent.subagents.pipeline_validator.PipelineValidator",
            return_value=mock_validator,
        ):
            tool = SpawnPipelineValidatorTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                    "connector": mock_connector,
                }
            )
            result = await tool.execute(pipeline={"stages": []})

        assert result["success"] is False
        assert "Validation crashed" in result["error"]

    @pytest.mark.asyncio()
    async def test_uncertainties_persisted(
        self, mock_session, mock_llm_factory, mock_llm_provider, mock_connector
    ):
        """Test that validation uncertainties are persisted."""
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conv-123"
        mock_conversation.source_analysis = {
            "by_extension": {".pdf": {"sample_uris": ["test.pdf"]}}
        }

        uncertainties = [
            MockUncertainty(
                severity="warning",
                message="Some chunks are very small",
            ),
        ]

        mock_result = MockSubAgentResult(
            success=True,
            data={"passed": True},
            uncertainties=uncertainties,
        )

        mock_validator = MagicMock()
        mock_validator.run = AsyncMock(return_value=mock_result)

        mock_conv_repo = MagicMock()
        mock_conv_repo.add_uncertainty = AsyncMock()

        with (
            patch(
                "webui.services.agent.subagents.pipeline_validator.PipelineValidator",
                return_value=mock_validator,
            ),
            patch(
                "webui.services.agent.repository.AgentConversationRepository",
                return_value=mock_conv_repo,
            ),
        ):
            tool = SpawnPipelineValidatorTool(
                context={
                    "session": mock_session,
                    "user_id": 1,
                    "llm_factory": mock_llm_factory,
                    "conversation": mock_conversation,
                    "connector": mock_connector,
                }
            )
            result = await tool.execute(pipeline={"stages": []})

        assert mock_conv_repo.add_uncertainty.call_count == 1
        assert result["uncertainties"][0]["severity"] == "warning"

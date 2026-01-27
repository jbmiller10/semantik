"""Unit tests for agent service models."""

from uuid import uuid4

from webui.services.agent.models import (
    AgentConversation,
    ConversationStatus,
    ConversationUncertainty,
    UncertaintySeverity,
)


class TestConversationStatus:
    """Tests for ConversationStatus enum."""

    def test_active_status(self):
        assert ConversationStatus.ACTIVE.value == "active"

    def test_applied_status(self):
        assert ConversationStatus.APPLIED.value == "applied"

    def test_abandoned_status(self):
        assert ConversationStatus.ABANDONED.value == "abandoned"


class TestUncertaintySeverity:
    """Tests for UncertaintySeverity enum."""

    def test_blocking_severity(self):
        assert UncertaintySeverity.BLOCKING.value == "blocking"

    def test_notable_severity(self):
        assert UncertaintySeverity.NOTABLE.value == "notable"

    def test_info_severity(self):
        assert UncertaintySeverity.INFO.value == "info"


class TestAgentConversation:
    """Tests for AgentConversation model."""

    def test_create_conversation(self):
        """Test creating a conversation instance."""
        conv_id = str(uuid4())
        conversation = AgentConversation(
            id=conv_id,
            user_id=1,
            status=ConversationStatus.ACTIVE,
        )

        assert conversation.id == conv_id
        assert conversation.user_id == 1
        assert conversation.status == ConversationStatus.ACTIVE
        assert conversation.source_id is None
        assert conversation.collection_id is None
        assert conversation.current_pipeline is None
        assert conversation.source_analysis is None
        assert conversation.summary is None

    def test_conversation_with_source(self):
        """Test creating a conversation with a source."""
        conversation = AgentConversation(
            id=str(uuid4()),
            user_id=1,
            source_id=42,
            status=ConversationStatus.ACTIVE,
        )

        assert conversation.source_id == 42

    def test_conversation_with_pipeline(self):
        """Test conversation with pipeline configuration."""
        pipeline_config = {
            "stages": [
                {"id": "extract", "type": "parser"},
                {"id": "chunk", "type": "chunker"},
            ]
        }
        conversation = AgentConversation(
            id=str(uuid4()),
            user_id=1,
            status=ConversationStatus.ACTIVE,
            current_pipeline=pipeline_config,
        )

        assert conversation.current_pipeline == pipeline_config

    def test_conversation_applied_status(self):
        """Test conversation with applied status and collection."""
        collection_id = str(uuid4())
        conversation = AgentConversation(
            id=str(uuid4()),
            user_id=1,
            status=ConversationStatus.APPLIED,
            collection_id=collection_id,
        )

        assert conversation.status == ConversationStatus.APPLIED
        assert conversation.collection_id == collection_id


class TestConversationUncertainty:
    """Tests for ConversationUncertainty model."""

    def test_create_blocking_uncertainty(self):
        """Test creating a blocking uncertainty."""
        conv_id = str(uuid4())
        uncertainty = ConversationUncertainty(
            id=str(uuid4()),
            conversation_id=conv_id,
            severity=UncertaintySeverity.BLOCKING,
            message="Cannot parse PDF files - OCR required",
            resolved=False,  # Explicit default since DB default doesn't apply in unit tests
        )

        assert uncertainty.conversation_id == conv_id
        assert uncertainty.severity == UncertaintySeverity.BLOCKING
        assert uncertainty.message == "Cannot parse PDF files - OCR required"
        assert uncertainty.resolved is False
        assert uncertainty.resolved_by is None

    def test_create_uncertainty_with_context(self):
        """Test creating an uncertainty with context."""
        context = {
            "affected_files": ["doc1.pdf", "doc2.pdf"],
            "parser_tried": "pypdf",
        }
        uncertainty = ConversationUncertainty(
            id=str(uuid4()),
            conversation_id=str(uuid4()),
            severity=UncertaintySeverity.NOTABLE,
            message="Some PDFs may be scanned images",
            context=context,
        )

        assert uncertainty.context == context

    def test_resolved_uncertainty(self):
        """Test a resolved uncertainty."""
        uncertainty = ConversationUncertainty(
            id=str(uuid4()),
            conversation_id=str(uuid4()),
            severity=UncertaintySeverity.BLOCKING,
            message="Encoding issue detected",
            resolved=True,
            resolved_by="user_confirmed",
        )

        assert uncertainty.resolved is True
        assert uncertainty.resolved_by == "user_confirmed"

    def test_info_severity(self):
        """Test info-level uncertainty."""
        uncertainty = ConversationUncertainty(
            id=str(uuid4()),
            conversation_id=str(uuid4()),
            severity=UncertaintySeverity.INFO,
            message="Found 3 empty files, will skip them",
        )

        assert uncertainty.severity == UncertaintySeverity.INFO

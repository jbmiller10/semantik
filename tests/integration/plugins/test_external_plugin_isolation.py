"""Integration test verifying external plugins work without semantik imports.

This test creates plugins in an isolated environment with NO imports from semantik,
verifies they can be loaded by the plugin system, and that their output is correctly
converted to internal types.

This is the key test proving Phase 8 of the Protocol-Based Plugin Decoupling works:
external plugins can be developed using ONLY Python standard library types.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

import pytest

# =============================================================================
# External Connector Plugin Code (NO semantik imports!)
# =============================================================================

EXTERNAL_CONNECTOR_CODE = '''
"""External connector plugin with zero semantik imports."""
from typing import ClassVar, Any, AsyncIterator
import hashlib


class ExternalTestConnector:
    """A connector that requires no semantik imports."""

    PLUGIN_ID: ClassVar[str] = "external-test-connector"
    PLUGIN_TYPE: ClassVar[str] = "connector"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    async def authenticate(self) -> bool:
        return True

    async def load_documents(
        self, source_id: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        content = "Test document content from external plugin"
        yield {
            "content": content,
            "unique_id": "external-test-doc-1",
            "source_type": "external-test-connector",
            "metadata": {"test": True, "external": True},
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        }

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        return [{"name": "api_key", "type": "password", "label": "API Key", "required": False}]

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        return [{"name": "api_key"}]

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": "External Test Connector",
            "description": "Test connector with no semantik imports",
        }
'''


# =============================================================================
# External Agent Plugin Code (NO semantik imports!)
# =============================================================================

EXTERNAL_AGENT_CODE = '''
"""External agent plugin with zero semantik imports."""
from typing import ClassVar, Any, AsyncIterator
from datetime import datetime, timezone
import uuid


class ExternalTestAgent:
    """An agent that requires no semantik imports."""

    PLUGIN_ID: ClassVar[str] = "external-test-agent"
    PLUGIN_TYPE: ClassVar[str] = "agent"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    async def execute(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "type": "text",
            "content": f"Response to: {prompt}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_partial": False,
            "sequence_number": 0,
        }

    @classmethod
    def get_capabilities(cls) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": False,
            "supports_sessions": False,
        }

    @classmethod
    def supported_use_cases(cls) -> list[str]:
        return ["assistant"]

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": "External Test Agent",
            "description": "Test agent with no semantik imports",
        }
'''


# =============================================================================
# External Embedding Plugin Code (NO semantik imports!)
# =============================================================================

EXTERNAL_EMBEDDING_CODE = '''
"""External embedding plugin with zero semantik imports."""
from typing import ClassVar, Any


class ExternalTestEmbedding:
    """An embedding provider that requires no semantik imports."""

    PLUGIN_ID: ClassVar[str] = "external-test-embedding"
    PLUGIN_TYPE: ClassVar[str] = "embedding"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"
    INTERNAL_NAME: ClassVar[str] = "external-test-embedding"
    API_ID: ClassVar[str] = "external-test-embedding"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    async def embed_texts(
        self,
        texts: list[str],
        mode: str = "document",
    ) -> list[list[float]]:
        # Return fixed-dimension embeddings (384d)
        return [[0.1] * 384 for _ in texts]

    @classmethod
    def get_definition(cls) -> dict[str, Any]:
        return {
            "api_id": cls.API_ID,
            "internal_id": cls.INTERNAL_NAME,
            "display_name": "External Test Embedding",
            "description": "Test embedding with no semantik imports",
            "provider_type": "local",
        }

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name == "external-test-model"

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": "External Test Embedding",
            "description": "Test embedding with no semantik imports",
        }
'''


# =============================================================================
# Tests
# =============================================================================


class TestExternalConnectorIsolation:
    """Test that connector plugins work without any semantik imports."""

    def test_external_connector_code_has_no_semantik_imports(self) -> None:
        """Verify the test code truly has no semantik imports."""
        assert "from semantik" not in EXTERNAL_CONNECTOR_CODE
        assert "import semantik" not in EXTERNAL_CONNECTOR_CODE
        assert "from shared" not in EXTERNAL_CONNECTOR_CODE
        assert "import shared" not in EXTERNAL_CONNECTOR_CODE

    def test_external_connector_satisfies_protocol(self) -> None:
        """Verify external connector class satisfies ConnectorProtocol structurally."""
        namespace: dict[str, Any] = {}
        exec(EXTERNAL_CONNECTOR_CODE, namespace)
        cls = namespace["ExternalTestConnector"]

        # Check protocol requirements
        assert hasattr(cls, "PLUGIN_ID")
        assert hasattr(cls, "PLUGIN_TYPE")
        assert hasattr(cls, "PLUGIN_VERSION")
        assert cls.PLUGIN_TYPE == "connector"
        assert callable(getattr(cls, "authenticate", None))
        assert callable(getattr(cls, "load_documents", None))
        assert callable(getattr(cls, "get_config_fields", None))
        assert callable(getattr(cls, "get_secret_fields", None))
        assert callable(getattr(cls, "get_manifest", None))

    def test_loader_validates_external_connector(self) -> None:
        """Verify the loader's _satisfies_protocol accepts external connector."""
        from shared.plugins.loader import _satisfies_protocol

        namespace: dict[str, Any] = {}
        exec(EXTERNAL_CONNECTOR_CODE, namespace)
        cls = namespace["ExternalTestConnector"]

        # The loader should recognize this as a valid connector
        assert _satisfies_protocol(cls, "connector")

    @pytest.mark.asyncio()
    async def test_external_connector_output_converts_correctly(self) -> None:
        """Verify external connector output converts to IngestedDocument."""
        from shared.plugins.dto_adapters import dict_to_ingested_document

        namespace: dict[str, Any] = {}
        exec(EXTERNAL_CONNECTOR_CODE, namespace)
        cls = namespace["ExternalTestConnector"]

        # Instantiate and get a document
        connector = cls(config={})
        async for doc_dict in connector.load_documents():
            # Convert using adapter
            doc = dict_to_ingested_document(doc_dict)

            # Verify conversion worked
            assert doc.content == "Test document content from external plugin"
            assert doc.unique_id == "external-test-doc-1"
            assert doc.source_type == "external-test-connector"
            assert doc.metadata["external"] is True
            assert len(doc.content_hash) == 64
            break

    def test_adapter_validates_missing_required_keys(self) -> None:
        """Verify adapter catches missing required fields from external plugins."""
        from shared.plugins.dto_adapters import (
            ValidationError,
            dict_to_ingested_document,
        )

        # Missing required keys
        with pytest.raises(ValidationError, match="missing required keys"):
            dict_to_ingested_document({"content": "test"})

    def test_adapter_validates_invalid_content_hash(self) -> None:
        """Verify adapter catches malformed content_hash."""
        from shared.plugins.dto_adapters import (
            ValidationError,
            dict_to_ingested_document,
        )

        # Invalid content_hash (not 64 hex chars)
        with pytest.raises(ValidationError, match="content_hash"):
            dict_to_ingested_document(
                {
                    "content": "test",
                    "unique_id": "id",
                    "source_type": "test",
                    "metadata": {},
                    "content_hash": "invalid",
                }
            )

    def test_get_manifest_returns_valid_dict(self) -> None:
        """Verify external connector get_manifest returns expected structure."""
        namespace: dict[str, Any] = {}
        exec(EXTERNAL_CONNECTOR_CODE, namespace)
        cls = namespace["ExternalTestConnector"]

        manifest = cls.get_manifest()
        assert manifest["id"] == "external-test-connector"
        assert manifest["type"] == "connector"
        assert manifest["version"] == "1.0.0"
        assert "display_name" in manifest
        assert "description" in manifest


class TestExternalAgentIsolation:
    """Test that agent plugins work without any semantik imports."""

    def test_external_agent_code_has_no_semantik_imports(self) -> None:
        """Verify the test code truly has no semantik imports."""
        assert "from semantik" not in EXTERNAL_AGENT_CODE
        assert "import semantik" not in EXTERNAL_AGENT_CODE
        assert "from shared" not in EXTERNAL_AGENT_CODE
        assert "import shared" not in EXTERNAL_AGENT_CODE

    def test_external_agent_satisfies_protocol(self) -> None:
        """Verify external agent class satisfies AgentProtocol structurally."""
        namespace: dict[str, Any] = {}
        exec(EXTERNAL_AGENT_CODE, namespace)
        cls = namespace["ExternalTestAgent"]

        assert hasattr(cls, "PLUGIN_ID")
        assert hasattr(cls, "PLUGIN_TYPE")
        assert hasattr(cls, "PLUGIN_VERSION")
        assert cls.PLUGIN_TYPE == "agent"
        assert callable(getattr(cls, "execute", None))
        assert callable(getattr(cls, "get_capabilities", None))
        assert callable(getattr(cls, "supported_use_cases", None))
        assert callable(getattr(cls, "get_manifest", None))

    def test_loader_validates_external_agent(self) -> None:
        """Verify the loader's _satisfies_protocol accepts external agent."""
        from shared.plugins.loader import _satisfies_protocol

        namespace: dict[str, Any] = {}
        exec(EXTERNAL_AGENT_CODE, namespace)
        cls = namespace["ExternalTestAgent"]

        assert _satisfies_protocol(cls, "agent")

    @pytest.mark.asyncio()
    async def test_external_agent_output_converts_correctly(self) -> None:
        """Verify external agent output converts to AgentMessage."""
        from shared.plugins.dto_adapters import dict_to_agent_message

        namespace: dict[str, Any] = {}
        exec(EXTERNAL_AGENT_CODE, namespace)
        cls = namespace["ExternalTestAgent"]

        agent = cls()
        async for msg_dict in agent.execute("Hello"):
            msg = dict_to_agent_message(msg_dict)

            assert msg.content == "Response to: Hello"
            assert msg.role.value == "assistant"
            assert msg.type.value == "text"
            break

    def test_adapter_validates_invalid_role(self) -> None:
        """Verify adapter catches invalid role strings from external plugins."""
        from datetime import datetime

        from shared.plugins.dto_adapters import ValidationError, dict_to_agent_message

        with pytest.raises(ValidationError, match="Invalid role"):
            dict_to_agent_message(
                {
                    "id": "msg-1",
                    "role": "invalid_role",  # Not in MESSAGE_ROLES
                    "type": "text",
                    "content": "test",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    def test_adapter_validates_invalid_message_type(self) -> None:
        """Verify adapter catches invalid message type strings."""
        from datetime import datetime

        from shared.plugins.dto_adapters import ValidationError, dict_to_agent_message

        with pytest.raises(ValidationError, match="Invalid type"):
            dict_to_agent_message(
                {
                    "id": "msg-1",
                    "role": "assistant",
                    "type": "invalid_type",  # Not in MESSAGE_TYPES
                    "content": "test",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    def test_get_capabilities_returns_valid_dict(self) -> None:
        """Verify external agent get_capabilities returns expected structure."""
        namespace: dict[str, Any] = {}
        exec(EXTERNAL_AGENT_CODE, namespace)
        cls = namespace["ExternalTestAgent"]

        caps = cls.get_capabilities()
        assert isinstance(caps, dict)
        assert "supports_streaming" in caps
        assert caps["supports_streaming"] is True


class TestExternalEmbeddingIsolation:
    """Test that embedding plugins work without any semantik imports."""

    def test_external_embedding_code_has_no_semantik_imports(self) -> None:
        """Verify the test code truly has no semantik imports."""
        assert "from semantik" not in EXTERNAL_EMBEDDING_CODE
        assert "import semantik" not in EXTERNAL_EMBEDDING_CODE
        assert "from shared" not in EXTERNAL_EMBEDDING_CODE
        assert "import shared" not in EXTERNAL_EMBEDDING_CODE

    def test_external_embedding_satisfies_protocol(self) -> None:
        """Verify external embedding class satisfies EmbeddingProtocol structurally."""
        namespace: dict[str, Any] = {}
        exec(EXTERNAL_EMBEDDING_CODE, namespace)
        cls = namespace["ExternalTestEmbedding"]

        assert hasattr(cls, "PLUGIN_ID")
        assert hasattr(cls, "PLUGIN_TYPE")
        assert hasattr(cls, "PLUGIN_VERSION")
        assert cls.PLUGIN_TYPE == "embedding"
        assert callable(getattr(cls, "embed_texts", None))
        assert callable(getattr(cls, "get_definition", None))
        assert callable(getattr(cls, "supports_model", None))
        assert callable(getattr(cls, "get_manifest", None))

    def test_loader_validates_external_embedding(self) -> None:
        """Verify the loader's _satisfies_protocol accepts external embedding."""
        from shared.plugins.loader import _satisfies_protocol

        namespace: dict[str, Any] = {}
        exec(EXTERNAL_EMBEDDING_CODE, namespace)
        cls = namespace["ExternalTestEmbedding"]

        assert _satisfies_protocol(cls, "embedding")

    @pytest.mark.asyncio()
    async def test_external_embedding_produces_valid_output(self) -> None:
        """Verify external embedding produces valid embedding vectors."""
        namespace: dict[str, Any] = {}
        exec(EXTERNAL_EMBEDDING_CODE, namespace)
        cls = namespace["ExternalTestEmbedding"]

        embedding = cls()
        result = await embedding.embed_texts(["test text"])

        assert len(result) == 1
        assert len(result[0]) == 384
        assert all(isinstance(v, float) for v in result[0])

    def test_get_definition_returns_valid_dict(self) -> None:
        """Verify external embedding get_definition returns expected structure."""
        namespace: dict[str, Any] = {}
        exec(EXTERNAL_EMBEDDING_CODE, namespace)
        cls = namespace["ExternalTestEmbedding"]

        definition = cls.get_definition()
        assert definition["api_id"] == "external-test-embedding"
        assert definition["internal_id"] == "external-test-embedding"
        assert "display_name" in definition
        assert "provider_type" in definition


class TestProtocolMixinsWithExternalPlugins:
    """Test that Protocol test mixins work with external plugins."""

    def test_connector_mixin_validates_external_plugin(self) -> None:
        """Verify ConnectorProtocolTestMixin validates external connector."""
        from shared.plugins.testing.contracts import ConnectorProtocolTestMixin

        namespace: dict[str, Any] = {}
        exec(EXTERNAL_CONNECTOR_CODE, namespace)
        connector_cls = namespace["ExternalTestConnector"]

        # Create a test class using the mixin
        class TestConnector(ConnectorProtocolTestMixin):
            plugin_class = connector_cls

        # Run the mixin tests
        test = TestConnector()
        test.test_satisfies_connector_protocol()
        test.test_has_required_class_vars()
        test.test_has_required_methods()
        test.test_get_manifest_returns_dict()

    def test_agent_mixin_validates_external_plugin(self) -> None:
        """Verify AgentProtocolTestMixin validates external agent."""
        from shared.plugins.testing.contracts import AgentProtocolTestMixin

        namespace: dict[str, Any] = {}
        exec(EXTERNAL_AGENT_CODE, namespace)
        agent_cls = namespace["ExternalTestAgent"]

        class TestAgent(AgentProtocolTestMixin):
            plugin_class = agent_cls

        test = TestAgent()
        test.test_satisfies_agent_protocol()
        test.test_has_required_class_vars()
        test.test_has_required_methods()
        test.test_get_manifest_returns_dict()

    def test_embedding_mixin_validates_external_plugin(self) -> None:
        """Verify EmbeddingProtocolTestMixin validates external embedding."""
        from shared.plugins.testing.contracts import EmbeddingProtocolTestMixin

        namespace: dict[str, Any] = {}
        exec(EXTERNAL_EMBEDDING_CODE, namespace)
        embedding_cls = namespace["ExternalTestEmbedding"]

        class TestEmbedding(EmbeddingProtocolTestMixin):
            plugin_class = embedding_cls

        test = TestEmbedding()
        test.test_satisfies_embedding_protocol()
        test.test_has_required_class_vars()
        test.test_has_required_methods()
        test.test_get_manifest_returns_dict()

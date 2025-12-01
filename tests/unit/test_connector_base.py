"""Unit tests for BaseConnector abstract class."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from shared.connectors.base import BaseConnector
from shared.dtos.ingestion import IngestedDocument
from shared.utils.hashing import compute_content_hash


class MockConnector(BaseConnector):
    """Mock connector for testing."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.auth_result = config.get("auth_result", True)
        self.documents: list[IngestedDocument] = config.get("documents", [])
        super().__init__(config)

    async def authenticate(self) -> bool:
        return self.auth_result

    async def load_documents(self) -> AsyncIterator[IngestedDocument]:
        for doc in self.documents:
            yield doc


class ValidatingConnector(BaseConnector):
    """Connector with config validation."""

    def validate_config(self) -> None:
        if "required_key" not in self._config:
            raise ValueError("Missing required_key in config")

    async def authenticate(self) -> bool:
        return True

    async def load_documents(self) -> AsyncIterator[IngestedDocument]:
        return
        yield  # Make this a generator


class TestBaseConnector:
    """Tests for BaseConnector ABC."""

    def test_config_stored(self) -> None:
        """Test that config is stored and accessible."""
        config: dict[str, Any] = {"key": "value"}
        connector = MockConnector(config)
        assert connector.config == config

    @pytest.mark.asyncio()
    async def test_authenticate_returns_true(self) -> None:
        """Test authenticate returns True on success."""
        connector = MockConnector({"auth_result": True})
        assert await connector.authenticate() is True

    @pytest.mark.asyncio()
    async def test_authenticate_returns_false(self) -> None:
        """Test authenticate returns False on failure."""
        connector = MockConnector({"auth_result": False})
        assert await connector.authenticate() is False

    @pytest.mark.asyncio()
    async def test_load_documents_yields_ingested_documents(self) -> None:
        """Test load_documents yields IngestedDocument objects."""
        content = "Test content"
        doc = IngestedDocument(
            content=content,
            unique_id="test://doc1",
            source_type="mock",
            metadata={"key": "value"},
            content_hash=compute_content_hash(content),
        )
        connector = MockConnector({"documents": [doc]})

        documents = [d async for d in connector.load_documents()]
        assert len(documents) == 1
        assert documents[0].unique_id == "test://doc1"
        assert documents[0].content == content

    @pytest.mark.asyncio()
    async def test_load_documents_empty(self) -> None:
        """Test load_documents yields nothing when no documents."""
        connector = MockConnector({"documents": []})

        documents = [d async for d in connector.load_documents()]
        assert len(documents) == 0

    @pytest.mark.asyncio()
    async def test_load_documents_multiple(self) -> None:
        """Test load_documents yields multiple documents."""
        docs = []
        for i in range(3):
            content = f"Test content {i}"
            docs.append(
                IngestedDocument(
                    content=content,
                    unique_id=f"test://doc{i}",
                    source_type="mock",
                    metadata={"index": i},
                    content_hash=compute_content_hash(content),
                )
            )
        connector = MockConnector({"documents": docs})

        documents = [d async for d in connector.load_documents()]
        assert len(documents) == 3
        assert documents[0].unique_id == "test://doc0"
        assert documents[1].unique_id == "test://doc1"
        assert documents[2].unique_id == "test://doc2"

    def test_validate_config_default_passes(self) -> None:
        """Test default validate_config accepts any config."""
        connector = MockConnector({})  # Empty config should be fine
        assert connector.config == {}

    def test_validate_config_raises_on_invalid(self) -> None:
        """Test validate_config can raise ValueError."""
        with pytest.raises(ValueError, match="Missing required_key"):
            ValidatingConnector({})

    def test_validate_config_passes_with_required_key(self) -> None:
        """Test validate_config passes when requirements met."""
        connector = ValidatingConnector({"required_key": "present"})
        assert connector.config["required_key"] == "present"

    def test_config_is_read_only_property(self) -> None:
        """Test config property returns the stored configuration."""
        config: dict[str, Any] = {"a": 1, "b": "two"}
        connector = MockConnector(config)
        assert connector.config is connector._config

"""Unit tests for BaseConnector abstract class."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from shared.connectors.base import BaseConnector
from shared.pipeline.types import FileReference


class MockConnector(BaseConnector):
    """Mock connector for testing."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.auth_result = config.get("auth_result", True)
        self.file_refs: list[FileReference] = config.get("file_refs", [])
        super().__init__(config)

    async def authenticate(self) -> bool:
        return self.auth_result

    async def enumerate(
        self,
        source_id: int | None = None,
    ) -> AsyncIterator[FileReference]:
        for ref in self.file_refs:
            yield ref

    async def load_content(self, file_ref: FileReference) -> bytes:
        return b"mock content"


class ValidatingConnector(BaseConnector):
    """Connector with config validation."""

    def validate_config(self) -> None:
        if "required_key" not in self._config:
            raise ValueError("Missing required_key in config")

    async def authenticate(self) -> bool:
        return True

    async def enumerate(
        self,
        source_id: int | None = None,
    ) -> AsyncIterator[FileReference]:
        return
        yield  # Make this a generator

    async def load_content(self, file_ref: FileReference) -> bytes:
        return b"mock content"


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
    async def test_enumerate_yields_file_references(self) -> None:
        """Test enumerate yields FileReference objects."""
        ref = FileReference(
            uri="test://doc1",
            source_type="mock",
            content_type="document",
            filename="doc1.txt",
            extension=".txt",
            mime_type="text/plain",
            size_bytes=100,
            change_hint="mtime:1234567890,size:100",
            metadata={"source": {"key": "value"}},
        )
        connector = MockConnector({"file_refs": [ref]})

        refs = [r async for r in connector.enumerate()]
        assert len(refs) == 1
        assert refs[0].uri == "test://doc1"
        assert refs[0].source_type == "mock"
        assert refs[0].content_type == "document"
        assert refs[0].filename == "doc1.txt"
        assert refs[0].change_hint == "mtime:1234567890,size:100"

    @pytest.mark.asyncio()
    async def test_enumerate_empty(self) -> None:
        """Test enumerate yields nothing when no files."""
        connector = MockConnector({"file_refs": []})

        refs = [r async for r in connector.enumerate()]
        assert len(refs) == 0

    @pytest.mark.asyncio()
    async def test_enumerate_multiple(self) -> None:
        """Test enumerate yields multiple file references."""
        refs_input = []
        for i in range(3):
            refs_input.append(
                FileReference(
                    uri=f"test://doc{i}",
                    source_type="mock",
                    content_type="document",
                    filename=f"doc{i}.txt",
                    extension=".txt",
                    mime_type="text/plain",
                    size_bytes=100 + i,
                    change_hint=f"mtime:123456789{i},size:{100 + i}",
                    metadata={"source": {"index": i}},
                )
            )
        connector = MockConnector({"file_refs": refs_input})

        refs = [r async for r in connector.enumerate()]
        assert len(refs) == 3
        assert refs[0].uri == "test://doc0"
        assert refs[1].uri == "test://doc1"
        assert refs[2].uri == "test://doc2"

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

    @pytest.mark.asyncio()
    async def test_load_documents_raises_not_implemented(self) -> None:
        """Test load_documents raises NotImplementedError with deprecation message."""
        connector = MockConnector({})
        with pytest.raises(NotImplementedError, match="deprecated"):
            [doc async for doc in connector.load_documents()]

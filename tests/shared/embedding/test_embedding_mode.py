"""Tests for EmbeddingMode enum and asymmetric embedding handling."""

from __future__ import annotations

from enum import Enum

import pytest

from shared.embedding.types import EmbeddingMode


class TestEmbeddingModeEnum:
    """Tests for the EmbeddingMode enum."""

    def test_embedding_mode_values(self) -> None:
        """Test that EmbeddingMode has expected values."""
        assert EmbeddingMode.QUERY.value == "query"
        assert EmbeddingMode.DOCUMENT.value == "document"

    def test_embedding_mode_is_enum(self) -> None:
        """Test that EmbeddingMode is an Enum."""
        assert isinstance(EmbeddingMode.QUERY, Enum)
        assert isinstance(EmbeddingMode.DOCUMENT, Enum)

    def test_embedding_mode_is_string_enum(self) -> None:
        """Test that EmbeddingMode is a string enum."""
        assert isinstance(EmbeddingMode.QUERY, str)
        assert isinstance(EmbeddingMode.DOCUMENT, str)

        # String comparison should work
        assert EmbeddingMode.QUERY == "query"
        assert EmbeddingMode.DOCUMENT == "document"

    def test_embedding_mode_members(self) -> None:
        """Test that only QUERY and DOCUMENT are defined."""
        members = list(EmbeddingMode)
        assert len(members) == 2
        assert EmbeddingMode.QUERY in members
        assert EmbeddingMode.DOCUMENT in members

    def test_embedding_mode_lookup_by_value(self) -> None:
        """Test looking up mode by value."""
        assert EmbeddingMode("query") == EmbeddingMode.QUERY
        assert EmbeddingMode("document") == EmbeddingMode.DOCUMENT

    def test_embedding_mode_invalid_value_raises(self) -> None:
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError, match="'invalid' is not a valid EmbeddingMode"):
            EmbeddingMode("invalid")


class TestEmbeddingModeUsage:
    """Tests for EmbeddingMode usage patterns."""

    def test_mode_in_function_signature(self) -> None:
        """Test that mode can be used as optional parameter."""

        def embed_with_mode(
            _text: str, mode: EmbeddingMode | None = None
        ) -> str:
            if mode is None:
                mode = EmbeddingMode.QUERY
            return f"Embedded with {mode.value} mode"

        result_default = embed_with_mode("test")
        assert "query" in result_default

        result_query = embed_with_mode("test", mode=EmbeddingMode.QUERY)
        assert "query" in result_query

        result_doc = embed_with_mode("test", mode=EmbeddingMode.DOCUMENT)
        assert "document" in result_doc

    def test_mode_comparison(self) -> None:
        """Test mode comparison operations."""
        mode = EmbeddingMode.QUERY

        assert mode == EmbeddingMode.QUERY
        assert mode != EmbeddingMode.DOCUMENT

        # String comparison
        assert mode == "query"
        assert mode != "document"

    def test_mode_in_dict(self) -> None:
        """Test that mode can be used as dict key."""
        config = {
            EmbeddingMode.QUERY: "query_prefix",
            EmbeddingMode.DOCUMENT: "doc_prefix",
        }

        assert config[EmbeddingMode.QUERY] == "query_prefix"
        assert config[EmbeddingMode.DOCUMENT] == "doc_prefix"

    def test_mode_serialization(self) -> None:
        """Test that mode can be serialized to string."""
        mode = EmbeddingMode.QUERY

        # Get string value via .value property
        assert mode.value == "query"

        # EmbeddingMode is a str enum, so direct comparison works
        assert mode == "query"

        # Can use .value in f-strings for clean output
        assert f"Mode: {mode.value}" == "Mode: query"

    def test_mode_from_string(self) -> None:
        """Test creating mode from string (e.g., from API input)."""
        # Simulate receiving string from API
        mode_str = "query"
        mode = EmbeddingMode(mode_str)
        assert mode == EmbeddingMode.QUERY

        mode_str = "document"
        mode = EmbeddingMode(mode_str)
        assert mode == EmbeddingMode.DOCUMENT

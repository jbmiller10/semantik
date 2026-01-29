"""Integration tests for the available-predicate-fields endpoint.

Tests the /api/v2/pipeline/available-predicate-fields endpoint which returns
dynamic predicate fields based on the source node of an edge in the DAG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from httpx import AsyncClient


@pytest.fixture()
def sample_dag_with_text_parser() -> dict:
    """DAG with a text parser node."""
    return {
        "id": "test-dag",
        "version": "1.0.0",
        "nodes": [
            {"id": "text_parser", "type": "parser", "plugin_id": "text", "config": {}},
            {"id": "recursive_chunker", "type": "chunker", "plugin_id": "recursive", "config": {}},
        ],
        "edges": [
            {"from_node": "_source", "to_node": "text_parser", "when": None},
            {"from_node": "text_parser", "to_node": "recursive_chunker", "when": None},
        ],
    }


@pytest.fixture()
def sample_dag_with_unstructured_parser() -> dict:
    """DAG with an unstructured parser node."""
    return {
        "id": "test-dag",
        "version": "1.0.0",
        "nodes": [
            {"id": "unstruct_parser", "type": "parser", "plugin_id": "unstructured", "config": {}},
            {"id": "recursive_chunker", "type": "chunker", "plugin_id": "recursive", "config": {}},
        ],
        "edges": [
            {"from_node": "_source", "to_node": "unstruct_parser", "when": None},
            {"from_node": "unstruct_parser", "to_node": "recursive_chunker", "when": None},
        ],
    }


@pytest.fixture()
def sample_dag_with_non_parser_node() -> dict:
    """DAG where we test routing from a non-parser node (chunker)."""
    return {
        "id": "test-dag",
        "version": "1.0.0",
        "nodes": [
            {"id": "text_parser", "type": "parser", "plugin_id": "text", "config": {}},
            {"id": "recursive_chunker", "type": "chunker", "plugin_id": "recursive", "config": {}},
            {"id": "embedder", "type": "embedder", "plugin_id": "default", "config": {}},
        ],
        "edges": [
            {"from_node": "_source", "to_node": "text_parser", "when": None},
            {"from_node": "text_parser", "to_node": "recursive_chunker", "when": None},
            {"from_node": "recursive_chunker", "to_node": "embedder", "when": None},
        ],
    }


class TestAvailablePredicateFieldsFromSource:
    """Tests for edges originating from _source node."""

    @pytest.mark.asyncio()
    async def test_fields_from_source_node_excludes_parsed(
        self, api_client: AsyncClient, api_auth_headers: dict, sample_dag_with_text_parser: dict
    ) -> None:
        """Verify that edges from _source do not include parsed.* fields."""
        response = await api_client.post(
            "/api/v2/pipeline/available-predicate-fields",
            headers=api_auth_headers,
            json={
                "dag": sample_dag_with_text_parser,
                "from_node": "_source",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "fields" in data

        # Should have source and detected fields
        categories = {f["category"] for f in data["fields"]}
        assert "source" in categories
        assert "detected" in categories

        # Should NOT have parsed fields (parser hasn't run yet)
        assert "parsed" not in categories

    @pytest.mark.asyncio()
    async def test_always_includes_source_fields(
        self, api_client: AsyncClient, api_auth_headers: dict, sample_dag_with_text_parser: dict
    ) -> None:
        """Verify that source.* fields are always included."""
        response = await api_client.post(
            "/api/v2/pipeline/available-predicate-fields",
            headers=api_auth_headers,
            json={
                "dag": sample_dag_with_text_parser,
                "from_node": "_source",
            },
        )

        assert response.status_code == 200
        data = response.json()

        source_fields = [f for f in data["fields"] if f["category"] == "source"]
        source_values = {f["value"] for f in source_fields}

        # Should include all expected source fields
        # NOTE: Source fields are top-level FileReference attributes, not nested under metadata.source
        expected_source_fields = {
            "mime_type",
            "extension",
            "source_type",
            "content_type",
        }
        assert expected_source_fields.issubset(source_values)

    @pytest.mark.asyncio()
    async def test_always_includes_detected_fields(
        self, api_client: AsyncClient, api_auth_headers: dict, sample_dag_with_text_parser: dict
    ) -> None:
        """Verify that detected.* fields are always included."""
        response = await api_client.post(
            "/api/v2/pipeline/available-predicate-fields",
            headers=api_auth_headers,
            json={
                "dag": sample_dag_with_text_parser,
                "from_node": "_source",
            },
        )

        assert response.status_code == 200
        data = response.json()

        detected_fields = [f for f in data["fields"] if f["category"] == "detected"]
        detected_values = {f["value"] for f in detected_fields}

        # Should include all expected detected fields
        expected_detected_fields = {
            "metadata.detected.is_scanned_pdf",
            "metadata.detected.is_code",
            "metadata.detected.is_structured_data",
        }
        assert expected_detected_fields.issubset(detected_values)


class TestAvailablePredicateFieldsFromParser:
    """Tests for edges originating from parser nodes."""

    @pytest.mark.asyncio()
    async def test_fields_from_text_parser_includes_parsed(
        self, api_client: AsyncClient, api_auth_headers: dict, sample_dag_with_text_parser: dict
    ) -> None:
        """Verify that edges from text parser include its parsed.* fields."""
        response = await api_client.post(
            "/api/v2/pipeline/available-predicate-fields",
            headers=api_auth_headers,
            json={
                "dag": sample_dag_with_text_parser,
                "from_node": "text_parser",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should have all three categories
        categories = {f["category"] for f in data["fields"]}
        assert "source" in categories
        assert "detected" in categories
        assert "parsed" in categories

        # Should include text parser's emitted fields
        parsed_fields = [f for f in data["fields"] if f["category"] == "parsed"]
        parsed_values = {f["value"] for f in parsed_fields}

        expected_text_parser_fields = {
            "metadata.parsed.detected_language",
            "metadata.parsed.approx_token_count",
            "metadata.parsed.line_count",
            "metadata.parsed.has_code_blocks",
        }
        assert expected_text_parser_fields == parsed_values

    @pytest.mark.asyncio()
    async def test_fields_from_unstructured_parser_includes_parsed(
        self, api_client: AsyncClient, api_auth_headers: dict, sample_dag_with_unstructured_parser: dict
    ) -> None:
        """Verify that edges from unstructured parser include its parsed.* fields."""
        response = await api_client.post(
            "/api/v2/pipeline/available-predicate-fields",
            headers=api_auth_headers,
            json={
                "dag": sample_dag_with_unstructured_parser,
                "from_node": "unstruct_parser",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should have all three categories
        categories = {f["category"] for f in data["fields"]}
        assert "source" in categories
        assert "detected" in categories
        assert "parsed" in categories

        # Should include unstructured parser's emitted fields
        parsed_fields = [f for f in data["fields"] if f["category"] == "parsed"]
        parsed_values = {f["value"] for f in parsed_fields}

        expected_unstructured_fields = {
            "metadata.parsed.page_count",
            "metadata.parsed.has_tables",
            "metadata.parsed.has_images",
            "metadata.parsed.element_types",
            "metadata.parsed.approx_token_count",
        }
        assert expected_unstructured_fields == parsed_values


class TestAvailablePredicateFieldsFromNonParser:
    """Tests for edges originating from non-parser nodes."""

    @pytest.mark.asyncio()
    async def test_fields_from_chunker_excludes_parsed(
        self, api_client: AsyncClient, api_auth_headers: dict, sample_dag_with_non_parser_node: dict
    ) -> None:
        """Verify that edges from non-parser nodes don't include parsed.* fields."""
        response = await api_client.post(
            "/api/v2/pipeline/available-predicate-fields",
            headers=api_auth_headers,
            json={
                "dag": sample_dag_with_non_parser_node,
                "from_node": "recursive_chunker",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should have source and detected fields
        categories = {f["category"] for f in data["fields"]}
        assert "source" in categories
        assert "detected" in categories

        # Should NOT have parsed fields (chunker is not a parser)
        assert "parsed" not in categories


class TestAvailablePredicateFieldsUnknownParser:
    """Tests for edges with unknown parser plugin IDs."""

    @pytest.mark.asyncio()
    async def test_fallback_for_unknown_parser(self, api_client: AsyncClient, api_auth_headers: dict) -> None:
        """Verify that unknown parser returns empty parsed.* fields."""
        dag = {
            "id": "test-dag",
            "version": "1.0.0",
            "nodes": [
                {"id": "custom_parser", "type": "parser", "plugin_id": "nonexistent_parser", "config": {}},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "custom_parser", "when": None},
            ],
        }

        response = await api_client.post(
            "/api/v2/pipeline/available-predicate-fields",
            headers=api_auth_headers,
            json={
                "dag": dag,
                "from_node": "custom_parser",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should have source and detected fields
        categories = {f["category"] for f in data["fields"]}
        assert "source" in categories
        assert "detected" in categories

        # Should NOT have parsed fields (unknown parser has no EMITTED_FIELDS)
        assert "parsed" not in categories


class TestAvailablePredicateFieldsAuthentication:
    """Tests for authentication requirements."""

    @pytest.mark.asyncio()
    async def test_requires_authentication(
        self, api_client_unauthenticated: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that the endpoint requires authentication."""
        # Disable the auth bypass that's enabled in the test environment
        monkeypatch.setattr("webui.auth.settings.DISABLE_AUTH", False)

        response = await api_client_unauthenticated.post(
            "/api/v2/pipeline/available-predicate-fields",
            json={
                "dag": {"id": "test", "version": "1.0", "nodes": [], "edges": []},
                "from_node": "_source",
            },
        )

        assert response.status_code == 401

"""Tests for v2 template endpoints.

Uses shared fixtures from conftest.py (api_client, api_client_unauthenticated).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.fixture(autouse=True)
def _load_plugins():
    """Ensure plugins are loaded for template validation."""
    from shared.pipeline.templates import clear_cache
    from shared.plugins.loader import load_plugins

    load_plugins(plugin_types={"embedding", "chunking", "connector", "extractor", "parser"})
    clear_cache()
    yield
    clear_cache()


class TestListTemplates:
    """Tests for GET /api/v2/templates endpoint."""

    @pytest.mark.asyncio()
    async def test_list_templates_returns_all(self, api_client: AsyncClient) -> None:
        """Should return all available templates."""
        response = await api_client.get("/api/v2/templates")

        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert "total" in data
        assert data["total"] == 5  # We have 5 built-in templates
        assert len(data["templates"]) == 5

    @pytest.mark.asyncio()
    async def test_list_templates_contains_expected_templates(
        self, api_client: AsyncClient
    ) -> None:
        """Should contain all expected template IDs."""
        response = await api_client.get("/api/v2/templates")

        assert response.status_code == 200
        data = response.json()
        template_ids = {t["id"] for t in data["templates"]}
        expected_ids = {
            "academic-papers",
            "codebase",
            "documentation",
            "email-archive",
            "mixed-documents",
        }
        assert template_ids == expected_ids

    @pytest.mark.asyncio()
    async def test_list_templates_summary_fields(
        self, api_client: AsyncClient
    ) -> None:
        """Each template summary should have required fields."""
        response = await api_client.get("/api/v2/templates")

        assert response.status_code == 200
        data = response.json()
        for template in data["templates"]:
            assert "id" in template
            assert "name" in template
            assert "description" in template
            assert "suggested_for" in template
            # Should NOT include pipeline DAG in list response
            assert "pipeline" not in template

    @pytest.mark.asyncio()
    async def test_list_templates_requires_auth(
        self, api_client_unauthenticated: AsyncClient
    ) -> None:
        """Should require authentication."""
        response = await api_client_unauthenticated.get("/api/v2/templates")
        assert response.status_code == 401


class TestGetTemplate:
    """Tests for GET /api/v2/templates/{template_id} endpoint."""

    @pytest.mark.asyncio()
    async def test_get_template_returns_full_details(
        self, api_client: AsyncClient
    ) -> None:
        """Should return full template with pipeline DAG."""
        response = await api_client.get("/api/v2/templates/academic-papers")

        assert response.status_code == 200
        data = response.json()

        # Basic fields
        assert data["id"] == "academic-papers"
        assert data["name"] == "Academic Papers"
        assert "description" in data
        assert "suggested_for" in data

        # Pipeline DAG
        assert "pipeline" in data
        pipeline = data["pipeline"]
        assert "id" in pipeline
        assert "version" in pipeline
        assert "nodes" in pipeline
        assert "edges" in pipeline
        assert len(pipeline["nodes"]) > 0
        assert len(pipeline["edges"]) > 0

        # Tunable parameters
        assert "tunable" in data
        assert isinstance(data["tunable"], list)

    @pytest.mark.asyncio()
    async def test_get_template_pipeline_nodes_have_required_fields(
        self, api_client: AsyncClient
    ) -> None:
        """Pipeline nodes should have all required fields."""
        response = await api_client.get("/api/v2/templates/codebase")

        assert response.status_code == 200
        data = response.json()

        for node in data["pipeline"]["nodes"]:
            assert "id" in node
            assert "type" in node
            assert "plugin_id" in node
            assert "config" in node

    @pytest.mark.asyncio()
    async def test_get_template_pipeline_edges_have_required_fields(
        self, api_client: AsyncClient
    ) -> None:
        """Pipeline edges should have required fields."""
        response = await api_client.get("/api/v2/templates/documentation")

        assert response.status_code == 200
        data = response.json()

        for edge in data["pipeline"]["edges"]:
            assert "from_node" in edge
            assert "to_node" in edge
            # 'when' is optional, may be null

    @pytest.mark.asyncio()
    async def test_get_template_tunable_params_have_required_fields(
        self, api_client: AsyncClient
    ) -> None:
        """Tunable parameters should have required fields."""
        response = await api_client.get("/api/v2/templates/academic-papers")

        assert response.status_code == 200
        data = response.json()

        # Academic papers template should have tunable params
        assert len(data["tunable"]) > 0

        for param in data["tunable"]:
            assert "path" in param
            assert "description" in param
            assert "default" in param
            # 'range' and 'options' are optional

    @pytest.mark.asyncio()
    async def test_get_template_not_found(self, api_client: AsyncClient) -> None:
        """Should return 404 for unknown template ID."""
        response = await api_client.get("/api/v2/templates/nonexistent-template")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "nonexistent-template" in data["detail"]

    @pytest.mark.asyncio()
    async def test_get_template_requires_auth(
        self, api_client_unauthenticated: AsyncClient
    ) -> None:
        """Should require authentication."""
        response = await api_client_unauthenticated.get(
            "/api/v2/templates/academic-papers"
        )
        assert response.status_code == 401

    @pytest.mark.asyncio()
    async def test_get_all_templates_individually(
        self, api_client: AsyncClient
    ) -> None:
        """Should be able to fetch each template by ID."""
        template_ids = [
            "academic-papers",
            "codebase",
            "documentation",
            "email-archive",
            "mixed-documents",
        ]

        for template_id in template_ids:
            response = await api_client.get(f"/api/v2/templates/{template_id}")
            assert response.status_code == 200, f"Failed for {template_id}"
            data = response.json()
            assert data["id"] == template_id


class TestTemplateValidation:
    """Tests for template validation behavior."""

    @pytest.mark.asyncio()
    async def test_all_templates_have_embedder_node(
        self, api_client: AsyncClient
    ) -> None:
        """All templates should have at least one embedder node."""
        response = await api_client.get("/api/v2/templates")
        assert response.status_code == 200

        for summary in response.json()["templates"]:
            detail_response = await api_client.get(
                f"/api/v2/templates/{summary['id']}"
            )
            assert detail_response.status_code == 200
            data = detail_response.json()

            node_types = [n["type"] for n in data["pipeline"]["nodes"]]
            assert (
                "embedder" in node_types
            ), f"Template {summary['id']} missing embedder node"

    @pytest.mark.asyncio()
    async def test_all_templates_have_source_edge(
        self, api_client: AsyncClient
    ) -> None:
        """All templates should have at least one edge from _source."""
        response = await api_client.get("/api/v2/templates")
        assert response.status_code == 200

        for summary in response.json()["templates"]:
            detail_response = await api_client.get(
                f"/api/v2/templates/{summary['id']}"
            )
            assert detail_response.status_code == 200
            data = detail_response.json()

            from_nodes = {e["from_node"] for e in data["pipeline"]["edges"]}
            assert (
                "_source" in from_nodes
            ), f"Template {summary['id']} missing _source edge"

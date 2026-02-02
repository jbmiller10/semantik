"""Unit tests for template discovery tools."""

from unittest.mock import patch

import pytest

from shared.pipeline.templates.types import PipelineTemplate, TunableParameter
from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode
from webui.services.agent.tools.templates import GetTemplateDetailsTool, ListTemplatesTool


@pytest.fixture()
def sample_pipeline_dag():
    """Create a sample pipeline DAG."""
    return PipelineDAG(
        id="test-template",
        version="1.0",
        nodes=[
            PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text-parser", config={}),
            PipelineNode(
                id="chunker",
                type=NodeType.CHUNKER,
                plugin_id="semantic-chunker",
                config={"max_tokens": 512},
            ),
            PipelineNode(
                id="embedder",
                type=NodeType.EMBEDDER,
                plugin_id="dense-embedding",
                config={"model": "BAAI/bge-base-en-v1.5"},
            ),
        ],
        edges=[
            PipelineEdge(from_node="_source", to_node="parser"),
            PipelineEdge(from_node="parser", to_node="chunker"),
            PipelineEdge(from_node="chunker", to_node="embedder"),
        ],
    )


@pytest.fixture()
def sample_template(sample_pipeline_dag):
    """Create a sample pipeline template."""
    return PipelineTemplate(
        id="academic-papers",
        name="Academic Papers",
        description="Template for processing academic papers and research documents",
        suggested_for=["PDF", "research", "academic"],
        pipeline=sample_pipeline_dag,
        tunable=[
            TunableParameter(
                path="nodes.chunker.config.max_tokens",
                description="Maximum tokens per chunk",
                default=512,
                range=(128, 2048),
            ),
            TunableParameter(
                path="nodes.chunker.config.overlap",
                description="Overlap between chunks",
                default=50,
                range=(0, 200),
            ),
        ],
    )


@pytest.fixture()
def sample_template_codebase():
    """Create a second sample template."""
    return PipelineTemplate(
        id="codebase",
        name="Codebase",
        description="Template for processing source code repositories",
        suggested_for=["code", "programming", "repository"],
        pipeline=PipelineDAG(
            id="codebase",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="code-parser"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="code-chunker"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="code-embedding"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        ),
        tunable=[],
    )


class TestListTemplatesTool:
    """Tests for ListTemplatesTool."""

    def test_schema_is_valid(self):
        """Test that the tool schema is properly defined."""
        tool = ListTemplatesTool(context={})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "list_templates"
        assert "suggested_for" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio()
    async def test_list_all_templates(self, sample_template, sample_template_codebase):
        """Test listing all templates without filter."""
        with patch("shared.pipeline.templates.list_templates") as mock_list:
            mock_list.return_value = [sample_template, sample_template_codebase]

            tool = ListTemplatesTool(context={})
            result = await tool.execute()

            assert result["count"] == 2
            assert len(result["templates"]) == 2
            assert result["filter"] is None

    @pytest.mark.asyncio()
    async def test_list_templates_with_filter(self, sample_template, sample_template_codebase):
        """Test listing templates filtered by suggested_for."""
        with patch("shared.pipeline.templates.list_templates") as mock_list:
            mock_list.return_value = [sample_template, sample_template_codebase]

            tool = ListTemplatesTool(context={})
            result = await tool.execute(suggested_for="PDF")

            assert result["count"] == 1
            assert result["filter"] == "PDF"
            assert result["templates"][0]["id"] == "academic-papers"

    @pytest.mark.asyncio()
    async def test_filter_case_insensitive(self, sample_template, sample_template_codebase):
        """Test that filter is case-insensitive."""
        with patch("shared.pipeline.templates.list_templates") as mock_list:
            mock_list.return_value = [sample_template, sample_template_codebase]

            tool = ListTemplatesTool(context={})
            result = await tool.execute(suggested_for="CODE")

            assert result["count"] == 1
            assert result["templates"][0]["id"] == "codebase"

    @pytest.mark.asyncio()
    async def test_template_info_includes_tunables(self, sample_template):
        """Test that tunable parameters are included."""
        with patch("shared.pipeline.templates.list_templates") as mock_list:
            mock_list.return_value = [sample_template]

            tool = ListTemplatesTool(context={})
            result = await tool.execute()

            template = result["templates"][0]
            assert len(template["tunable_parameters"]) == 2
            assert template["tunable_parameters"][0]["path"] == "nodes.chunker.config.max_tokens"
            assert template["tunable_parameters"][0]["default"] == 512
            assert template["tunable_parameters"][0]["range"] == [128, 2048]

    @pytest.mark.asyncio()
    async def test_template_info_includes_structure_counts(self, sample_template):
        """Test that node and edge counts are included."""
        with patch("shared.pipeline.templates.list_templates") as mock_list:
            mock_list.return_value = [sample_template]

            tool = ListTemplatesTool(context={})
            result = await tool.execute()

            template = result["templates"][0]
            assert template["node_count"] == 3
            assert template["edge_count"] == 3

    @pytest.mark.asyncio()
    async def test_handles_validation_error(self):
        """Test that validation errors are handled."""
        with patch("shared.pipeline.templates.list_templates") as mock_list:
            mock_list.side_effect = ValueError("Template validation failed")

            tool = ListTemplatesTool(context={})
            result = await tool.execute()

            assert "error" in result
            assert "validation" in result["error"].lower()
            assert result["count"] == 0

    @pytest.mark.asyncio()
    async def test_handles_general_exception(self):
        """Test that general exceptions are handled."""
        with patch("shared.pipeline.templates.list_templates") as mock_list:
            mock_list.side_effect = Exception("Unknown error")

            tool = ListTemplatesTool(context={})
            result = await tool.execute()

            assert "error" in result
            assert result["count"] == 0


class TestGetTemplateDetailsTool:
    """Tests for GetTemplateDetailsTool."""

    def test_schema_is_valid(self):
        """Test that the tool schema is properly defined."""
        tool = GetTemplateDetailsTool(context={})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_template_details"
        assert "template_id" in schema["function"]["parameters"]["properties"]
        assert "template_id" in schema["function"]["parameters"]["required"]

    @pytest.mark.asyncio()
    async def test_get_existing_template(self, sample_template):
        """Test getting details for an existing template."""
        with patch("shared.pipeline.templates.load_template") as mock_load:
            mock_load.return_value = sample_template

            tool = GetTemplateDetailsTool(context={})
            result = await tool.execute(template_id="academic-papers")

            assert result["found"] is True
            assert result["id"] == "academic-papers"
            assert result["name"] == "Academic Papers"
            assert "PDF" in result["suggested_for"]

    @pytest.mark.asyncio()
    async def test_includes_full_pipeline(self, sample_template):
        """Test that full pipeline structure is included."""
        with patch("shared.pipeline.templates.load_template") as mock_load:
            mock_load.return_value = sample_template

            tool = GetTemplateDetailsTool(context={})
            result = await tool.execute(template_id="academic-papers")

            # Check pipeline field
            assert "pipeline" in result
            assert result["pipeline"]["id"] == "test-template"

            # Check nodes
            assert len(result["nodes"]) == 3
            node_ids = [n["id"] for n in result["nodes"]]
            assert "parser" in node_ids
            assert "chunker" in node_ids
            assert "embedder" in node_ids

            # Check node details
            chunker = next(n for n in result["nodes"] if n["id"] == "chunker")
            assert chunker["type"] == "chunker"
            assert chunker["plugin_id"] == "semantic-chunker"
            assert chunker["config"]["max_tokens"] == 512

    @pytest.mark.asyncio()
    async def test_includes_edges(self, sample_template):
        """Test that edges are included with routing info."""
        with patch("shared.pipeline.templates.load_template") as mock_load:
            mock_load.return_value = sample_template

            tool = GetTemplateDetailsTool(context={})
            result = await tool.execute(template_id="academic-papers")

            assert len(result["edges"]) == 3
            edge_pairs = [(e["from"], e["to"]) for e in result["edges"]]
            assert ("_source", "parser") in edge_pairs
            assert ("parser", "chunker") in edge_pairs
            assert ("chunker", "embedder") in edge_pairs

    @pytest.mark.asyncio()
    async def test_includes_tunable_parameters(self, sample_template):
        """Test that tunable parameters are included with full details."""
        with patch("shared.pipeline.templates.load_template") as mock_load:
            mock_load.return_value = sample_template

            tool = GetTemplateDetailsTool(context={})
            result = await tool.execute(template_id="academic-papers")

            assert len(result["tunable_parameters"]) == 2
            tunable = result["tunable_parameters"][0]
            assert tunable["path"] == "nodes.chunker.config.max_tokens"
            assert tunable["description"] == "Maximum tokens per chunk"
            assert tunable["default"] == 512
            assert tunable["range"] == [128, 2048]

    @pytest.mark.asyncio()
    async def test_get_nonexistent_template(self, sample_template):
        """Test getting details for a non-existent template."""
        with (
            patch("shared.pipeline.templates.load_template") as mock_load,
            patch("shared.pipeline.templates.list_templates") as mock_list,
        ):
            mock_load.return_value = None
            mock_list.return_value = [sample_template]

            tool = GetTemplateDetailsTool(context={})
            result = await tool.execute(template_id="nonexistent")

            assert result["found"] is False
            assert "not found" in result["error"]
            assert "academic-papers" in result["available_templates"]

    @pytest.mark.asyncio()
    async def test_handles_validation_error(self):
        """Test that validation errors are handled."""
        with patch("shared.pipeline.templates.load_template") as mock_load:
            mock_load.side_effect = ValueError("Template validation failed")

            tool = GetTemplateDetailsTool(context={})
            result = await tool.execute(template_id="test")

            assert result["found"] is False
            assert "validation" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_handles_general_exception(self):
        """Test that general exceptions are handled."""
        with patch("shared.pipeline.templates.load_template") as mock_load:
            mock_load.side_effect = Exception("Unknown error")

            tool = GetTemplateDetailsTool(context={})
            result = await tool.execute(template_id="test")

            assert result["found"] is False
            assert "error" in result

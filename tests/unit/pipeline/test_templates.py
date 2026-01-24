"""Unit tests for pipeline templates."""

from __future__ import annotations

import pytest

from shared.pipeline.templates.types import (
    PipelineTemplate,
    TunableParameter,
    resolve_tunable_path,
)
from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode


class TestTunableParameter:
    """Tests for TunableParameter dataclass."""

    def test_basic_creation(self) -> None:
        """Should create a tunable parameter with required fields."""
        param = TunableParameter(
            path="nodes.chunker.config.max_tokens",
            description="Maximum tokens per chunk",
            default=512,
        )
        assert param.path == "nodes.chunker.config.max_tokens"
        assert param.description == "Maximum tokens per chunk"
        assert param.default == 512
        assert param.range is None
        assert param.options is None

    def test_with_range(self) -> None:
        """Should create a tunable parameter with numeric range."""
        param = TunableParameter(
            path="nodes.chunker.config.max_tokens",
            description="Maximum tokens per chunk",
            default=512,
            range=(256, 1024),
        )
        assert param.range == (256, 1024)

    def test_with_options(self) -> None:
        """Should create a tunable parameter with string options."""
        param = TunableParameter(
            path="nodes.parser.config.strategy",
            description="Parsing strategy",
            default="fast",
            options=["fast", "hi_res", "auto"],
        )
        assert param.options == ["fast", "hi_res", "auto"]

    def test_to_dict_basic(self) -> None:
        """Should serialize to dict with required fields only."""
        param = TunableParameter(
            path="nodes.chunker.config.max_tokens",
            description="Maximum tokens per chunk",
            default=512,
        )
        result = param.to_dict()
        assert result == {
            "path": "nodes.chunker.config.max_tokens",
            "description": "Maximum tokens per chunk",
            "default": 512,
        }
        # Should not include None values for range/options
        assert "range" not in result
        assert "options" not in result

    def test_to_dict_with_all_fields(self) -> None:
        """Should serialize all fields when present."""
        param = TunableParameter(
            path="nodes.chunker.config.max_tokens",
            description="Maximum tokens per chunk",
            default=512,
            range=(256, 1024),
            options=None,
        )
        result = param.to_dict()
        assert result["range"] == [256, 1024]  # tuple becomes list

    def test_from_dict(self) -> None:
        """Should deserialize from dict."""
        data = {
            "path": "nodes.chunker.config.max_tokens",
            "description": "Maximum tokens per chunk",
            "default": 512,
            "range": [256, 1024],
        }
        param = TunableParameter.from_dict(data)
        assert param.path == "nodes.chunker.config.max_tokens"
        assert param.default == 512
        assert param.range == (256, 1024)  # list becomes tuple

    def test_immutability(self) -> None:
        """Should be immutable (frozen dataclass)."""
        param = TunableParameter(
            path="nodes.chunker.config.max_tokens",
            description="Maximum tokens per chunk",
            default=512,
        )
        with pytest.raises(AttributeError):
            param.default = 1024  # type: ignore[misc]


class TestPipelineTemplate:
    """Tests for PipelineTemplate dataclass."""

    @pytest.fixture()
    def simple_dag(self) -> PipelineDAG:
        """Create a simple DAG for testing."""
        return PipelineDAG(
            id="test-dag",
            version="1.0",
            nodes=[
                PipelineNode(
                    id="parser",
                    type=NodeType.PARSER,
                    plugin_id="text",
                    config={"encoding": "utf-8"},
                ),
                PipelineNode(
                    id="chunker",
                    type=NodeType.CHUNKER,
                    plugin_id="recursive",
                    config={"chunk_size": 1000, "chunk_overlap": 100},
                ),
                PipelineNode(
                    id="embedder",
                    type=NodeType.EMBEDDER,
                    plugin_id="dense_local",
                    config={},
                ),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )

    def test_basic_creation(self, simple_dag: PipelineDAG) -> None:
        """Should create a template with required fields."""
        template = PipelineTemplate(
            id="test-template",
            name="Test Template",
            description="A template for testing",
            suggested_for=["test", "demo"],
            pipeline=simple_dag,
        )
        assert template.id == "test-template"
        assert template.name == "Test Template"
        assert template.description == "A template for testing"
        assert template.suggested_for == ["test", "demo"]
        assert template.pipeline == simple_dag
        assert template.tunable == []

    def test_with_tunable_params(self, simple_dag: PipelineDAG) -> None:
        """Should create a template with tunable parameters."""
        tunable = [
            TunableParameter(
                path="nodes.chunker.config.chunk_size",
                description="Chunk size",
                default=1000,
                range=(500, 2000),
            ),
        ]
        template = PipelineTemplate(
            id="test-template",
            name="Test Template",
            description="A template for testing",
            suggested_for=["test"],
            pipeline=simple_dag,
            tunable=tunable,
        )
        assert len(template.tunable) == 1
        assert template.tunable[0].path == "nodes.chunker.config.chunk_size"

    def test_to_dict(self, simple_dag: PipelineDAG) -> None:
        """Should serialize template to dict."""
        template = PipelineTemplate(
            id="test-template",
            name="Test Template",
            description="A template for testing",
            suggested_for=["test", "demo"],
            pipeline=simple_dag,
            tunable=[
                TunableParameter(
                    path="nodes.chunker.config.chunk_size",
                    description="Chunk size",
                    default=1000,
                ),
            ],
        )
        result = template.to_dict()
        assert result["id"] == "test-template"
        assert result["name"] == "Test Template"
        assert result["suggested_for"] == ["test", "demo"]
        assert "pipeline" in result
        assert result["pipeline"]["id"] == "test-dag"
        assert len(result["tunable"]) == 1

    def test_from_dict(self, simple_dag: PipelineDAG) -> None:
        """Should deserialize template from dict."""
        data = {
            "id": "test-template",
            "name": "Test Template",
            "description": "A template for testing",
            "suggested_for": ["test"],
            "pipeline": simple_dag.to_dict(),
            "tunable": [],
        }
        template = PipelineTemplate.from_dict(data)
        assert template.id == "test-template"
        assert template.pipeline.id == "test-dag"

    def test_round_trip_serialization(self, simple_dag: PipelineDAG) -> None:
        """Should round-trip through serialization."""
        original = PipelineTemplate(
            id="test-template",
            name="Test Template",
            description="A template for testing",
            suggested_for=["test", "demo"],
            pipeline=simple_dag,
            tunable=[
                TunableParameter(
                    path="nodes.chunker.config.chunk_size",
                    description="Chunk size",
                    default=1000,
                    range=(500, 2000),
                ),
            ],
        )
        data = original.to_dict()
        restored = PipelineTemplate.from_dict(data)
        assert restored.id == original.id
        assert restored.name == original.name
        assert len(restored.tunable) == len(original.tunable)


class TestResolveTunablePath:
    """Tests for resolve_tunable_path function."""

    @pytest.fixture()
    def template_with_nodes(self) -> PipelineTemplate:
        """Create a template with multiple nodes for path resolution tests."""
        dag = PipelineDAG(
            id="test-dag",
            version="1.0",
            nodes=[
                PipelineNode(
                    id="parser",
                    type=NodeType.PARSER,
                    plugin_id="text",
                    config={"encoding": "utf-8"},
                ),
                PipelineNode(
                    id="chunker",
                    type=NodeType.CHUNKER,
                    plugin_id="recursive",
                    config={"chunk_size": 1000, "chunk_overlap": 100},
                ),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
            ],
        )
        return PipelineTemplate(
            id="test-template",
            name="Test Template",
            description="A template for testing",
            suggested_for=["test"],
            pipeline=dag,
        )

    def test_valid_path(self, template_with_nodes: PipelineTemplate) -> None:
        """Should resolve valid path to node and config key."""
        node, config_key = resolve_tunable_path(
            template_with_nodes, "nodes.chunker.config.chunk_size"
        )
        assert node is not None
        assert node.id == "chunker"
        assert config_key == "chunk_size"

    def test_valid_path_different_node(
        self, template_with_nodes: PipelineTemplate
    ) -> None:
        """Should resolve path for different node."""
        node, config_key = resolve_tunable_path(
            template_with_nodes, "nodes.parser.config.encoding"
        )
        assert node is not None
        assert node.id == "parser"
        assert config_key == "encoding"

    def test_invalid_path_wrong_format(
        self, template_with_nodes: PipelineTemplate
    ) -> None:
        """Should return None for paths with wrong format."""
        # Too few parts
        node, config_key = resolve_tunable_path(template_with_nodes, "nodes.chunker")
        assert node is None
        assert config_key is None

        # Too many parts
        node, config_key = resolve_tunable_path(
            template_with_nodes, "nodes.chunker.config.chunk_size.extra"
        )
        assert node is None
        assert config_key is None

    def test_invalid_path_wrong_prefix(
        self, template_with_nodes: PipelineTemplate
    ) -> None:
        """Should return None for paths not starting with 'nodes'."""
        node, config_key = resolve_tunable_path(
            template_with_nodes, "edges.chunker.config.chunk_size"
        )
        assert node is None
        assert config_key is None

    def test_invalid_path_missing_config(
        self, template_with_nodes: PipelineTemplate
    ) -> None:
        """Should return None for paths without 'config' in position 2."""
        node, config_key = resolve_tunable_path(
            template_with_nodes, "nodes.chunker.settings.chunk_size"
        )
        assert node is None
        assert config_key is None

    def test_invalid_path_unknown_node(
        self, template_with_nodes: PipelineTemplate
    ) -> None:
        """Should return None for paths referencing unknown nodes."""
        node, config_key = resolve_tunable_path(
            template_with_nodes, "nodes.unknown_node.config.chunk_size"
        )
        assert node is None
        assert config_key is None

    def test_empty_path(self, template_with_nodes: PipelineTemplate) -> None:
        """Should return None for empty path."""
        node, config_key = resolve_tunable_path(template_with_nodes, "")
        assert node is None
        assert config_key is None


class TestTemplateModule:
    """Tests for template module functions (list_templates, load_template)."""

    def test_list_templates_returns_list(self) -> None:
        """Should return a list of templates."""
        # Note: This will trigger template loading and validation,
        # so we need the plugin registry to be available.
        # For unit tests, we may need to mock the registry.
        # Let's test that it returns a non-empty list when plugins are loaded.
        from shared.plugins.loader import load_plugins

        # Load plugins so validation can succeed
        load_plugins(plugin_types={"embedding", "chunking", "connector", "extractor", "parser"})

        from shared.pipeline.templates import clear_cache, list_templates

        clear_cache()  # Clear any cached templates
        templates = list_templates()

        assert isinstance(templates, list)
        assert len(templates) == 5  # We created 5 templates

    def test_load_template_by_id(self) -> None:
        """Should load a specific template by ID."""
        from shared.plugins.loader import load_plugins

        load_plugins(plugin_types={"embedding", "chunking", "connector", "extractor", "parser"})

        from shared.pipeline.templates import clear_cache, load_template

        clear_cache()
        template = load_template("academic-papers")

        assert template is not None
        assert template.id == "academic-papers"
        assert template.name == "Academic Papers"

    def test_load_template_unknown_id(self) -> None:
        """Should return None for unknown template ID."""
        from shared.plugins.loader import load_plugins

        load_plugins(plugin_types={"embedding", "chunking", "connector", "extractor", "parser"})

        from shared.pipeline.templates import clear_cache, load_template

        clear_cache()
        template = load_template("nonexistent-template")

        assert template is None

    def test_template_ids_are_unique(self) -> None:
        """All templates should have unique IDs."""
        from shared.plugins.loader import load_plugins

        load_plugins(plugin_types={"embedding", "chunking", "connector", "extractor", "parser"})

        from shared.pipeline.templates import clear_cache, list_templates

        clear_cache()
        templates = list_templates()
        template_ids = [t.id for t in templates]

        assert len(template_ids) == len(set(template_ids))

    def test_all_templates_have_valid_pipelines(self) -> None:
        """All templates should have valid pipeline DAGs."""
        from shared.plugins.loader import load_plugins

        load_plugins(plugin_types={"embedding", "chunking", "connector", "extractor", "parser"})

        from shared.pipeline.templates import clear_cache, list_templates

        clear_cache()
        templates = list_templates()

        for template in templates:
            # Pipeline should have at least one node
            assert len(template.pipeline.nodes) > 0
            # Pipeline should have at least one edge
            assert len(template.pipeline.edges) > 0
            # Should have an embedder node
            embedder_nodes = [
                n for n in template.pipeline.nodes if n.type == NodeType.EMBEDDER
            ]
            assert len(embedder_nodes) >= 1


class TestBuiltinTemplates:
    """Tests for built-in template definitions."""

    @pytest.fixture(autouse=True)
    def _load_plugins_and_clear_cache(self) -> None:
        """Load plugins and clear template cache before each test."""
        from shared.plugins.loader import load_plugins

        load_plugins(plugin_types={"embedding", "chunking", "connector", "extractor", "parser"})

        from shared.pipeline.templates import clear_cache

        clear_cache()

    def test_academic_papers_template(self) -> None:
        """Academic papers template should be properly configured."""
        from shared.pipeline.templates import load_template

        template = load_template("academic-papers")
        assert template is not None
        assert "PDF" in template.suggested_for
        assert "research" in template.suggested_for

    def test_codebase_template(self) -> None:
        """Codebase template should be properly configured."""
        from shared.pipeline.templates import load_template

        template = load_template("codebase")
        assert template is not None
        assert "code" in template.suggested_for

    def test_documentation_template(self) -> None:
        """Documentation template should be properly configured."""
        from shared.pipeline.templates import load_template

        template = load_template("documentation")
        assert template is not None
        assert "documentation" in template.suggested_for

    def test_email_archive_template(self) -> None:
        """Email archive template should be properly configured."""
        from shared.pipeline.templates import load_template

        template = load_template("email-archive")
        assert template is not None
        assert "email" in template.suggested_for

    def test_mixed_documents_template(self) -> None:
        """Mixed documents template should be properly configured."""
        from shared.pipeline.templates import load_template

        template = load_template("mixed-documents")
        assert template is not None
        assert "mixed" in template.suggested_for

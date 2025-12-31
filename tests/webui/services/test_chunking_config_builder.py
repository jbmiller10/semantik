import pytest

from webui.api.v2.chunking_schemas import ChunkingStrategy as ChunkingStrategyEnum
from webui.services.chunking_config_builder import ChunkingConfigBuilder


def test_build_config_unknown_strategy_returns_error():
    builder = ChunkingConfigBuilder()
    result = builder.build_config("unknown-strategy")

    assert result.strategy == ChunkingStrategyEnum.RECURSIVE
    assert result.validation_errors == ["Unknown strategy: unknown-strategy"]


def test_build_config_semantic_invalid_values_returns_errors_and_warnings():
    builder = ChunkingConfigBuilder()

    result = builder.build_config(
        "semantic",
        {
            "chunk_size": "50",
            "chunk_overlap": "40",
            "similarity_threshold": 1.5,
            "min_chunk_size": 200,
            "max_chunk_size": 100,
            "embedding_model": "",
        },
    )

    assert result.validation_errors
    assert "similarity_threshold must be between 0 and 1" in result.validation_errors
    assert "min_chunk_size must be <= max_chunk_size" in result.validation_errors
    assert result.warnings
    assert "Very small chunk_size may impact search quality" in result.warnings
    assert any("Large overlap ratio" in warning for warning in result.warnings)
    assert "No embedding model specified, using default" in result.warnings


def test_merge_configs_coerces_boolean_strings():
    builder = ChunkingConfigBuilder()

    result = builder.build_config(
        "recursive",
        {
            "keep_separator": "false",
        },
    )

    assert result.config["keep_separator"] is False


def test_coerce_bool_invalid_string_raises():
    builder = ChunkingConfigBuilder()

    with pytest.raises(ValueError, match="Invalid boolean string"):
        builder._coerce_bool("maybe")


def test_validate_parameter_separators_requires_list():
    builder = ChunkingConfigBuilder()

    error = builder.validate_parameter("separators", "not-a-list", ChunkingStrategyEnum.RECURSIVE)

    assert error == "separators must be a list"


def test_suggest_config_prefers_document_structure_for_markdown():
    builder = ChunkingConfigBuilder()

    result = builder.suggest_config(file_type=".md", content_size=40000)

    assert result.strategy == ChunkingStrategyEnum.DOCUMENT_STRUCTURE
    assert result.config["chunk_size"] == 800
    assert result.config["chunk_overlap"] == 100

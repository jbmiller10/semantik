#!/usr/bin/env python3

"""Unit tests for the shared error classifier."""

from packages.webui.api.chunking_exceptions import ChunkingDependencyError, ChunkingMemoryError
from packages.webui.services.chunking_error_handler import ChunkingErrorType
from packages.webui.utils.error_classifier import (
    ClassificationRule,
    build_chunking_error_classifier,
    get_default_chunking_error_classifier,
)


def test_type_based_classification() -> None:
    """Chunking exceptions produce expected enum and string codes."""

    classifier = get_default_chunking_error_classifier()

    memory_result = classifier.classify(ChunkingMemoryError("oom"))
    assert memory_result.error_type == ChunkingErrorType.MEMORY_ERROR
    assert memory_result.code == "memory_error"
    assert memory_result.matched_rule == "chunking_memory_error"

    dependency_result = classifier.classify(ChunkingDependencyError("vector store down"))
    assert dependency_result.error_type == ChunkingErrorType.DEPENDENCY_ERROR
    assert dependency_result.code == "dependency_error"


def test_keyword_classification() -> None:
    """Keyword heuristics fall back when type matching fails."""

    classifier = get_default_chunking_error_classifier()

    permission_result = classifier.classify(Exception("Permission denied for resource"))
    assert permission_result.error_type == ChunkingErrorType.PERMISSION_ERROR
    assert permission_result.code == "permission_error"
    assert permission_result.confidence < 1.0


def test_unknown_classification_defaults() -> None:
    """Unknown errors fall back to the default bucket."""

    classifier = get_default_chunking_error_classifier()
    result = classifier.classify(Exception("no hint here"))
    assert result.error_type == ChunkingErrorType.UNKNOWN_ERROR
    assert result.code == "unknown"
    assert result.matched_rule is None


def test_extra_rule_injection() -> None:
    """Additional rules can override the default behaviour."""

    extra_rule = ClassificationRule(
        name="custom_logic",
        predicate=lambda exc, _msg: str(exc) == "custom",
        error_type=ChunkingErrorType.STRATEGY_ERROR,
        confidence=0.95,
        code="strategy_error",
    )

    classifier = build_chunking_error_classifier(extra_rules=[extra_rule])
    result = classifier.classify(Exception("custom"))
    assert result.error_type == ChunkingErrorType.STRATEGY_ERROR
    assert result.matched_rule == "custom_logic"

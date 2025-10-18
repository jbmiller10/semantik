"""Shared error classification utilities for chunking services.

This module provides a reusable classifier that translates raw exceptions into
coordinated enum + string code representations.  It centralises the rule set
so Celery tasks and service-layer handlers stay in sync when new error
categories are introduced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

RulePredicate = Callable[[Exception, str], bool]
MetadataFactory = Callable[[Exception], dict[str, Any] | None]


@dataclass(frozen=True)
class ErrorClassificationResult:
    """Outcome of classifying an exception.

    Attributes:
        error_type: Enum value describing the classification (e.g., ChunkingErrorType).
        code: Stable string identifier used by Celery retry state and logs.
        matched_rule: Optional rule identifier that produced the classification.
        confidence: Heuristic confidence score (1.0 == high, 0.5 == low).
        metadata: Optional structured payload giving callers extra insight.
    """

    error_type: Any
    code: str
    matched_rule: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ClassificationRule:
    """Single classification rule evaluated against an exception."""

    name: str
    predicate: RulePredicate
    error_type: Any
    confidence: float = 1.0
    code: str | None = None
    metadata_factory: MetadataFactory | None = None


class ErrorClassifier:
    """Evaluate exceptions against an ordered rule set."""

    def __init__(
        self,
        *,
        rules: Sequence[ClassificationRule],
        enum_to_code: Mapping[Any, str],
        default_error_type: Any,
        default_code: str | None = None,
    ) -> None:
        self._rules = list(rules)
        self._enum_to_code = dict(enum_to_code)
        self._default_error_type = default_error_type
        self._default_code = default_code or self._enum_to_code.get(default_error_type, "unknown")

    def classify(self, error: Exception) -> ErrorClassificationResult:
        """Return the full classification result for an exception."""

        message = str(error)
        for rule in self._rules:
            if rule.predicate(error, message):
                code = rule.code or self._enum_to_code.get(rule.error_type, self._default_code)
                metadata = rule.metadata_factory(error) if rule.metadata_factory else None
                return ErrorClassificationResult(
                    error_type=rule.error_type,
                    code=code,
                    matched_rule=rule.name,
                    confidence=rule.confidence,
                    metadata=metadata,
                )

        return ErrorClassificationResult(
            error_type=self._default_error_type,
            code=self._default_code,
            matched_rule=None,
            confidence=0.1,
            metadata=None,
        )

    def as_enum(self, error: Exception) -> Any:
        """Return the enum value for an exception."""

        return self.classify(error).error_type

    def as_code(self, error: Exception) -> str:
        """Return the stable string code for an exception."""

        return self.classify(error).code

    def with_rules(self, extra_rules: Iterable[ClassificationRule]) -> ErrorClassifier:
        """Return a new classifier with additional rules prepended."""

        combined_rules = list(extra_rules) + self._rules
        return ErrorClassifier(
            rules=combined_rules,
            enum_to_code=self._enum_to_code,
            default_error_type=self._default_error_type,
            default_code=self._default_code,
        )


def build_chunking_rules() -> list[ClassificationRule]:
    """Construct the canonical rule list for chunking errors."""

    from celery.exceptions import SoftTimeLimitExceeded

    from packages.webui.api.chunking_exceptions import (
        ChunkingDependencyError,
        ChunkingMemoryError,
        ChunkingPartialFailureError,
        ChunkingResourceLimitError,
        ChunkingStrategyError,
        ChunkingTimeoutError,
        ChunkingValidationError,
    )
    from packages.webui.services.chunking_error_handler import ChunkingErrorType

    def _is_instance(expected: tuple[type[Exception], ...]) -> RulePredicate:
        return lambda exc, _msg: isinstance(exc, expected)

    def _contains(*needles: str) -> RulePredicate:
        lowered_needles = tuple(needle.lower() for needle in needles)

        def predicate(_exc: Exception, message: str) -> bool:
            lowered = message.lower()
            return any(needle in lowered for needle in lowered_needles)

        return predicate

    rules: list[ClassificationRule] = [
        ClassificationRule(
            name="chunking_memory_error",
            predicate=_is_instance((ChunkingMemoryError, MemoryError)),
            error_type=ChunkingErrorType.MEMORY_ERROR,
            confidence=1.0,
        ),
        ClassificationRule(
            name="chunking_timeout_error",
            predicate=_is_instance((ChunkingTimeoutError, TimeoutError, SoftTimeLimitExceeded)),
            error_type=ChunkingErrorType.TIMEOUT_ERROR,
            confidence=1.0,
        ),
        ClassificationRule(
            name="chunking_validation_error",
            predicate=_is_instance((ChunkingValidationError,)),
            error_type=ChunkingErrorType.VALIDATION_ERROR,
            confidence=1.0,
        ),
        ClassificationRule(
            name="chunking_strategy_error",
            predicate=_is_instance((ChunkingStrategyError,)),
            error_type=ChunkingErrorType.STRATEGY_ERROR,
            confidence=1.0,
        ),
        ClassificationRule(
            name="chunking_dependency_error",
            predicate=_is_instance((ChunkingDependencyError,)),
            error_type=ChunkingErrorType.DEPENDENCY_ERROR,
            confidence=1.0,
        ),
        ClassificationRule(
            name="chunking_resource_limit_error",
            predicate=_is_instance((ChunkingResourceLimitError,)),
            error_type=ChunkingErrorType.RESOURCE_LIMIT_ERROR,
            confidence=1.0,
        ),
        ClassificationRule(
            name="chunking_partial_failure_error",
            predicate=_is_instance((ChunkingPartialFailureError,)),
            error_type=ChunkingErrorType.PARTIAL_FAILURE,
            confidence=0.9,
        ),
        ClassificationRule(
            name="permission_error_type",
            predicate=_is_instance((PermissionError,)),
            error_type=ChunkingErrorType.PERMISSION_ERROR,
            confidence=0.9,
        ),
        ClassificationRule(
            name="unicode_error_type",
            predicate=_is_instance((UnicodeDecodeError, UnicodeError)),
            error_type=ChunkingErrorType.INVALID_ENCODING,
            confidence=0.9,
        ),
        ClassificationRule(
            name="connection_error_type",
            predicate=_is_instance((ConnectionError,)),
            error_type=ChunkingErrorType.NETWORK_ERROR,
            confidence=0.9,
        ),
        ClassificationRule(
            name="memory_keyword",
            predicate=_contains("memory", "out of memory"),
            error_type=ChunkingErrorType.MEMORY_ERROR,
            confidence=0.7,
        ),
        ClassificationRule(
            name="encoding_keyword",
            predicate=_contains("encoding", "codec"),
            error_type=ChunkingErrorType.INVALID_ENCODING,
            confidence=0.7,
        ),
        ClassificationRule(
            name="permission_keyword",
            predicate=_contains("permission", "access denied"),
            error_type=ChunkingErrorType.PERMISSION_ERROR,
            confidence=0.7,
        ),
        ClassificationRule(
            name="network_keyword",
            predicate=_contains("connection", "network", "host unreachable"),
            error_type=ChunkingErrorType.NETWORK_ERROR,
            confidence=0.7,
        ),
        ClassificationRule(
            name="timeout_keyword",
            predicate=_contains("timeout", "timed out"),
            error_type=ChunkingErrorType.TIMEOUT_ERROR,
            confidence=0.7,
        ),
        ClassificationRule(
            name="validation_keyword",
            predicate=_contains("validation", "invalid"),
            error_type=ChunkingErrorType.VALIDATION_ERROR,
            confidence=0.6,
        ),
        ClassificationRule(
            name="strategy_keyword",
            predicate=_contains("strategy", "chunker"),
            error_type=ChunkingErrorType.STRATEGY_ERROR,
            confidence=0.6,
        ),
    ]

    return rules


def chunking_enum_code_map() -> dict[Any, str]:
    """Return canonical enum→code mapping for chunking errors."""

    from packages.webui.services.chunking_error_handler import ChunkingErrorType

    return {
        ChunkingErrorType.MEMORY_ERROR: "memory_error",
        ChunkingErrorType.TIMEOUT_ERROR: "timeout_error",
        ChunkingErrorType.INVALID_ENCODING: "invalid_encoding",
        ChunkingErrorType.STRATEGY_ERROR: "strategy_error",
        ChunkingErrorType.PARTIAL_FAILURE: "partial_failure",
        ChunkingErrorType.VALIDATION_ERROR: "validation_error",
        ChunkingErrorType.NETWORK_ERROR: "connection_error",
        ChunkingErrorType.PERMISSION_ERROR: "permission_error",
        ChunkingErrorType.DEPENDENCY_ERROR: "dependency_error",
        ChunkingErrorType.RESOURCE_LIMIT_ERROR: "resource_limit_error",
        ChunkingErrorType.UNKNOWN_ERROR: "unknown",
    }


@lru_cache(maxsize=1)
def get_default_chunking_error_classifier() -> ErrorClassifier:
    """Return the shared chunking error classifier singleton."""

    from packages.webui.services.chunking_error_handler import ChunkingErrorType

    return ErrorClassifier(
        rules=build_chunking_rules(),
        enum_to_code=chunking_enum_code_map(),
        default_error_type=ChunkingErrorType.UNKNOWN_ERROR,
        default_code="unknown",
    )


def build_chunking_error_classifier(
    *, extra_rules: Sequence[ClassificationRule] | None = None
) -> ErrorClassifier:
    """Create a new classifier instance with optional extra rules."""

    base_classifier = get_default_chunking_error_classifier()
    if not extra_rules:
        return base_classifier
    return base_classifier.with_rules(extra_rules)

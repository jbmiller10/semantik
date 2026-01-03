"""Tests for ExtractionResult helpers."""

from __future__ import annotations

from shared.plugins.types.extractor import Entity, ExtractionResult


def test_to_searchable_dict_groups_entities() -> None:
    """Should group entities by type and include entity_types."""
    result = ExtractionResult(
        entities=[
            Entity(text="Alice", type="PERSON", start=0, end=5),
            Entity(text="Bob", type="PERSON", start=10, end=13),
            Entity(text="Acme", type="ORG", start=20, end=24),
        ],
        keywords=["alpha", "beta"],
        language="en",
        topics=["topic1"],
        sentiment=0.25,
        custom={"foo": "bar"},
    )

    searchable = result.to_searchable_dict()

    assert searchable["entities"] == {"PERSON": ["Alice", "Bob"], "ORG": ["Acme"]}
    assert searchable["entity_types"] == ["PERSON", "ORG"]
    assert searchable["keywords"] == ["alpha", "beta"]
    assert searchable["language"] == "en"
    assert searchable["topics"] == ["topic1"]
    assert searchable["sentiment"] == 0.25
    assert searchable["custom"] == {"foo": "bar"}


def test_to_searchable_dict_empty() -> None:
    """Empty ExtractionResult should serialize to empty dict."""
    result = ExtractionResult()
    assert result.to_searchable_dict() == {}


def test_merge_dedupes_and_prefers_non_none() -> None:
    """Merge should dedupe lists and prefer non-None scalar values."""
    base = ExtractionResult(
        entities=[Entity(text="Alice", type="PERSON", start=0, end=5)],
        keywords=["alpha", "beta"],
        language="en",
        sentiment=None,
        custom={"shared": 2, "base": True},
    )
    other = ExtractionResult(
        entities=[
            Entity(text="Alice", type="PERSON", start=0, end=5),
            Entity(text="Acme", type="ORG", start=6, end=10),
        ],
        keywords=["beta", "gamma"],
        language="fr",
        sentiment=0.7,
        summary="other summary",
        custom={"shared": 1, "other": True},
    )

    merged = base.merge(other)

    assert [(e.text, e.type) for e in merged.entities] == [("Alice", "PERSON"), ("Acme", "ORG")]
    assert merged.keywords == ["alpha", "beta", "gamma"]
    assert merged.language == "en"  # base takes precedence
    assert merged.sentiment == 0.7  # other provides first non-None
    assert merged.summary == "other summary"
    assert merged.custom == {"shared": 2, "other": True, "base": True}

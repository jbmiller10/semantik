"""Tests for KeywordExtractorPlugin."""

from __future__ import annotations

import pytest

from shared.plugins.builtins.keyword_extractor import KeywordExtractorPlugin, extract_keywords_rake
from shared.plugins.types.extractor import ExtractionResult, ExtractionType


def test_extract_keywords_rake_empty_text() -> None:
    assert extract_keywords_rake("") == []
    assert extract_keywords_rake("   ") == []


def test_extract_keywords_rake_limits_top_k() -> None:
    text = "Machine learning improves accuracy. Machine learning models scale."
    keywords = extract_keywords_rake(text, top_k=2, min_chars=3, max_words=3)
    assert len(keywords) <= 2
    assert all(isinstance(keyword, str) for keyword in keywords)


@pytest.mark.asyncio()
async def test_keyword_extractor_respects_extraction_types() -> None:
    plugin = KeywordExtractorPlugin(config={"top_k": 5})
    result = await plugin.extract("alpha beta gamma", extraction_types=[ExtractionType.ENTITIES])
    assert result == ExtractionResult()


@pytest.mark.asyncio()
async def test_keyword_extractor_options_override_config() -> None:
    plugin = KeywordExtractorPlugin(config={"top_k": 1})
    result = await plugin.extract(
        "alpha beta gamma delta",
        extraction_types=[ExtractionType.KEYWORDS],
        options={"top_k": 3, "min_chars": 3, "max_words": 2},
    )
    assert len(result.keywords) <= 3

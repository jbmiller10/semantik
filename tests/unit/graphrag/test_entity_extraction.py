"""Unit tests for EntityExtractionService.

Tests entity extraction service logic with mocked spaCy model.
All tests use mocks - no real spaCy or database required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEntityExtractionService:
    """Tests for EntityExtractionService."""

    @pytest.fixture
    def mock_spacy_doc(self):
        """Create a factory for mock spaCy documents with entities."""

        def _create_doc(entities=None):
            """Create a mock doc with specified entities.

            Args:
                entities: List of tuples (text, label, start_char, end_char)
            """
            doc = MagicMock()
            mock_ents = []

            if entities:
                for text, label, start, end in entities:
                    ent = MagicMock()
                    ent.text = text
                    ent.label_ = label
                    ent.start_char = start
                    ent.end_char = end
                    mock_ents.append(ent)

            doc.ents = mock_ents
            return doc

        return _create_doc

    @pytest.fixture
    def mock_nlp(self, mock_spacy_doc):
        """Create a mock spaCy nlp model."""
        mock = MagicMock()

        # Default behavior: parse text and detect common entities
        def process_text(text):
            entities = []

            # Simple entity detection simulation
            if "John Smith" in text:
                idx = text.find("John Smith")
                entities.append(("John Smith", "PERSON", idx, idx + 10))
            if "Microsoft" in text:
                idx = text.find("Microsoft")
                entities.append(("Microsoft", "ORG", idx, idx + 9))
            if "Seattle" in text:
                idx = text.find("Seattle")
                entities.append(("Seattle", "GPE", idx, idx + 7))
            if "Apple" in text:
                idx = text.find("Apple")
                entities.append(("Apple", "ORG", idx, idx + 5))
            if "Mount Rainier" in text:
                idx = text.find("Mount Rainier")
                entities.append(("Mount Rainier", "LOC", idx, idx + 13))

            return mock_spacy_doc(entities)

        mock.side_effect = process_text

        # Mock pipe for batch processing
        def mock_pipe(texts, batch_size=32):
            for text in texts:
                yield process_text(text)

        mock.pipe = mock_pipe

        return mock

    @pytest.fixture
    def extraction_service(self, mock_nlp):
        """Create EntityExtractionService with mocked NLP."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            yield service

    @pytest.mark.asyncio
    async def test_extract_from_text_finds_person(self, mock_nlp):
        """Should extract PERSON entities from text."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            entities = await service.extract_from_text(
                text="John Smith works at the company.",
                document_id="doc-1",
                chunk_id=1,
            )

        persons = [e for e in entities if e["entity_type"] == "PERSON"]
        assert len(persons) >= 1
        assert persons[0]["name"] == "John Smith"
        assert persons[0]["document_id"] == "doc-1"
        assert persons[0]["chunk_id"] == 1

    @pytest.mark.asyncio
    async def test_extract_from_text_finds_org(self, mock_nlp):
        """Should extract ORG entities from text."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            entities = await service.extract_from_text(
                text="Microsoft is a technology company.",
                document_id="doc-1",
            )

        orgs = [e for e in entities if e["entity_type"] == "ORG"]
        assert len(orgs) >= 1
        assert orgs[0]["name"] == "Microsoft"

    @pytest.mark.asyncio
    async def test_extract_from_text_finds_gpe(self, mock_nlp):
        """Should extract GPE (geopolitical entity) from text."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            entities = await service.extract_from_text(
                text="Seattle is a beautiful city.",
                document_id="doc-1",
            )

        gpes = [e for e in entities if e["entity_type"] == "GPE"]
        assert len(gpes) >= 1
        assert gpes[0]["name"] == "Seattle"

    @pytest.mark.asyncio
    async def test_extract_from_text_includes_positions(self, mock_nlp):
        """Should include character positions in extracted entities."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            entities = await service.extract_from_text(
                text="John Smith is here.",
                document_id="doc-1",
            )

        assert len(entities) > 0
        entity = entities[0]
        assert "start_offset" in entity
        assert "end_offset" in entity
        assert entity["start_offset"] >= 0
        assert entity["end_offset"] > entity["start_offset"]
        # Verify positions are correct
        assert entity["start_offset"] == 0  # "John Smith" starts at position 0
        assert entity["end_offset"] == 10  # "John Smith" is 10 chars

    @pytest.mark.asyncio
    async def test_extract_from_chunks_batch_processing(self, mock_nlp):
        """Should process multiple chunks and associate entities with chunk IDs."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            chunks = [
                {"id": 1, "text": "John Smith works at Microsoft."},
                {"id": 2, "text": "Seattle is on the west coast."},
                {"id": 3, "text": "Just some text with no recognized entities."},
            ]

            entities = await service.extract_from_chunks(
                chunks=chunks,
                document_id="doc-1",
                collection_id="col-1",
            )

        # Should find entities from multiple chunks
        chunk_ids = {e["chunk_id"] for e in entities}
        assert 1 in chunk_ids  # John Smith, Microsoft
        assert 2 in chunk_ids  # Seattle

        # Verify we found all expected entities
        names = {e["name"] for e in entities}
        assert "John Smith" in names
        assert "Microsoft" in names
        assert "Seattle" in names

    @pytest.mark.asyncio
    async def test_extract_filters_unsupported_types(self, mock_spacy_doc):
        """Should filter out unsupported entity types like DATE, CARDINAL."""
        # Create mock that returns unsupported entity types
        mock_nlp = MagicMock()

        def process_text(text):
            return mock_spacy_doc([
                ("Monday", "DATE", 0, 6),
                ("100", "CARDINAL", 10, 13),
                ("5%", "PERCENT", 20, 22),
            ])

        mock_nlp.side_effect = process_text

        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            entities = await service.extract_from_text(
                text="Monday we have 100 items at 5% discount.",
                document_id="doc-1",
            )

        # DATE, CARDINAL, PERCENT are not in SUPPORTED_ENTITY_TYPES
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_extract_handles_empty_text(self, mock_spacy_doc):
        """Should handle empty text gracefully."""
        mock_nlp = MagicMock()
        mock_nlp.side_effect = lambda text: mock_spacy_doc([])

        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()

            # Empty string
            entities = await service.extract_from_text(
                text="",
                document_id="doc-1",
            )
            assert entities == []

            # Whitespace only
            entities = await service.extract_from_text(
                text="   \n\t  ",
                document_id="doc-1",
            )
            assert entities == []

    @pytest.mark.asyncio
    async def test_extract_default_confidence(self, mock_nlp):
        """Should set default confidence score of 0.85."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            entities = await service.extract_from_text(
                text="John Smith is here.",
                document_id="doc-1",
            )

        assert len(entities) > 0
        assert entities[0]["confidence"] == 0.85

    def test_supported_entity_types_constant(self):
        """Should have expected entity types in SUPPORTED_ENTITY_TYPES."""
        from packages.vecpipe.graphrag.entity_extraction import SUPPORTED_ENTITY_TYPES

        assert "PERSON" in SUPPORTED_ENTITY_TYPES
        assert "ORG" in SUPPORTED_ENTITY_TYPES
        assert "GPE" in SUPPORTED_ENTITY_TYPES
        assert "LOC" in SUPPORTED_ENTITY_TYPES
        assert "PRODUCT" in SUPPORTED_ENTITY_TYPES
        assert "EVENT" in SUPPORTED_ENTITY_TYPES
        assert "WORK_OF_ART" in SUPPORTED_ENTITY_TYPES
        assert "LAW" in SUPPORTED_ENTITY_TYPES
        assert "NORP" in SUPPORTED_ENTITY_TYPES

        # These should NOT be in supported types
        assert "DATE" not in SUPPORTED_ENTITY_TYPES
        assert "TIME" not in SUPPORTED_ENTITY_TYPES
        assert "CARDINAL" not in SUPPORTED_ENTITY_TYPES
        assert "PERCENT" not in SUPPORTED_ENTITY_TYPES
        assert "MONEY" not in SUPPORTED_ENTITY_TYPES

    @pytest.mark.asyncio
    async def test_extract_from_text_multiple_entities(self, mock_nlp):
        """Should extract multiple entities from a single text."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            entities = await service.extract_from_text(
                text="John Smith works at Microsoft in Seattle.",
                document_id="doc-1",
            )

        assert len(entities) == 3
        names = {e["name"] for e in entities}
        assert "John Smith" in names
        assert "Microsoft" in names
        assert "Seattle" in names

    @pytest.mark.asyncio
    async def test_extract_from_chunks_empty_list(self, mock_nlp):
        """Should handle empty chunk list gracefully."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            entities = await service.extract_from_chunks(
                chunks=[],
                document_id="doc-1",
                collection_id="col-1",
            )

        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_from_chunks_with_empty_text_chunks(self, mock_nlp):
        """Should skip chunks with empty text."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            chunks = [
                {"id": 1, "text": ""},  # Empty text
                {"id": 2, "text": "John Smith is here."},
                {"id": 3, "text": "   "},  # Whitespace only
            ]

            entities = await service.extract_from_chunks(
                chunks=chunks,
                document_id="doc-1",
                collection_id="col-1",
            )

        # Should only have entities from chunk 2
        assert len(entities) == 1
        assert entities[0]["chunk_id"] == 2
        assert entities[0]["name"] == "John Smith"

    @pytest.mark.asyncio
    async def test_extract_preserves_entity_type(self, mock_nlp):
        """Should preserve the correct entity type for each entity."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()
            entities = await service.extract_from_text(
                text="John Smith visited Mount Rainier near Seattle.",
                document_id="doc-1",
            )

        entity_by_name = {e["name"]: e for e in entities}
        assert entity_by_name["John Smith"]["entity_type"] == "PERSON"
        assert entity_by_name["Mount Rainier"]["entity_type"] == "LOC"
        assert entity_by_name["Seattle"]["entity_type"] == "GPE"

    def test_get_entity_type_description(self, mock_nlp):
        """Should return human-readable descriptions for entity types."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            service = EntityExtractionService()

        assert service.get_entity_type_description("PERSON") == "Person (real or fictional)"
        assert service.get_entity_type_description("ORG") == "Organization"
        assert service.get_entity_type_description("GPE") == "Country, city, or state"
        assert service.get_entity_type_description("LOC") == "Location (non-political)"
        # Unknown type returns itself
        assert service.get_entity_type_description("UNKNOWN") == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_custom_entity_types_filter(self, mock_nlp):
        """Should allow custom entity types in constructor."""
        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            from packages.vecpipe.graphrag.entity_extraction import (
                EntityExtractionService,
            )

            # Only extract PERSON entities
            service = EntityExtractionService(entity_types={"PERSON"})
            entities = await service.extract_from_text(
                text="John Smith works at Microsoft in Seattle.",
                document_id="doc-1",
            )

        # Should only find PERSON, not ORG or GPE
        assert len(entities) == 1
        assert entities[0]["name"] == "John Smith"
        assert entities[0]["entity_type"] == "PERSON"

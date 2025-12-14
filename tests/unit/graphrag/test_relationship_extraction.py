"""Unit tests for RelationshipExtractionService.

Tests relationship extraction service logic with mocked spaCy model.
All tests use mocks - no real spaCy or database required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRelationshipExtractionService:
    """Tests for RelationshipExtractionService."""

    @pytest.fixture
    def mock_token(self):
        """Factory for creating mock spaCy tokens."""

        def _create_token(
            text,
            lemma=None,
            pos="NOUN",
            dep="ROOT",
            i=0,
            head=None,
            children=None,
        ):
            token = MagicMock()
            token.text = text
            token.lemma_ = lemma if lemma else text.lower()
            token.pos_ = pos
            token.dep_ = dep
            token.i = i
            token.head = head if head else token  # Default to self
            token.children = children if children else []
            token.subtree = [token]
            return token

        return _create_token

    @pytest.fixture
    def mock_span(self):
        """Factory for creating mock spaCy spans."""

        def _create_span(text, start_char, end_char, root=None):
            span = MagicMock()
            span.text = text
            span.start_char = start_char
            span.end_char = end_char
            span.root = root if root else MagicMock()
            span.root.i = 0
            span.root.head = MagicMock()
            return span

        return _create_span

    @pytest.fixture
    def relationship_service(self):
        """Create RelationshipExtractionService."""
        from packages.vecpipe.graphrag.relationship_extraction import (
            RelationshipExtractionService,
        )

        return RelationshipExtractionService()

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_with_no_entities(
        self, relationship_service
    ):
        """Should return empty list when no entities are provided."""
        relationships = await relationship_service.extract_relationships(
            text="Some text without entities.",
            entities=[],
        )

        assert relationships == []

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_with_one_entity(
        self, relationship_service
    ):
        """Should return empty list when fewer than 2 entities are provided."""
        entities = [
            {
                "name": "John",
                "entity_type": "PERSON",
                "start_offset": 0,
                "end_offset": 4,
            },
        ]

        relationships = await relationship_service.extract_relationships(
            text="John is here.",
            entities=entities,
        )

        assert relationships == []

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_text(self, relationship_service):
        """Should return empty list for empty text even with entities."""
        entities = [
            {"name": "John", "entity_type": "PERSON", "start_offset": 0, "end_offset": 4},
            {"name": "Microsoft", "entity_type": "ORG", "start_offset": 10, "end_offset": 19},
        ]

        relationships = await relationship_service.extract_relationships(
            text="",
            entities=entities,
        )

        assert relationships == []

        relationships = await relationship_service.extract_relationships(
            text="   \n\t  ",
            entities=entities,
        )

        assert relationships == []

    def test_normalize_verb_known_mappings(self, relationship_service):
        """Should normalize known verbs to relationship types."""
        test_cases = [
            ("work", "WORKS_FOR"),
            ("employ", "EMPLOYS"),
            ("own", "OWNS"),
            ("found", "FOUNDED"),
            ("create", "CREATED"),
            ("build", "BUILT"),
            ("lead", "LEADS"),
            ("manage", "MANAGES"),
            ("acquire", "ACQUIRED"),
            ("buy", "ACQUIRED"),  # Maps to ACQUIRED
            ("locate", "LOCATED_IN"),
            ("base", "BASED_IN"),
            ("headquarter", "HEADQUARTERED_IN"),
        ]

        for lemma, expected_type in test_cases:
            token = MagicMock()
            token.lemma_ = lemma
            result = relationship_service._normalize_verb(token)
            assert result == expected_type, f"Expected {lemma} to map to {expected_type}, got {result}"

    def test_normalize_verb_unknown_returns_uppercase(self, relationship_service):
        """Should return uppercased lemma for unknown verbs."""
        token = MagicMock()
        token.lemma_ = "collaborate"
        result = relationship_service._normalize_verb(token)
        assert result == "COLLABORATE"

        token.lemma_ = "partner"
        result = relationship_service._normalize_verb(token)
        assert result == "PARTNER"

    def test_calculate_confidence_direct_deps(
        self, relationship_service, mock_token
    ):
        """Should calculate higher confidence for direct dependencies."""
        # Create verb token
        verb = mock_token("works", pos="VERB", i=1)

        # Create subject span with root directly connected to verb
        subj_span = MagicMock()
        subj_span.root = MagicMock()
        subj_span.root.i = 0  # Close to verb (distance 1)
        subj_span.root.head = verb  # Direct connection

        # Create object span with root directly connected to verb
        obj_span = MagicMock()
        obj_span.root = MagicMock()
        obj_span.root.i = 2  # Close to verb (distance 1)
        obj_span.root.head = verb  # Direct connection

        confidence = relationship_service._calculate_confidence(verb, subj_span, obj_span)

        # Base confidence (0.7) + 0.1 for direct subj + 0.1 for direct obj = 0.9
        assert abs(confidence - 0.9) < 0.01

    def test_calculate_confidence_distant_deps(
        self, relationship_service, mock_token
    ):
        """Should calculate lower confidence for distant dependencies."""
        # Create verb token
        verb = mock_token("works", pos="VERB", i=5)

        # Create subject span with root far from verb
        subj_span = MagicMock()
        subj_span.root = MagicMock()
        subj_span.root.i = 0  # Distance 5 from verb
        subj_span.root.head = MagicMock()  # Not connected to verb

        # Create object span with root far from verb
        obj_span = MagicMock()
        obj_span.root = MagicMock()
        obj_span.root.i = 12  # Distance 7 from verb
        obj_span.root.head = MagicMock()  # Not connected to verb

        confidence = relationship_service._calculate_confidence(verb, subj_span, obj_span)

        # Base confidence (0.7) - 0.2 for long distance (total dist = 12 > 10) = 0.5
        assert abs(confidence - 0.5) < 0.01

    def test_calculate_confidence_bounds(
        self, relationship_service, mock_token, mock_span
    ):
        """Confidence should be bounded between 0.0 and 1.0."""
        verb = mock_token("works", pos="VERB", i=50)

        # Very distant tokens
        subj_root = mock_token("John", i=0)
        subj_root.head = MagicMock()
        subj_span = mock_span("John", 0, 4, root=subj_root)

        obj_root = mock_token("Microsoft", i=100)
        obj_root.head = MagicMock()
        obj_span = mock_span("Microsoft", 500, 509, root=obj_root)

        confidence = relationship_service._calculate_confidence(verb, subj_span, obj_span)

        assert 0.0 <= confidence <= 1.0

    def test_deduplicate_relationships_keeps_highest_confidence(
        self, relationship_service
    ):
        """Should keep the relationship with highest confidence when duplicates exist."""
        relationships = [
            {
                "source_entity": {"name": "John"},
                "target_entity": {"name": "Microsoft"},
                "relationship_type": "WORKS_FOR",
                "confidence": 0.7,
            },
            {
                "source_entity": {"name": "John"},
                "target_entity": {"name": "Microsoft"},
                "relationship_type": "WORKS_FOR",
                "confidence": 0.9,
            },
            {
                "source_entity": {"name": "John"},
                "target_entity": {"name": "Microsoft"},
                "relationship_type": "WORKS_FOR",
                "confidence": 0.6,
            },
        ]

        deduped = relationship_service._deduplicate_relationships(relationships)

        assert len(deduped) == 1
        assert deduped[0]["confidence"] == 0.9

    def test_deduplicate_relationships_different_types(self, relationship_service):
        """Should keep relationships with different types."""
        relationships = [
            {
                "source_entity": {"name": "John"},
                "target_entity": {"name": "Microsoft"},
                "relationship_type": "WORKS_FOR",
                "confidence": 0.7,
            },
            {
                "source_entity": {"name": "John"},
                "target_entity": {"name": "Microsoft"},
                "relationship_type": "FOUNDED",
                "confidence": 0.8,
            },
        ]

        deduped = relationship_service._deduplicate_relationships(relationships)

        assert len(deduped) == 2

    def test_deduplicate_relationships_different_entities(self, relationship_service):
        """Should keep relationships between different entity pairs."""
        relationships = [
            {
                "source_entity": {"name": "John"},
                "target_entity": {"name": "Microsoft"},
                "relationship_type": "WORKS_FOR",
                "confidence": 0.7,
            },
            {
                "source_entity": {"name": "Jane"},
                "target_entity": {"name": "Microsoft"},
                "relationship_type": "WORKS_FOR",
                "confidence": 0.8,
            },
            {
                "source_entity": {"name": "John"},
                "target_entity": {"name": "Apple"},
                "relationship_type": "WORKS_FOR",
                "confidence": 0.6,
            },
        ]

        deduped = relationship_service._deduplicate_relationships(relationships)

        assert len(deduped) == 3

    def test_deduplicate_relationships_empty_list(self, relationship_service):
        """Should handle empty list gracefully."""
        deduped = relationship_service._deduplicate_relationships([])
        assert deduped == []

    def test_build_entity_index(self, relationship_service):
        """Should build index mapping (start, end) to entity dict."""
        entities = [
            {"name": "John", "entity_type": "PERSON", "start_offset": 0, "end_offset": 4},
            {"name": "Microsoft", "entity_type": "ORG", "start_offset": 15, "end_offset": 24},
        ]

        index = relationship_service._build_entity_index(entities)

        assert (0, 4) in index
        assert index[(0, 4)]["name"] == "John"
        assert (15, 24) in index
        assert index[(15, 24)]["name"] == "Microsoft"

    def test_build_entity_index_skips_invalid(self, relationship_service):
        """Should skip entities with invalid offsets."""
        entities = [
            {"name": "John", "entity_type": "PERSON", "start_offset": 0, "end_offset": 4},
            {"name": "Invalid1", "entity_type": "ORG"},  # No offsets
            {"name": "Invalid2", "entity_type": "ORG", "start_offset": -1, "end_offset": 5},  # Negative start
            {"name": "Invalid3", "entity_type": "ORG", "start_offset": 10, "end_offset": 5},  # End before start
        ]

        index = relationship_service._build_entity_index(entities)

        assert len(index) == 1
        assert (0, 4) in index

    def test_verb_mappings_constant(self, relationship_service):
        """Should have expected verb mappings defined."""
        from packages.vecpipe.graphrag.relationship_extraction import (
            RelationshipExtractionService,
        )

        mappings = RelationshipExtractionService.VERB_MAPPINGS

        assert "WORK" in mappings
        assert mappings["WORK"] == "WORKS_FOR"
        assert "OWN" in mappings
        assert mappings["OWN"] == "OWNS"
        assert "BUY" in mappings
        assert mappings["BUY"] == "ACQUIRED"

    @pytest.mark.asyncio
    async def test_extract_from_chunks_groups_by_chunk(self, relationship_service):
        """Should process chunks and group entities by chunk_id."""
        # Create mock for get_nlp
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.text = "John works at Microsoft"
        mock_doc.__iter__ = lambda self: iter([])
        mock_nlp.return_value = mock_doc

        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            chunks = [
                {"id": 1, "text": "John works at Microsoft."},
                {"id": 2, "text": "Jane leads Google."},
            ]

            all_entities = [
                {"name": "John", "entity_type": "PERSON", "start_offset": 0, "end_offset": 4, "chunk_id": 1},
                {"name": "Microsoft", "entity_type": "ORG", "start_offset": 15, "end_offset": 24, "chunk_id": 1},
                {"name": "Jane", "entity_type": "PERSON", "start_offset": 0, "end_offset": 4, "chunk_id": 2},
                {"name": "Google", "entity_type": "ORG", "start_offset": 11, "end_offset": 17, "chunk_id": 2},
            ]

            relationships = await relationship_service.extract_from_chunks(
                chunks=chunks,
                all_entities=all_entities,
            )

        # The method was called and returned a list
        assert isinstance(relationships, list)

    @pytest.mark.asyncio
    async def test_extract_from_chunks_skips_single_entity_chunks(
        self, relationship_service
    ):
        """Should skip chunks with fewer than 2 entities."""
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([])
        mock_nlp.return_value = mock_doc

        with patch(
            "packages.vecpipe.graphrag.entity_extraction.get_nlp",
            new_callable=AsyncMock,
            return_value=mock_nlp,
        ):
            chunks = [
                {"id": 1, "text": "John is here."},  # Only 1 entity
                {"id": 2, "text": "No entities here."},  # 0 entities
            ]

            all_entities = [
                {"name": "John", "entity_type": "PERSON", "start_offset": 0, "end_offset": 4, "chunk_id": 1},
            ]

            relationships = await relationship_service.extract_from_chunks(
                chunks=chunks,
                all_entities=all_entities,
            )

        # No relationships should be found (need 2+ entities per chunk)
        assert relationships == []

    def test_min_confidence_threshold(self):
        """Should respect min_confidence threshold in constructor."""
        from packages.vecpipe.graphrag.relationship_extraction import (
            RelationshipExtractionService,
        )

        service = RelationshipExtractionService(min_confidence=0.8)
        assert service.min_confidence == 0.8

        service = RelationshipExtractionService(min_confidence=0.5)
        assert service.min_confidence == 0.5

        # Default should be 0.5
        service = RelationshipExtractionService()
        assert service.min_confidence == 0.5

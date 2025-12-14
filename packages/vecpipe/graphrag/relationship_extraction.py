"""Relationship extraction service using spaCy dependency parsing.

This service extracts relationships between entities by analyzing
grammatical structure (subject-verb-object patterns).
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from packages.vecpipe.graphrag.entity_extraction import get_nlp

if TYPE_CHECKING:
    from collections.abc import Callable

    from spacy.language import Language
    from spacy.tokens import Span, Token

logger = logging.getLogger(__name__)


class RelationshipExtractionService:
    """Service for extracting relationships between entities.

    Uses spaCy's dependency parser to find subject-verb-object patterns
    and maps them to relationships between extracted entities.

    Usage:
        service = RelationshipExtractionService()
        relationships = await service.extract_relationships(text, entities)
    """

    # Common verb mappings for clearer relationship types
    VERB_MAPPINGS: dict[str, str] = {
        "BE": "IS",
        "WORK": "WORKS_FOR",
        "EMPLOY": "EMPLOYS",
        "OWN": "OWNS",
        "FOUND": "FOUNDED",
        "CREATE": "CREATED",
        "BUILD": "BUILT",
        "LEAD": "LEADS",
        "MANAGE": "MANAGES",
        "ACQUIRE": "ACQUIRED",
        "BUY": "ACQUIRED",
        "LOCATE": "LOCATED_IN",
        "BASE": "BASED_IN",
        "HEADQUARTER": "HEADQUARTERED_IN",
    }

    def __init__(self, min_confidence: float = 0.5) -> None:
        """Initialize the relationship extraction service.

        Args:
            min_confidence: Minimum confidence threshold for relationships
        """
        self.min_confidence = min_confidence
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def extract_relationships(
        self,
        text: str,
        entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract relationships between entities in text.

        Args:
            text: Source text (typically a chunk or document)
            entities: List of entity dicts from EntityExtractionService

        Returns:
            List of relationship dicts with keys:
                - source_entity: Source entity dict
                - target_entity: Target entity dict
                - relationship_type: Relationship type (verb lemma, uppercased)
                - confidence: Confidence score
                - extraction_method: 'dependency'
        """
        if not entities or len(entities) < 2:
            return []

        if not text or not text.strip():
            return []

        nlp = await get_nlp()
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            self._executor,
            self._extract_sync,
            nlp,
            text,
            entities,
        )

    def _extract_sync(
        self,
        nlp: Language,
        text: str,
        entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Synchronous relationship extraction (runs in thread pool).

        Args:
            nlp: spaCy Language model
            text: Source text
            entities: Entity dicts

        Returns:
            List of relationship dicts
        """
        doc = nlp(text)
        relationships: list[dict[str, Any]] = []

        # Build entity span index for matching
        entity_index = self._build_entity_index(entities)

        # Find subject-verb-object patterns
        for token in doc:
            if token.pos_ != "VERB":
                continue

            # Find subjects
            subjects = self._find_subjects(token)
            # Find objects
            objects = self._find_objects(token)

            # Match subjects and objects to entities
            for subj in subjects:
                subj_entity = self._match_span_to_entity(subj, entity_index)
                if not subj_entity:
                    continue

                for obj in objects:
                    obj_entity = self._match_span_to_entity(obj, entity_index)
                    if not obj_entity:
                        continue

                    # Don't create self-relationships
                    if subj_entity["name"] == obj_entity["name"]:
                        continue

                    # Calculate confidence based on dependency distance
                    confidence = self._calculate_confidence(token, subj, obj)

                    if confidence >= self.min_confidence:
                        relationships.append({
                            "source_entity": subj_entity,
                            "target_entity": obj_entity,
                            "relationship_type": self._normalize_verb(token),
                            "confidence": confidence,
                            "extraction_method": "dependency",
                        })

        # Deduplicate relationships
        return self._deduplicate_relationships(relationships)

    def _build_entity_index(
        self,
        entities: list[dict[str, Any]],
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Build index of entity spans for fast matching.

        Args:
            entities: List of entity dicts

        Returns:
            Dict mapping (start, end) to entity dict
        """
        index: dict[tuple[int, int], dict[str, Any]] = {}
        for entity in entities:
            start = entity.get("start_offset", -1)
            end = entity.get("end_offset", -1)
            if start >= 0 and end > start:
                index[(start, end)] = entity
        return index

    def _find_subjects(self, verb_token: Token) -> list[Span]:
        """Find subject spans for a verb.

        Args:
            verb_token: spaCy Token for the verb

        Returns:
            List of spaCy Spans that are subjects
        """
        subjects: list[Span] = []
        for child in verb_token.children:
            # Get the full noun phrase if it's a subject and has dependency annotation
            if child.dep_ in ("nsubj", "nsubjpass") and child.doc.has_annotation("DEP"):
                span = self._get_noun_phrase_span(child)
                subjects.append(span)
        return subjects

    def _find_objects(self, verb_token: Token) -> list[Span]:
        """Find object spans for a verb.

        Args:
            verb_token: spaCy Token for the verb

        Returns:
            List of spaCy Spans that are objects
        """
        objects: list[Span] = []
        for child in verb_token.children:
            if child.dep_ in ("dobj", "pobj", "attr", "dative"):
                span = self._get_noun_phrase_span(child)
                objects.append(span)
            # Also check prepositional objects
            elif child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        span = self._get_noun_phrase_span(pobj)
                        objects.append(span)
        return objects

    def _get_noun_phrase_span(self, token: Token) -> Span:
        """Get the full noun phrase span for a token.

        Args:
            token: spaCy Token

        Returns:
            spaCy Span covering the noun phrase
        """
        # Get subtree span
        start = token.i
        end = token.i + 1

        # Extend to include modifiers
        for child in token.subtree:
            if child.i < start:
                start = child.i
            if child.i >= end:
                end = child.i + 1

        return token.doc[start:end]

    def _match_span_to_entity(
        self,
        span: Span,
        entity_index: dict[tuple[int, int], dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Match a spaCy span to an extracted entity.

        Args:
            span: spaCy Span
            entity_index: Entity index from _build_entity_index

        Returns:
            Matching entity dict, or None
        """
        span_start = span.start_char
        span_end = span.end_char

        # Exact match
        if (span_start, span_end) in entity_index:
            return entity_index[(span_start, span_end)]

        # Overlap match (span contains entity or entity contains span)
        for (ent_start, ent_end), entity in entity_index.items():
            # Check for significant overlap
            overlap_start = max(span_start, ent_start)
            overlap_end = min(span_end, ent_end)

            if overlap_start < overlap_end:
                # At least some overlap exists
                overlap_len = overlap_end - overlap_start
                span_len = span_end - span_start
                ent_len = ent_end - ent_start

                # Accept if overlap is >50% of either span
                if overlap_len > span_len * 0.5 or overlap_len > ent_len * 0.5:
                    return entity

        return None

    def _normalize_verb(self, token: Token) -> str:
        """Normalize a verb token to a relationship type.

        Args:
            token: spaCy Token (verb)

        Returns:
            Normalized relationship type string
        """
        # Use lemma and uppercase
        verb = token.lemma_.upper()

        return self.VERB_MAPPINGS.get(verb, verb)

    def _calculate_confidence(
        self,
        verb_token: Token,
        subj_span: Span,
        obj_span: Span,
    ) -> float:
        """Calculate confidence score for a relationship.

        Based on:
        - Dependency distance (closer = higher confidence)
        - Sentence structure complexity

        Args:
            verb_token: Verb token
            subj_span: Subject span
            obj_span: Object span

        Returns:
            Confidence score (0-1)
        """
        # Base confidence
        confidence = 0.7

        # Boost for direct dependencies (subject/object are direct children)
        subj_head = subj_span.root
        obj_head = obj_span.root

        if subj_head.head == verb_token:
            confidence += 0.1
        if obj_head.head == verb_token:
            confidence += 0.1

        # Penalize for long dependencies
        subj_dist = abs(subj_head.i - verb_token.i)
        obj_dist = abs(obj_head.i - verb_token.i)
        total_dist = subj_dist + obj_dist

        if total_dist > 10:
            confidence -= 0.2
        elif total_dist > 5:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _deduplicate_relationships(
        self,
        relationships: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove duplicate relationships, keeping highest confidence.

        Args:
            relationships: List of relationship dicts

        Returns:
            Deduplicated list
        """
        seen: dict[tuple[str, str, str], dict[str, Any]] = {}

        for rel in relationships:
            key = (
                rel["source_entity"]["name"],
                rel["target_entity"]["name"],
                rel["relationship_type"],
            )

            if key not in seen or rel["confidence"] > seen[key]["confidence"]:
                seen[key] = rel

        return list(seen.values())

    async def extract_from_chunks(
        self,
        chunks: list[dict[str, Any]],
        all_entities: list[dict[str, Any]],
        progress_callback: Callable[[str], Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract relationships from multiple chunks.

        Args:
            chunks: List of chunk dicts with 'id' and 'text'
            all_entities: All entities extracted from the chunks
            progress_callback: Optional progress callback

        Returns:
            List of all relationship dicts
        """
        all_relationships: list[dict[str, Any]] = []
        total_chunks = len(chunks)

        # Group entities by chunk
        entities_by_chunk: dict[int, list[dict[str, Any]]] = {}
        for entity in all_entities:
            chunk_id = entity.get("chunk_id")
            if chunk_id is not None:
                if chunk_id not in entities_by_chunk:
                    entities_by_chunk[chunk_id] = []
                entities_by_chunk[chunk_id].append(entity)

        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id")
            chunk_text = chunk.get("text", "")

            # Skip chunks without valid ID
            if chunk_id is None:
                continue

            chunk_entities = entities_by_chunk.get(chunk_id, [])

            if len(chunk_entities) >= 2:
                chunk_rels = await self.extract_relationships(
                    chunk_text,
                    chunk_entities,
                )
                all_relationships.extend(chunk_rels)

            if progress_callback and (i + 1) % 10 == 0:
                try:
                    result = progress_callback(
                        f"Relationship extraction: {i + 1}/{total_chunks} chunks"
                    )
                    # Support both sync and async callbacks
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass

        logger.info(
            f"Extracted {len(all_relationships)} relationships "
            f"from {total_chunks} chunks"
        )
        return all_relationships

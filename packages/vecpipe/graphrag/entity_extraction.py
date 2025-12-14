"""Entity extraction service using spaCy NER.

This service extracts named entities from document chunks locally using spaCy.
No external API calls are made - all processing happens on the local machine.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from spacy.language import Language
    from spacy.tokens import Span

logger = logging.getLogger(__name__)

# Lazy load spaCy to avoid import-time overhead
_nlp: Language | None = None
_nlp_lock = asyncio.Lock()


async def get_nlp() -> Language:
    """Get or initialize the spaCy model (lazy singleton).

    Uses thread-safe lazy initialization with double-checked locking.
    Falls back from en_core_web_md to en_core_web_sm if the medium
    model is not available.

    Returns:
        spaCy Language model

    Raises:
        OSError: If no spaCy model can be loaded
    """
    global _nlp
    if _nlp is None:
        async with _nlp_lock:
            if _nlp is None:  # Double-check after acquiring lock
                import spacy

                try:
                    _nlp = spacy.load("en_core_web_md")
                    logger.info("Loaded spaCy model: en_core_web_md")
                except OSError:
                    # Fallback to small model
                    logger.warning(
                        "en_core_web_md not found, falling back to en_core_web_sm"
                    )
                    _nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy model: en_core_web_sm")
    return _nlp


# Entity types we extract (spaCy labels)
SUPPORTED_ENTITY_TYPES: frozenset[str] = frozenset({
    "PERSON",      # People, including fictional
    "ORG",         # Organizations, companies, agencies
    "GPE",         # Geopolitical entities (countries, cities, states)
    "LOC",         # Non-GPE locations (mountains, water bodies)
    "PRODUCT",     # Products (objects, vehicles, foods, etc.)
    "EVENT",       # Named events (hurricanes, battles, wars, sports events)
    "WORK_OF_ART", # Titles of books, songs, etc.
    "LAW",         # Named laws, bills, etc.
    "NORP",        # Nationalities, religious/political groups
})


# Entity type descriptions for human-readable output
_ENTITY_TYPE_DESCRIPTIONS: dict[str, str] = {
    "PERSON": "Person (real or fictional)",
    "ORG": "Organization",
    "GPE": "Country, city, or state",
    "LOC": "Location (non-political)",
    "PRODUCT": "Product",
    "EVENT": "Named event",
    "WORK_OF_ART": "Creative work",
    "LAW": "Law or legal document",
    "NORP": "Nationality or group",
}


class EntityExtractionService:
    """Service for extracting named entities from text using spaCy.

    This service is designed to be:
    - Async-compatible (runs spaCy in thread pool)
    - Batch-efficient (processes multiple texts at once using nlp.pipe())
    - Memory-efficient (lazy loads model)

    Usage:
        service = EntityExtractionService()
        entities = await service.extract_from_chunks(chunks, doc_id, collection_id)

    Attributes:
        entity_types: Set of entity types to extract
        min_confidence: Minimum confidence threshold (reserved for future use)
        batch_size: Batch size for spaCy processing
    """

    def __init__(
        self,
        entity_types: set[str] | None = None,
        min_confidence: float = 0.0,
        batch_size: int = 32,
    ) -> None:
        """Initialize the entity extraction service.

        Args:
            entity_types: Set of entity types to extract.
                Defaults to SUPPORTED_ENTITY_TYPES.
            min_confidence: Minimum confidence threshold.
                Reserved for future use as spaCy doesn't provide
                per-entity confidence scores.
            batch_size: Batch size for spaCy processing.
                Larger batches are more efficient but use more memory.
        """
        self.entity_types: frozenset[str] = frozenset(
            entity_types if entity_types is not None else SUPPORTED_ENTITY_TYPES
        )
        self.min_confidence: float = min_confidence
        self.batch_size: int = batch_size
        # Use limited workers to prevent memory issues with concurrent model loads
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)

    async def extract_from_text(
        self,
        text: str,
        document_id: str,
        chunk_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract entities from a single text.

        Args:
            text: Text to extract entities from
            document_id: Source document ID
            chunk_id: Optional source chunk ID

        Returns:
            List of entity dicts with keys:
                - name: Entity text
                - entity_type: Entity type (PERSON, ORG, etc.)
                - start_offset: Start character position
                - end_offset: End character position
                - document_id: Source document
                - chunk_id: Source chunk (if provided)
                - confidence: Confidence score (0.85 default for spaCy)
        """
        if not text or not text.strip():
            return []

        nlp = await get_nlp()
        loop = asyncio.get_event_loop()

        # Run spaCy in thread pool (it's CPU-bound)
        return await loop.run_in_executor(
            self._executor,
            self._extract_sync,
            nlp,
            text,
            document_id,
            chunk_id,
        )

    def _extract_sync(
        self,
        nlp: Language,
        text: str,
        document_id: str,
        chunk_id: int | None,
    ) -> list[dict[str, Any]]:
        """Synchronous entity extraction (runs in thread pool).

        Args:
            nlp: spaCy Language model
            text: Text to process
            document_id: Source document ID
            chunk_id: Source chunk ID

        Returns:
            List of entity dicts
        """
        doc = nlp(text)
        entities: list[dict[str, Any]] = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entities.append(self._span_to_entity_dict(
                    ent, document_id, chunk_id
                ))

        return entities

    def _span_to_entity_dict(
        self,
        ent: Span,
        document_id: str,
        chunk_id: int | None,
    ) -> dict[str, Any]:
        """Convert a spaCy entity span to our entity dict format.

        Args:
            ent: spaCy entity span
            document_id: Source document ID
            chunk_id: Source chunk ID

        Returns:
            Entity dict with standardized fields
        """
        return {
            "name": ent.text,
            "entity_type": ent.label_,
            "start_offset": ent.start_char,
            "end_offset": ent.end_char,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "confidence": 0.85,  # spaCy doesn't provide per-entity confidence
        }

    async def extract_from_chunks(
        self,
        chunks: list[dict[str, Any]],
        document_id: str,
        collection_id: str,
        progress_callback: Callable[[str], Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract entities from multiple chunks (batch processing).

        Uses spaCy's nlp.pipe() for efficient batch processing of texts.

        Args:
            chunks: List of chunk dicts with 'id' and 'text' keys
            document_id: Source document ID
            collection_id: Collection ID (for logging/progress)
            progress_callback: Optional async callback for progress updates.
                Called with a status message string.

        Returns:
            List of all extracted entity dicts
        """
        if not chunks:
            return []

        nlp = await get_nlp()
        loop = asyncio.get_event_loop()

        # Extract texts and IDs, filtering out empty texts
        valid_chunks: list[tuple[int | None, str]] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if text and text.strip():
                valid_chunks.append((chunk.get("id"), text))

        if not valid_chunks:
            logger.debug(
                f"No valid text chunks to process for document {document_id}"
            )
            return []

        chunk_ids = [c[0] for c in valid_chunks]
        texts = [c[1] for c in valid_chunks]

        # Process in batches
        all_entities: list[dict[str, Any]] = []
        total_chunks = len(texts)

        for batch_start in range(0, total_chunks, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_chunks)
            batch_texts = texts[batch_start:batch_end]
            batch_ids = chunk_ids[batch_start:batch_end]

            # Run batch in thread pool
            batch_entities = await loop.run_in_executor(
                self._executor,
                self._extract_batch_sync,
                nlp,
                batch_texts,
                batch_ids,
                document_id,
            )
            all_entities.extend(batch_entities)

            # Progress update
            if progress_callback is not None:
                progress = (batch_end / total_chunks) * 100
                try:
                    result = progress_callback(
                        f"Entity extraction: {batch_end}/{total_chunks} chunks "
                        f"({progress:.0f}%)"
                    )
                    # Support both sync and async callbacks
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    # Don't fail extraction if progress callback fails
                    logger.debug("Progress callback failed", exc_info=True)

        logger.info(
            f"Extracted {len(all_entities)} entities from {total_chunks} chunks "
            f"in collection {collection_id}"
        )

        return all_entities

    def _extract_batch_sync(
        self,
        nlp: Language,
        texts: list[str],
        chunk_ids: list[int | None],
        document_id: str,
    ) -> list[dict[str, Any]]:
        """Synchronous batch extraction (runs in thread pool).

        Uses spaCy's nlp.pipe() for efficient batch processing.

        Args:
            nlp: spaCy Language model
            texts: List of texts to process
            chunk_ids: Corresponding chunk IDs
            document_id: Source document ID

        Returns:
            List of all entity dicts from the batch
        """
        all_entities: list[dict[str, Any]] = []

        # nlp.pipe() is much faster than processing one at a time
        for doc, chunk_id in zip(
            nlp.pipe(texts, batch_size=self.batch_size),
            chunk_ids,
            strict=True,
        ):
            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    all_entities.append(self._span_to_entity_dict(
                        ent, document_id, chunk_id
                    ))

        return all_entities

    async def extract_from_document(
        self,
        full_text: str,
        document_id: str,
    ) -> list[dict[str, Any]]:
        """Extract entities from a full document (not chunked).

        Useful for getting document-level entity overview.

        Args:
            full_text: Full document text
            document_id: Document ID

        Returns:
            List of entity dicts (without chunk_id)
        """
        return await self.extract_from_text(full_text, document_id, chunk_id=None)

    def get_entity_type_description(self, entity_type: str) -> str:
        """Get human-readable description of an entity type.

        Args:
            entity_type: spaCy entity type label (e.g., "PERSON", "ORG")

        Returns:
            Human-readable description string.
            If the entity type is unknown, returns the type itself.
        """
        return _ENTITY_TYPE_DESCRIPTIONS.get(entity_type, entity_type)

"""Graph-enhanced search service.

Enhances vector search results by incorporating entity relationship context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from packages.vecpipe.graphrag.entity_extraction import EntityExtractionService
from shared.database.repositories.entity_repository import EntityRepository
from shared.database.repositories.relationship_repository import RelationshipRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class GraphEnhancedSearchService:
    """Service for enhancing search results with graph context.

    This service:
    1. Extracts entities from the search query
    2. Finds matching entities in the collection's graph
    3. Traverses relationships to find related entities
    4. Boosts search results containing related entities

    Usage:
        service = GraphEnhancedSearchService(db_session)
        enhanced_results = await service.enhance_results(
            query="Who works at Microsoft?",
            vector_results=vector_search_results,
            collection_id="my-collection",
        )
    """

    def __init__(
        self,
        db_session: AsyncSession,
        graph_weight: float = 0.2,
        max_hops: int = 2,
    ) -> None:
        """Initialize the graph-enhanced search service.

        Args:
            db_session: Database session for repository access
            graph_weight: Weight for graph score in final ranking (0-1).
                         Final score = (1-graph_weight)*vector_score + graph_weight*graph_score
            max_hops: Maximum graph traversal depth
        """
        self.db_session = db_session
        self.graph_weight = graph_weight
        self.max_hops = max_hops

        self.entity_repo = EntityRepository(db_session)
        self.rel_repo = RelationshipRepository(db_session)
        self.entity_service = EntityExtractionService()

    async def enhance_results(
        self,
        query: str,
        vector_results: list[dict[str, Any]],
        collection_id: str,
        graph_weight: float | None = None,
        max_hops: int | None = None,
    ) -> list[dict[str, Any]]:
        """Enhance vector search results with graph context.

        Args:
            query: Original search query
            vector_results: Results from vector search, each with:
                - chunk_id: Chunk identifier
                - score: Vector similarity score
                - text: Chunk text
                - (other fields preserved)
            collection_id: Collection to search graph in
            graph_weight: Override default graph weight
            max_hops: Override default max hops

        Returns:
            Enhanced results with additional fields:
                - original_score: Original vector score
                - graph_score: Score from graph enhancement
                - score: Combined score
                - matched_entities: List of matched entity info
        """
        if not vector_results:
            return []

        weight = graph_weight if graph_weight is not None else self.graph_weight
        hops = max_hops if max_hops is not None else self.max_hops

        try:
            # Step 1: Extract entities from query
            query_entities = await self._extract_query_entities(query)

            if not query_entities:
                # No entities in query, return original results unchanged
                logger.debug(f"No entities found in query: {query[:50]}...")
                return self._add_empty_graph_fields(vector_results)

            logger.debug(f"Query entities: {[e['name'] for e in query_entities]}")

            # Step 2: Find matching entities in collection graph
            matched_entity_ids = await self._find_matching_entities(
                query_entities, collection_id
            )

            if not matched_entity_ids:
                # No matching entities in graph
                logger.debug("No matching entities found in collection graph")
                return self._add_empty_graph_fields(vector_results)

            # Step 3: Expand via graph traversal
            related_entity_ids = await self._expand_via_graph(
                matched_entity_ids, collection_id, hops
            )

            logger.debug(
                f"Graph expansion: {len(matched_entity_ids)} direct matches, "
                f"{len(related_entity_ids)} total after {hops}-hop expansion"
            )

            # Step 4: Get entities by chunk for scoring
            chunk_ids = [r.get("chunk_id") for r in vector_results if r.get("chunk_id")]
            chunk_entity_map = await self._get_entities_by_chunks(
                chunk_ids, collection_id
            )

            # Step 5: Score and rerank results
            enhanced_results = self._score_results(
                vector_results=vector_results,
                related_entity_ids=related_entity_ids,
                direct_match_ids=matched_entity_ids,
                chunk_entity_map=chunk_entity_map,
                weight=weight,
            )

            # Sort by combined score
            enhanced_results.sort(key=lambda x: x["score"], reverse=True)

            return enhanced_results

        except Exception as e:
            # Graceful degradation: return original results if graph enhancement fails
            logger.warning(f"Graph enhancement failed, returning original results: {e}")
            return self._add_empty_graph_fields(vector_results)

    async def _extract_query_entities(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """Extract entities from the search query.

        Args:
            query: Search query text

        Returns:
            List of entity dicts
        """
        try:
            return await self.entity_service.extract_from_text(
                text=query,
                document_id="query",  # Dummy ID
                chunk_id=None,
            )
        except Exception as e:
            logger.warning(f"Failed to extract query entities: {e}")
            return []

    async def _find_matching_entities(
        self,
        query_entities: list[dict[str, Any]],
        collection_id: str,
    ) -> set[int]:
        """Find entities in collection matching query entities.

        Uses name matching to find corresponding entities.

        Args:
            query_entities: Entities extracted from query
            collection_id: Collection to search

        Returns:
            Set of matching entity IDs
        """
        matched_ids: set[int] = set()

        for query_entity in query_entities:
            try:
                # Search by name prefix
                matches = await self.entity_repo.search_by_name(
                    collection_id=collection_id,
                    query=query_entity["name"],
                    entity_types=[query_entity["entity_type"]],
                    limit=5,
                )

                for match in matches:
                    # Accept if name is similar enough
                    if self._names_match(query_entity["name"], match.name):
                        matched_ids.add(match.id)
            except Exception as e:
                logger.warning(f"Failed to find matching entity for '{query_entity['name']}': {e}")
                continue

        return matched_ids

    def _names_match(self, query_name: str, entity_name: str) -> bool:
        """Check if two entity names match.

        Uses simple case-insensitive comparison.
        Could be enhanced with fuzzy matching.

        Args:
            query_name: Name from query
            entity_name: Name from database

        Returns:
            True if names match
        """
        q_norm = query_name.lower().strip()
        e_norm = entity_name.lower().strip()

        # Exact match
        if q_norm == e_norm:
            return True

        # One contains the other
        if q_norm in e_norm or e_norm in q_norm:
            return True

        return False

    async def _expand_via_graph(
        self,
        entity_ids: set[int],
        collection_id: str,
        max_hops: int,
    ) -> set[int]:
        """Expand entity set via graph traversal.

        Args:
            entity_ids: Starting entity IDs
            collection_id: Collection ID
            max_hops: Maximum traversal depth

        Returns:
            Expanded set of entity IDs
        """
        if not entity_ids:
            return set()

        try:
            neighbor_data = await self.rel_repo.get_neighbors(
                entity_ids=list(entity_ids),
                collection_id=collection_id,
                max_hops=max_hops,
            )
            # get_neighbors returns {"entities": {entity_id: hop_distance, ...}, "relationships": [...]}
            return set(neighbor_data["entities"].keys())
        except Exception as e:
            logger.warning(f"Graph expansion failed: {e}")
            return entity_ids

    async def _get_entities_by_chunks(
        self,
        chunk_ids: list[int | None],
        collection_id: str,
    ) -> dict[int, list[dict[str, Any]]]:
        """Get entities associated with each chunk.

        Args:
            chunk_ids: List of chunk IDs
            collection_id: Collection ID

        Returns:
            Dict mapping chunk_id to list of entity dicts
        """
        chunk_entity_map: dict[int, list[dict[str, Any]]] = {}

        for chunk_id in chunk_ids:
            if chunk_id is None:
                continue

            try:
                entities = await self.entity_repo.get_by_chunk(
                    chunk_id=chunk_id,
                    collection_id=collection_id,
                )
                chunk_entity_map[chunk_id] = [
                    {"id": e.id, "name": e.name, "type": e.entity_type}
                    for e in entities
                ]
            except Exception as e:
                logger.debug(f"Failed to get entities for chunk {chunk_id}: {e}")
                chunk_entity_map[chunk_id] = []

        return chunk_entity_map

    def _score_results(
        self,
        vector_results: list[dict[str, Any]],
        related_entity_ids: set[int],
        direct_match_ids: set[int],
        chunk_entity_map: dict[int, list[dict[str, Any]]],
        weight: float,
    ) -> list[dict[str, Any]]:
        """Calculate combined scores for results.

        Graph score is based on:
        - Whether chunk contains directly matched entities (higher)
        - Whether chunk contains related entities (lower)
        - Number of relevant entities in chunk

        Args:
            vector_results: Original results
            related_entity_ids: All related entity IDs (includes direct matches)
            direct_match_ids: Directly matched entity IDs (subset of related)
            chunk_entity_map: Entities by chunk
            weight: Graph weight in final score

        Returns:
            Enhanced results with scores
        """
        enhanced: list[dict[str, Any]] = []

        for result in vector_results:
            chunk_id = result.get("chunk_id")
            vector_score = result.get("score", 0.0)

            # Get entities in this chunk
            chunk_entities = chunk_entity_map.get(chunk_id, []) if chunk_id else []
            chunk_entity_ids = {e["id"] for e in chunk_entities}

            # Calculate graph score
            graph_score = 0.0
            matched_entities: list[dict[str, Any]] = []

            if chunk_entities:
                direct_matches = chunk_entity_ids & direct_match_ids
                related_matches = chunk_entity_ids & related_entity_ids

                # Direct matches worth more
                direct_score = len(direct_matches) * 0.3
                # Related matches worth less (exclude direct matches to avoid double-counting)
                related_score = (len(related_matches) - len(direct_matches)) * 0.1

                graph_score = min(1.0, direct_score + related_score)

                # Record matched entities for transparency
                for entity in chunk_entities:
                    if entity["id"] in related_entity_ids:
                        matched_entities.append({
                            "name": entity["name"],
                            "type": entity["type"],
                            "direct_match": entity["id"] in direct_match_ids,
                        })

            # Combined score formula: (1 - weight) * vector_score + weight * graph_score
            combined_score = (1 - weight) * vector_score + weight * graph_score

            enhanced.append({
                **result,
                "original_score": vector_score,
                "graph_score": graph_score,
                "score": combined_score,
                "matched_entities": matched_entities,
            })

        return enhanced

    def _add_empty_graph_fields(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add empty graph fields to results when enhancement not applied.

        Args:
            results: Original results

        Returns:
            Results with empty graph fields added
        """
        return [
            {
                **r,
                "original_score": r.get("score", 0.0),
                "graph_score": 0.0,
                "matched_entities": [],
            }
            for r in results
        ]


async def create_graph_search_service(
    db_session: AsyncSession,
    graph_weight: float = 0.2,
    max_hops: int = 2,
) -> GraphEnhancedSearchService:
    """Factory function for creating graph search service.

    Args:
        db_session: Database session
        graph_weight: Weight for graph scores (0-1)
        max_hops: Maximum graph traversal depth

    Returns:
        Configured GraphEnhancedSearchService instance
    """
    return GraphEnhancedSearchService(
        db_session=db_session,
        graph_weight=graph_weight,
        max_hops=max_hops,
    )

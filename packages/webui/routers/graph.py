"""Graph API endpoints for entity browsing and graph traversal."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from shared.contracts.graph import (
    EntityResponse,
    EntitySearchRequest,
    EntitySearchResponse,
    GraphEdge,
    GraphNode,
    GraphResponse,
    GraphStatsResponse,
    GraphTraversalRequest,
)
from shared.database import get_db
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.models import Collection
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.entity_repository import EntityRepository
from shared.database.repositories.relationship_repository import RelationshipRepository
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


async def get_collection_with_access_check(
    collection_id: str,
    user_id: int,
    db: AsyncSession,
) -> Collection:
    """Get collection and verify user access.

    Args:
        collection_id: Collection UUID
        user_id: User ID
        db: Database session

    Returns:
        Collection instance

    Raises:
        HTTPException: If collection not found or access denied
    """
    repo = CollectionRepository(db)

    try:
        collection = await repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id,
            user_id=user_id,
        )
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found",
        ) from e
    except AccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this collection",
        ) from e

    return collection


@router.get("/collections/{collection_id}/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    collection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> GraphStatsResponse:
    """Get graph statistics for a collection.

    Returns entity and relationship counts, broken down by type.
    If graph is not enabled for the collection, returns zero counts.
    """
    collection = await get_collection_with_access_check(collection_id, int(current_user["id"]), db)

    # Return empty stats if graph is not enabled
    if not collection.graph_enabled:
        return GraphStatsResponse(
            total_entities=0,
            entities_by_type={},
            total_relationships=0,
            relationships_by_type={},
            graph_enabled=False,
        )

    entity_repo = EntityRepository(db)
    rel_repo = RelationshipRepository(db)

    total_entities = await entity_repo.count_by_collection(collection_id)
    entities_by_type = await entity_repo.count_by_type(collection_id)
    total_relationships = await rel_repo.count_by_collection(collection_id)
    relationships_by_type = await rel_repo.count_by_type(collection_id)

    return GraphStatsResponse(
        total_entities=total_entities,
        entities_by_type=entities_by_type,
        total_relationships=total_relationships,
        relationships_by_type=relationships_by_type,
        graph_enabled=collection.graph_enabled,
    )


@router.post(
    "/collections/{collection_id}/entities/search",
    response_model=EntitySearchResponse,
)
async def search_entities(
    collection_id: str,
    request: EntitySearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> EntitySearchResponse:
    """Search entities in a collection.

    Supports name prefix search and filtering by entity type.
    Returns empty results if graph is not enabled for the collection.
    """
    collection = await get_collection_with_access_check(collection_id, int(current_user["id"]), db)

    if not collection.graph_enabled:
        return EntitySearchResponse(entities=[], total=0, has_more=False)

    entity_repo = EntityRepository(db)

    if request.query:
        # Search by name prefix
        entities = await entity_repo.search_by_name(
            collection_id=collection_id,
            query=request.query,
            entity_types=request.entity_types,
            limit=request.limit + 1,  # +1 to check has_more
        )
    elif request.entity_types:
        # Filter by type only
        entities = []
        for entity_type in request.entity_types:
            type_entities = await entity_repo.get_by_type(
                collection_id=collection_id,
                entity_type=entity_type,
                limit=request.limit + 1,
                offset=request.offset,
            )
            entities.extend(type_entities)
        entities = entities[: request.limit + 1]
    else:
        # Get all entities (paginated) - use empty string query
        entities = await entity_repo.search_by_name(
            collection_id=collection_id,
            query="",  # Empty query returns all
            limit=request.limit + 1,
        )

    has_more = len(entities) > request.limit
    if has_more:
        entities = entities[: request.limit]

    total = await entity_repo.count_by_collection(collection_id)

    return EntitySearchResponse(
        entities=[
            EntityResponse(
                id=e.id,
                name=e.name,
                entity_type=e.entity_type,
                document_id=e.document_id,
                chunk_id=e.chunk_id,
                confidence=e.confidence,
                created_at=e.created_at.isoformat() if e.created_at else "",
            )
            for e in entities
        ],
        total=total,
        has_more=has_more,
    )


@router.get(
    "/collections/{collection_id}/entities/{entity_id}",
    response_model=EntityResponse,
)
async def get_entity(
    collection_id: str,
    entity_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> EntityResponse:
    """Get a single entity by ID.

    Returns 404 if the entity is not found or graph is not enabled.
    """
    collection = await get_collection_with_access_check(collection_id, int(current_user["id"]), db)

    if not collection.graph_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found (graph not enabled)",
        )

    entity_repo = EntityRepository(db)

    try:
        entity = await entity_repo.get_by_id(entity_id, collection_id)
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found",
        ) from e

    return EntityResponse(
        id=entity.id,
        name=entity.name,
        entity_type=entity.entity_type,
        document_id=entity.document_id,
        chunk_id=entity.chunk_id,
        confidence=entity.confidence,
        created_at=entity.created_at.isoformat() if entity.created_at else "",
    )


@router.post(
    "/collections/{collection_id}/traverse",
    response_model=GraphResponse,
)
async def traverse_graph(
    collection_id: str,
    request: GraphTraversalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> GraphResponse:
    """Traverse the graph from a starting entity.

    Returns nodes and edges suitable for visualization.
    Returns empty graph if graph is not enabled for the collection.
    """
    collection = await get_collection_with_access_check(collection_id, int(current_user["id"]), db)

    if not collection.graph_enabled:
        return GraphResponse(nodes=[], edges=[])

    rel_repo = RelationshipRepository(db)

    try:
        subgraph = await rel_repo.get_subgraph(
            entity_ids=[request.entity_id],
            collection_id=collection_id,
            max_hops=request.max_hops,
            relationship_types=request.relationship_types,
        )
    except Exception as e:
        logger.error(f"Graph traversal error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error traversing graph",
        ) from e

    return GraphResponse(
        nodes=[GraphNode(id=n["id"], name=n["name"], type=n["type"], hop=n["hop"]) for n in subgraph["nodes"]],
        edges=[
            GraphEdge(
                id=e["id"],
                source=e["source"],
                target=e["target"],
                type=e["type"],
                confidence=e["confidence"],
            )
            for e in subgraph["edges"]
        ],
    )


@router.get("/collections/{collection_id}/entity-types")
async def get_entity_types(
    collection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, int]:
    """Get entity types and counts for a collection.

    Useful for building filter UI.
    Returns empty dict if graph is not enabled.
    """
    collection = await get_collection_with_access_check(collection_id, int(current_user["id"]), db)

    if not collection.graph_enabled:
        return {}

    entity_repo = EntityRepository(db)
    result: dict[str, int] = await entity_repo.count_by_type(collection_id)
    return result


@router.get("/collections/{collection_id}/relationship-types")
async def get_relationship_types(
    collection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, int]:
    """Get relationship types and counts for a collection.

    Useful for building filter UI.
    Returns empty dict if graph is not enabled.
    """
    collection = await get_collection_with_access_check(collection_id, int(current_user["id"]), db)

    if not collection.graph_enabled:
        return {}

    rel_repo = RelationshipRepository(db)
    result: dict[str, int] = await rel_repo.count_by_type(collection_id)
    return result

"""
Dimension validation utilities for embeddings and Qdrant collections.

This module provides utilities to validate that embedding dimensions match
the expected dimensions of Qdrant collections to prevent indexing and search failures.
"""

import logging
from typing import cast

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from shared.database.exceptions import DimensionMismatchError
from shared.embedding.models import get_model_config

logger = logging.getLogger(__name__)


def get_collection_dimension(client: QdrantClient, collection_name: str) -> int | None:
    """
    Get the vector dimension configured for a Qdrant collection.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection

    Returns:
        The vector dimension, or None if the collection doesn't exist

    Raises:
        Exception: If there's an error accessing the collection
    """
    try:
        collection_info = client.get_collection(collection_name)

        # Extract vector dimension from collection configuration
        if hasattr(collection_info.config, "params") and hasattr(collection_info.config.params, "vectors"):
            vectors_config = collection_info.config.params.vectors
            if vectors_config is not None and hasattr(vectors_config, "size"):
                return cast(int, vectors_config.size)
            if isinstance(vectors_config, dict) and "size" in vectors_config:
                return cast(int, vectors_config["size"])

        logger.warning(f"Could not extract vector dimension from collection {collection_name}")
        return None

    except UnexpectedResponse as e:
        if e.status_code == 404:
            logger.info(f"Collection {collection_name} does not exist")
            return None
        raise
    except Exception as e:
        logger.error(f"Error getting collection dimension for {collection_name}: {e}")
        raise


def get_model_dimension(model_name: str) -> int | None:
    """
    Get the expected dimension for an embedding model.

    Args:
        model_name: Name of the embedding model

    Returns:
        The model's output dimension, or None if unknown
    """
    model_config = get_model_config(model_name)
    if model_config:
        dimension: int = model_config.dimension
        return dimension

    logger.warning(f"Unknown model {model_name}, cannot determine dimension")
    return None


def validate_dimension_compatibility(
    expected_dimension: int,
    actual_dimension: int,
    collection_name: str | None = None,
    model_name: str | None = None,
) -> None:
    """
    Validate that dimensions match between expected and actual values.

    Args:
        expected_dimension: Expected dimension (e.g., from Qdrant collection)
        actual_dimension: Actual dimension (e.g., from embedding model)
        collection_name: Optional collection name for error context
        model_name: Optional model name for error context

    Raises:
        DimensionMismatchError: If dimensions don't match
    """
    if expected_dimension != actual_dimension:
        raise DimensionMismatchError(
            expected_dimension=expected_dimension,
            actual_dimension=actual_dimension,
            collection_name=collection_name,
            model_name=model_name,
        )


def validate_embedding_dimensions(embeddings: list[list[float]], expected_dimension: int) -> None:
    """
    Validate that all embeddings have the expected dimension.

    Args:
        embeddings: List of embedding vectors
        expected_dimension: Expected dimension for each embedding

    Raises:
        DimensionMismatchError: If any embedding has incorrect dimension
        ValueError: If embeddings list is empty
    """
    if not embeddings:
        raise ValueError("No embeddings provided for validation")

    for embedding in embeddings:
        if len(embedding) != expected_dimension:
            raise DimensionMismatchError(
                expected_dimension=expected_dimension,
                actual_dimension=len(embedding),
                collection_name=None,
                model_name=None,
            )


async def validate_collection_model_compatibility(
    client: QdrantClient,
    collection_name: str,
    model_name: str,
) -> None:
    """
    Validate that a model is compatible with a Qdrant collection's configured dimension.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        model_name: Name of the embedding model

    Raises:
        DimensionMismatchError: If model dimension doesn't match collection dimension
        ValueError: If collection doesn't exist or model is unknown
    """
    # Get collection dimension
    collection_dim = get_collection_dimension(client, collection_name)
    if collection_dim is None:
        raise ValueError(f"Collection {collection_name} does not exist or has no vector configuration")

    # Get model dimension
    model_dim = get_model_dimension(model_name)
    if model_dim is None:
        raise ValueError(f"Unknown model {model_name}, cannot validate dimension compatibility")

    # Validate compatibility
    validate_dimension_compatibility(
        expected_dimension=collection_dim,
        actual_dimension=model_dim,
        collection_name=collection_name,
        model_name=model_name,
    )

    logger.info(
        f"Validated dimension compatibility: collection {collection_name} "
        f"(dim={collection_dim}) matches model {model_name} (dim={model_dim})"
    )


def adjust_embeddings_dimension(
    embeddings: list[list[float]],
    target_dimension: int,
    normalize: bool = True,
) -> list[list[float]]:
    """
    Adjust embeddings to match a target dimension by truncating or padding.

    This should only be used when absolutely necessary, as it can affect search quality.

    Args:
        embeddings: List of embedding vectors
        target_dimension: Target dimension
        normalize: Whether to renormalize after adjustment

    Returns:
        Adjusted embeddings with the target dimension
    """
    if not embeddings:
        return embeddings

    adjusted_embeddings = []

    for embedding in embeddings:
        current_dim = len(embedding)

        if current_dim == target_dimension:
            adjusted = embedding
        elif current_dim > target_dimension:
            # Truncate
            adjusted = embedding[:target_dimension]
            logger.warning(f"Truncating embedding from {current_dim} to {target_dimension} dimensions")
        else:
            # Pad with zeros
            adjusted = embedding + [0.0] * (target_dimension - current_dim)
            logger.warning(f"Padding embedding from {current_dim} to {target_dimension} dimensions")

        if normalize and any(v != 0 for v in adjusted):
            # Renormalize to unit length
            norm = sum(v**2 for v in adjusted) ** 0.5
            adjusted = [v / norm for v in adjusted]

        adjusted_embeddings.append(adjusted)

    return adjusted_embeddings

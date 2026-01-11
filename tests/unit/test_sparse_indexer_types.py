from __future__ import annotations

import pytest


def test_sparse_vector_validates_length_match() -> None:
    from shared.plugins.types.sparse_indexer import SparseVector

    with pytest.raises(ValueError, match="same length"):
        SparseVector(indices=(1, 2), values=(0.1,), chunk_id="c1")


def test_sparse_query_vector_validates_length_match() -> None:
    from shared.plugins.types.sparse_indexer import SparseQueryVector

    with pytest.raises(ValueError, match="same length"):
        SparseQueryVector(indices=(1, 2), values=(0.1,))


def test_sparse_indexer_capabilities_validates_sparse_type() -> None:
    from shared.plugins.types.sparse_indexer import SparseIndexerCapabilities

    with pytest.raises(ValueError, match="Invalid sparse_type"):
        SparseIndexerCapabilities(sparse_type="bogus", max_tokens=1)

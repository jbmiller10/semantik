from __future__ import annotations


def test_reciprocal_rank_fusion_empty_inputs() -> None:
    from vecpipe.search.sparse_search import _reciprocal_rank_fusion

    assert _reciprocal_rank_fusion([], [], k=10) == []


def test_reciprocal_rank_fusion_prefers_dense_payload_and_adds_ranks() -> None:
    from vecpipe.search.sparse_search import _reciprocal_rank_fusion

    dense = [
        {"chunk_id": "a", "score": 0.9, "payload": {"from": "dense"}},
        {"chunk_id": "b", "score": 0.8, "payload": {"from": "dense"}},
    ]
    sparse = [
        {"chunk_id": "b", "score": 2.0, "payload": {"from": "sparse"}},
        {"chunk_id": "c", "score": 1.0, "payload": {"from": "sparse"}},
    ]

    fused = _reciprocal_rank_fusion(dense, sparse, k=10, rrf_k=60)

    ids = {r["chunk_id"] for r in fused}
    assert ids == {"a", "b", "c"}

    b = next(r for r in fused if r["chunk_id"] == "b")
    assert b["payload"]["from"] == "dense"
    assert b["_dense_rank"] == 2
    assert b["_sparse_rank"] == 1
    assert b["score"] > 0.0


def test_reciprocal_rank_fusion_normalizes_when_all_scores_equal() -> None:
    from vecpipe.search.sparse_search import _reciprocal_rank_fusion

    # Make two independent single-source results with identical rank contribution.
    dense = [{"chunk_id": "a", "score": 0.9}]
    sparse = [{"chunk_id": "b", "score": 0.1}]

    fused = _reciprocal_rank_fusion(dense, sparse, k=10, rrf_k=1_000_000)
    assert len(fused) == 2
    assert fused[0]["score"] == fused[1]["score"]
    assert 0.0 < fused[0]["score"] < 1.0

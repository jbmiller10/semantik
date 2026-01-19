from __future__ import annotations

import hashlib
import json

import pytest

from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository


@pytest.mark.asyncio()
async def test_add_relevance_computes_hash_and_list_unresolved_is_stable(
    db_session, test_user_db, collection_factory, document_factory
) -> None:
    repo = BenchmarkDatasetRepository(db_session)

    dataset = await repo.create(
        name="ds-1",
        owner_id=test_user_db.id,
        query_count=0,
        description=None,
        schema_version="1.0",
        metadata={},
    )
    collection = await collection_factory(owner_id=test_user_db.id)
    document = await document_factory(collection_id=collection.id)

    mapping = await repo.create_mapping(dataset_id=str(dataset.id), collection_id=collection.id)
    query = await repo.add_query(dataset_id=str(dataset.id), query_key="q1", query_text="hello", metadata={})

    doc_ref_a = {"uri": "file:///tmp/a.txt"}
    rel_a = await repo.add_relevance(
        query_id=int(query.id),
        mapping_id=int(mapping.id),
        doc_ref=doc_ref_a,
        relevance_grade=2,
        doc_ref_hash=None,
    )
    rel_b = await repo.add_relevance(
        query_id=int(query.id),
        mapping_id=int(mapping.id),
        doc_ref={"uri": "file:///tmp/b.txt"},
        relevance_grade=1,
        doc_ref_hash=None,
    )
    rel_c = await repo.add_relevance(
        query_id=int(query.id),
        mapping_id=int(mapping.id),
        doc_ref={"uri": "file:///tmp/c.txt"},
        relevance_grade=3,
        doc_ref_hash=None,
    )
    await db_session.commit()

    expected_hash = hashlib.sha256(json.dumps(doc_ref_a, sort_keys=True).encode()).hexdigest()
    assert rel_a.doc_ref_hash == expected_hash

    await repo.resolve_relevance(relevance_id=int(rel_a.id), document_id=str(document.id))
    await db_session.commit()

    unresolved = await repo.list_unresolved_relevance_for_mapping(int(mapping.id), after_id=0, limit=100)
    unresolved_ids = [int(r.id) for r in unresolved]
    assert unresolved_ids == sorted(unresolved_ids)
    assert int(rel_a.id) not in unresolved_ids
    assert set(unresolved_ids) == {int(rel_b.id), int(rel_c.id)}

    after_id = unresolved_ids[-1]
    assert await repo.list_unresolved_relevance_for_mapping(int(mapping.id), after_id=after_id, limit=100) == []

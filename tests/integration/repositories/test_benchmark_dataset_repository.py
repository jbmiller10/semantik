from __future__ import annotations

import hashlib
import json

import pytest

from shared.database.exceptions import AccessDeniedError, EntityAlreadyExistsError, ValidationError
from shared.database.models import MappingStatus
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


@pytest.mark.asyncio()
async def test_dataset_update_merges_metadata_and_trims_name(db_session, test_user_db) -> None:
    repo = BenchmarkDatasetRepository(db_session)
    dataset = await repo.create(
        name=" ds ",
        owner_id=test_user_db.id,
        query_count=1,
        description=None,
        schema_version="1.0",
        metadata={"a": 1},
    )
    await db_session.commit()

    updated = await repo.update(str(dataset.id), name="  ds2  ", metadata={"b": 2})
    await db_session.commit()
    assert updated.name == "ds2"
    assert updated.meta == {"a": 1, "b": 2}

    with pytest.raises(ValidationError):
        await repo.update(str(dataset.id), name="   ")


@pytest.mark.asyncio()
async def test_get_by_uuid_for_user_enforces_ownership(db_session, test_user_db, other_user_db) -> None:
    repo = BenchmarkDatasetRepository(db_session)
    dataset = await repo.create(
        name="ds-ownership",
        owner_id=test_user_db.id,
        query_count=0,
        description=None,
        schema_version="1.0",
        metadata={},
    )
    await db_session.commit()

    with pytest.raises(AccessDeniedError):
        await repo.get_by_uuid_for_user(str(dataset.id), other_user_db.id)


@pytest.mark.asyncio()
async def test_create_mapping_rejects_duplicates_and_update_sets_resolved_at(
    db_session, test_user_db, collection_factory
) -> None:
    repo = BenchmarkDatasetRepository(db_session)
    dataset = await repo.create(
        name="ds-map",
        owner_id=test_user_db.id,
        query_count=0,
        description=None,
        schema_version="1.0",
        metadata={},
    )
    collection = await collection_factory(owner_id=test_user_db.id)
    mapping = await repo.create_mapping(dataset_id=str(dataset.id), collection_id=collection.id)
    await db_session.commit()

    with pytest.raises(EntityAlreadyExistsError):
        await repo.create_mapping(dataset_id=str(dataset.id), collection_id=collection.id)

    updated = await repo.update_mapping_status(int(mapping.id), MappingStatus.RESOLVED, mapped_count=1, total_count=1)
    await db_session.commit()
    assert updated.mapping_status == MappingStatus.RESOLVED.value
    assert updated.resolved_at is not None


@pytest.mark.asyncio()
async def test_relevance_count_helpers(db_session, test_user_db, collection_factory, document_factory) -> None:
    repo = BenchmarkDatasetRepository(db_session)
    dataset = await repo.create(
        name="ds-counts",
        owner_id=test_user_db.id,
        query_count=0,
        description=None,
        schema_version="1.0",
        metadata={},
    )
    collection = await collection_factory(owner_id=test_user_db.id)
    mapping = await repo.create_mapping(dataset_id=str(dataset.id), collection_id=collection.id)
    query = await repo.add_query(dataset_id=str(dataset.id), query_key="q1", query_text="hello", metadata={})

    rel_a = await repo.add_relevance(
        query_id=int(query.id),
        mapping_id=int(mapping.id),
        doc_ref={"uri": "file:///tmp/a.txt"},
        relevance_grade=2,
    )
    rel_b = await repo.add_relevance(
        query_id=int(query.id),
        mapping_id=int(mapping.id),
        doc_ref={"uri": "file:///tmp/b.txt"},
        relevance_grade=1,
    )
    await db_session.commit()

    assert await repo.count_relevance_for_mapping(int(mapping.id)) == 2
    assert await repo.count_resolved_relevance_for_mapping(int(mapping.id)) == 0

    document = await document_factory(collection_id=collection.id)
    await repo.resolve_relevance(relevance_id=int(rel_a.id), document_id=str(document.id))
    await db_session.commit()

    assert await repo.count_resolved_relevance_for_mapping(int(mapping.id)) == 1

    relevances = await repo.get_relevance_for_mapping(int(mapping.id))
    assert {int(r.id) for r in relevances} == {int(rel_a.id), int(rel_b.id)}

    by_query = await repo.get_relevance_for_query(int(query.id), int(mapping.id))
    assert [int(r.relevance_grade) for r in by_query] == sorted(
        [int(r.relevance_grade) for r in by_query],
        reverse=True,
    )

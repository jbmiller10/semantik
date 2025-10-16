"""Unit coverage for pure helpers in partition utilities."""

from __future__ import annotations

from uuid import uuid4

import pytest
from packages.shared.database.partition_utils import PartitionValidation


class TestPartitionValidationUnit:
    """Pure validation tests that do not require database access."""

    def test_validate_uuid_variants(self) -> None:
        valid_uuid = str(uuid4())
        assert PartitionValidation.validate_uuid(valid_uuid) == valid_uuid
        assert PartitionValidation.validate_uuid(valid_uuid.upper()) == valid_uuid

    def test_validate_uuid_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            PartitionValidation.validate_uuid("not-a-uuid")
        with pytest.raises(ValueError):
            PartitionValidation.validate_uuid("", "document_id")
        with pytest.raises(TypeError):
            PartitionValidation.validate_uuid(12345)  # type: ignore[arg-type]

    def test_validate_chunk_data_normalizes_ids(self) -> None:
        payload = {
            "collection_id": str(uuid4()).upper(),
            "document_id": str(uuid4()).upper(),
            "chunk_index": 0,
            "content": "hello",
            "metadata": {"source": "unit"},
        }
        validated = PartitionValidation.validate_chunk_data(payload)
        assert validated["collection_id"].islower()
        assert validated["document_id"].islower()

    def test_validate_chunk_data_errors(self) -> None:
        with pytest.raises(ValueError):
            PartitionValidation.validate_chunk_data({})
        with pytest.raises(TypeError):
            PartitionValidation.validate_chunk_data("wrong")
        with pytest.raises(ValueError):
            PartitionValidation.validate_chunk_data({"collection_id": str(uuid4()), "chunk_index": -1})

    def test_validate_batch_size_limits(self) -> None:
        PartitionValidation.validate_batch_size(list(range(PartitionValidation.MAX_BATCH_SIZE)))
        with pytest.raises(ValueError):
            PartitionValidation.validate_batch_size(list(range(PartitionValidation.MAX_BATCH_SIZE + 1)))

    def test_sanitize_string_behaviour(self) -> None:
        assert PartitionValidation.sanitize_string("abc\x00def") == "abcdef"
        assert PartitionValidation.sanitize_string(42) == "42"

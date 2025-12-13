"""Repository-level tests for document artifacts (lightweight, no DB)."""

import pytest

from shared.database.repositories.document_artifact_repository import DocumentArtifactRepository
from shared.utils.hashing import compute_content_hash


class _Session:
    def __init__(self):
        self.added = []
        self.executed = []
        self.flushed = False

    async def execute(self, stmt):
        self.executed.append(stmt)
        return

    async def flush(self):  # pragma: no cover - trivial
        self.flushed = True

    def add(self, obj):
        self.added.append(obj)


@pytest.mark.asyncio()
async def test_create_or_replace_recomputes_hash_for_truncated_text():
    session = _Session()
    repo = DocumentArtifactRepository(session, max_artifact_bytes=5)

    original = "ab🙂c"  # 2 ASCII bytes + 4-byte emoji + 1 byte = 7 bytes
    original_hash = compute_content_hash(original)

    artifact = await repo.create_or_replace(
        document_id="doc",
        collection_id="col",
        content=original,
        mime_type="text/plain",
        content_hash=original_hash,
    )

    assert artifact.is_truncated is True
    assert artifact.content_text == "ab"
    assert artifact.size_bytes == 2
    assert artifact.content_hash == compute_content_hash("ab")


@pytest.mark.asyncio()
async def test_create_or_replace_recomputes_hash_for_truncated_bytes():
    session = _Session()
    repo = DocumentArtifactRepository(session, max_artifact_bytes=5)

    original = b"abcdef"
    original_hash = compute_content_hash(original)

    artifact = await repo.create_or_replace(
        document_id="doc",
        collection_id="col",
        content=original,
        mime_type="application/octet-stream",
        content_hash=original_hash,
    )

    assert artifact.is_truncated is True
    assert artifact.content_bytes == b"abcde"
    assert artifact.size_bytes == 5
    assert artifact.content_hash == compute_content_hash(b"abcde")

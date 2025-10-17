"""Integration tests for DocumentScanningService with real repositories."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from packages.shared.database.models import Document
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.webui.services.document_scanning_service import DocumentScanningService
from sqlalchemy import select


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestDocumentScanningServiceIntegration:
    """Verify directory scanning persists documents and deduplicates content."""

    @pytest.fixture()
    def service(self, db_session):
        return DocumentScanningService(db_session, DocumentRepository(db_session))

    async def test_scan_directory_registers_documents(self, service, db_session, collection_factory, tmp_path: Path, test_user_db):
        collection = await collection_factory(owner_id=test_user_db.id)
        source = tmp_path / "docs"
        source.mkdir()
        unique_marker = uuid4().hex
        (source / f"one_{unique_marker}.pdf").write_bytes(("pdf" + unique_marker).encode() * 100)
        (source / f"two_{unique_marker}.txt").write_text(f"document two {unique_marker}")
        (source / "ignore.bin").write_bytes(b"not supported")

        stats = await service.scan_directory_and_register_documents(collection.id, str(source))
        await db_session.commit()

        assert stats["new_documents_registered"] == 2
        assert stats["duplicate_documents_skipped"] == 0
        assert stats["total_documents_found"] == 2

        result = await db_session.execute(select(Document).where(Document.collection_id == collection.id))
        documents = result.scalars().all()
        assert len(documents) == 2
        paths = {doc.file_path for doc in documents}
        expected_paths = {
            str(source / f"one_{unique_marker}.pdf"),
            str(source / f"two_{unique_marker}.txt"),
        }
        assert paths == expected_paths

    async def test_scan_directory_skips_duplicates(self, service, db_session, collection_factory, tmp_path: Path, test_user_db):
        collection = await collection_factory(owner_id=test_user_db.id)
        source = tmp_path / "docs"
        source.mkdir()
        file_path = source / "duplicate.txt"
        unique_marker = uuid4().hex
        file_path.write_text(f"duplicate content {unique_marker}")

        first = await service.scan_directory_and_register_documents(collection.id, str(source))
        await db_session.commit()
        second = await service.scan_directory_and_register_documents(collection.id, str(source))
        await db_session.commit()

        assert first["new_documents_registered"] == 1
        assert second["duplicate_documents_skipped"] == 1

        result = await db_session.execute(select(Document).where(Document.collection_id == collection.id))
        documents = result.scalars().all()
        assert len(documents) == 1

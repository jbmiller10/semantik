"""Integration tests for DocumentScanningService."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import Document
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.webui.services.document_scanning_service import DocumentScanningService

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


@pytest.fixture()
def document_service(db_session: AsyncSession) -> DocumentScanningService:
    repo = DocumentRepository(db_session)
    return DocumentScanningService(db_session, repo)


async def _write_documents(root: Path) -> None:
    (root / "files").mkdir(parents=True, exist_ok=True)
    (root / "files" / "introduction.pdf").write_bytes(b"pdf" * 100)
    (root / "files" / "notes.txt").write_text("notes")
    (root / "files" / "skip.xyz").write_text("unsupported")


async def test_scan_directory_registers_documents(
    document_service: DocumentScanningService,
        db_session: AsyncSession,
    collection_factory,
    test_user_db,
    tmp_path: Path,
) -> None:
    await _write_documents(tmp_path)
    collection = await collection_factory(owner_id=test_user_db.id)

    stats = await document_service.scan_directory_and_register_documents(
        collection_id=collection.id,
        source_path=str(tmp_path / "files"),
        recursive=False,
    )

    assert stats["total_documents_found"] == 2
    assert stats["new_documents_registered"] == 2
    assert stats["duplicate_documents_skipped"] == 0

    stored = await db_session.execute(select(Document).where(Document.collection_id == collection.id))
    documents = stored.scalars().all()
    assert len(documents) == 2
    assert {doc.file_name for doc in documents} == {"introduction.pdf", "notes.txt"}


async def test_duplicate_documents_are_skipped(
    document_service: DocumentScanningService,
    db_session: AsyncSession,
    collection_factory,
    test_user_db,
    tmp_path: Path,
) -> None:
    await _write_documents(tmp_path)
    collection = await collection_factory(owner_id=test_user_db.id)

    await document_service.scan_directory_and_register_documents(
        collection_id=collection.id,
        source_path=str(tmp_path / "files"),
        recursive=False,
    )
    stats = await document_service.scan_directory_and_register_documents(
        collection_id=collection.id,
        source_path=str(tmp_path / "files"),
        recursive=False,
    )

    assert stats["duplicate_documents_skipped"] == 2

    stored = await db_session.execute(select(Document).where(Document.collection_id == collection.id))
    assert len(stored.scalars().all()) == 2


async def test_recursive_scan_discovers_nested_files(
    document_service: DocumentScanningService,
    collection_factory,
    test_user_db,
    tmp_path: Path,
) -> None:
    await _write_documents(tmp_path)
    nested_dir = tmp_path / "files" / "nested"
    nested_dir.mkdir()
    (nested_dir / "slides.pptx").write_bytes(b"pptx" * 10)

    collection = await collection_factory(owner_id=test_user_db.id)

    stats = await document_service.scan_directory_and_register_documents(
        collection_id=collection.id,
        source_path=str(tmp_path / "files"),
        recursive=True,
    )

    assert stats["total_documents_found"] == 3
    assert stats["new_documents_registered"] == 3

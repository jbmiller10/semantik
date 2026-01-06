"""Unit tests for DocumentScanningService helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from webui.services.document_scanning_service import MAX_FILE_SIZE, DocumentScanningService


@pytest.fixture()
def service() -> DocumentScanningService:
    db_session = MagicMock()
    document_repo = MagicMock()
    return DocumentScanningService(db_session, document_repo)


def test_get_mime_type_fallback(monkeypatch: pytest.MonkeyPatch, service: DocumentScanningService) -> None:
    monkeypatch.setattr("webui.services.document_scanning_service.mimetypes.guess_type", lambda _path: (None, None))

    file_path = Path("example.md")
    assert service._get_mime_type(file_path) == "text/markdown"


@pytest.mark.asyncio()
async def test_scan_document_rejects_unsupported_type(tmp_path: Path, service: DocumentScanningService) -> None:
    file_path = tmp_path / "image.bin"
    file_path.write_text("data")

    with pytest.raises(ValueError, match="Unsupported document type"):
        await service.scan_document("collection", str(file_path))


@pytest.mark.asyncio()
async def test_scan_document_success(tmp_path: Path, service: DocumentScanningService) -> None:
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello")

    service._register_file = AsyncMock(return_value={"document_id": "doc-1", "is_new": True, "file_size": 5})

    result = await service.scan_document("collection", str(file_path))

    assert result["document_id"] == "doc-1"
    assert result["is_new"] is True
    assert result["file_name"] == "doc.txt"
    assert result["mime_type"] == "text/plain"


@pytest.mark.asyncio()
async def test_register_file_rejects_large_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, service: DocumentScanningService
) -> None:
    file_path = tmp_path / "big.pdf"
    file_path.write_text("data")

    original_stat = Path.stat

    def _fake_stat(self: Path):
        if self == file_path:

            class _Stat:
                st_size = MAX_FILE_SIZE + 1

            return _Stat()
        return original_stat(self)

    monkeypatch.setattr(Path, "stat", _fake_stat)

    with pytest.raises(ValueError, match="Document too large"):
        await service._register_file("collection", file_path)

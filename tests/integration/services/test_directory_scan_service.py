"""Integration coverage for DirectoryScanService using the real filesystem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from shared.config import settings
from webui.api.schemas import DirectoryScanProgress
from webui.services.directory_scan_service import MAX_FILE_SIZE, DirectoryScanService

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio()
class TestDirectoryScanServiceIntegration:
    """Exercise DirectoryScanService end-to-end with filesystem interactions."""

    @pytest.fixture(autouse=True)
    def allow_test_scan_roots(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Allow scans within each test's temp directory."""
        monkeypatch.setattr(settings, "_document_allowed_roots", (tmp_path.resolve(),), raising=False)

    @pytest.fixture()
    def service(self) -> DirectoryScanService:
        return DirectoryScanService()

    @pytest.fixture()
    def capture_progress(self, monkeypatch):
        """Capture WebSocket progress events emitted by the service."""
        messages: list[DirectoryScanProgress] = []

        async def fake_send(_scan_id: str, payload: dict[str, Any]) -> None:  # matches send_to_operation signature
            messages.append(DirectoryScanProgress(**payload))

        monkeypatch.setattr(
            "webui.services.directory_scan_service.ws_manager.send_to_operation",
            fake_send,
        )
        return messages

    async def test_scan_directory_preview_collects_supported_files(
        self, service, tmp_path: Path, capture_progress
    ) -> None:
        base = tmp_path / "docs"
        base.mkdir()
        (base / "one.pdf").write_bytes(b"pdf" * 200)
        (base / "two.docx").write_bytes(b"docx" * 180)
        (base / "ignore.xyz").write_text("skip")
        subdir = base / "nested"
        subdir.mkdir()
        (subdir / "note.md").write_text("markdown")

        result = await service.scan_directory_preview(
            path=str(base),
            scan_id=f"scan-{uuid4().hex[:8]}",
            user_id=123,
            recursive=True,
        )

        file_paths = {file.file_path for file in result.files}
        assert file_paths == {str(base / "one.pdf"), str(base / "two.docx"), str(subdir / "note.md")}
        assert result.total_files == 3
        assert result.total_size > 0
        assert all(isinstance(msg, DirectoryScanProgress) for msg in capture_progress)

    async def test_non_recursive_scan_ignores_subdirectories(self, service, tmp_path: Path, capture_progress) -> None:
        base = tmp_path / "docs"
        base.mkdir()
        (base / "root.txt").write_text("root")
        nested = base / "nested"
        nested.mkdir()
        (nested / "inner.txt").write_text("inner")

        result = await service.scan_directory_preview(
            path=str(base),
            scan_id=f"scan-{uuid4().hex[:8]}",
            user_id=456,
            recursive=False,
        )

        assert {file.file_path for file in result.files} == {str(base / "root.txt")}
        assert capture_progress  # progress events still emitted

    async def test_large_file_emits_warning(self, service, tmp_path: Path, capture_progress) -> None:
        base = tmp_path / "docs"
        base.mkdir()
        (base / "large.pdf").write_bytes(b"x" * (MAX_FILE_SIZE + 1))

        result = await service.scan_directory_preview(
            path=str(base),
            scan_id=f"scan-{uuid4().hex[:8]}",
            user_id=789,
            recursive=True,
        )

        assert not result.files  # file excluded
        assert any("File too large" in warning for warning in result.warnings)

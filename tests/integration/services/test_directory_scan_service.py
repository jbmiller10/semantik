"""Integration tests for DirectoryScanService using the real filesystem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from packages.webui.services.directory_scan_service import DirectoryScanService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from packages.webui.api.schemas import DirectoryScanFile

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


class RecordingDirectoryScanService(DirectoryScanService):
    """Directory scan service that records progress events for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def _send_progress(
        self,
        *,
        channel_id: str,
        scan_id: str,
        msg_type: str,
        data: dict[str, Any],
    ) -> None:
        # Record progress instead of sending over WebSocket
        self.events.append((msg_type, data))

    async def _scan_recursive(
        self,
        directory: Path,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> AsyncIterator[tuple[DirectoryScanFile | None, str | None]]:
        async for item in super()._scan_recursive(directory, include_patterns, exclude_patterns):
            yield item


@pytest.fixture()
def directory_service() -> RecordingDirectoryScanService:
    return RecordingDirectoryScanService()


async def _create_files(root: Path) -> None:
    (root / "nested").mkdir()
    (root / "document1.pdf").write_bytes(b"pdf" * 100)
    (root / "document2.docx").write_bytes(b"docx" * 120)
    (root / "skip.xyz").write_bytes(b"xyz")
    (root / "nested" / "notes.md").write_bytes(b"markdown" * 50)
    (root / "nested" / "too_large.pdf").write_bytes(b"x" * (600 * 1024 * 1024))


async def test_scan_directory_recursive(directory_service: RecordingDirectoryScanService, tmp_path: Path) -> None:
    await _create_files(tmp_path)

    response = await directory_service.scan_directory_preview(
        path=str(tmp_path),
        scan_id="scan-recursive",
        user_id=1,
        recursive=True,
    )

    returned_paths = {file.file_path for file in response.files}
    assert returned_paths == {
        str(tmp_path / "document1.pdf"),
        str(tmp_path / "document2.docx"),
        str(tmp_path / "nested" / "notes.md"),
    }
    assert response.total_files == 3
    assert all(event[0] in {"counting", "progress", "completed", "warning"} for event in directory_service.events)


async def test_scan_directory_non_recursive(directory_service: RecordingDirectoryScanService, tmp_path: Path) -> None:
    await _create_files(tmp_path)

    response = await directory_service.scan_directory_preview(
        path=str(tmp_path),
        scan_id="scan-non-recursive",
        user_id=1,
        recursive=False,
    )

    returned_paths = {file.file_path for file in response.files}
    assert returned_paths == {
        str(tmp_path / "document1.pdf"),
        str(tmp_path / "document2.docx"),
    }


async def test_scan_directory_with_patterns(directory_service: RecordingDirectoryScanService, tmp_path: Path) -> None:
    await _create_files(tmp_path)

    response = await directory_service.scan_directory_preview(
        path=str(tmp_path),
        scan_id="scan-pattern",
        user_id=1,
        recursive=True,
        include_patterns=["*.pdf"],
        exclude_patterns=["*too_large*"],
    )

    returned_paths = {file.file_path for file in response.files}
    assert returned_paths == {str(tmp_path / "document1.pdf")}

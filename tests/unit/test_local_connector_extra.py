"""Additional unit tests for LocalFileConnector helpers."""

from __future__ import annotations

from pathlib import Path

from shared.connectors import local as local_module
from shared.connectors.local import MAX_FILE_SIZE, LocalFileConnector, _process_file_worker
from shared.text_processing.parsers import ExtractionFailedError, ParseResult


def test_is_safe_path_allows_within_base(tmp_path: Path) -> None:
    connector = LocalFileConnector({"path": str(tmp_path)})
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello")

    assert connector._is_safe_path(file_path, tmp_path) is True


def test_is_safe_path_blocks_outside_base(tmp_path: Path) -> None:
    connector = LocalFileConnector({"path": str(tmp_path)})
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope")

    assert connector._is_safe_path(outside, tmp_path) is False


def test_should_include_file_patterns(tmp_path: Path) -> None:
    connector = LocalFileConnector(
        {
            "path": str(tmp_path),
            "include_patterns": ["*.md"],
            "exclude_patterns": ["secret*"],
        }
    )

    included = tmp_path / "notes.md"
    excluded = tmp_path / "secret.md"
    other = tmp_path / "notes.txt"

    assert connector._should_include_file(included) is True
    assert connector._should_include_file(excluded) is False
    assert connector._should_include_file(other) is False


def test_process_file_worker_success(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello")

    def mock_parse_content(_bytes, **_kw):
        return ParseResult(
            text="hello",
            elements=[],
            metadata={
                "filename": "doc.txt",
                "file_extension": ".txt",
                "file_type": "txt",
                "mime_type": None,
                "parser": "text",
                "foo": "bar",
            },
        )

    monkeypatch.setattr(local_module, "parse_content", mock_parse_content)
    monkeypatch.setattr(local_module, "compute_content_hash", lambda _content: "hash")

    result = _process_file_worker(str(file_path))

    assert result["status"] == "success"
    data = result["data"]
    assert data["content"] == "hello"
    assert data["metadata"]["file_size"] == file_path.stat().st_size
    assert data["content_hash"] == "hash"


def test_process_file_worker_empty_content(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("ignored")

    def mock_parse_content(_bytes, **_kw):
        return ParseResult(
            text="   ",
            elements=[],
            metadata={},
        )

    monkeypatch.setattr(local_module, "parse_content", mock_parse_content)

    result = _process_file_worker(str(file_path))

    assert result["status"] == "skipped"
    assert result["reason"] == "empty_content"


def test_process_file_worker_parse_error(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "bad.txt"
    file_path.write_text("ignored")

    def _boom(_bytes, **_kw):
        raise ExtractionFailedError("parse failed")

    monkeypatch.setattr(local_module, "parse_content", _boom)

    result = _process_file_worker(str(file_path))

    assert result["status"] == "error"
    assert "Failed to parse" in result["reason"]


def test_process_file_worker_skips_large_files(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "big.txt"
    file_path.write_text("ignored")

    original_stat = Path.stat

    def _fake_stat(self: Path):
        if self == file_path:

            class _Stat:
                st_size = MAX_FILE_SIZE + 1

            return _Stat()
        return original_stat(self)

    monkeypatch.setattr(Path, "stat", _fake_stat)

    result = _process_file_worker(str(file_path))

    assert result["status"] == "skipped"
    assert result["reason"] == "file_too_large"

"""Additional unit tests for LocalFileConnector helpers."""

from __future__ import annotations

from pathlib import Path

from shared.connectors.local import LocalFileConnector


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


def test_should_include_file_no_patterns(tmp_path: Path) -> None:
    """Test that all files are included when no patterns specified."""
    connector = LocalFileConnector({"path": str(tmp_path)})

    txt_file = tmp_path / "file.txt"
    md_file = tmp_path / "file.md"
    py_file = tmp_path / "file.py"

    assert connector._should_include_file(txt_file) is True
    assert connector._should_include_file(md_file) is True
    assert connector._should_include_file(py_file) is True


def test_should_include_file_only_exclude(tmp_path: Path) -> None:
    """Test exclude pattern without include pattern."""
    connector = LocalFileConnector(
        {
            "path": str(tmp_path),
            "exclude_patterns": ["*.log", "temp*"],
        }
    )

    normal_file = tmp_path / "data.txt"
    log_file = tmp_path / "debug.log"
    temp_file = tmp_path / "temp_cache.txt"

    assert connector._should_include_file(normal_file) is True
    assert connector._should_include_file(log_file) is False
    assert connector._should_include_file(temp_file) is False


def test_infer_content_type_code_extensions(tmp_path: Path) -> None:
    """Test _infer_content_type for code files."""
    connector = LocalFileConnector({"path": str(tmp_path)})

    assert connector._infer_content_type(Path("file.py")) == "code"
    assert connector._infer_content_type(Path("file.js")) == "code"
    assert connector._infer_content_type(Path("file.ts")) == "code"
    assert connector._infer_content_type(Path("file.java")) == "code"
    assert connector._infer_content_type(Path("file.go")) == "code"
    assert connector._infer_content_type(Path("file.rs")) == "code"
    assert connector._infer_content_type(Path("file.c")) == "code"
    assert connector._infer_content_type(Path("file.cpp")) == "code"
    assert connector._infer_content_type(Path("file.h")) == "code"
    assert connector._infer_content_type(Path("file.rb")) == "code"
    assert connector._infer_content_type(Path("file.php")) == "code"


def test_infer_content_type_document_extensions(tmp_path: Path) -> None:
    """Test _infer_content_type for document files."""
    connector = LocalFileConnector({"path": str(tmp_path)})

    assert connector._infer_content_type(Path("file.pdf")) == "document"
    assert connector._infer_content_type(Path("file.docx")) == "document"
    assert connector._infer_content_type(Path("file.txt")) == "document"
    assert connector._infer_content_type(Path("file.md")) == "document"
    assert connector._infer_content_type(Path("file.html")) == "document"
    assert connector._infer_content_type(Path("file.unknown")) == "document"

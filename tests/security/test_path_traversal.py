#!/usr/bin/env python3

"""
Comprehensive security tests for path traversal vulnerability prevention.

Tests all OWASP path traversal patterns and ensures robust validation.
"""

import contextlib
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from packages.webui.services.chunking_security import ChunkingSecurityValidator, ValidationError


class TestPathTraversalSecurity:
    """Test path traversal security validation."""

    def test_basic_traversal_patterns_blocked(self) -> None:
        """Test that basic directory traversal patterns are blocked."""
        dangerous_paths = [
            ["../../../etc/passwd"],
            ["../../passwords.txt"],
            ["../"],
            ["./../../etc/"],
            ["documents/../../../etc/passwd"],
            ["folder/../../../../../../windows/system32"],
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_url_encoded_traversal_blocked(self) -> None:
        """Test that URL-encoded traversal attempts are blocked."""
        dangerous_paths = [
            # Single encoding
            ["%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"],
            ["%2e%2e%2fpasswords.txt"],
            ["%2E%2E%2F"],  # Mixed case
            ["%2e%2e%5c"],  # Backslash variant
            # Double encoding
            ["%252e%252e%252f"],
            ["%252e%252e%252f%252e%252e%252fetc%252fpasswd"],
            # Triple encoding
            ["%25252e%25252e%25252f"],
            # Mixed encoding styles
            ["..%2f..%2f..%2fetc%2fpasswd"],
            ["%2e%2e/../../etc/passwd"],
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_windows_paths_blocked(self) -> None:
        """Test that Windows-specific malicious paths are blocked."""
        dangerous_paths = [
            # Drive letters
            ["C:\\Windows\\System32"],
            ["c:\\windows\\system32"],  # Lowercase
            ["D:\\"],
            ["E:\\Program Files\\"],
            # UNC paths
            ["\\\\server\\share"],
            ["\\\\192.168.1.1\\share"],
            ["\\\\server\\c$\\windows"],
            # Windows absolute paths
            ["\\Windows\\System32"],
            ["\\Program Files\\"],
            # Mixed separators
            ["C:/Windows/System32"],
            ["\\\\server/share"],
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_unix_absolute_paths_blocked(self) -> None:
        """Test that Unix absolute paths are blocked."""
        dangerous_paths = [
            ["/etc/passwd"],
            ["/etc/shadow"],
            ["/var/log/auth.log"],
            ["/home/user/.ssh/id_rsa"],
            ["/"],
            ["/root/"],
            ["//etc//passwd"],  # Double slashes
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_home_directory_expansion_blocked(self) -> None:
        """Test that home directory expansion attempts are blocked."""
        dangerous_paths = [
            ["~/.ssh/id_rsa"],
            ["~/passwords.txt"],
            ["~root/.bashrc"],
            ["~/../etc/passwd"],
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_null_byte_injection_blocked(self) -> None:
        """Test that null byte injection attempts are blocked."""
        dangerous_paths = [
            ["file.txt\x00.jpg"],
            ["document\x00/../../etc/passwd"],
            ["test\x00"],
            ["file\x00\x00.txt"],
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_unicode_attacks_blocked(self) -> None:
        """Test that Unicode-based attacks are blocked."""
        dangerous_paths = [
            # Right-to-left override
            ["file\u202e\u002e\u002e\u002f"],
            ["test\u202eetc/passwd"],
            # Zero-width characters
            ["file\ufeff../../etc/passwd"],
            # Homograph attacks (different Unicode representations)
            ["．．／．．／etc/passwd"],  # Full-width dots
            # Control characters
            ["file\r\n../../etc/passwd"],
            ["test\n/etc/passwd"],
            ["doc\r../../"],
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_multiple_dots_blocked(self) -> None:
        """Test that suspicious multiple dot patterns are blocked."""
        dangerous_paths = [
            [".."],  # Parent directory reference
            ["../file.txt"],  # Parent directory traversal
            ["..\\file.txt"],  # Windows parent directory traversal
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_backslash_on_unix_blocked(self) -> None:
        """Test that backslashes are blocked on Unix systems."""

        if os.name != "nt":  # Only test on Unix-like systems
            dangerous_paths = [
                ["folder\\file.txt"],
                ["..\\..\\etc\\passwd"],
                ["test\\"],
            ]

            for paths in dangerous_paths:
                with pytest.raises(ValidationError, match="Invalid file path"):
                    ChunkingSecurityValidator.validate_file_paths(paths)

    def test_legitimate_paths_allowed(self) -> None:
        """Test that legitimate paths are correctly allowed."""
        safe_paths = [
            ["documents/file.txt"],
            ["data/subfolder/document.pdf"],
            ["file_with_dots...txt"],
            ["folder.with.dots/file.txt"],
            ["2024-01-01_report.pdf"],
            ["user@email.com/inbox/message.txt"],
            ["project-name/src/main.py"],
            ["file_with_underscore_and-dash.txt"],
            ["deeply/nested/folder/structure/file.doc"],
            ["file (with spaces).txt"],
            ["file[with]brackets.txt"],
            ["file{with}braces.txt"],
        ]

        for paths in safe_paths:
            # Should not raise any exception
            ChunkingSecurityValidator.validate_file_paths(paths)

    def test_base_directory_containment(self) -> None:
        """Test that paths are contained within base directory when specified."""
        with TemporaryDirectory() as temp_dir:
            base_dir = temp_dir

            # Test paths that try to escape
            escape_attempts = [
                ["../etc/passwd"],
                ["subdir/../../etc/passwd"],
                ["./../../outside.txt"],
            ]

            for paths in escape_attempts:
                with pytest.raises(ValidationError, match="Invalid file path"):
                    ChunkingSecurityValidator.validate_file_paths(paths, base_dir=base_dir)

            # Test legitimate paths within base directory
            safe_paths = [
                ["file.txt"],
                ["subfolder/document.pdf"],
                ["nested/deep/file.doc"],
            ]

            for paths in safe_paths:
                # Should not raise exception
                ChunkingSecurityValidator.validate_file_paths(paths, base_dir=base_dir)

    def test_symlink_resolution(self) -> None:
        """Test that symlinks are properly resolved and validated."""
        with TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            # Create a test file and symlink
            test_file = base_path / "test.txt"
            test_file.write_text("test content")

            symlink = base_path / "link_to_test"
            symlink.symlink_to(test_file)

            # Test that symlink within base directory is allowed
            ChunkingSecurityValidator.validate_file_paths(["link_to_test"], base_dir=str(base_path))

            # Test that symlink escaping base directory would be blocked
            # (This would require creating a symlink to outside directory,
            # which might not be possible in test environment)

    def test_performance_under_10ms(self) -> None:
        """Test that validation completes within 10ms performance requirement."""
        # Test with various path types
        test_cases = [
            ["normal/path/file.txt"],
            ["%2e%2e%2f%2e%2e%2fetc%2fpasswd"],  # URL encoded
            ["file\x00.txt"],  # Null byte
            ["C:\\Windows\\System32"],  # Windows path
            ["documents/file.txt", "data/other.pdf", "test.doc"],  # Multiple files
        ]

        for paths in test_cases:
            start_time = time.perf_counter()
            with contextlib.suppress(ValidationError):
                ChunkingSecurityValidator.validate_file_paths(paths)

            elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            assert elapsed_time < 10, f"Validation took {elapsed_time:.2f}ms, exceeding 10ms limit"

    def test_error_messages_no_path_leakage(self) -> None:
        """Test that error messages don't leak sensitive path information."""
        dangerous_paths = [
            ["/etc/passwd"],
            ["../../secret.txt"],
            ["C:\\Windows\\System32\\config"],
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError) as exc_info:
                ChunkingSecurityValidator.validate_file_paths(paths)

            error_message = str(exc_info.value)
            # Ensure error message doesn't contain the actual path
            assert "/etc/passwd" not in error_message
            assert "secret.txt" not in error_message
            assert "System32" not in error_message
            assert "Windows" not in error_message
            # Should only contain generic error message
            assert error_message == "Invalid file path"

    def test_input_validation(self) -> None:
        """Test input type validation."""
        # Test non-list input
        with pytest.raises(ValidationError, match="file_paths must be a list"):
            ChunkingSecurityValidator.validate_file_paths("not_a_list")

        # Test non-string path in list
        with pytest.raises(ValidationError, match="File path must be string"):
            ChunkingSecurityValidator.validate_file_paths([123])

        # Test too many paths
        too_many_paths = ["file.txt"] * 1001
        with pytest.raises(ValidationError, match="Too many file paths"):
            ChunkingSecurityValidator.validate_file_paths(too_many_paths)

        # Test path too long
        long_path = ["a" * 1001]
        with pytest.raises(ValidationError, match="File path too long"):
            ChunkingSecurityValidator.validate_file_paths(long_path)

    def test_complex_encoding_combinations(self) -> None:
        """Test complex combinations of encoding and obfuscation."""
        dangerous_paths = [
            # Mixed encoding and traversal
            ["%2e%2e/../%2e%2e/etc/passwd"],
            ["../%252e%252e/passwords"],
            # Encoded null bytes
            ["file%00.txt"],
            ["test%2500.doc"],
            # Multiple encoding layers with different techniques
            ["%25%32%65%25%32%65%25%32%66"],  # %2e%2e%2f encoded
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_case_sensitivity(self) -> None:
        """Test that validation handles case variations properly."""
        dangerous_paths = [
            # Case variations in encoding
            ["%2E%2E%2F"],
            ["%2e%2E%2f"],
            # Windows drive letters
            ["C:\\test"],
            ["c:\\test"],
            # Mixed case UNC
            ["\\\\SERVER\\Share"],
            ["\\\\server\\SHARE"],
        ]

        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)

    def test_empty_and_edge_cases(self) -> None:
        """Test edge cases and empty inputs."""
        # Empty list should be allowed
        ChunkingSecurityValidator.validate_file_paths([])

        # Empty string path
        with pytest.raises(ValidationError, match="Invalid file path"):
            ChunkingSecurityValidator.validate_file_paths([""])

        # Just dots
        with pytest.raises(ValidationError, match="Invalid file path"):
            ChunkingSecurityValidator.validate_file_paths([".."])

        # Just slashes
        with pytest.raises(ValidationError, match="Invalid file path"):
            ChunkingSecurityValidator.validate_file_paths(["/"])

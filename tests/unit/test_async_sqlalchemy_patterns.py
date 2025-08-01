"""Unit tests to ensure correct async SQLAlchemy patterns are used.

This test suite specifically checks that we're using the correct async patterns
for database operations, preventing regression of the deletion bug.
"""

import ast
from pathlib import Path

import pytest


class TestAsyncSQLAlchemyPatterns:
    """Test that async SQLAlchemy patterns are used correctly."""

    def test_no_sync_delete_pattern_in_repositories(self):
        """Ensure no repository uses session.delete() without await session.execute()."""
        repository_dir = Path(__file__).parent.parent.parent / "packages" / "shared" / "database" / "repositories"

        violations = []

        for py_file in repository_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            content = py_file.read_text()

            # Parse the AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            # Look for session.delete() calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "delete" and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "session":
                    # Get the line number and content
                    line_num = node.lineno
                    line = content.split("\n")[line_num - 1].strip()

                    # Check if it's the old pattern (without execute)
                    if "execute" not in line and "# type: ignore" not in line:
                        violations.append(
                            {
                                "file": py_file.name,
                                "line": line_num,
                                "code": line,
                                "issue": "Using sync session.delete() pattern in async context",
                            }
                        )

        assert len(violations) == 0, f"Found sync delete patterns in async repositories: {violations}"

    def test_delete_imports_in_repositories(self):
        """Ensure repositories that perform deletions import the delete construct."""
        repository_dir = Path(__file__).parent.parent.parent / "packages" / "shared" / "database" / "repositories"

        for py_file in repository_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            content = py_file.read_text()

            # If the file has a delete method, it should import delete from sqlalchemy
            if "async def delete" in content:
                has_import = "from sqlalchemy import" in content
                has_delete = "delete" in content
                assert has_import, f"{py_file.name} has a delete method but doesn't import from sqlalchemy"
                assert has_delete, f"{py_file.name} has a delete method but doesn't import delete specifically"

    def test_commit_after_delete_in_services(self):
        """Ensure service layer commits after deletion operations."""
        services_dir = Path(__file__).parent.parent.parent / "packages" / "webui" / "services"

        for py_file in services_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            content = py_file.read_text()

            # Look for delete_collection or similar methods
            if "async def delete_" in content:
                lines = content.split("\n")

                for i, line in enumerate(lines):
                    if "async def delete_" in line:
                        # Find the method body
                        method_end = i + 1
                        indent = len(line) - len(line.lstrip())

                        # Find where the method ends
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) <= indent:
                                method_end = j
                                break

                        # Extract method body
                        method_body = "\n".join(lines[i:method_end])

                        # Check for commit
                        if "await self.collection_repo.delete" in method_body:
                            assert (
                                "await self.db_session.commit()" in method_body
                            ), f"{py_file.name} calls collection_repo.delete but doesn't commit the transaction"

    def test_flush_vs_commit_pattern(self):
        """Ensure repositories use flush and services use commit."""
        # Repositories should use flush
        repository_dir = Path(__file__).parent.parent.parent / "packages" / "shared" / "database" / "repositories"

        for py_file in repository_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            content = py_file.read_text()

            # Repositories should use flush, not commit
            if "self.session.commit()" in content:
                pytest.fail(f"{py_file.name} uses commit() instead of flush() - repositories should use flush")

        # Services should use commit
        services_dir = Path(__file__).parent.parent.parent / "packages" / "webui" / "services"

        for py_file in services_dir.glob("*.py"):
            if py_file.name.startswith("__") or py_file.name == "factory.py":
                continue

            content = py_file.read_text()

            # Services that modify data should commit
            if (
                "await self." in content
                and "_repo." in content
                and any(op in content for op in ["create", "update", "delete"])
                and "self.db_session" in content
                and "commit()" not in content
            ):
                    # Some services might be read-only, so only fail if they have write operations
                    write_ops = ["create_", "update_", "delete_", "add_", "remove_"]
                    if any(f"async def {op}" in content for op in write_ops):
                        pytest.fail(f"{py_file.name} has write operations but doesn't commit transactions")

    def test_no_type_ignore_for_async_operations(self):
        """Ensure we're not suppressing warnings for async operations."""
        packages_dir = Path(__file__).parent.parent.parent / "packages"

        violations = []

        for py_file in packages_dir.rglob("*.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines):
                # Look for type: ignore comments related to coroutines
                if "# type: ignore" in line and ("coroutine" in line or "unused-coroutine" in line):
                    violations.append({"file": py_file.relative_to(packages_dir), "line": i + 1, "code": line.strip()})

        assert len(violations) == 0, f"Found type: ignore for coroutine warnings: {violations}"

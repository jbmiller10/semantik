#!/usr/bin/env python3
"""
Update remaining imports in test files for CORE-002 refactoring.
Specifically handles test-specific import patterns.
"""

import argparse
import re
from pathlib import Path


class TestImportUpdater:
    """Update test file imports for the refactoring"""

    IMPORT_MAPPINGS = {
        # Embedding service imports
        r"from packages\.webui\.embedding_service import EmbeddingService": "from shared.embedding import EmbeddingService",
        r"from webui\.embedding_service import EmbeddingService": "from shared.embedding import EmbeddingService",
        r"from packages\.webui\.embedding_service import embedding_service": "from shared.embedding import embedding_service",
        r"from webui\.embedding_service import embedding_service": "from shared.embedding import embedding_service",
        r"from packages\.webui\.embedding_service import": "from shared.embedding import",
        r"from webui\.embedding_service import": "from shared.embedding import",
        r"import packages\.webui\.embedding_service": "import shared.embedding",
        r"import webui\.embedding_service": "import shared.embedding",
        # Config imports
        r"from packages\.vecpipe\.config import": "from shared.config import",
        r"from vecpipe\.config import": "from shared.config import",
        r"import packages\.vecpipe\.config": "import shared.config",
        r"import vecpipe\.config": "import shared.config",
        # Metrics imports
        r"from packages\.vecpipe\.metrics import": "from shared.metrics.prometheus import",
        r"from vecpipe\.metrics import": "from shared.metrics.prometheus import",
        r"import packages\.vecpipe\.metrics": "import shared.metrics.prometheus",
        r"import vecpipe\.metrics": "import shared.metrics.prometheus",
        # Extract chunks imports
        r"from packages\.vecpipe\.extract_chunks import TokenChunker": "from shared.text_processing.chunking import TokenChunker",
        r"from vecpipe\.extract_chunks import TokenChunker": "from shared.text_processing.chunking import TokenChunker",
        r"from packages\.vecpipe\.extract_chunks import extract_text": "from shared.text_processing.extraction import extract_text",
        r"from vecpipe\.extract_chunks import extract_text": "from shared.text_processing.extraction import extract_text",
        r"from packages\.vecpipe\.extract_chunks import extract_and_serialize": "from shared.text_processing.extraction import extract_and_serialize",
        r"from vecpipe\.extract_chunks import extract_and_serialize": "from shared.text_processing.extraction import extract_and_serialize",
        # Generic extract_chunks imports (for other functions)
        r"from packages\.vecpipe\.extract_chunks import": "from shared.text_processing import",
        r"from vecpipe\.extract_chunks import": "from shared.text_processing import",
        # Module references in code
        r"packages\.vecpipe\.config": "shared.config",
        r"vecpipe\.config": "shared.config",
        r"packages\.vecpipe\.metrics": "shared.metrics.prometheus",
        r"vecpipe\.metrics": "shared.metrics.prometheus",
        r"packages\.vecpipe\.extract_chunks": "vecpipe.extract_chunks",  # Keep for non-migrated functions
        r"packages\.webui\.embedding_service": "shared.embedding",
        r"webui\.embedding_service": "shared.embedding",
        # Registry specific updates
        r"from shared.metrics.prometheus import registry": "from shared.metrics.prometheus import registry",
    }

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.changes_made = 0
        self.files_changed = 0

    def update_file(self, file_path: Path) -> list[str]:
        """Update imports in a single file"""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        changes = []

        for pattern, replacement in self.IMPORT_MAPPINGS.items():
            if re.search(pattern, content):
                # Count occurrences
                occurrences = len(re.findall(pattern, content))
                content = re.sub(pattern, replacement, content)
                change_desc = f"{pattern} -> {replacement} ({occurrences} occurrence{'s' if occurrences > 1 else ''})"
                changes.append(change_desc)
                self.changes_made += occurrences

        # Special handling for test mock paths
        mock_patterns = [
            (r"@patch\(['\"]packages\.webui\.embedding_service", "@patch('shared.embedding"),
            (r"@patch\(['\"]webui\.embedding_service", "@patch('shared.embedding"),
            (r"@patch\(['\"]packages\.vecpipe\.config", "@patch('shared.config"),
            (r"@patch\(['\"]vecpipe\.config", "@patch('shared.config"),
            (r"@patch\(['\"]packages\.vecpipe\.metrics", "@patch('shared.metrics.prometheus"),
            (r"@patch\(['\"]vecpipe\.metrics", "@patch('shared.metrics.prometheus"),
            (
                r"@patch\(['\"]packages\.vecpipe\.extract_chunks\.TokenChunker",
                "@patch('shared.text_processing.chunking.TokenChunker",
            ),
            (
                r"@patch\(['\"]vecpipe\.extract_chunks\.TokenChunker",
                "@patch('shared.text_processing.chunking.TokenChunker",
            ),
        ]

        for pattern, replacement in mock_patterns:
            if re.search(pattern, content):
                occurrences = len(re.findall(pattern, content))
                content = re.sub(pattern, replacement, content)
                changes.append(f"Mock path: {pattern} -> {replacement}")
                self.changes_made += occurrences

        if changes and not self.dry_run:
            try:
                file_path.write_text(content, encoding="utf-8")
                self.files_changed += 1
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return []

        if self.verbose and changes:
            print(f"\n{file_path}:")
            for change in changes:
                print(f"  - {change}")

        return changes

    def update_directory(self, directory: Path, include_all: bool = False) -> dict[str, list[str]]:
        """Update all Python files in directory"""
        results = {}

        # Patterns to process
        patterns = ["**/*.py"] if include_all else ["**/test_*.py", "**/*_test.py", "**/conftest.py"]

        for pattern in patterns:
            for py_file in directory.glob(pattern):
                # Skip __pycache__ directories
                if "__pycache__" in str(py_file):
                    continue

                changes = self.update_file(py_file)
                if changes:
                    results[str(py_file)] = changes

        return results

    def print_summary(self, results: dict[str, list[str]]):
        """Print a summary of changes"""
        if not results:
            print("No import changes needed in test files.")
            return

        print(f"\n{'DRY RUN: ' if self.dry_run else ''}Test Import Update Summary")
        print("=" * 60)

        if not self.verbose:
            for file_path, changes in sorted(results.items()):
                print(f"\n{file_path}:")
                for change in changes:
                    print(f"  - {change}")

        print(f"\nTotal: {self.changes_made} imports updated in {len(results)} files")

        if self.dry_run:
            print("\nThis was a dry run. No files were modified.")
            print("Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(description="Update test imports for CORE-002 refactoring")
    parser.add_argument(
        "--directory", "-d", type=Path, default=Path("tests"), help="Directory to process (default: tests)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--all-files", action="store_true", help="Process all Python files, not just test files")

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist")
        return 1

    print(f"Processing test files in: {args.directory}")
    if args.dry_run:
        print("Running in dry-run mode - no files will be modified")

    updater = TestImportUpdater(dry_run=args.dry_run, verbose=args.verbose)
    results = updater.update_directory(args.directory, include_all=args.all_files)
    updater.print_summary(results)

    # Also check for any Python files in the entire project that might have been missed
    if not args.all_files:
        print("\nChecking for any missed imports in the entire project...")
        all_results = {}
        for directory in [Path("packages"), Path("scripts")]:
            if directory.exists():
                dir_results = updater.update_directory(directory, include_all=True)
                all_results.update(dir_results)

        if all_results:
            print(f"\nFound {len(all_results)} additional files with outdated imports:")
            for file_path in sorted(all_results.keys()):
                print(f"  - {file_path}")
            print("\nRun with --directory <dir> --all-files to update these files.")

    return 0 if updater.changes_made > 0 or args.dry_run else 1


if __name__ == "__main__":
    exit(main())

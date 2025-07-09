#!/usr/bin/env python3
"""
Automated import path updater for CORE-002 refactoring.
Updates import statements to use the new shared package structure.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


class ImportUpdater:
    """Automated import path updater"""

    IMPORT_MAPPINGS = {
        # Config moves
        r"from vecpipe\.config import": "from shared.config import",
        r"from packages\.vecpipe\.config import": "from shared.config import",
        r"import vecpipe\.config": "import shared.config",
        
        # Metrics moves
        r"from vecpipe\.metrics import": "from shared.metrics.prometheus import",
        r"from packages\.vecpipe\.metrics import": "from shared.metrics.prometheus import",
        r"import vecpipe\.metrics": "import shared.metrics.prometheus",
        
        # Text processing moves - TokenChunker
        r"from vecpipe\.extract_chunks import TokenChunker": "from shared.text_processing.chunking import TokenChunker",
        r"from packages\.vecpipe\.extract_chunks import TokenChunker": "from shared.text_processing.chunking import TokenChunker",
        
        # Text processing moves - extract functions
        r"from vecpipe\.extract_chunks import extract_text": "from shared.text_processing.extraction import extract_text",
        r"from packages\.vecpipe\.extract_chunks import extract_text": "from shared.text_processing.extraction import extract_text",
        r"from vecpipe\.extract_chunks import extract_and_serialize": "from shared.text_processing.extraction import extract_and_serialize",
        r"from packages\.vecpipe\.extract_chunks import extract_and_serialize": "from shared.text_processing.extraction import extract_and_serialize",
        
        # Import multiple items from extract_chunks
        r"from vecpipe\.extract_chunks import": "from shared.text_processing import",
        r"from packages\.vecpipe\.extract_chunks import": "from shared.text_processing import",
    }

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.changes_made = 0
        self.files_changed = 0

    def update_file(self, file_path: Path) -> List[str]:
        """Update imports in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        original_content = content
        changes = []

        for pattern, replacement in self.IMPORT_MAPPINGS.items():
            if re.search(pattern, content):
                # Count occurrences
                occurrences = len(re.findall(pattern, content))
                content = re.sub(pattern, replacement, content)
                change_desc = f"{pattern} -> {replacement} ({occurrences} occurrence{'s' if occurrences > 1 else ''})"
                changes.append(change_desc)
                self.changes_made += occurrences

        if changes and not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.files_changed += 1
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return []

        return changes

    def update_directory(self, directory: Path, exclude_patterns: List[str] = None) -> Dict[str, List[str]]:
        """Update all Python files in directory"""
        results = {}
        exclude_patterns = exclude_patterns or []

        for py_file in directory.rglob("*.py"):
            # Skip files matching exclude patterns
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            # Skip the shared package itself
            if "packages/shared" in str(py_file):
                continue

            changes = self.update_file(py_file)
            if changes:
                results[str(py_file)] = changes

        return results

    def print_summary(self, results: Dict[str, List[str]]):
        """Print a summary of changes"""
        if not results:
            print("No import changes needed.")
            return

        print(f"\n{'DRY RUN: ' if self.dry_run else ''}Import Update Summary")
        print("=" * 60)
        
        for file_path, changes in sorted(results.items()):
            print(f"\n{file_path}:")
            for change in changes:
                print(f"  - {change}")

        print(f"\nTotal: {self.changes_made} imports updated in {len(results)} files")
        
        if self.dry_run:
            print("\nThis was a dry run. No files were modified.")
            print("Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(description="Update imports for CORE-002 refactoring")
    parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        default=Path("packages"),
        help="Directory to process (default: packages)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=["__pycache__", ".git", "venv", "env"],
        help="Patterns to exclude from processing"
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist")
        return 1

    print(f"Processing Python files in: {args.directory}")
    if args.dry_run:
        print("Running in dry-run mode - no files will be modified")
    
    updater = ImportUpdater(dry_run=args.dry_run)
    results = updater.update_directory(args.directory, exclude_patterns=args.exclude)
    updater.print_summary(results)

    return 0 if updater.changes_made > 0 or args.dry_run else 1


if __name__ == "__main__":
    exit(main())
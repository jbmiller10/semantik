#!/usr/bin/env python3
"""
Verify all migration files are syntactically correct.
"""

import ast
import os
import sys
from pathlib import Path


def verify_migration_file(filepath: Path) -> tuple[bool, str]:
    """Verify a single migration file is syntactically correct."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Try to compile the file
        compile(content, str(filepath), 'exec')
        
        # Also parse with AST to check structure
        tree = ast.parse(content, filename=str(filepath))
        
        # Check for required functions
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        
        if 'upgrade' not in functions:
            return False, "Missing 'upgrade' function"
        
        if 'downgrade' not in functions:
            return False, "Missing 'downgrade' function"
        
        return True, "OK"
        
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def main():
    """Main function to verify all migrations."""
    migrations_dir = Path("/home/john/semantik/alembic/versions")
    
    if not migrations_dir.exists():
        print(f"Error: Migrations directory not found: {migrations_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("MIGRATION FILES SYNTAX VERIFICATION")
    print("=" * 60)
    
    # Get all Python files in migrations directory
    migration_files = sorted(migrations_dir.glob("*.py"))
    
    # Filter out __init__.py
    migration_files = [f for f in migration_files if f.name != "__init__.py"]
    
    results = []
    for filepath in migration_files:
        is_valid, message = verify_migration_file(filepath)
        results.append((filepath.name, is_valid, message))
        
        status = "✅" if is_valid else "❌"
        print(f"{status} {filepath.name}: {message}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    valid_count = sum(1 for _, is_valid, _ in results if is_valid)
    total_count = len(results)
    
    print(f"Valid migrations: {valid_count}/{total_count}")
    
    if valid_count == total_count:
        print("\n✅ All migration files are syntactically correct!")
        return 0
    else:
        print("\n❌ Some migration files have syntax errors!")
        failed = [name for name, is_valid, _ in results if not is_valid]
        print(f"Failed files: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
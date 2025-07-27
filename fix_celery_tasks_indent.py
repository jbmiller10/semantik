#!/usr/bin/env python3
"""Fix indentation issue in test_celery_tasks.py at line 745"""

import py_compile
from pathlib import Path

# Read the file
file_path = Path("tests/webui/test_celery_tasks.py")
with file_path.open() as f:
    lines = f.readlines()

# Fix the indentation issue at line 744-750
# These lines have extra indentation that needs to be removed
for i in range(743, 751):  # Lines 744-751 (0-indexed)
    if i < len(lines) and lines[i].startswith("            "):
        lines[i] = lines[i][4:]  # Remove 4 spaces
        print(f"Fixed line {i+1}: {lines[i].rstrip()}")

# Write the fixed content
with file_path.open("w") as f:
    f.writelines(lines)

print("\nFixed indentation in test_celery_tasks.py")

# Verify it's syntactically correct

try:
    py_compile.compile("tests/webui/test_celery_tasks.py", doraise=True)
    print("✓ File is now syntactically valid!")
except py_compile.PyCompileError as e:
    print(f"✗ Still has syntax error: {e}")

#!/usr/bin/env python3
"""Fix indentation issue in test_celery_tasks.py at line 745"""

# Read the file
with open("tests/webui/test_celery_tasks.py", "r") as f:
    lines = f.readlines()

# Fix the indentation issue at line 744-750
# These lines have extra indentation that needs to be removed
for i in range(743, 751):  # Lines 744-751 (0-indexed)
    if i < len(lines):
        # Remove 4 spaces from the beginning of these lines
        if lines[i].startswith("            "):
            lines[i] = lines[i][4:]  # Remove 4 spaces
            print(f"Fixed line {i+1}: {lines[i].rstrip()}")

# Write the fixed content
with open("tests/webui/test_celery_tasks.py", "w") as f:
    f.writelines(lines)

print("\nFixed indentation in test_celery_tasks.py")

# Verify it's syntactically correct
import py_compile
try:
    py_compile.compile("tests/webui/test_celery_tasks.py", doraise=True)
    print("✓ File is now syntactically valid!")
except py_compile.PyCompileError as e:
    print(f"✗ Still has syntax error: {e}")
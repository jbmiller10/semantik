#!/usr/bin/env python3
"""Fix all indentation issues in test_search_api.py"""

# Read the file
with open("tests/unit/test_search_api.py", "r") as f:
    lines = f.readlines()

# Fix lines with excessive indentation (16 spaces where there should be 8)
fixed_lines = []
for i, line in enumerate(lines):
    # If line starts with exactly 16 spaces (but not 20 or more), reduce to 8
    if line.startswith("                ") and not line.startswith("                    "):
        fixed_line = line[8:]  # Remove 8 spaces
        fixed_lines.append(fixed_line)
        print(f"Fixed line {i+1}: {line.rstrip()} -> {fixed_line.rstrip()}")
    else:
        fixed_lines.append(line)

# Write the fixed content
with open("tests/unit/test_search_api.py", "w") as f:
    f.writelines(fixed_lines)

print("\nFixed all indentation issues")

# Verify it's syntactically correct
import py_compile
try:
    py_compile.compile("tests/unit/test_search_api.py", doraise=True)
    print("✓ test_search_api.py is now syntactically valid!")
except py_compile.PyCompileError as e:
    print(f"✗ Still has syntax error: {e}")
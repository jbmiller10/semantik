#!/usr/bin/env python3
"""Fix ALL indentation issues in test_search_api.py"""

# Read the file
with open("tests/unit/test_search_api.py") as f:
    content = f.read()

# Fix all lines that start with 16 spaces (but not 20 or more)
# These should have 8 spaces instead
import re

# Pattern to match lines with exactly 16 spaces at the start
pattern = re.compile(r"^                (?!    )", re.MULTILINE)
content = pattern.sub("        ", content)

# Write the fixed content
with open("tests/unit/test_search_api.py", "w") as f:
    f.write(content)

# Verify it's syntactically correct
try:
    compile(content, "tests/unit/test_search_api.py", "exec")
    print("✓ File is now syntactically valid!")
except SyntaxError as e:
    print(f"✗ Still has syntax error at line {e.lineno}: {e.msg}")
    # Show the problematic line
    lines = content.split("\n")
    if e.lineno and e.lineno <= len(lines):
        print(f"Line {e.lineno}: {lines[e.lineno-1]}")

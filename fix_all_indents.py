#!/usr/bin/env python3
"""Fix ALL indentation issues in test_search_api.py"""

import re
from pathlib import Path

# Read the file
file_path = Path("tests/unit/test_search_api.py")
with file_path.open() as f:
    content = f.read()

# Pattern to match lines with exactly 16 spaces at the start
pattern = re.compile(r"^                (?!    )", re.MULTILINE)
content = pattern.sub("        ", content)

# Write the fixed content
with file_path.open("w") as f:
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

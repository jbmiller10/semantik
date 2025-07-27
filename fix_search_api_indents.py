#!/usr/bin/env python3
"""Fix indentation issues in test_search_api.py"""

from pathlib import Path

# Read the file
file_path = Path("tests/unit/test_search_api.py")
with file_path.open() as f:
    lines = f.readlines()

# Fix lines with excessive indentation (16 spaces where there should be 8)
fixed_lines = []
for i, line in enumerate(lines):
    # Check if this line has 16 spaces at the start (should be 8)
    if line.startswith("                ") and not line.startswith("                    "):
        # If the previous line ends with """ or is a function definition, this should have 8 spaces
        if i > 0 and (lines[i - 1].strip().endswith('"""') or "def test_" in lines[i - 1]):
            fixed_line = line[8:]  # Remove 8 spaces
            fixed_lines.append(fixed_line)
            print(f"Fixed line {i+1}: {line.rstrip()} -> {fixed_line.rstrip()}")
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write the fixed content
with file_path.open("w") as f:
    f.writelines(fixed_lines)

print("\nFixed indentation issues in test_search_api.py")

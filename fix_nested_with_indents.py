#!/usr/bin/env python3
"""Fix nested with statement indentation issues."""

# Read the file
with open("tests/unit/test_search_api.py") as f:
    lines = f.readlines()

# Fix specific indentation issues
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # Check if this line and next line form a nested with statement pattern
    if i + 1 < len(lines):
        next_line = lines[i + 1]
        # If current line has 'with' and is properly indented, and next line also has 'with'
        if "with patch" in line and "with patch" in next_line:
            # Count indentation of current line
            current_indent = len(line) - len(line.lstrip())
            next_indent = len(next_line) - len(next_line.lstrip())

            # If next line has same or less indentation, it should be indented more
            if next_indent <= current_indent:
                fixed_lines.append(line)
                # Add proper indentation to next line
                fixed_next = " " * (current_indent + 4) + next_line.lstrip()
                fixed_lines.append(fixed_next)
                print(f"Fixed nested with at line {i+2}: {next_line.rstrip()}")
                i += 2
                continue

    # Check if line after a with statement is not properly indented
    if i > 0 and "with" in lines[i - 1] and ":" in lines[i - 1]:
        prev_indent = len(lines[i - 1]) - len(lines[i - 1].lstrip())
        curr_indent = len(line) - len(line.lstrip())

        # If current line is not indented more than the with statement
        if curr_indent <= prev_indent and line.strip() and not line.strip().startswith(("except", "finally", "else")):
            # Fix indentation
            fixed_line = " " * (prev_indent + 4) + line.lstrip()
            fixed_lines.append(fixed_line)
            print(f"Fixed line {i+1} after with statement: {line.rstrip()}")
            i += 1
            continue

    fixed_lines.append(line)
    i += 1

# Write the fixed content
with open("tests/unit/test_search_api.py", "w") as f:
    f.writelines(fixed_lines)

print("\nFixed nested with statement indentation issues")

# Verify it's syntactically correct
import py_compile

try:
    py_compile.compile("tests/unit/test_search_api.py", doraise=True)
    print("✓ test_search_api.py is now syntactically valid!")
except py_compile.PyCompileError as e:
    print(f"✗ Still has syntax error: {e}")

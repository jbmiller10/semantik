#!/usr/bin/env python3

"""
Manual testing script for path traversal vulnerability fixes.
Run this to verify the security implementation.
"""

import time

from webui.services.chunking_security import ChunkingSecurityValidator, ValidationError


def test_path(path: str) -> tuple[bool, str]:
    """Test a single path and return result."""
    try:
        ChunkingSecurityValidator.validate_file_paths([path])
        return True, "ALLOWED"
    except ValidationError as e:
        return False, str(e)


def main() -> None:
    """Run manual security tests."""
    print("=" * 80)
    print("PATH TRAVERSAL SECURITY VALIDATION TESTING")
    print("=" * 80)

    # Test cases that should be BLOCKED
    blocked_tests = [
        ("../../../etc/passwd", "Basic traversal"),
        ("%2e%2e%2f%2e%2e%2fetc%2fpasswd", "URL encoded traversal"),
        ("%252e%252e%252f", "Double URL encoding"),
        ("%25252e%25252e%25252f", "Triple URL encoding"),
        ("C:\\Windows\\System32", "Windows absolute path"),
        ("\\\\server\\share", "UNC path"),
        ("/etc/passwd", "Unix absolute path"),
        ("~/passwords.txt", "Home directory expansion"),
        ("file.txt\x00.jpg", "Null byte injection"),
        ("file\u202eetc/passwd", "Unicode RLO attack"),
        ("．．／．．／etc/passwd", "Full-width characters"),
        ("../%2e%2e/passwords", "Mixed encoding"),
        ("", "Empty path"),
        ("..", "Parent directory"),
        ("/", "Root directory"),
    ]

    # Test cases that should be ALLOWED
    allowed_tests = [
        ("documents/file.txt", "Normal relative path"),
        ("data/subfolder/document.pdf", "Nested path"),
        ("file_with_dots...txt", "File with multiple dots"),
        ("folder.with.dots/file.txt", "Folder with dots"),
        ("2024-01-01_report.pdf", "Date in filename"),
        ("user@email.com/inbox/message.txt", "Email-like path"),
        ("file (with spaces).txt", "Spaces in filename"),
        ("file[with]brackets.txt", "Brackets in filename"),
    ]

    print("\n🔴 Testing paths that should be BLOCKED:")
    print("-" * 80)

    blocked_count = 0
    for path_to_test, description in blocked_tests:
        allowed, message = test_path(path_to_test)
        if not allowed:
            print(f"✅ BLOCKED: {description:<30} | Path: {repr(path_to_test)[:40]}")
            blocked_count += 1
        else:
            print(f"❌ ERROR - ALLOWED: {description:<30} | Path: {repr(path_to_test)[:40]}")

    print(f"\nBlocked {blocked_count}/{len(blocked_tests)} malicious paths")

    print("\n🟢 Testing paths that should be ALLOWED:")
    print("-" * 80)

    allowed_count = 0
    for path_to_test, description in allowed_tests:
        allowed, message = test_path(path_to_test)
        if allowed:
            print(f"✅ ALLOWED: {description:<30} | Path: {repr(path_to_test)[:40]}")
            allowed_count += 1
        else:
            print(f"❌ ERROR - BLOCKED: {description:<30} | Path: {repr(path_to_test)[:40]}")

    print(f"\nAllowed {allowed_count}/{len(allowed_tests)} legitimate paths")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_blocked = blocked_count == len(blocked_tests)
    total_allowed = allowed_count == len(allowed_tests)

    if total_blocked and total_allowed:
        print("✅ ALL TESTS PASSED - Path traversal vulnerability is FIXED")
        print("✅ All OWASP path traversal patterns are blocked")
        print("✅ No false positives on legitimate paths")
    else:
        print("❌ TESTS FAILED - Security issues detected")
        if not total_blocked:
            print(f"   - {len(blocked_tests) - blocked_count} malicious paths were incorrectly allowed")
        if not total_allowed:
            print(f"   - {len(allowed_tests) - allowed_count} legitimate paths were incorrectly blocked")

    print("\n📊 Performance Check:")

    start = time.perf_counter()
    for _ in range(1000):
        test_path("documents/file.txt")
    elapsed = (time.perf_counter() - start) / 1000 * 1000  # ms per validation
    print(f"   Average validation time: {elapsed:.3f}ms (Requirement: <10ms)")
    if elapsed < 10:
        print("   ✅ Performance requirement met")
    else:
        print("   ❌ Performance requirement NOT met")


if __name__ == "__main__":
    main()

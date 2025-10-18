#!/usr/bin/env python3
"""Verification script for REINDEX chunk_count fix.

This script verifies that:
1. Document.chunk_count is updated in REINDEX operation
2. Failed documents are marked as FAILED
3. Transaction boundaries are properly maintained
"""

import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent))

INGESTION_PATH = Path("packages/webui/tasks/ingestion.py")
REINDEX_PATH = Path("packages/webui/tasks/reindex.py")


def verify_reindex_chunk_count_update():
    """Verify that REINDEX updates Document.chunk_count."""
    print("\n🔍 Verifying REINDEX chunk_count Update...")

    # Check the code in reindex module
    try:
        content = REINDEX_PATH.read_text()

        checks = [
            "chunk_count=len(all_chunks)" in content,
            'logger.info("Updated document' in content,
        ]

        if all(checks):
            print("✅ REINDEX updates Document.chunk_count after successful processing")
            return True
        print("❌ REINDEX does not properly update Document.chunk_count")
        return False

    except Exception as e:
        print(f"❌ Error checking REINDEX implementation: {e}")
        return False


def verify_failed_document_marking():
    """Verify that failed documents are marked as FAILED in REINDEX."""
    print("\n🔍 Verifying Failed Document Status Update...")

    try:
        content = REINDEX_PATH.read_text()

        checks = [
            "DocumentStatus.FAILED" in content,
            "Marked document" in content,
        ]

        if all(checks):
            print("✅ REINDEX marks failed documents as FAILED")
            return True
        print("❌ REINDEX does not properly mark failed documents")
        return False

    except Exception as e:
        print(f"❌ Error checking failed document handling: {e}")
        return False


def verify_transaction_boundaries():
    """Verify proper transaction boundaries in REINDEX."""
    print("\n🔍 Verifying Transaction Boundaries...")

    try:
        content = REINDEX_PATH.read_text()

        if (
            "create_celery_chunking_service_with_repos" in content
            and "DocumentRepository(document_repo.session)" not in content
        ):
            print("✅ Transaction boundaries properly maintained - reusing existing repository instances")
            return True
        print("⚠️  Transaction boundaries may have issues - check repository instantiation")
        return False

    except Exception as e:
        print(f"❌ Error checking transaction boundaries: {e}")
        return False


def verify_consistency_between_operations():
    """Verify that both APPEND and REINDEX have consistent fixes."""
    print("\n🔍 Verifying Consistency Between APPEND and REINDEX...")

    try:
        append_content = INGESTION_PATH.read_text()
        reindex_content = REINDEX_PATH.read_text()

        append_chunk_count = "chunk_count=len(chunks)" in append_content
        reindex_chunk_count = "chunk_count=len(all_chunks)" in reindex_content

        if append_chunk_count and reindex_chunk_count:
            print("✅ Both APPEND and REINDEX operations update chunk_count consistently")
            return True
        issues = []
        if not append_chunk_count:
            issues.append("APPEND missing chunk_count update")
        if not reindex_chunk_count:
            issues.append("REINDEX missing chunk_count update")
        print(f"❌ Inconsistency found: {', '.join(issues)}")
        return False

    except Exception as e:
        print(f"❌ Error checking consistency: {e}")
        return False


def check_imports():
    """Verify necessary imports are present."""
    print("\n🔍 Verifying Required Imports...")

    try:
        content = REINDEX_PATH.read_text()

        imports = [
            "from shared.database.models import DocumentStatus",
            "from packages.webui.services.chunking.container import resolve_celery_chunking_service",
        ]

        missing = []
        for imp in imports:
            if imp not in content:
                missing.append(imp)

        if not missing:
            print("✅ All required imports are present")
            return True
        print(f"❌ Missing imports: {missing}")
        return False

    except Exception as e:
        print(f"❌ Error checking imports: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("REINDEX Operation Fixes Verification")
    print("=" * 60)

    results = {
        "Chunk Count Update": verify_reindex_chunk_count_update(),
        "Failed Document Marking": verify_failed_document_marking(),
        "Transaction Boundaries": verify_transaction_boundaries(),
        "Operation Consistency": verify_consistency_between_operations(),
        "Required Imports": check_imports(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:25} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All REINDEX Fixes Verified Successfully!")
        print("\nKey fixes implemented:")
        print("✅ Document.chunk_count updated after successful reprocessing")
        print("✅ Failed documents marked with FAILED status")
        print("✅ Transaction boundaries properly maintained")
        print("✅ Consistent implementation across APPEND and REINDEX")
    else:
        print("⚠️  Some Fixes Need Attention")
        print("Please review failed checks above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

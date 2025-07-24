#!/usr/bin/env python3
"""
Test document processing pipeline to debug embedding generation issues.
This script tests text extraction and chunking for a specific file.
"""

import sys
from pathlib import Path

# Add the packages directory to the Python path
sys.path.insert(0, "/home/dockertest/semantik/packages")

from shared.chunking.token_chunker import TokenChunker
from shared.text_extraction.text_extractor import extract_text_and_serialize


def test_text_extraction(file_path: str):
    """Test text extraction from a file."""
    print(f"\n{'='*60}")
    print(f"Testing text extraction for: {file_path}")
    print(f"{'='*60}\n")

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return None

    try:
        # Extract text
        print("Extracting text...")
        text_blocks = extract_text_and_serialize(file_path)

        if not text_blocks:
            print("❌ No text extracted from file")
            return None

        print(f"✓ Extracted {len(text_blocks)} text block(s)")

        total_chars = 0
        for i, (text, metadata) in enumerate(text_blocks):
            char_count = len(text)
            total_chars += char_count
            print(f"\n  Block {i+1}:")
            print(f"    Characters: {char_count}")
            print(f"    Metadata: {metadata}")
            print(f"    Preview: {text[:200]}..." if len(text) > 200 else f"    Text: {text}")

        print(f"\nTotal characters extracted: {total_chars}")
        return text_blocks

    except Exception as e:
        print(f"❌ Error during text extraction: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_chunking(text_blocks, chunk_size=1000, chunk_overlap=200):
    """Test chunking of extracted text."""
    print(f"\n{'='*60}")
    print(f"Testing chunking (size={chunk_size}, overlap={chunk_overlap})")
    print(f"{'='*60}\n")

    try:
        # Create chunker
        chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Process each text block
        all_chunks = []
        for i, (text, metadata) in enumerate(text_blocks):
            if not text.strip():
                print(f"  Skipping empty block {i+1}")
                continue

            print(f"\n  Processing block {i+1} ({len(text)} chars)...")
            chunks = chunker.chunk_text(text, f"test_doc_{i}", metadata)

            print(f"    Created {len(chunks)} chunks")
            all_chunks.extend(chunks)

            # Show chunk details
            for j, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"\n    Chunk {j+1}:")
                print(f"      ID: {chunk['chunk_id']}")
                print(f"      Length: {len(chunk['text'])} chars")
                print(f"      Preview: {chunk['text'][:100]}...")

        print(f"\n✓ Total chunks created: {len(all_chunks)}")

        # Check for issues
        if not all_chunks:
            print("\n⚠️  No chunks were created! This would cause embedding generation to fail.")

        return all_chunks

    except Exception as e:
        print(f"❌ Error during chunking: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run document processing tests."""
    print("Document Processing Pipeline Test")
    print("=" * 80)

    # Get file path from command line or use a default
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Try to find a sample file
        sample_files = ["/mnt/data/test.txt", "/mnt/data/sample.pdf", "/mnt/data/example.md"]

        test_file = None
        for f in sample_files:
            if Path(f).exists():
                test_file = f
                break

        if not test_file:
            print("\nUsage: python test_document_processing.py <file_path>")
            print("\nNo test file specified and no sample files found.")
            print("Please provide a file path to test.")
            return

    # Test text extraction
    text_blocks = test_text_extraction(test_file)

    if text_blocks:
        # Test chunking with default settings
        chunks = test_chunking(text_blocks)

        # Test with different chunk sizes
        print("\n" + "=" * 80)
        print("Testing with smaller chunks...")
        test_chunking(text_blocks, chunk_size=500, chunk_overlap=100)

    print("\n" + "=" * 80)
    print("Test complete.")

    # Provide diagnosis
    print("\nDIAGNOSIS:")
    print("-" * 40)

    if not text_blocks:
        print("❌ Text extraction failed - embeddings cannot be generated without text")
        print("   Possible causes:")
        print("   - File format not supported")
        print("   - File is corrupted")
        print("   - Missing dependencies for this file type")
    elif not chunks:
        print("❌ Chunking failed - embeddings cannot be generated without chunks")
        print("   Possible causes:")
        print("   - Text is too short for the chunk size")
        print("   - TokenChunker configuration issue")
    else:
        print("✓ Document processing pipeline is working correctly")
        print("   Text extraction: OK")
        print("   Chunking: OK")
        print("\n   If embeddings are still not being generated, check:")
        print("   - Vecpipe service is running")
        print("   - Celery workers are processing tasks")
        print("   - No errors in operation status")


if __name__ == "__main__":
    main()

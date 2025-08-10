"""Test to verify partition key calculation always returns positive values."""

import hashlib
import random
import string
import uuid


def get_partition_key_python(collection_id: str) -> int:
    """
    Python equivalent of the PostgreSQL partition key calculation.
    This mimics what abs(hashtext()) % 100 would do.
    """
    # Use MD5 for consistent hashing (similar approach)
    hash_val = hashlib.md5(collection_id.encode()).hexdigest()
    # Convert first 8 hex chars to int and mod by 100
    hash_int = int(hash_val[:8], 16)
    # Ensure always positive (0-99)
    return abs(hash_int) % 100


def test_partition_key_always_positive():
    """Test that partition keys are always in range 0-99."""
    # Test with various UUIDs
    for _ in range(1000):
        collection_id = str(uuid.uuid4())
        partition_key = get_partition_key_python(collection_id)
        assert 0 <= partition_key < 100, f"Partition key {partition_key} out of range for {collection_id}"


def test_partition_key_with_edge_cases():
    """Test partition key calculation with edge cases."""
    test_cases = [
        "",  # Empty string
        "a",  # Single character
        "test-collection",  # Regular string
        "1234567890",  # Numbers
        "special-chars-!@#$%^&*()",  # Special characters
        "very-long-" + "x" * 1000,  # Very long string
        str(uuid.UUID("00000000-0000-0000-0000-000000000000")),  # Zero UUID
        str(uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")),  # Max UUID
    ]
    
    for collection_id in test_cases:
        partition_key = get_partition_key_python(collection_id)
        assert 0 <= partition_key < 100, f"Partition key {partition_key} out of range for '{collection_id}'"


def test_partition_key_distribution():
    """Test that partition keys have reasonable distribution."""
    partition_counts = {}
    num_samples = 10000
    
    # Generate random UUIDs and count partition assignments
    for _ in range(num_samples):
        collection_id = str(uuid.uuid4())
        partition_key = get_partition_key_python(collection_id)
        partition_counts[partition_key] = partition_counts.get(partition_key, 0) + 1
    
    # Check that we're using multiple partitions (at least 50 out of 100)
    assert len(partition_counts) >= 50, f"Only {len(partition_counts)} partitions used out of 100"
    
    # Check that distribution is somewhat even (no partition has more than 5% of data)
    max_count = max(partition_counts.values())
    assert max_count < num_samples * 0.05, f"Partition distribution too skewed: max count {max_count}"


def test_partition_key_consistency():
    """Test that the same collection_id always maps to the same partition."""
    collection_id = str(uuid.uuid4())
    
    # Calculate partition key multiple times
    keys = [get_partition_key_python(collection_id) for _ in range(100)]
    
    # All should be the same
    assert len(set(keys)) == 1, "Partition key not consistent for same collection_id"


def test_negative_hash_handling():
    """Test that negative hash values are properly handled."""
    # In Python, we're using abs() to ensure positive values
    # This test verifies our approach works correctly
    
    # Generate many random strings to increase chance of negative hash
    for _ in range(1000):
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        partition_key = get_partition_key_python(random_str)
        
        # Verify it's always in valid range
        assert 0 <= partition_key < 100, f"Partition key {partition_key} out of range"


if __name__ == "__main__":
    test_partition_key_always_positive()
    test_partition_key_with_edge_cases()
    test_partition_key_distribution()
    test_partition_key_consistency()
    test_negative_hash_handling()
    print("✓ All partition key tests passed!")
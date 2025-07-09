#!/usr/bin/env python3
"""
Test script to verify bitsandbytes and CUDA setup for INT8 quantization.

Usage:
    docker exec semantik-webui python /app/scripts/test-bitsandbytes.py
"""

import os
import sys


def check_cuda():
    """Check CUDA availability and configuration"""
    print("=" * 60)
    print("CUDA Environment Check")
    print("=" * 60)

    # Check environment variables
    env_vars = ["CUDA_HOME", "CUDA_VERSION", "BNB_CUDA_VERSION", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"]

    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"{var}: {value}")

    print("\n" + "-" * 60 + "\n")

    # Check PyTorch CUDA
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")

            # Check memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")

            return True
        print("\n⚠️  CUDA is not available!")
        print("This means INT8 quantization will not work.")
        return False

    except ImportError:
        print("❌ PyTorch not installed!")
        return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_bitsandbytes():
    """Check bitsandbytes installation and functionality"""
    print("\n" + "=" * 60)
    print("Bitsandbytes Check")
    print("=" * 60)

    try:
        import bitsandbytes as bnb

        print(f"✓ Bitsandbytes version: {bnb.__version__}")

        # Try to import key components
        try:
            import bitsandbytes.nn  # Test import availability

            print("✓ Linear8bitLt available")
        except ImportError as e:
            print(f"❌ Linear8bitLt not available: {e}")
            return False

        # Test 8-bit operations
        print("\nTesting INT8 operations...")
        try:
            import torch

            if torch.cuda.is_available():
                # Create test tensors
                test_size = 128
                input_tensor = torch.randn(32, test_size).cuda()

                # Create 8-bit linear layer
                linear_8bit = bnb.nn.Linear8bitLt(test_size, test_size).cuda()

                # Forward pass
                output = linear_8bit(input_tensor)

                print("✓ INT8 forward pass successful!")
                print(f"  Input shape: {input_tensor.shape}")
                print(f"  Output shape: {output.shape}")
                print(
                    f"  Memory used: ~{linear_8bit.weight.element_size() * linear_8bit.weight.nelement() / 1024**2:.2f} MB"
                )

                return True
            print("⚠️  Cannot test INT8 operations without CUDA")
            return False

        except Exception as e:
            print(f"❌ INT8 operation test failed: {e}")
            return False

    except ImportError as e:
        print(f"❌ Bitsandbytes not installed or not working: {e}")
        print("\nPossible causes:")
        print("1. Not using the CUDA-enabled Docker image")
        print("2. Missing CUDA libraries (libcusparse, libcublas)")
        print("3. Python/CUDA version mismatch")
        return False


def test_embedding_service():
    """Test the embedding service with INT8 quantization"""
    print("\n" + "=" * 60)
    print("Embedding Service INT8 Test")
    print("=" * 60)

    try:
        # Add packages to path
        sys.path.insert(0, "/app/packages")

        from webui.embedding_service import EmbeddingService

        service = EmbeddingService()

        # Try loading a small model with INT8
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"\nTesting model: {model_name}")
        print("Quantization: int8")

        if service.load_model(model_name, quantization="int8"):
            print("✓ Model loaded successfully with INT8!")

            # Test embedding generation
            test_text = ["This is a test sentence for INT8 quantization."]
            embeddings = service.generate_embeddings(
                test_text, model_name, quantization="int8", batch_size=1, show_progress=False
            )

            if embeddings is not None:
                print("✓ Embeddings generated successfully!")
                print(f"  Shape: {embeddings.shape}")
                print(f"  Dimension: {embeddings.shape[1]}")
                return True
            print("❌ Failed to generate embeddings")
            return False
        print("❌ Failed to load model with INT8")
        print("\nThis might mean:")
        print("1. Bitsandbytes is not properly installed")
        print("2. CUDA libraries are not accessible")
        print("3. The model doesn't support INT8 quantization")
        return False

    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Semantik Bitsandbytes INT8 Quantization Test")
    print("=" * 60)

    cuda_ok = check_cuda()
    bnb_ok = check_bitsandbytes()

    if cuda_ok and bnb_ok:
        embed_ok = test_embedding_service()
    else:
        embed_ok = False
        print("\n⚠️  Skipping embedding service test due to previous failures")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"CUDA available: {'✓' if cuda_ok else '❌'}")
    print(f"Bitsandbytes working: {'✓' if bnb_ok else '❌'}")
    print(f"INT8 embeddings: {'✓' if embed_ok else '❌'}")

    if cuda_ok and bnb_ok and embed_ok:
        print("\n✅ All tests passed! INT8 quantization is ready to use.")
        print("\nTo use INT8 in production:")
        print("1. Set DEFAULT_QUANTIZATION=int8 in your .env file")
        print("2. Restart the containers")
        return 0
    print("\n❌ Some tests failed. INT8 quantization may not work properly.")
    print("\nTroubleshooting:")
    print("1. Ensure you're using: docker compose -f docker-compose.yml -f docker-compose.cuda.yml up")
    print("2. Check that your GPU has compute capability 7.0+")
    print("3. Verify NVIDIA Docker runtime is installed")
    print("4. Check the logs: docker logs semantik-webui")
    return 1


if __name__ == "__main__":
    sys.exit(main())

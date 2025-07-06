#!/usr/bin/env python3
"""
Validate search API setup before starting
Helps diagnose configuration and model loading issues
"""

import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_environment():
    """Check environment configuration"""
    print("=" * 60)
    print("Search API Configuration Validation")
    print("=" * 60)

    # Check environment variables
    print("\n1. Environment Variables:")
    use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"
    model = os.getenv("DEFAULT_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    quantization = os.getenv("DEFAULT_QUANTIZATION", "float16")

    print(f"   USE_MOCK_EMBEDDINGS: {use_mock}")
    print(f"   DEFAULT_EMBEDDING_MODEL: {model}")
    print(f"   DEFAULT_QUANTIZATION: {quantization}")

    return use_mock, model, quantization


def check_dependencies():
    """Check required dependencies"""
    print("\n2. Dependencies:")

    required = {"transformers": "4.51.0", "torch": None, "sentence_transformers": None, "accelerate": None}

    all_good = True
    for package, min_version in required.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")

            if min_version and version < min_version:
                print(f"   ✗ {package}: {version} (need >= {min_version})")
                all_good = False
            else:
                print(f"   ✓ {package}: {version}")
        except ImportError:
            print(f"   ✗ {package}: NOT INSTALLED")
            all_good = False

    return all_good


def check_hardware():
    """Check hardware capabilities"""
    print("\n3. Hardware:")

    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")

        # Check memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3

        print(f"   ✓ GPU memory: {free_memory:.1f}GB free / {total_memory:.1f}GB total")

        # Memory requirements
        print("\n   Memory requirements by model:")
        print("   - Qwen3-0.6B (float16): ~1.2GB")
        print("   - Qwen3-0.6B (int8): ~0.6GB")
        print("   - Qwen3-4B (float16): ~8GB")
        print("   - Qwen3-4B (int8): ~4GB")

        return True, free_memory
    else:
        print("   ⚠ No CUDA GPU available - will use CPU (slower)")
        return False, 0


def test_model_loading(model_name, quantization, has_gpu):
    """Test loading the embedding model"""
    print(f"\n4. Testing Model Loading: {model_name}")

    try:
        from webui.embedding_service import EmbeddingService

        print(f"   Loading with {quantization} quantization...")
        service = EmbeddingService()

        if service.load_model(model_name, quantization):
            print("   ✓ Model loaded successfully")

            # Test embedding generation
            test_text = "This is a test"
            embedding = service.generate_single_embedding(test_text, model_name, quantization)

            if embedding:
                print(f"   ✓ Test embedding generated (dimension: {len(embedding)})")
                return True
            else:
                print("   ✗ Failed to generate test embedding")
                return False
        else:
            print("   ✗ Failed to load model")
            return False

    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return False


def suggest_fixes(use_mock, model, quantization, deps_ok, has_gpu, gpu_memory, model_loads):
    """Suggest fixes for common issues"""
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)

    if use_mock:
        print("\n✓ Currently configured for MOCK embeddings - no model needed")
        print("  To use real embeddings: unset USE_MOCK_EMBEDDINGS or set it to 'false'")
        return

    if not deps_ok:
        print("\n✗ Missing dependencies. Install with:")
        print("  pip install transformers>=4.51.0 torch sentence-transformers accelerate")

    if not model_loads:
        if not has_gpu:
            print("\n⚠ No GPU detected. Options:")
            print("  1. Use CPU (slower): The model should still work")
            print("  2. Use mock embeddings: export USE_MOCK_EMBEDDINGS=true")

        elif gpu_memory < 1.0:
            print("\n⚠ Low GPU memory. Options:")
            print("  1. Use int8 quantization: export DEFAULT_QUANTIZATION=int8")
            print("  2. Use smaller model: export DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2")
            print("  3. Use mock embeddings: export USE_MOCK_EMBEDDINGS=true")

        else:
            print("\n⚠ Model failed to load despite adequate resources. Try:")
            print("  1. Check internet connection (for downloading model)")
            print("  2. Clear cache: rm -rf ~/.cache/huggingface")
            print("  3. Try a different model: export DEFAULT_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5")
    else:
        print("\n✓ Everything looks good! The API should start successfully.")


def main():
    # Check configuration
    use_mock, model, quantization = check_environment()

    # If using mock, no need to check further
    if use_mock:
        suggest_fixes(use_mock, model, quantization, True, True, 0, True)
        return

    # Check dependencies
    deps_ok = check_dependencies()

    # Check hardware
    has_gpu, gpu_memory = check_hardware()

    # Test model loading
    model_loads = False
    if deps_ok:
        model_loads = test_model_loading(model, quantization, has_gpu)

    # Provide recommendations
    suggest_fixes(use_mock, model, quantization, deps_ok, has_gpu, gpu_memory, model_loads)

    # Exit code
    if use_mock or model_loads:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()

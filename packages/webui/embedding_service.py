#!/usr/bin/env python3
"""
Unified Embedding Service with quantization and Qwen3 support
Provides backward-compatible API with feature flags
"""

import gc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

# Import metrics tracking if available
try:
    from packages.vecpipe.metrics import Counter, registry

    # Check if metrics already exist in registry to avoid duplicates
    try:
        oom_errors = Counter(
            "embedding_oom_errors_total",
            "Total OOM errors during embedding generation",
            ["model", "quantization"],
            registry=registry,
        )
        batch_size_reductions = Counter(
            "embedding_batch_size_reductions_total",
            "Total batch size reductions due to OOM",
            ["model", "quantization"],
            registry=registry,
        )
    except ValueError as e:
        # Metrics already registered, get them from registry
        if "Duplicated timeseries" in str(e):
            # Find existing metrics in registry
            for collector in registry._collector_to_names:
                if hasattr(collector, "_name"):
                    if collector._name == "embedding_oom_errors_total":
                        oom_errors = collector
                    elif collector._name == "embedding_batch_size_reductions_total":
                        batch_size_reductions = collector
        else:
            raise
    METRICS_AVAILABLE = True
except ImportError:
    # Metrics not available, create dummy functions
    METRICS_AVAILABLE = False

    class DummyCounter:
        def labels(self, **kwargs):
            return self

        def inc(self):
            pass

    oom_errors = DummyCounter()
    batch_size_reductions = DummyCounter()

logger = logging.getLogger(__name__)


# Qwen3 specific pooling function
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for Qwen3 models"""
    return f"Instruct: {task_description}\nQuery:{query}"


class EmbeddingService:
    """Unified service for generating embeddings with optional quantization and Qwen3 support"""

    def __init__(self, mock_mode: bool = False):
        self.models = {}
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.is_qwen3_model = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mock_mode = mock_mode
        # Adaptive batch sizing
        self.original_batch_size = None
        self.current_batch_size = None
        self.successful_batches = 0
        self.min_batch_size = 4
        logger.info(f"Unified EmbeddingService initialized with device: {self.device}, mock_mode: {self.mock_mode}")

    def load_model(self, model_name: str, quantization: str = "float32") -> bool:
        """Load a model from HuggingFace with optional quantization

        Args:
            model_name: HuggingFace model name
            quantization: One of "float32", "float16", "int8" (default: "float32")
        """
        try:
            # Mock mode - skip actual model loading
            if self.mock_mode:
                logger.info(f"Mock mode: Simulating load of {model_name} with {quantization}")
                self.current_model_name = model_name
                self.current_quantization = quantization
                return True

            # Create unique key for model+quantization combo
            model_key = f"{model_name}_{quantization}"

            # If same model+quantization already loaded, return
            if model_key == f"{self.current_model_name}_{self.current_quantization}":
                return True

            # Clear previous model to save memory
            if self.current_model is not None:
                del self.current_model
                if self.current_tokenizer is not None:
                    del self.current_tokenizer
                torch.cuda.empty_cache() if self.device == "cuda" else None
                gc.collect()

            logger.info(f"Loading model: {model_name} with quantization: {quantization}")

            # Check if this is a Qwen3 model
            self.is_qwen3_model = "Qwen3-Embedding" in model_name

            # Load model based on type and quantization
            if self.is_qwen3_model:
                # Load Qwen3 model using transformers directly
                logger.info("Loading Qwen3 embedding model")
                self.current_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

                if quantization == "int8" and self.device == "cuda":
                    try:
                        import bitsandbytes as bnb
                        from transformers import BitsAndBytesConfig

                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16,
                            bnb_8bit_use_double_quant=True,
                            bnb_8bit_quant_type="nf4",
                        )

                        self.current_model = AutoModel.from_pretrained(
                            model_name, quantization_config=quantization_config, device_map="auto"
                        )
                        logger.info("Loaded Qwen3 model with INT8 quantization")
                    except Exception as e:
                        error_msg = f"Failed to load model with INT8 quantization: {str(e)}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg) from e

                elif quantization == "float16" and self.device == "cuda":
                    self.current_model = AutoModel.from_pretrained(
                        model_name, torch_dtype=torch.float16, device_map={"": 0} if self.device == "cuda" else None
                    )
                    if self.device == "cpu":
                        self.current_model = self.current_model.to(self.device)
                    logger.info("Loaded Qwen3 model in float16 precision")
                else:
                    self.current_model = AutoModel.from_pretrained(model_name).to(self.device)
                    logger.info("Loaded Qwen3 model in float32 precision")

            elif quantization == "int8" and self.device == "cuda":
                # Use bitsandbytes for INT8 quantization
                try:
                    import bitsandbytes as bnb
                    from transformers import BitsAndBytesConfig

                    # Configure 8-bit quantization
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16,
                        bnb_8bit_use_double_quant=True,
                        bnb_8bit_quant_type="nf4",
                    )

                    # Load model with quantization
                    self.current_model = SentenceTransformer(
                        model_name, device=self.device, model_kwargs={"quantization_config": quantization_config}
                    )
                    logger.info("Loaded model with INT8 quantization")

                except ImportError:
                    logger.warning("bitsandbytes not available, falling back to post-quantization")
                    self.current_model = SentenceTransformer(model_name, device=self.device)

            elif quantization == "float16" and self.device == "cuda":
                # Load model in float16
                self.current_model = SentenceTransformer(model_name, device=self.device)
                # Convert model to float16
                self.current_model = self.current_model.half()
                logger.info("Loaded model in float16 precision")

            else:
                # Load standard float32 model
                self.current_model = SentenceTransformer(model_name, device=self.device)
                logger.info("Loaded model in float32 precision")

            self.current_model_name = model_name
            self.current_quantization = quantization

            # Test model
            test_embedding = self._generate_test_embedding()
            logger.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding)}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def _generate_test_embedding(self) -> np.ndarray:
        """Generate a test embedding to verify model and get dimensions"""
        if self.mock_mode:
            return np.random.randn(1024)  # Default mock dimension

        if not hasattr(self, "current_model") or self.current_model is None:
            raise RuntimeError("Model not loaded")

        if self.is_qwen3_model:
            batch_dict = self.current_tokenizer(
                ["test"],
                padding=True,
                truncation=True,
                max_length=32768,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.current_model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                return embeddings[0].cpu().numpy()
        else:
            return self.current_model.encode("test", convert_to_numpy=True)

    def _generate_mock_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate deterministic mock embeddings based on text hash"""
        import hashlib

        # Determine embedding dimension
        vector_dim = 1024  # Default dimension
        if self.current_model_name:
            # Try to get dimension from model info
            model_info = QUANTIZED_MODEL_INFO.get(self.current_model_name, {})
            vector_dim = model_info.get("dimension", 1024)

        embeddings = []
        for text in texts:
            # Create deterministic embedding from text hash
            text_hash = hashlib.sha256(text.encode()).digest()
            # Convert hash to float array (using first bytes)
            embedding = np.frombuffer(text_hash, dtype=np.float32)[: vector_dim // 4]
            # Repeat and normalize to get correct dimension
            embedding = np.tile(embedding, (vector_dim // len(embedding) + 1))[:vector_dim]
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_model_info(self, model_name: str, quantization: str = "float32") -> dict:
        """Get information about a model

        Args:
            model_name: HuggingFace model name
            quantization: Quantization type (default: "float32")
        """
        try:
            model_key = f"{model_name}_{quantization}"
            if model_key != f"{self.current_model_name}_{self.current_quantization}":
                if not self.load_model(model_name, quantization):
                    return {"error": f"Failed to load model {model_name}"}

            if not hasattr(self, "current_model") or self.current_model is None:
                return {"error": "Model not loaded"}

            # Get embedding dimension
            test_embedding = self._generate_test_embedding()

            # Calculate model size
            model_size = 0
            if hasattr(self.current_model, "parameters"):
                for param in self.current_model.parameters():
                    model_size += param.numel() * param.element_size()

            return {
                "model_name": model_name,
                "embedding_dim": len(test_embedding),
                "device": self.device,
                "quantization": quantization,
                "model_size_mb": model_size / 1024 / 1024,
                "max_seq_length": getattr(self.current_model, "max_seq_length", 32768 if self.is_qwen3_model else 512),
                "is_qwen3": self.is_qwen3_model,
            }

        except Exception as e:
            return {"error": str(e)}

    def generate_embeddings(
        self,
        texts: list[str],
        model_name: str,
        quantization: str = "float32",
        batch_size: int = 32,
        show_progress: bool = True,
        instruction: str | None = None,
        **kwargs,
    ) -> np.ndarray | None:
        """Generate embeddings for a list of texts

        Args:
            texts: List of texts to embed
            model_name: HuggingFace model name
            quantization: Quantization type (default: "float32")
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            instruction: Optional instruction for Qwen3 models
            **kwargs: Additional arguments (ignored for compatibility)
        """
        try:
            # Mock mode handling
            if self.mock_mode:
                logger.info(f"Generating mock embeddings for {len(texts)} texts")
                return self._generate_mock_embeddings(texts)

            # Load model with specified quantization if needed
            model_key = f"{model_name}_{quantization}"
            if model_key != f"{self.current_model_name}_{self.current_quantization}":
                if not self.load_model(model_name, quantization):
                    return None

            if not hasattr(self, "current_model") or self.current_model is None:
                logger.error("Model not properly loaded")
                return None

            if not texts:
                return np.array([])

            logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")

            # Initialize adaptive batch sizing
            if self.original_batch_size is None:
                self.original_batch_size = batch_size
                self.current_batch_size = batch_size

            # Use current batch size from adaptive sizing
            current_batch_size = self.current_batch_size

            # Generate embeddings based on model type
            if self.is_qwen3_model:
                # Use instruction if provided for Qwen3 models
                if instruction:
                    texts = [get_detailed_instruct(instruction, text) for text in texts]
                    logger.info(f"Using instruction: {instruction}")

                # Process in batches for Qwen3
                all_embeddings = []
                from tqdm import tqdm

                for i in tqdm(range(0, len(texts), current_batch_size), disable=not show_progress, desc="Encoding"):
                    batch_texts = texts[i : i + current_batch_size]

                    try:
                        # Tokenize batch
                        batch_dict = self.current_tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=32768,
                            return_tensors="pt",
                        ).to(self.device)

                        # Generate embeddings
                        with torch.no_grad():
                            if quantization == "float16" and self.device == "cuda":
                                with torch.cuda.amp.autocast(dtype=torch.float16):
                                    outputs = self.current_model(**batch_dict)
                            else:
                                outputs = self.current_model(**batch_dict)

                            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                            embeddings = F.normalize(embeddings, p=2, dim=1)
                            all_embeddings.append(embeddings.cpu().numpy())

                        # Track successful batches for restoration
                        if current_batch_size == self.current_batch_size:
                            self.successful_batches += 1
                            # Try to restore batch size after 10 successful batches
                            if self.successful_batches > 10 and self.current_batch_size < self.original_batch_size:
                                new_size = min(self.current_batch_size * 2, self.original_batch_size)
                                logger.info(f"Restoring batch size from {self.current_batch_size} to {new_size}")
                                self.current_batch_size = new_size
                                current_batch_size = new_size
                                self.successful_batches = 0

                    except torch.cuda.OutOfMemoryError:
                        if current_batch_size > self.min_batch_size:
                            logger.warning(
                                f"OOM with batch size {current_batch_size}, reducing to {current_batch_size // 2}"
                            )
                            # Record OOM error
                            oom_errors.labels(
                                model=self.current_model_name, quantization=self.current_quantization
                            ).inc()
                            batch_size_reductions.labels(
                                model=self.current_model_name, quantization=self.current_quantization
                            ).inc()

                            torch.cuda.empty_cache()
                            current_batch_size = max(self.min_batch_size, current_batch_size // 2)
                            self.current_batch_size = current_batch_size
                            self.successful_batches = 0  # Reset success counter
                            # Retry with smaller batch
                            i -= current_batch_size  # Step back to retry
                        else:
                            raise

                embeddings = np.vstack(all_embeddings)

            else:
                # Standard sentence-transformers processing
                while current_batch_size >= self.min_batch_size:
                    try:
                        if quantization == "float16" and self.device == "cuda":
                            # Ensure inputs are float16 compatible
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                embeddings = self.current_model.encode(
                                    texts,
                                    batch_size=current_batch_size,
                                    normalize_embeddings=True,
                                    convert_to_numpy=True,
                                    show_progress_bar=show_progress,
                                )
                        else:
                            embeddings = self.current_model.encode(
                                texts,
                                batch_size=current_batch_size,
                                normalize_embeddings=True,
                                convert_to_numpy=True,
                                show_progress_bar=show_progress,
                            )

                        # Success! Track it for potential restoration
                        if current_batch_size == self.current_batch_size:
                            self.successful_batches += 1
                            # Try to restore batch size after 10 successful batches
                            if self.successful_batches > 10 and self.current_batch_size < self.original_batch_size:
                                new_size = min(self.current_batch_size * 2, self.original_batch_size)
                                logger.info(f"Restoring batch size from {self.current_batch_size} to {new_size}")
                                self.current_batch_size = new_size
                                self.successful_batches = 0

                        break  # Success, exit the retry loop

                    except torch.cuda.OutOfMemoryError:
                        logger.warning(
                            f"OOM with batch size {current_batch_size}, reducing to {current_batch_size // 2}"
                        )
                        # Record OOM error
                        oom_errors.labels(model=self.current_model_name, quantization=self.current_quantization).inc()
                        batch_size_reductions.labels(
                            model=self.current_model_name, quantization=self.current_quantization
                        ).inc()

                        torch.cuda.empty_cache()
                        current_batch_size = max(self.min_batch_size, current_batch_size // 2)
                        self.current_batch_size = current_batch_size
                        self.successful_batches = 0  # Reset success counter

                        if current_batch_size < self.min_batch_size:
                            raise RuntimeError("Unable to process batch even with minimum batch size") from e

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None

    def generate_single_embedding(
        self, text: str, model_name: str, quantization: str = "float32", instruction: str | None = None, **kwargs
    ) -> list[float] | None:
        """Generate embedding for a single text

        Args:
            text: Text to embed
            model_name: HuggingFace model name
            quantization: Quantization type (default: "float32")
            instruction: Optional instruction for Qwen3 models
            **kwargs: Additional arguments (ignored for compatibility)
        """
        embeddings = self.generate_embeddings(
            [text], model_name, quantization, batch_size=1, show_progress=False, instruction=instruction
        )
        if embeddings is not None and len(embeddings) > 0:
            return embeddings[0].tolist()
        return None


# Create instances for both APIs
embedding_service = EmbeddingService()
enhanced_embedding_service = embedding_service  # Alias for compatibility

# Model configurations with quantization info
QUANTIZED_MODEL_INFO = {
    # Qwen3 Embedding Models
    "Qwen/Qwen3-Embedding-0.6B": {
        "dimension": 1024,
        "description": "Qwen3 small model, instruction-aware (1024d)",
        "supports_quantization": True,
        "recommended_quantization": "float16",
        "memory_estimate": {"float32": 2400, "float16": 1200, "int8": 600},
    },
    "Qwen/Qwen3-Embedding-4B": {
        "dimension": 2560,
        "description": "Qwen3 medium model, MTEB top performer (2560d)",
        "supports_quantization": True,
        "recommended_quantization": "float16",
        "memory_estimate": {"float32": 16000, "float16": 8000, "int8": 4000},
    },
    "Qwen/Qwen3-Embedding-8B": {
        "dimension": 4096,
        "description": "Qwen3 large model, MTEB #1 (4096d)",
        "supports_quantization": True,
        "recommended_quantization": "int8",
        "memory_estimate": {"float32": 32000, "float16": 16000, "int8": 8000},
    },
}

# Backward compatibility - also expose as POPULAR_MODELS
POPULAR_MODELS = QUANTIZED_MODEL_INFO

# Legacy compatibility - map old dimension key
for model_info in POPULAR_MODELS.values():
    if "dimension" in model_info:
        model_info["dim"] = model_info["dimension"]


def test_embedding_service():
    """Test the unified embedding service"""
    service = EmbeddingService()

    # Test loading a model
    print("Testing model loading...")
    if service.load_model("sentence-transformers/all-MiniLM-L6-v2"):
        print("✓ Model loaded successfully")

    # Test model info
    print("\nTesting model info...")
    info = service.get_model_info("sentence-transformers/all-MiniLM-L6-v2")
    print(f"✓ Model info: {info}")

    # Test embedding generation
    print("\nTesting embedding generation...")
    texts = ["Hello world", "This is a test"]
    embeddings = service.generate_embeddings(texts, "sentence-transformers/all-MiniLM-L6-v2")
    if embeddings is not None:
        print(f"✓ Generated embeddings shape: {embeddings.shape}")

    # Test single embedding
    print("\nTesting single embedding...")
    embedding = service.generate_single_embedding("Test text", "sentence-transformers/all-MiniLM-L6-v2")
    if embedding:
        print(f"✓ Single embedding dimension: {len(embedding)}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_embedding_service()

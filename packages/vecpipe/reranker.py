"""
Cross-encoder reranking service for improved search relevance
Uses Qwen3-Reranker models for state-of-the-art reranking
"""

import logging
import threading
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Handles cross-encoder reranking using Qwen3-Reranker models

    The Qwen3-Reranker models use a special format where they predict
    the probability of "yes" vs "no" tokens to determine relevance.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda",
        quantization: str = "float16",
        max_length: int = 512,
    ):
        """
        Initialize the reranker

        Args:
            model_name: Name of the Qwen3-Reranker model
            device: Device to run on ('cuda' or 'cpu')
            quantization: Quantization type ('float32', 'float16', 'int8')
            max_length: Maximum sequence length for input
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.quantization = quantization
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()  # Thread safety lock

        # Model size to batch size mapping
        self.batch_size_config = {
            "0.6B": {"float32": 64, "float16": 128, "int8": 256},
            "4B": {"float32": 16, "float16": 32, "int8": 64},
            "8B": {"float32": 8, "float16": 16, "int8": 32},
        }

    def _get_model_size(self) -> str:
        """Extract model size from model name"""
        if "0.6B" in self.model_name:
            return "0.6B"
        if "4B" in self.model_name:
            return "4B"
        if "8B" in self.model_name:
            return "8B"
        return "4B"  # Default

    def get_batch_size(self) -> int:
        """Get optimal batch size based on model and quantization"""
        model_size = self._get_model_size()
        return self.batch_size_config.get(model_size, {}).get(self.quantization, 32)

    def load_model(self) -> None:
        """Load the reranker model and tokenizer"""
        with self._lock:
            if self.model is not None:
                logger.info(f"Model {self.model_name} already loaded")
                return

            logger.info(f"Loading reranker model: {self.model_name}")
            start_time = time.time()

            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="left",  # Important for batch processing
                )
                if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None):
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Determine torch dtype based on quantization
                torch_dtype = torch.float32
                if self.quantization == "float16":
                    torch_dtype = torch.float16
                elif self.quantization == "bfloat16":
                    torch_dtype = torch.bfloat16

                # Load model with appropriate settings
                # Use explicit device_map={"": 0} instead of "auto" to ensure
                # proper cleanup when offloading to CPU. device_map="auto" creates
                # internal Accelerate hooks that don't release GPU memory properly
                # with .to("cpu") calls.
                load_kwargs: dict[str, Any] = {
                    "torch_dtype": torch_dtype,
                    "device_map": {"": 0} if self.device == "cuda" else None,
                    "trust_remote_code": True,
                }

                # Add flash attention if available
                # Note: attn_implementation parameter is not available in transformers 4.53.0
                # Flash attention will be auto-detected if flash_attn package is installed
                import importlib.util

                if importlib.util.find_spec("flash_attn") is not None:
                    logger.info("Flash Attention package detected, will be used automatically if supported by model")
                else:
                    logger.debug("Flash Attention not available")

                # Apply int8 quantization if requested
                if self.quantization == "int8":
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16,
                    )
                    load_kwargs["quantization_config"] = quantization_config
                    # Remove torch_dtype when using quantization_config
                    load_kwargs.pop("torch_dtype", None)

                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

                # Move to device if not using device_map
                if self.model is not None and (self.device == "cpu" or self.quantization != "int8"):
                    self.model = self.model.to(self.device)

                if self.model is not None:
                    self.model.eval()

                load_time = time.time() - start_time
                logger.info(f"Reranker model loaded in {load_time:.2f}s")

            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                raise RuntimeError(f"Failed to load reranker model: {e}") from e

    def unload_model(self) -> None:
        """Unload the model to free memory"""
        with self._lock:
            if self.model is not None:
                logger.info(f"Unloading reranker model: {self.model_name}")
                del self.model
                self.model = None
                del self.tokenizer
                self.tokenizer = None

                # Synchronize and clear GPU cache
                # CRITICAL: CUDA operations are asynchronous. We must synchronize
                # before empty_cache() to ensure model deletion is complete.
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

    def format_input(self, query: str, document: str, instruction: str | None = None) -> str:
        """
        Format input for Qwen3-Reranker models

        Args:
            query: Search query
            document: Document to score
            instruction: Optional custom instruction

        Returns:
            Formatted input string
        """
        if instruction is None:
            instruction = "Given the query and document, determine if the document is relevant to the query."

        # Qwen3-Reranker format
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    @torch.no_grad()
    def compute_relevance_scores(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
        batch_size: int | None = None,
    ) -> list[float]:
        """
        Compute relevance scores for documents given a query

        Args:
            query: Search query
            documents: List of documents to score
            instruction: Optional custom instruction
            batch_size: Override default batch size

        Returns:
            List of relevance scores (0-1 range)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            return []

        # Clean documents and track indices
        valid_docs = []
        for i, doc in enumerate(documents):
            if not doc or not doc.strip():
                logger.warning(f"Document at index {i} is empty, using placeholder")
                valid_docs.append(".")  # Use minimal placeholder to maintain indices
            else:
                valid_docs.append(doc)

        if batch_size is None:
            batch_size = self.get_batch_size()

        all_scores = []

        # Process in batches
        for i in range(0, len(valid_docs), batch_size):
            batch_docs = valid_docs[i : i + batch_size]

            # Format inputs
            inputs = [self.format_input(query, doc, instruction) for doc in batch_docs]

            # Tokenize
            assert self.tokenizer is not None
            encoded = self.tokenizer(
                inputs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)

            # Get model outputs
            # Qwen3-Reranker is a causal LM; by default it produces logits for every
            # token in the sequence, which can be extremely memory hungry
            # (batch, seq_len, vocab). We only need the last token's logits to
            # score Yes/No, so ask the model to keep just the final step when
            # supported.
            try:
                outputs = self.model(
                    **encoded,
                    output_hidden_states=False,
                    use_cache=False,
                    logits_to_keep=1,
                )
            except TypeError:
                # Older transformers / other model implementations may not support
                # logits_to_keep; fall back to the full logits.
                outputs = self.model(
                    **encoded,
                    output_hidden_states=False,
                    use_cache=False,
                )
            logits = outputs.logits

            # Get the logits for "yes" and "no" tokens
            # Token IDs for Qwen models
            yes_tokens = self.tokenizer.encode("Yes", add_special_tokens=False)
            no_tokens = self.tokenizer.encode("No", add_special_tokens=False)

            if len(yes_tokens) != 1 or len(no_tokens) != 1:
                # Fallback to lowercase if capitalized versions don't work
                yes_tokens = self.tokenizer.encode("yes", add_special_tokens=False)
                no_tokens = self.tokenizer.encode("no", add_special_tokens=False)

                if len(yes_tokens) != 1 or len(no_tokens) != 1:
                    raise ValueError(
                        f"Yes/No tokens must encode to single tokens. Got Yes: {yes_tokens}, No: {no_tokens}"
                    )

            yes_token_id = yes_tokens[0]
            no_token_id = no_tokens[0]

            # Get last token logits
            last_token_logits = logits[:, -1, :]

            # Extract yes/no logits
            yes_logits = last_token_logits[:, yes_token_id]
            no_logits = last_token_logits[:, no_token_id]

            # Compute probability of "yes" using softmax over yes/no
            yes_no_logits = torch.stack([no_logits, yes_logits], dim=1)
            probs = torch.nn.functional.softmax(yes_no_logits, dim=1)
            yes_probs = probs[:, 1]  # Probability of "yes"

            # Convert to list and add to results
            batch_scores = yes_probs.cpu().tolist()
            all_scores.extend(batch_scores)

        return all_scores

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int,
        instruction: str | None = None,
        return_scores: bool = False,  # noqa: ARG002
    ) -> list[tuple[int, float]]:
        """
        Rerank documents based on relevance to query

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            instruction: Optional custom instruction
            return_scores: Whether to return scores with indices

        Returns:
            List of (index, score) tuples sorted by relevance (highest first)
        """
        start_time = time.time()

        # Compute relevance scores
        scores = self.compute_relevance_scores(query, documents, instruction)

        # Create index-score pairs and sort
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        top_results = indexed_scores[:top_k]

        rerank_time = time.time() - start_time
        if top_results:
            logger.info(
                f"Reranked {len(documents)} documents in {rerank_time:.2f}s (top score: {top_results[0][1]:.3f})"
            )
        else:
            logger.info(f"Reranked {len(documents)} documents in {rerank_time:.2f}s (no results)")

        return top_results

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {
                "loaded": False,
                "model_name": self.model_name,
                "device": self.device,
                "quantization": self.quantization,
            }

        # Calculate model size in memory
        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_gb = param_count * 4 / (1024**3)  # Assuming float32

        if self.quantization == "float16" or self.quantization == "bfloat16":
            param_size_gb /= 2
        elif self.quantization == "int8":
            param_size_gb /= 4

        return {
            "loaded": True,
            "model_name": self.model_name,
            "device": self.device,
            "quantization": self.quantization,
            "param_count": param_count,
            "estimated_size_gb": round(param_size_gb, 2),
            "batch_size": self.get_batch_size(),
            "max_length": self.max_length,
        }

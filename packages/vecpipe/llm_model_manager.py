"""Local LLM model lifecycle manager with GPU memory governance.

Manages local LLM models using HuggingFace Transformers with bitsandbytes
quantization. Integrates with GPUMemoryGovernor for coordinated memory
management alongside embedding and reranker models.

Key design decisions:
- Phase 1 uses bitsandbytes int4/int8 quantization only
- No CPU warm offload for LLMs (unload-only eviction)
- One in-flight generation per model key (serialized via per-model locks)
- Blocking HuggingFace operations run in dedicated ThreadPoolExecutor
"""

from __future__ import annotations

import asyncio
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from shared.config import settings
from vecpipe.llm_memory import get_llm_memory_requirement
from vecpipe.memory_governor import GPUMemoryGovernor, ModelType

logger = logging.getLogger(__name__)


class LLMModelManager:
    """Manages local LLM lifecycle with GPU memory governance.

    Follows the same pattern as SparseModelManager:
    - Lazy loading on demand
    - Governor callbacks for unload (offload is future/optional per backend)
    - LRU-based eviction coordination

    Attributes:
        _governor: Optional GPU memory governor for coordinated memory management
        _models: Dict mapping model keys to (model, tokenizer) tuples
        _inflight_loads: Dict tracking in-flight load tasks to prevent duplicate loads
        _global_lock: Protects load/unload bookkeeping
        _model_locks: Per-model locks for serialized generation
        _executor: ThreadPoolExecutor for blocking HuggingFace operations
    """

    def __init__(
        self,
        governor: GPUMemoryGovernor | None = None,
        unload_after_seconds: int | None = None,  # noqa: ARG002
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        """Initialize the LLM model manager.

        Args:
            governor: Optional GPU memory governor for coordinated memory management
            unload_after_seconds: Unused in Phase 1 (governor handles eviction)
            executor: Optional ThreadPoolExecutor for blocking operations
        """
        # unload_after_seconds is accepted for API consistency but unused in Phase 1
        _ = unload_after_seconds
        self._governor = governor
        self._models: dict[str, tuple[Any, Any]] = {}  # key -> (model, tokenizer)
        self._inflight_loads: dict[str, asyncio.Task[tuple[Any, Any]]] = {}
        self._global_lock = asyncio.Lock()
        self._model_locks: dict[str, asyncio.Lock] = {}
        self._executor = executor or ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="llm-gen-"
        )
        self._owns_executor = executor is None

        # Register callbacks with governor
        if governor:
            governor.register_callbacks(
                ModelType.LLM,
                unload_fn=self._governor_unload_callback,
                # NOTE: Phase 1 (bitsandbytes int4/int8) does NOT register offload_fn.
                # LLM eviction under memory pressure is unload-only.
            )
            logger.info("LLMModelManager initialized with governor callbacks")
        else:
            logger.info("LLMModelManager initialized without memory governor")

    def _get_model_lock(self, key: str) -> asyncio.Lock:
        """Get or create a per-model lock for serialized generation."""
        if key not in self._model_locks:
            self._model_locks[key] = asyncio.Lock()
        return self._model_locks[key]

    async def generate(
        self,
        model_name: str,
        quantization: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        """Generate text, loading model if needed.

        Concurrency rules:
        - Exactly one in-flight generation per model key (serialize via per-model lock)
        - Governor unload callback also acquires the same per-model lock, so eviction
          waits for in-flight generation to complete

        Args:
            model_name: HuggingFace model ID
            quantization: Quantization type ("int4", "int8", "float16")
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with "content", "prompt_tokens", "completion_tokens"

        Raises:
            RuntimeError: If insufficient GPU memory or model loading fails
        """
        key = f"{model_name}:{quantization}"
        async with self._get_model_lock(key):
            model, tokenizer = await self._ensure_model_loaded(model_name, quantization)

            # Run generation in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            content, prompt_tokens, completion_tokens = await loop.run_in_executor(
                self._executor,
                self._generate_sync,
                model,
                tokenizer,
                prompt,
                system_prompt,
                temperature,
                max_tokens,
            )

            # Touch for LRU update
            if self._governor:
                await self._governor.touch(model_name, ModelType.LLM, quantization)

            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

    def _generate_sync(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Synchronous generation helper (runs in executor).

        Args:
            model: Loaded HuggingFace model
            tokenizer: Loaded tokenizer
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens)
        """
        import torch

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "Tokenizer has no chat template; use curated chat models only in Phase 1"
            )

        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        prompt_tokens = int(input_ids.shape[-1])
        input_ids = input_ids.to(model.device)

        pad_token_id = getattr(tokenizer, "eos_token_id", None)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=pad_token_id,
            )

        response_ids = outputs[0][input_ids.shape[1] :]
        content = tokenizer.decode(response_ids, skip_special_tokens=True)
        completion_tokens = int(response_ids.shape[-1])

        return content, prompt_tokens, completion_tokens

    async def _ensure_model_loaded(
        self, model_name: str, quantization: str
    ) -> tuple[Any, Any]:
        """Load model with governor memory management.

        Uses an in-flight task map so concurrent requests don't double-load.

        Args:
            model_name: HuggingFace model ID
            quantization: Quantization type

        Returns:
            Tuple of (model, tokenizer)
        """
        key = f"{model_name}:{quantization}"

        async with self._global_lock:
            if key in self._models:
                return self._models[key]

            inflight = self._inflight_loads.get(key)
            if inflight is None:
                inflight = asyncio.create_task(
                    self._load_and_register_model(model_name, quantization)
                )
                self._inflight_loads[key] = inflight

        try:
            model, tokenizer = await inflight
            return model, tokenizer
        finally:
            async with self._global_lock:
                # Ensure we don't leak tasks in the in-flight map
                self._inflight_loads.pop(key, None)

    async def _load_and_register_model(
        self, model_name: str, quantization: str
    ) -> tuple[Any, Any]:
        """Load model, then register it with the governor and local cache.

        Args:
            model_name: HuggingFace model ID
            quantization: Quantization type

        Returns:
            Tuple of (model, tokenizer)
        """
        key = f"{model_name}:{quantization}"

        # Request memory from governor (admission control + evictions if needed)
        required_mb = get_llm_memory_requirement(model_name, quantization)
        if self._governor:
            can_allocate = await self._governor.request_memory(
                model_name=model_name,
                model_type=ModelType.LLM,
                quantization=quantization,
                required_mb=required_mb,
            )
            if not can_allocate:
                raise RuntimeError(f"Insufficient GPU memory for {model_name}")

        try:
            model, tokenizer = await self._load_model(model_name, quantization)
        except Exception as e:
            # Clean up governor tracking on failure
            if self._governor:
                await self._governor.mark_unloaded(model_name, ModelType.LLM, quantization)
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e

        async with self._global_lock:
            self._models[key] = (model, tokenizer)

        if self._governor:
            # Pass explicit memory_mb; the governor otherwise calls vecpipe.memory_utils,
            # which does not know LLM sizes/quantizations by default.
            await self._governor.mark_loaded(
                model_name=model_name,
                model_type=ModelType.LLM,
                quantization=quantization,
                model_ref=model,
                memory_mb=required_mb,
            )

        logger.info(
            "Loaded LLM model %s (%s) - estimated %dMB",
            model_name,
            quantization,
            required_mb,
        )

        return model, tokenizer

    async def _load_model(
        self, model_name: str, quantization: str
    ) -> tuple[Any, Any]:
        """Load HuggingFace model with bitsandbytes quantization.

        Runs in executor since model loading is blocking.

        Args:
            model_name: HuggingFace model ID
            quantization: Quantization type

        Returns:
            Tuple of (model, tokenizer)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._load_model_sync,
            model_name,
            quantization,
        )

    def _load_model_sync(
        self, model_name: str, quantization: str
    ) -> tuple[Any, Any]:
        """Synchronous model loading helper (runs in executor).

        Args:
            model_name: HuggingFace model ID
            quantization: Quantization type

        Returns:
            Tuple of (model, tokenizer)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading LLM model %s with %s quantization...", model_name, quantization)

        # Configure quantization
        if quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "int8":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            # float16 - no quantization
            bnb_config = None

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=settings.LLM_TRUST_REMOTE_CODE,
        )

        # Load model with quantization config
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": settings.LLM_TRUST_REMOTE_CODE,
            "device_map": {"": 0},  # Single GPU, avoid auto for better control
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        logger.info("Successfully loaded %s", model_name)
        return model, tokenizer

    async def _governor_unload_callback(
        self, model_name: str, quantization: str
    ) -> None:
        """Fully unload model from memory (governor callback).

        Args:
            model_name: Model identifier
            quantization: Quantization type
        """
        import torch

        key = f"{model_name}:{quantization}"

        # Ensure eviction waits for any in-flight generation
        async with self._get_model_lock(key), self._global_lock:
            if key in self._models:
                model, tokenizer = self._models.pop(key)
                del model
                del tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                logger.info("Governor-initiated unload of LLM '%s'", key)
            else:
                logger.warning(
                    "Governor requested unload of '%s' but not loaded", key
                )

    async def shutdown(self) -> None:
        """Shutdown the manager and cleanup resources."""
        logger.info("Shutting down LLMModelManager...")

        async with self._global_lock:
            # Unload all models
            for key in list(self._models.keys()):
                model_name, quantization = key.rsplit(":", 1)
                await self._governor_unload_callback(model_name, quantization)
            self._models.clear()

        # Shutdown executor if we own it
        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None  # type: ignore[assignment]

        logger.info("LLMModelManager shutdown complete")


__all__ = ["LLMModelManager"]

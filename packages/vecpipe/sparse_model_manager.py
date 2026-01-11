"""Sparse Model Manager - Plugin lifecycle management with GPU memory governance.

This module provides a manager for sparse indexer plugins (SPLADE) that integrates
with the GPU memory governor for coordinated memory management.

The manager:
- Lazily loads sparse indexer plugins on demand
- Tracks loaded plugins and their memory usage
- Integrates with GPUMemoryGovernor for memory coordination
- Supports CPU offloading for warm model pool
- Handles plugin initialization and cleanup

Note: BM25 is CPU-only and doesn't need governor coordination. This manager
focuses on GPU-based sparse indexers like SPLADE.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

from shared.plugins.loader import load_plugins
from shared.plugins.registry import plugin_registry

if TYPE_CHECKING:
    from shared.plugins.types.sparse_indexer import (
        SparseIndexerPlugin,
        SparseQueryVector,
        SparseVector,
    )

from .cpu_offloader import get_offloader
from .memory_governor import GPUMemoryGovernor, ModelType

logger = logging.getLogger(__name__)

# Memory estimates for SPLADE models (conservative, in MB)
# Base model is ~400MB, with overhead for inference
SPLADE_BASE_MEMORY_MB = 400
SPLADE_BATCH_OVERHEAD_MB = 50  # Per batch_size=8 increment


def _estimate_splade_memory(config: dict[str, Any] | None) -> int:
    """Estimate SPLADE memory requirement based on config.

    Args:
        config: Plugin configuration with optional quantization and batch_size.

    Returns:
        Estimated memory in MB.
    """
    if config is None:
        config = {}

    base_mb = SPLADE_BASE_MEMORY_MB
    quantization = config.get("quantization", "float16")
    batch_size: int = config.get("batch_size", 32)

    # Adjust for quantization
    if quantization == "float32":
        base_mb *= 2
    elif quantization == "int8":
        base_mb = int(base_mb * 0.6)

    # Add batch overhead
    batch_overhead = (batch_size // 8) * SPLADE_BATCH_OVERHEAD_MB

    return base_mb + batch_overhead


def _config_hash(config: dict[str, Any] | None) -> str:
    """Generate a hash for plugin config to detect config changes.

    Args:
        config: Plugin configuration dict.

    Returns:
        Short hash string (8 chars).
    """
    if config is None:
        return "default"
    # Sort keys for consistent hashing
    import json

    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


class SparseModelManager:
    """Manages sparse indexer plugins with GPU memory governance.

    This manager handles the lifecycle of sparse indexer plugins (primarily SPLADE)
    with integration into the GPU memory governor for coordinated memory management.

    Example:
        governor = GPUMemoryGovernor(budget=budget)
        manager = SparseModelManager(governor=governor)

        # Encode documents
        vectors = await manager.encode_documents(
            plugin_id="splade-local",
            texts=["hello world"],
            chunk_ids=["chunk-1"],
            config={"batch_size": 32},
        )

        # Cleanup
        await manager.shutdown()
    """

    def __init__(self, governor: GPUMemoryGovernor | None = None) -> None:
        """Initialize the sparse model manager.

        Args:
            governor: Optional GPU memory governor for coordinated memory management.
                     If None, plugins are loaded without memory governance.
        """
        self._governor = governor
        self._lock = asyncio.Lock()

        # Track loaded plugins: key = "plugin_id:config_hash"
        self._loaded_plugins: dict[str, SparseIndexerPlugin] = {}
        self._plugin_configs: dict[str, dict[str, Any]] = {}  # Store config for each plugin key

        # CPU offloader for warm model pool
        self._offloader = get_offloader()

        # Register callbacks with governor if available
        if self._governor is not None:
            self._governor.register_callbacks(
                ModelType.SPARSE,
                unload_fn=self._governor_unload_sparse,
                offload_fn=self._governor_offload_sparse,
            )

        logger.info(
            "SparseModelManager initialized (governor=%s)",
            "enabled" if governor else "disabled",
        )

    def _get_plugin_key(self, plugin_id: str, config: dict[str, Any] | None) -> str:
        """Generate unique key for a plugin with its config.

        Args:
            plugin_id: The sparse indexer plugin ID.
            config: Plugin configuration.

        Returns:
            Unique key string "plugin_id:config_hash".
        """
        return f"{plugin_id}:{_config_hash(config)}"

    def _get_governor_model_name(self, plugin_id: str, config: dict[str, Any] | None) -> str:
        """Generate model name for governor tracking.

        Args:
            plugin_id: The sparse indexer plugin ID.
            config: Plugin configuration.

        Returns:
            Model name for governor (e.g., "splade-local:abc12345").
        """
        return self._get_plugin_key(plugin_id, config)

    async def _load_plugin(
        self,
        plugin_id: str,
        config: dict[str, Any] | None,
    ) -> SparseIndexerPlugin:
        """Load and initialize a sparse indexer plugin.

        Args:
            plugin_id: The sparse indexer plugin ID.
            config: Plugin configuration.

        Returns:
            Initialized sparse indexer plugin.

        Raises:
            ValueError: If plugin not found or not a sparse indexer.
            RuntimeError: If initialization fails.
        """
        # Ensure sparse_indexer plugins are loaded
        load_plugins(plugin_types={"sparse_indexer"})

        # Get the plugin class from registry
        record = plugin_registry.find_by_id(plugin_id)
        if record is None:
            raise ValueError(f"Sparse indexer plugin '{plugin_id}' not found")

        if record.plugin_type != "sparse_indexer":
            raise ValueError(f"Plugin '{plugin_id}' is not a sparse indexer (type: {record.plugin_type})")

        # Instantiate the plugin
        plugin_cls = record.plugin_class
        plugin = plugin_cls()

        # Initialize with config
        init_config = config or {}
        if hasattr(plugin, "initialize") and callable(plugin.initialize):
            await plugin.initialize(init_config)

        logger.info("Loaded sparse indexer plugin '%s' with config hash %s", plugin_id, _config_hash(config))

        return plugin

    async def _unload_plugin(self, plugin_key: str) -> None:
        """Unload a sparse indexer plugin and free resources.

        Args:
            plugin_key: The plugin key (plugin_id:config_hash).
        """
        plugin = self._loaded_plugins.pop(plugin_key, None)
        self._plugin_configs.pop(plugin_key, None)

        if plugin is not None:
            # Cleanup plugin
            if hasattr(plugin, "cleanup") and callable(plugin.cleanup):
                await plugin.cleanup()

            # Force garbage collection
            del plugin
            gc.collect()

            # Clear CUDA cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Unloaded sparse indexer plugin '%s'", plugin_key)

    async def get_or_load_plugin(
        self,
        plugin_id: str,
        config: dict[str, Any] | None = None,
    ) -> SparseIndexerPlugin:
        """Get a loaded plugin or load it if not already loaded.

        Args:
            plugin_id: The sparse indexer plugin ID.
            config: Plugin configuration.

        Returns:
            The sparse indexer plugin instance.

        Raises:
            RuntimeError: If memory cannot be allocated or plugin fails to load.
        """
        plugin_key = self._get_plugin_key(plugin_id, config)

        async with self._lock:
            # Fast path: already loaded
            if plugin_key in self._loaded_plugins:
                plugin = self._loaded_plugins[plugin_key]

                # Check if model was offloaded to CPU and restore if needed
                offloader_key = f"sparse:{plugin_key}"
                if self._offloader.is_offloaded(offloader_key):
                    logger.info("Restoring offloaded sparse plugin '%s' to GPU", plugin_key)
                    self._offloader.restore_to_gpu(offloader_key)

                    # Update plugin's device reference after restoration
                    model = getattr(plugin, "_model", None)
                    if model is not None:
                        # Get the device the model is now on
                        try:
                            first_param = next(model.parameters())
                            actual_device = str(first_param.device)
                            if hasattr(plugin, "_actual_device"):
                                plugin._actual_device = actual_device
                                logger.debug("Updated plugin device to %s", actual_device)
                        except StopIteration:
                            pass

                # Touch in governor to update LRU
                if self._governor is not None:
                    model_name = self._get_governor_model_name(plugin_id, config)
                    quantization = (config or {}).get("quantization", "float16")
                    await self._governor.touch(model_name, ModelType.SPARSE, quantization)
                return plugin

            # Request memory from governor
            if self._governor is not None:
                model_name = self._get_governor_model_name(plugin_id, config)
                quantization = (config or {}).get("quantization", "float16")
                required_mb = _estimate_splade_memory(config)

                can_allocate = await self._governor.request_memory(
                    model_name=model_name,
                    model_type=ModelType.SPARSE,
                    quantization=quantization,
                    required_mb=required_mb,
                )

                if not can_allocate:
                    stats = self._governor.get_memory_stats()
                    raise RuntimeError(
                        f"Cannot allocate memory for sparse indexer '{plugin_id}' "
                        f"({required_mb}MB required). Memory stats: {stats}"
                    )

            # Load the plugin
            try:
                plugin = await self._load_plugin(plugin_id, config)
                self._loaded_plugins[plugin_key] = plugin
                self._plugin_configs[plugin_key] = config or {}

                # Mark as loaded in governor
                if self._governor is not None:
                    model_name = self._get_governor_model_name(plugin_id, config)
                    quantization = (config or {}).get("quantization", "float16")
                    model_ref = getattr(plugin, "_model", None)
                    await self._governor.mark_loaded(
                        model_name=model_name,
                        model_type=ModelType.SPARSE,
                        quantization=quantization,
                        model_ref=model_ref,
                    )

                return plugin

            except Exception as e:
                # Clean up governor tracking on failure
                if self._governor is not None:
                    model_name = self._get_governor_model_name(plugin_id, config)
                    quantization = (config or {}).get("quantization", "float16")
                    await self._governor.mark_unloaded(model_name, ModelType.SPARSE, quantization)
                raise RuntimeError(f"Failed to load sparse indexer '{plugin_id}': {e}") from e

    async def encode_documents(
        self,
        plugin_id: str,
        texts: list[str],
        chunk_ids: list[str],
        config: dict[str, Any] | None = None,
    ) -> list[SparseVector]:
        """Encode documents to sparse vectors.

        Args:
            plugin_id: The sparse indexer plugin ID.
            texts: Document texts to encode.
            chunk_ids: Chunk IDs for each text.
            config: Plugin configuration.

        Returns:
            List of SparseVector instances.

        Raises:
            RuntimeError: If encoding fails.
            ValueError: If texts and chunk_ids have different lengths.
        """
        if len(texts) != len(chunk_ids):
            raise ValueError(f"texts ({len(texts)}) and chunk_ids ({len(chunk_ids)}) must have same length")

        if not texts:
            return []

        # Get or load plugin
        plugin = await self.get_or_load_plugin(plugin_id, config)

        # Build documents in expected format
        documents = [
            {"content": text, "chunk_id": chunk_id, "metadata": {}}
            for text, chunk_id in zip(texts, chunk_ids, strict=True)
        ]

        # Encode documents
        start_time = time.time()
        vectors = await plugin.encode_documents(documents)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Encoded %d documents with '%s' in %.1fms",
            len(documents),
            plugin_id,
            elapsed_ms,
        )

        return list(vectors)

    async def encode_query(
        self,
        plugin_id: str,
        query: str,
        config: dict[str, Any] | None = None,
    ) -> SparseQueryVector:
        """Encode a query to sparse vector.

        Args:
            plugin_id: The sparse indexer plugin ID.
            query: Query text to encode.
            config: Plugin configuration.

        Returns:
            SparseQueryVector instance.

        Raises:
            RuntimeError: If encoding fails.
        """
        # Get or load plugin
        plugin = await self.get_or_load_plugin(plugin_id, config)

        # Encode query
        start_time = time.time()
        vector = await plugin.encode_query(query)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Encoded query with '%s' in %.1fms (vector size: %d)",
            plugin_id,
            elapsed_ms,
            len(vector.indices),
        )

        return vector

    async def unload_plugin(self, plugin_id: str, config: dict[str, Any] | None = None) -> bool:
        """Unload a specific plugin.

        Args:
            plugin_id: The sparse indexer plugin ID.
            config: Plugin configuration (to match the loaded instance).

        Returns:
            True if plugin was unloaded, False if not found.
        """
        plugin_key = self._get_plugin_key(plugin_id, config)

        async with self._lock:
            if plugin_key not in self._loaded_plugins:
                return False

            # Notify governor
            if self._governor is not None:
                model_name = self._get_governor_model_name(plugin_id, config)
                quantization = (config or {}).get("quantization", "float16")
                await self._governor.mark_unloaded(model_name, ModelType.SPARSE, quantization)

            await self._unload_plugin(plugin_key)
            return True

    async def shutdown(self) -> None:
        """Shutdown manager and unload all plugins."""
        async with self._lock:
            # Unload all plugins
            for plugin_key in list(self._loaded_plugins.keys()):
                await self._unload_plugin(plugin_key)

            self._loaded_plugins.clear()
            self._plugin_configs.clear()

        logger.info("SparseModelManager shutdown complete")

    def get_loaded_plugins(self) -> list[dict[str, Any]]:
        """Get information about loaded plugins.

        Returns:
            List of dicts with plugin_key, plugin_id, config info.
        """
        result = []
        for plugin_key, plugin in self._loaded_plugins.items():
            config = self._plugin_configs.get(plugin_key, {})
            parts = plugin_key.split(":", 1)
            plugin_id = parts[0] if parts else plugin_key

            result.append(
                {
                    "plugin_key": plugin_key,
                    "plugin_id": plugin_id,
                    "config_hash": parts[1] if len(parts) > 1 else "default",
                    "sparse_type": getattr(plugin, "SPARSE_TYPE", "unknown"),
                    "model_name": config.get("model_name", "default"),
                    "quantization": config.get("quantization", "float16"),
                }
            )
        return result

    # === Governor callbacks ===

    async def _governor_unload_sparse(
        self,
        model_name: str,
        quantization: str,
    ) -> None:
        """Callback for governor to unload a sparse model.

        Args:
            model_name: The model name in governor format (plugin_id:config_hash).
            quantization: Model quantization (unused, but required by callback signature).
        """
        _ = quantization  # Unused in this callback

        # model_name is "plugin_id:config_hash"
        plugin_key = model_name

        if plugin_key in self._loaded_plugins:
            # Discard from offloader
            offloader_key = f"sparse:{plugin_key}"
            self._offloader.discard(offloader_key)

            # Unload without notifying governor (it already knows)
            await self._unload_plugin(plugin_key)
            logger.info("Governor-initiated unload of sparse plugin '%s'", plugin_key)
        else:
            logger.warning(
                "Governor requested unload of sparse '%s' but it's not loaded",
                model_name,
            )

    async def _governor_offload_sparse(
        self,
        model_name: str,
        quantization: str,
        target_device: str,
    ) -> None:
        """Callback for governor to offload/restore a sparse model.

        Args:
            model_name: The model name in governor format (plugin_id:config_hash).
            quantization: Model quantization (unused).
            target_device: Target device ("cpu" for offload, "cuda" for restore).
        """
        _ = quantization  # Unused

        plugin_key = model_name
        offloader_key = f"sparse:{plugin_key}"

        if plugin_key not in self._loaded_plugins:
            raise RuntimeError(f"Cannot offload sparse '{plugin_key}': not loaded")

        plugin = self._loaded_plugins[plugin_key]
        model = getattr(plugin, "_model", None)

        if model is None:
            raise RuntimeError(f"Cannot offload sparse '{plugin_key}': no model attribute")

        if target_device == "cpu":
            self._offloader.offload_to_cpu(offloader_key, model)
            # Update plugin's device reference
            if hasattr(plugin, "_actual_device"):
                plugin._actual_device = "cpu"
            logger.info("Offloaded sparse plugin '%s' to CPU", plugin_key)

        elif target_device == "cuda":
            if self._offloader.is_offloaded(offloader_key):
                self._offloader.restore_to_gpu(offloader_key)
                # Update plugin's device reference
                if hasattr(plugin, "_actual_device"):
                    # Get actual device from model parameters
                    try:
                        first_param = next(model.parameters())
                        plugin._actual_device = str(first_param.device)
                    except StopIteration:
                        plugin._actual_device = "cuda:0"
                logger.info("Restored sparse plugin '%s' to GPU", plugin_key)
            else:
                raise RuntimeError(
                    f"Cannot restore sparse '{plugin_key}': not in offloaded models "
                    "(state mismatch between governor and offloader)"
                )

"""
Qwen3 Search Optimization Configuration
Best practices and configurations for optimal search performance
"""

# Qwen3 model recommendations for different use cases
QWEN3_MODEL_RECOMMENDATIONS = {
    "high_quality": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "quantization": "int8",  # Balance quality and memory
        "description": "Best quality, MTEB #1, 4096d embeddings",
        "use_cases": ["research", "legal", "medical", "high-precision"],
    },
    "balanced": {
        "model": "Qwen/Qwen3-Embedding-4B",
        "quantization": "float16",
        "description": "Great balance of quality and speed, 2560d embeddings",
        "use_cases": ["general", "documentation", "knowledge-base"],
    },
    "fast": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "description": "Fast inference, good quality, 1024d embeddings",
        "use_cases": ["real-time", "chat", "autocomplete", "high-volume"],
    },
}

# Optimized search instructions for different domains
DOMAIN_INSTRUCTIONS = {
    "general": {
        "index": "Represent this document for retrieval:",
        "query": "Represent this sentence for searching relevant passages:",
    },
    "technical": {
        "index": "Represent this technical documentation for retrieval:",
        "query": "Represent this technical question for finding relevant documentation:",
    },
    "code": {
        "index": "Represent this code snippet for similarity search:",
        "query": "Represent this code query for finding similar implementations:",
    },
    "qa": {
        "index": "Represent this answer for question-answering retrieval:",
        "query": "Represent this question for retrieving supporting answers:",
    },
    "semantic": {
        "index": "Generate a comprehensive semantic embedding for this text:",
        "query": "Generate a semantic embedding for finding conceptually similar content:",
    },
}

# Batch processing configurations
BATCH_CONFIGS = {
    "Qwen/Qwen3-Embedding-8B": {
        "float32": {"batch_size": 8, "max_length": 8192},
        "float16": {"batch_size": 16, "max_length": 8192},
        "int8": {"batch_size": 32, "max_length": 8192},
    },
    "Qwen/Qwen3-Embedding-4B": {
        "float32": {"batch_size": 16, "max_length": 8192},
        "float16": {"batch_size": 32, "max_length": 8192},
        "int8": {"batch_size": 64, "max_length": 8192},
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "float32": {"batch_size": 64, "max_length": 32768},
        "float16": {"batch_size": 128, "max_length": 32768},
        "int8": {"batch_size": 256, "max_length": 32768},
    },
}

# Search optimization settings
SEARCH_OPTIMIZATIONS = {
    "enable_instruction_tuning": True,  # Use task-specific instructions
    "normalize_embeddings": True,  # L2 normalize for cosine similarity
    "use_last_token_pooling": True,  # Qwen3's optimal pooling strategy
    "enable_caching": True,  # Cache frequently used embeddings
    "parallel_encoding": True,  # Parallel batch processing
    "adaptive_batch_sizing": True,  # Adjust batch size based on GPU memory
}

# Reranking configurations
RERANK_CONFIG = {
    "enabled": True,
    "top_k_candidates": 50,  # Retrieve more candidates
    "final_k": 10,  # Return top-k after reranking
    "cross_encoder_model": "BAAI/bge-reranker-v2-m3",
    "use_hybrid_scoring": True,  # Combine vector similarity with reranking scores
}

# Performance monitoring
MONITORING_CONFIG = {
    "track_latencies": True,
    "log_slow_queries": True,
    "slow_query_threshold_ms": 1000,
    "collect_usage_stats": True,
    "profile_memory_usage": True,
}


def get_optimal_config(use_case: str = "balanced", gpu_memory_gb: int = 16):
    """Get optimal configuration based on use case and available resources"""

    base_config = QWEN3_MODEL_RECOMMENDATIONS.get(use_case, QWEN3_MODEL_RECOMMENDATIONS["balanced"])

    # Adjust based on GPU memory
    if gpu_memory_gb < 8:
        # Limited GPU memory - use smaller model or more aggressive quantization
        if base_config["model"] == "Qwen/Qwen3-Embedding-8B":
            base_config = QWEN3_MODEL_RECOMMENDATIONS["balanced"].copy()
            base_config["quantization"] = "int8"
        elif base_config["model"] == "Qwen/Qwen3-Embedding-4B":
            base_config["quantization"] = "int8"
    elif gpu_memory_gb >= 24:
        # Plenty of GPU memory - can use less quantization
        if base_config["quantization"] == "int8":
            base_config["quantization"] = "float16"

    # Add batch config
    model_batch_config = BATCH_CONFIGS.get(base_config["model"], {})
    base_config["batch_config"] = model_batch_config.get(
        base_config["quantization"], {"batch_size": 32, "max_length": 8192}
    )

    return base_config


# Example usage configurations
EXAMPLE_CONFIGS = {
    "high_volume_api": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
        "batch_size": 128,
        "cache_enabled": True,
        "instruction": DOMAIN_INSTRUCTIONS["general"]["query"],
    },
    "research_quality": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "quantization": "float16",
        "batch_size": 16,
        "reranking_enabled": True,
        "instruction": DOMAIN_INSTRUCTIONS["technical"]["query"],
    },
    "code_search": {
        "model": "Qwen/Qwen3-Embedding-4B",
        "quantization": "float16",
        "batch_size": 32,
        "instruction": DOMAIN_INSTRUCTIONS["code"]["query"],
    },
}

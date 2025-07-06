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

# Qwen3 Reranker model mapping - matches embedding models with their reranker counterparts
QWEN3_RERANKER_MAPPING = {
    "Qwen/Qwen3-Embedding-0.6B": "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Embedding-4B": "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Embedding-8B": "Qwen/Qwen3-Reranker-8B",
}

# Reranking configurations
RERANK_CONFIG = {
    "enabled": True,
    "candidate_multiplier": 5,  # Retrieve N * multiplier candidates for reranking
    "min_candidates": 20,  # Minimum candidates to retrieve (even for small k)
    "max_candidates": 200,  # Maximum candidates to retrieve (cap for large k)
    "default_model": "Qwen/Qwen3-Reranker-0.6B",  # Default reranker model
    "use_hybrid_scoring": True,  # Combine vector similarity with reranking scores
    "hybrid_weight": 0.3,  # Weight for original vector score (0.3 vector + 0.7 rerank)
    "batch_sizes": {
        "Qwen/Qwen3-Reranker-0.6B": {
            "float32": 64,
            "float16": 128,
            "int8": 256,
        },
        "Qwen/Qwen3-Reranker-4B": {
            "float32": 16,
            "float16": 32,
            "int8": 64,
        },
        "Qwen/Qwen3-Reranker-8B": {
            "float32": 8,
            "float16": 16,
            "int8": 32,
        },
    },
}

# Reranking instructions for different domains
RERANKING_INSTRUCTIONS = {
    "general": "Given the query and document, determine if the document is relevant to the query.",
    "technical": "Assess if this technical document provides useful information for the technical query.",
    "code": "Determine if this code snippet is relevant to the programming query.",
    "qa": "Check if this document contains information that answers the question.",
    "semantic": "Evaluate the semantic relevance between the query and document.",
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
    elif gpu_memory_gb >= 24 and base_config["quantization"] == "int8":
        # Plenty of GPU memory - can use less quantization
        base_config["quantization"] = "float16"

    # Add batch config
    model_batch_config = BATCH_CONFIGS.get(base_config["model"], {})
    base_config["batch_config"] = model_batch_config.get(
        base_config["quantization"], {"batch_size": 32, "max_length": 8192}
    )

    return base_config


def get_reranker_for_embedding_model(embedding_model: str) -> str:
    """
    Get the appropriate reranker model for a given embedding model

    Args:
        embedding_model: Name of the embedding model

    Returns:
        Name of the corresponding reranker model
    """
    # Direct mapping if available
    if embedding_model in QWEN3_RERANKER_MAPPING:
        return QWEN3_RERANKER_MAPPING[embedding_model]

    # Try to match by size
    if "0.6B" in embedding_model:
        return "Qwen/Qwen3-Reranker-0.6B"
    if "4B" in embedding_model:
        return "Qwen/Qwen3-Reranker-4B"
    if "8B" in embedding_model:
        return "Qwen/Qwen3-Reranker-8B"

    # Default to smallest model
    return RERANK_CONFIG["default_model"]


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

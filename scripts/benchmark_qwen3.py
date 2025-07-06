#!/usr/bin/env python3
"""
Test script for Qwen3 search optimizations
Demonstrates performance improvements and best practices
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from vecpipe.qwen3_search_config import DOMAIN_INSTRUCTIONS, get_optimal_config
from webui.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SearchBenchmark:
    """Benchmark different search configurations"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.results = []

    def benchmark_embedding_generation(self, texts: list[str], model_config: dict):
        """Benchmark embedding generation with different configurations"""

        model_name = model_config["model"]
        quantization = model_config.get("quantization", "float32")
        instruction = model_config.get("instruction")

        # Load model
        logger.info(f"Loading model: {model_name} with {quantization}")
        if not self.embedding_service.load_model(model_name, quantization):
            logger.error(f"Failed to load model: {model_name}")
            return None

        # Get model info
        model_info = self.embedding_service.get_model_info(model_name, quantization)
        logger.info(f"Model info: {model_info}")

        # Benchmark embedding generation
        start_time = time.time()

        embeddings = self.embedding_service.generate_embeddings(
            texts, model_name, quantization, batch_size=model_config.get("batch_size", 32), instruction=instruction
        )

        end_time = time.time()
        total_time = end_time - start_time

        if embeddings is not None:
            result = {
                "model": model_name,
                "quantization": quantization,
                "num_texts": len(texts),
                "total_time": total_time,
                "avg_time_per_text": total_time / len(texts),
                "texts_per_second": len(texts) / total_time,
                "embedding_dim": embeddings.shape[1],
                "memory_estimate_mb": model_info.get("model_size_mb", "N/A"),
            }

            logger.info(f"Results: {result}")
            return result
        logger.error("Failed to generate embeddings")
        return None

    def compare_search_quality(self, query: str, documents: list[str], configs: list[dict]):
        """Compare search quality across different configurations"""

        results = []

        for config in configs:
            logger.info(f"\nTesting config: {config['name']}")

            # Generate query embedding
            query_instruction = config.get("query_instruction", DOMAIN_INSTRUCTIONS["general"]["query"])
            query_embedding = self.embedding_service.generate_single_embedding(
                query, config["model"], config.get("quantization", "float32"), instruction=query_instruction
            )

            if query_embedding is None:
                logger.error(f"Failed to generate query embedding for {config['name']}")
                continue

            # Generate document embeddings
            doc_instruction = config.get("doc_instruction", DOMAIN_INSTRUCTIONS["general"]["index"])
            doc_embeddings = self.embedding_service.generate_embeddings(
                documents,
                config["model"],
                config.get("quantization", "float32"),
                batch_size=config.get("batch_size", 32),
                instruction=doc_instruction,
            )

            if doc_embeddings is None:
                logger.error(f"Failed to generate document embeddings for {config['name']}")
                continue

            # Calculate similarities
            query_vec = np.array(query_embedding)
            similarities = np.dot(doc_embeddings, query_vec)

            # Get top results
            top_indices = np.argsort(similarities)[::-1][:5]
            top_scores = similarities[top_indices]

            result = {
                "config": config["name"],
                "model": config["model"],
                "quantization": config.get("quantization", "float32"),
                "top_5_indices": top_indices.tolist(),
                "top_5_scores": top_scores.tolist(),
                "avg_top_5_score": np.mean(top_scores),
            }

            results.append(result)

            logger.info(f"Top 5 results for {config['name']}:")
            for idx, score in zip(top_indices, top_scores, strict=False):
                logger.info(f"  Doc {idx}: {documents[idx][:50]}... (score: {score:.4f})")

        return results


def run_benchmarks():
    """Run comprehensive benchmarks"""

    benchmark = SearchBenchmark()

    # Test documents
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning has revolutionized computer vision.",
        "Natural language processing enables machines to understand text.",
        "Transformers have become the dominant architecture in NLP.",
        "BERT and GPT are examples of transformer models.",
        "Reinforcement learning trains agents through rewards.",
        "Computer vision tasks include image classification and detection.",
        "Data preprocessing is crucial for model performance.",
    ] * 10  # Multiply for larger benchmark

    logger.info("=" * 80)
    logger.info("BENCHMARK 1: Embedding Generation Performance")
    logger.info("=" * 80)

    # Test different configurations
    configs_to_test = [
        {"name": "Qwen3-0.6B-FP16", "model": "Qwen/Qwen3-Embedding-0.6B", "quantization": "float16", "batch_size": 128},
        {"name": "Qwen3-0.6B-INT8", "model": "Qwen/Qwen3-Embedding-0.6B", "quantization": "int8", "batch_size": 256},
        {"name": "BGE-Base", "model": "BAAI/bge-base-en-v1.5", "quantization": "float32", "batch_size": 64},
    ]

    perf_results = []
    for config in configs_to_test:
        result = benchmark.benchmark_embedding_generation(test_texts, config)
        if result:
            perf_results.append(result)

    # Print performance comparison
    logger.info("\nPerformance Summary:")
    logger.info("-" * 80)
    logger.info(f"{'Model':<30} {'Quant':<8} {'Texts/s':<12} {'Avg ms/text':<12} {'Mem (MB)':<10}")
    logger.info("-" * 80)

    for result in perf_results:
        logger.info(
            f"{result['model']:<30} "
            f"{result['quantization']:<8} "
            f"{result['texts_per_second']:<12.2f} "
            f"{result['avg_time_per_text']*1000:<12.2f} "
            f"{str(result['memory_estimate_mb']):<10}"
        )

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK 2: Search Quality Comparison")
    logger.info("=" * 80)

    # Test search quality
    search_query = "What are transformer models used for in AI?"
    search_docs = [
        "Transformers are a type of neural network architecture that has revolutionized NLP.",
        "BERT and GPT are popular transformer models used for various language tasks.",
        "Computer vision has been transformed by convolutional neural networks.",
        "Transformer models excel at understanding context in sequential data.",
        "Machine learning algorithms can be supervised or unsupervised.",
        "The attention mechanism in transformers allows processing of long sequences.",
        "Deep learning requires large amounts of data for training.",
        "Natural language processing uses transformers for tasks like translation.",
        "Reinforcement learning is different from supervised learning.",
        "Pre-trained transformer models can be fine-tuned for specific tasks.",
    ]

    search_configs = [
        {
            "name": "Qwen3-with-instruction",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "query_instruction": DOMAIN_INSTRUCTIONS["qa"]["query"],
            "doc_instruction": DOMAIN_INSTRUCTIONS["qa"]["index"],
        },
        {
            "name": "Qwen3-no-instruction",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "query_instruction": None,
            "doc_instruction": None,
        },
        {"name": "BGE-baseline", "model": "BAAI/bge-base-en-v1.5", "quantization": "float32"},
    ]

    quality_results = benchmark.compare_search_quality(search_query, search_docs, search_configs)

    logger.info(f"\nSearch Quality Summary for query: '{search_query}'")
    logger.info("-" * 80)

    for result in quality_results:
        logger.info(f"\n{result['config']}:")
        logger.info(f"  Average top-5 score: {result['avg_top_5_score']:.4f}")
        logger.info(f"  Top document indices: {result['top_5_indices']}")

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK 3: Optimal Configuration Selection")
    logger.info("=" * 80)

    # Test optimal config selection
    use_cases = ["high_quality", "balanced", "fast"]
    gpu_memories = [8, 16, 24]

    logger.info("\nOptimal configurations for different scenarios:")
    logger.info("-" * 80)

    for use_case in use_cases:
        for gpu_mem in gpu_memories:
            config = get_optimal_config(use_case, gpu_mem)
            logger.info(
                f"Use case: {use_case:<12} GPU: {gpu_mem}GB -> "
                f"Model: {config['model'].split('/')[-1]:<20} "
                f"Quant: {config['quantization']:<8} "
                f"Batch: {config['batch_config']['batch_size']}"
            )


def main():
    """Main entry point"""
    logger.info("Starting Qwen3 Search Optimization Tests")

    try:
        run_benchmarks()
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

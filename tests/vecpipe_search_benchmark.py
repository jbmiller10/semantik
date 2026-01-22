#!/usr/bin/env python
"""Standalone benchmark harness for VecPipe search API.

This is a CLI script (not pytest) for measuring VecPipe search performance.

Usage:
    uv run python tests/vecpipe_search_benchmark.py \
        --base-url http://localhost:8000 \
        --collection my_collection \
        --mode hybrid \
        --rerank \
        --requests 200 \
        --concurrency 20
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

DEFAULT_QUERIES = [
    "What is semantic search?",
    "How does vector embedding work?",
    "Explain document chunking strategies",
    "What are the benefits of hybrid search?",
    "How to optimize search performance?",
    "What is reciprocal rank fusion?",
    "Explain dense vs sparse retrieval",
    "How does reranking improve results?",
    "What is a cross-encoder model?",
    "Explain embedding model quantization",
]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark request."""

    latency_ms: float
    success: bool
    error: str | None = None
    result_count: int = 0


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark statistics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies_ms: list[float]
    errors: list[str]
    total_duration_s: float

    @property
    def error_rate(self) -> float:
        """Error rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    @property
    def throughput(self) -> float:
        """Requests per second."""
        if self.total_duration_s == 0:
            return 0.0
        return self.successful_requests / self.total_duration_s

    @property
    def p50_ms(self) -> float:
        """50th percentile latency in ms."""
        if not self.latencies_ms:
            return 0.0
        return statistics.quantiles(self.latencies_ms, n=100)[49]

    @property
    def p95_ms(self) -> float:
        """95th percentile latency in ms."""
        if not self.latencies_ms:
            return 0.0
        return statistics.quantiles(self.latencies_ms, n=100)[94]

    @property
    def p99_ms(self) -> float:
        """99th percentile latency in ms."""
        if not self.latencies_ms:
            return 0.0
        return statistics.quantiles(self.latencies_ms, n=100)[98]

    @property
    def mean_ms(self) -> float:
        """Mean latency in ms."""
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def stddev_ms(self) -> float:
        """Standard deviation of latency in ms."""
        if len(self.latencies_ms) < 2:
            return 0.0
        return statistics.stdev(self.latencies_ms)


async def make_search_request(
    client: httpx.AsyncClient,
    base_url: str,
    collection: str,
    query: str,
    mode: Literal["dense", "sparse", "hybrid"],
    rerank: bool,
    k: int,
    api_key: str | None,
) -> BenchmarkResult:
    """Make a single search request and measure latency."""
    headers = {}
    if api_key:
        headers["X-Internal-Api-Key"] = api_key

    payload = {
        "query": query,
        "collection": collection,
        "k": k,
        "search_mode": mode,
        "use_reranker": rerank,
    }

    start = time.perf_counter()
    try:
        response = await client.post(
            f"{base_url}/search",
            json=payload,
            headers=headers,
            timeout=60.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            result_count = len(data.get("results", []))
            return BenchmarkResult(
                latency_ms=latency_ms,
                success=True,
                result_count=result_count,
            )
        return BenchmarkResult(
            latency_ms=latency_ms,
            success=False,
            error=f"HTTP {response.status_code}: {response.text[:200]}",
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )


async def run_benchmark(
    base_url: str,
    collection: str,
    mode: Literal["dense", "sparse", "hybrid"],
    rerank: bool,
    concurrency: int,
    num_requests: int,
    k: int,
    api_key: str | None,
    queries: list[str],
) -> BenchmarkSummary:
    """Run the benchmark with specified parameters."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[BenchmarkResult] = []

    async def bounded_request(query: str) -> BenchmarkResult:
        async with semaphore:
            async with httpx.AsyncClient() as client:
                return await make_search_request(
                    client=client,
                    base_url=base_url,
                    collection=collection,
                    query=query,
                    mode=mode,
                    rerank=rerank,
                    k=k,
                    api_key=api_key,
                )

    # Generate queries for the specified number of requests
    query_list = [queries[i % len(queries)] for i in range(num_requests)]

    start_time = time.perf_counter()
    tasks = [bounded_request(q) for q in query_list]
    results = await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - start_time

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    return BenchmarkSummary(
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        latencies_ms=[r.latency_ms for r in successful],
        errors=[r.error for r in failed if r.error],
        total_duration_s=total_duration,
    )


def print_summary(summary: BenchmarkSummary, as_json: bool = False) -> None:
    """Print benchmark summary."""
    if as_json:
        output = {
            "total_requests": summary.total_requests,
            "successful_requests": summary.successful_requests,
            "failed_requests": summary.failed_requests,
            "error_rate_percent": round(summary.error_rate, 2),
            "throughput_rps": round(summary.throughput, 2),
            "latency_ms": {
                "p50": round(summary.p50_ms, 2),
                "p95": round(summary.p95_ms, 2),
                "p99": round(summary.p99_ms, 2),
                "mean": round(summary.mean_ms, 2),
                "stddev": round(summary.stddev_ms, 2),
            },
            "total_duration_s": round(summary.total_duration_s, 2),
        }
        if summary.errors:
            output["sample_errors"] = summary.errors[:5]
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "=" * 60)
        print("VecPipe Search Benchmark Results")
        print("=" * 60)
        print(f"\nRequests:    {summary.successful_requests}/{summary.total_requests} successful")
        print(f"Error rate:  {summary.error_rate:.1f}%")
        print(f"Duration:    {summary.total_duration_s:.2f}s")
        print(f"Throughput:  {summary.throughput:.2f} req/s")
        print("\nLatency (ms):")
        print(f"  p50:    {summary.p50_ms:.2f}")
        print(f"  p95:    {summary.p95_ms:.2f}")
        print(f"  p99:    {summary.p99_ms:.2f}")
        print(f"  mean:   {summary.mean_ms:.2f}")
        print(f"  stddev: {summary.stddev_ms:.2f}")

        if summary.errors:
            print(f"\nSample errors ({len(summary.errors)} total):")
            for err in summary.errors[:3]:
                print(f"  - {err[:100]}")

        print("=" * 60 + "\n")


def load_queries_from_file(filepath: str) -> list[str]:
    """Load queries from a file (one per line or JSON array)."""
    with Path(filepath).open() as f:
        content = f.read().strip()

    # Try JSON array first
    try:
        queries = json.loads(content)
        if isinstance(queries, list):
            return [str(q) for q in queries if q]
    except json.JSONDecodeError:
        pass

    # Fall back to line-by-line
    return [line.strip() for line in content.split("\n") if line.strip()]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark VecPipe search API performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="VecPipe API base URL",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Collection name to search",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid"],
        default="dense",
        help="Search mode",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable reranking",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests to make",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to request",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("X_INTERNAL_API_KEY"),
        help="API key (or set X_INTERNAL_API_KEY env var)",
    )
    parser.add_argument(
        "--queries-file",
        help="Path to file with queries (one per line or JSON array)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Load queries
    if args.queries_file:
        try:
            queries = load_queries_from_file(args.queries_file)
            if not queries:
                print(f"Error: No queries found in {args.queries_file}", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"Error loading queries file: {e}", file=sys.stderr)
            return 1
    else:
        queries = DEFAULT_QUERIES

    if not args.json:
        print(f"Running benchmark against {args.base_url}")
        print(f"Collection: {args.collection}")
        print(f"Mode: {args.mode}, Rerank: {args.rerank}")
        print(f"Requests: {args.requests}, Concurrency: {args.concurrency}, k: {args.k}")
        print(f"Queries: {len(queries)} unique")

    # Run benchmark
    summary = asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            collection=args.collection,
            mode=args.mode,
            rerank=args.rerank,
            concurrency=args.concurrency,
            num_requests=args.requests,
            k=args.k,
            api_key=args.api_key,
            queries=queries,
        )
    )

    print_summary(summary, as_json=args.json)

    # Exit with error if all requests failed
    if summary.successful_requests == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

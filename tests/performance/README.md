# Chunking Performance Testing Framework

This directory contains a comprehensive performance testing framework for the Semantik chunking strategies. The framework provides tools for benchmarking, monitoring, baseline tracking, and regression detection.

## Overview

The performance testing framework consists of:

1. **test_chunking_performance.py** - Main performance test suite using pytest-benchmark
2. **chunking_benchmarks.py** - Enhanced benchmarking utilities with continuous monitoring
3. **performance_utils.py** - Utilities for baseline tracking, reporting, and analysis
4. **generate_baseline.py** - Script to generate performance baselines

## Performance Targets

Based on our token-based chunking implementations, the current performance targets are:

| Strategy   | Target (chunks/sec) | Memory (MB/MB) | Description |
|------------|-------------------|----------------|-------------|
| Character  | 100               | 50             | Token-based splitting with overlap |
| Recursive  | 80                | 60             | Sentence-aware splitting |
| Markdown   | 60                | 80             | Structure-aware parsing |

## Running Performance Tests

### Prerequisites

```bash
# Install development dependencies
poetry install --with dev

# Ensure testing environment is set
export TESTING=true
```

### Run All Performance Tests

```bash
# Run all performance tests
poetry run pytest tests/performance/test_chunking_performance.py --benchmark-only

# Run specific test group
poetry run pytest tests/performance/test_chunking_performance.py --benchmark-only --benchmark-group-by=group

# Run tests for specific strategy
poetry run pytest tests/performance/test_chunking_performance.py -k "character" --benchmark-only
```

### Generate HTML Report

```bash
# Generate detailed HTML report
poetry run pytest tests/performance/test_chunking_performance.py --benchmark-only --benchmark-autosave --benchmark-html=performance_report.html
```

## Performance Monitoring

The enhanced `PerformanceMonitor` class provides continuous monitoring with:

- **Memory sampling**: Tracks memory usage throughout the test
- **CPU monitoring**: Records CPU utilization
- **Timeline data**: Captures performance metrics over time

Example usage:

```python
from tests.performance.chunking_benchmarks import PerformanceMonitor

monitor = PerformanceMonitor(sample_interval=0.05)  # 50ms sampling
monitor.start()

# Run your chunking operation
chunks = await chunker.chunk_text_async(document, "test_doc")

metrics = monitor.stop()
print(f"Peak memory: {metrics['peak_memory_mb']:.1f} MB")
print(f"Average CPU: {metrics['cpu_stats']['avg']:.1f}%")
```

## Baseline Management

### Generate Baseline

```bash
# Generate performance baseline for all strategies
poetry run python tests/performance/generate_baseline.py
```

This creates `baseline/baseline_results.json` with current performance metrics.

### Check for Regressions

```python
from tests.performance.performance_utils import PerformanceBaselineManager

manager = PerformanceBaselineManager()
has_regression, message = manager.check_regression(
    strategy="character",
    current_performance=145.5,
    metric="chunks_per_second",
    threshold=0.1  # 10% tolerance
)
```

## Test Categories

### 1. Single-threaded Performance
Tests the raw performance of each chunking strategy:
- Synchronous chunking
- Asynchronous chunking
- Warmup rounds for accurate measurements

### 2. Memory Usage
Tests memory consumption with different document sizes:
- Peak memory tracking
- Memory per MB of input
- Continuous memory monitoring

### 3. Parallel Scalability
Tests how well chunking scales with multiple workers:
- 1, 2, 4, and 8 worker configurations
- Efficiency calculations
- Scalability metrics

### 4. Edge Cases
Tests performance with extreme inputs:
- Empty documents
- Tiny documents (10 bytes)
- Huge documents (100MB)

### 5. Memory Profiling
Detailed memory analysis using memory_profiler:
- Line-by-line memory usage
- Memory leak detection
- Resource cleanup verification

## Performance Utilities

### Document Generation

```python
from tests.performance.performance_utils import DocumentGenerator

# Generate realistic test documents
doc = DocumentGenerator.generate_realistic_document(
    size_bytes=1024*1024,  # 1MB
    doc_type="mixed",      # mixed, technical, narrative, code
    seed=42               # For reproducibility
)
```

### Memory Leak Detection

```python
from tests.performance.performance_utils import MemoryLeakDetector

detector = MemoryLeakDetector(threshold_mb=10.0)
has_leak, metrics = detector.check_memory_leak(
    run_function=lambda: chunker.chunk_text(doc, "test"),
    iterations=5,
    warmup=1
)
```

### Performance Reporting

```python
from tests.performance.performance_utils import PerformanceReporter

reporter = PerformanceReporter()
report_path = reporter.generate_report(
    benchmark_results,
    output_format="markdown"  # or "json"
)
```

## CI/CD Integration

To integrate performance tests in CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run Performance Tests
  run: |
    poetry run pytest tests/performance/test_chunking_performance.py \
      --benchmark-only \
      --benchmark-json=performance.json
    
- name: Check Performance Regression
  run: |
    poetry run python tests/performance/check_regression.py \
      --results performance.json \
      --threshold 0.15  # 15% tolerance
```

## Interpreting Results

### Benchmark Output

```
----- benchmark 'single-thread': 3 tests -----
Name                              Min      Mean     StdDev   Rounds
test_performance[character]    1.68s    1.72s    0.024s        5
test_performance[recursive]    0.38s    0.41s    0.051s        5
test_performance[markdown]     4.30s    4.35s    0.043s        5
```

### Key Metrics

1. **Chunks per second**: Primary performance metric
2. **Memory per MB**: Memory efficiency (MB used per MB of input)
3. **CPU utilization**: Average and peak CPU usage
4. **Scalability efficiency**: Parallel performance vs single-threaded

## Troubleshooting

### Common Issues

1. **Low performance numbers**
   - Ensure `TESTING=true` environment variable is set
   - Check if mock embeddings are being used
   - Verify no other processes are consuming resources

2. **Memory test failures**
   - Token-based chunkers require more memory than simple splitting
   - Adjust memory limits based on your implementation
   - Consider streaming for very large documents

3. **Inconsistent results**
   - Use `--benchmark-warmup` for stable results
   - Increase rounds with `--benchmark-min-rounds=10`
   - Ensure consistent system load during testing

## Future Enhancements

1. **Advanced Strategies**: Add tests for semantic, hierarchical, and hybrid chunkers
2. **GPU Monitoring**: Track GPU usage for embedding-based strategies
3. **Distributed Testing**: Test performance across multiple machines
4. **Real-world Documents**: Benchmark with actual user documents
5. **Streaming Performance**: Test streaming chunking for large files

## Contributing

When adding new performance tests:

1. Follow the existing test structure
2. Include appropriate performance targets
3. Add edge case coverage
4. Update baseline after significant changes
5. Document any new utilities or patterns

For questions or issues, please refer to the main project documentation or create an issue in the repository.
#!/usr/bin/env python3
"""
Performance testing utilities for chunking benchmarks.

This module provides utilities for baseline tracking, performance reporting,
regression detection, and test data generation.
"""

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline data for a specific strategy."""

    strategy: str
    chunks_per_second: float
    memory_per_mb: float
    cpu_avg_percent: float
    test_date: str
    git_commit: str | None = None
    hardware_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceBaseline":
        """Create from dictionary."""
        return cls(**data)


class PerformanceBaselineManager:
    """Manage performance baselines for regression detection."""

    def __init__(self, baseline_dir: Path | None = None) -> None:
        """Initialize baseline manager.

        Args:
            baseline_dir: Directory to store baseline files
        """
        self.baseline_dir = baseline_dir or Path("tests/performance/baseline")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_file = self.baseline_dir / "baseline_results.json"

    def save_baseline(self, baselines: list[PerformanceBaseline]) -> None:
        """Save performance baselines to file.

        Args:
            baselines: List of performance baselines to save
        """
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "baselines": [b.to_dict() for b in baselines],
        }

        with self.baseline_file.open("w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(baselines)} baselines to {self.baseline_file}")

    def load_baseline(self) -> dict[str, PerformanceBaseline]:
        """Load performance baselines from file.

        Returns:
            Dictionary mapping strategy name to baseline
        """
        if not self.baseline_file.exists():
            logger.warning(f"No baseline file found at {self.baseline_file}")
            return {}

        try:
            with self.baseline_file.open() as f:
                data = json.load(f)

            baselines = {}
            for b_data in data.get("baselines", []):
                baseline = PerformanceBaseline.from_dict(b_data)
                baselines[baseline.strategy] = baseline

            logger.info(f"Loaded {len(baselines)} baselines from {self.baseline_file}")
            return baselines

        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")
            return {}

    def check_regression(
        self, strategy: str, current_performance: float, metric: str = "chunks_per_second", threshold: float = 0.1
    ) -> tuple[bool, str]:
        """Check if current performance shows regression.

        Args:
            strategy: Strategy name
            current_performance: Current performance value
            metric: Metric to check (chunks_per_second, memory_per_mb)
            threshold: Regression threshold (0.1 = 10% tolerance)

        Returns:
            Tuple of (has_regression, message)
        """
        baselines = self.load_baseline()

        if strategy not in baselines:
            return False, f"No baseline found for {strategy}"

        baseline = baselines[strategy]
        baseline_value = getattr(baseline, metric)

        # For memory, lower is better
        if metric == "memory_per_mb":
            regression = current_performance > baseline_value * (1 + threshold)
            change_pct = ((current_performance - baseline_value) / baseline_value) * 100
        else:
            # For other metrics, higher is better
            regression = current_performance < baseline_value * (1 - threshold)
            change_pct = ((baseline_value - current_performance) / baseline_value) * 100

        if regression:
            message = (
                f"Performance regression detected for {strategy}:\n"
                f"  Baseline {metric}: {baseline_value:.2f}\n"
                f"  Current {metric}: {current_performance:.2f}\n"
                f"  Degradation: {abs(change_pct):.1f}%"
            )
        else:
            message = (
                f"Performance acceptable for {strategy}:\n"
                f"  Baseline {metric}: {baseline_value:.2f}\n"
                f"  Current {metric}: {current_performance:.2f}\n"
                f"  Change: {change_pct:+.1f}%"
            )

        return regression, message


class PerformanceReporter:
    """Generate performance reports from benchmark results."""

    def __init__(self, results_dir: Path | None = None) -> None:
        """Initialize reporter.

        Args:
            results_dir: Directory to store reports
        """
        self.results_dir = results_dir or Path("tests/performance/reports")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, benchmark_results: list[dict[str, Any]], output_format: str = "markdown") -> Path:
        """Generate performance report from benchmark results.

        Args:
            benchmark_results: List of benchmark result dictionaries
            output_format: Report format (markdown, json)

        Returns:
            Path to generated report
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        if output_format == "markdown":
            report_path = self.results_dir / f"performance_report_{timestamp}.md"
            content = self._generate_markdown_report(benchmark_results)
        else:
            report_path = self.results_dir / f"performance_report_{timestamp}.json"
            content = json.dumps(benchmark_results, indent=2)

        with report_path.open("w") as f:
            f.write(content)

        logger.info(f"Generated performance report: {report_path}")
        return report_path

    def _generate_markdown_report(self, results: list[dict[str, Any]]) -> str:
        """Generate markdown-formatted report.

        Args:
            results: Benchmark results

        Returns:
            Markdown report content
        """
        report = ["# Chunking Performance Report", ""]
        report.append(f"Generated: {datetime.now(UTC).isoformat()}")
        report.append("")

        # Group results by test group
        groups: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            group = result.get("group", "ungrouped")
            if group not in groups:
                groups[group] = []
            groups[group].append(result)

        # Generate sections for each group
        for group, group_results in sorted(groups.items()):
            report.append(f"## {group.replace('-', ' ').title()}")
            report.append("")

            # Create table
            if group_results:
                # Get all unique keys for columns
                all_keys = set()
                for r in group_results:
                    all_keys.update(r.get("extra_info", {}).keys())

                # Table header
                report.append("| Test | Mean Time | Std Dev | " + " | ".join(sorted(all_keys)) + " |")
                report.append("|" + "---|" * (3 + len(all_keys)))

                # Table rows
                for result in sorted(group_results, key=lambda x: x.get("name", "")):
                    name = result.get("name", "unknown")
                    stats = result.get("stats", {})
                    mean = stats.get("mean", 0) * 1000  # Convert to ms
                    stddev = stats.get("stddev", 0) * 1000
                    extra = result.get("extra_info", {})

                    row = [
                        name,
                        f"{mean:.2f}ms",
                        f"{stddev:.2f}ms",
                    ]

                    for key in sorted(all_keys):
                        value = extra.get(key, "N/A")
                        if isinstance(value, float):
                            row.append(f"{value:.2f}")
                        else:
                            row.append(str(value))

                    report.append("| " + " | ".join(row) + " |")

            report.append("")

        # Add summary statistics
        report.extend(self._generate_summary_stats(results))

        return "\n".join(report)

    def _generate_summary_stats(self, results: list[dict[str, Any]]) -> list[str]:
        """Generate summary statistics section.

        Args:
            results: Benchmark results

        Returns:
            List of report lines
        """
        summary = ["## Summary Statistics", ""]

        # Extract performance metrics by strategy
        strategy_metrics: dict[str, dict[str, list[float]]] = {}

        for result in results:
            extra = result.get("extra_info", {})
            if "chunks_per_second" in extra:
                # Extract strategy from test name
                name = result.get("name", "")
                for strategy in ["character", "recursive", "markdown"]:
                    if strategy in name:
                        if strategy not in strategy_metrics:
                            strategy_metrics[strategy] = {
                                "chunks_per_second": [],
                                "memory_used_mb": [],
                                "cpu_percent": [],
                            }

                        metrics = strategy_metrics[strategy]
                        metrics["chunks_per_second"].append(extra["chunks_per_second"])

                        if "memory_used_mb" in extra:
                            metrics["memory_used_mb"].append(extra["memory_used_mb"])

                        if "avg_cpu_percent" in extra:
                            metrics["cpu_percent"].append(extra["avg_cpu_percent"])

                        break

        # Generate summary table
        if strategy_metrics:
            summary.append("| Strategy | Avg Chunks/sec | Avg Memory (MB) | Avg CPU % |")
            summary.append("|----------|----------------|-----------------|-----------|")

            for strategy, metrics in sorted(strategy_metrics.items()):
                chunks_per_sec = statistics.mean(metrics["chunks_per_second"]) if metrics["chunks_per_second"] else 0
                memory_mb = statistics.mean(metrics["memory_used_mb"]) if metrics["memory_used_mb"] else 0
                cpu_pct = statistics.mean(metrics["cpu_percent"]) if metrics["cpu_percent"] else 0

                summary.append(f"| {strategy} | {chunks_per_sec:.1f} | {memory_mb:.1f} | {cpu_pct:.1f}% |")

        return summary


class MemoryLeakDetector:
    """Detect potential memory leaks in chunking operations."""

    def __init__(self, threshold_mb: float = 10.0) -> None:
        """Initialize leak detector.

        Args:
            threshold_mb: Memory growth threshold to consider as leak
        """
        self.threshold_mb = threshold_mb
        self.baseline_memory: float | None = None

    def check_memory_leak(self, run_function: Any, iterations: int = 5, warmup: int = 1) -> tuple[bool, dict[str, Any]]:
        """Check for memory leaks by running function multiple times.

        Args:
            run_function: Function to test for leaks
            iterations: Number of iterations to run
            warmup: Number of warmup iterations

        Returns:
            Tuple of (has_leak, metrics)
        """
        import gc

        import psutil

        process = psutil.Process()
        memory_readings = []

        # Warmup runs
        for _ in range(warmup):
            run_function()
            gc.collect()

        # Baseline memory
        gc.collect()
        self.baseline_memory = process.memory_info().rss / (1024 * 1024)

        # Test runs
        for _ in range(iterations):
            run_function()

            # Force garbage collection
            gc.collect()

            # Measure memory
            current_memory = process.memory_info().rss / (1024 * 1024)
            memory_readings.append(current_memory)

        # Analyze results
        memory_growth = memory_readings[-1] - self.baseline_memory
        avg_growth_per_iteration = memory_growth / iterations

        # Check for consistent growth pattern
        is_growing = all(
            memory_readings[i] <= memory_readings[i + 1] + 1.0  # Allow 1MB fluctuation
            for i in range(len(memory_readings) - 1)
        )

        has_leak = memory_growth > self.threshold_mb and is_growing and avg_growth_per_iteration > 1.0

        metrics = {
            "baseline_memory_mb": self.baseline_memory,
            "final_memory_mb": memory_readings[-1],
            "total_growth_mb": memory_growth,
            "avg_growth_per_iteration_mb": avg_growth_per_iteration,
            "is_consistently_growing": is_growing,
            "memory_readings": memory_readings,
        }

        return has_leak, metrics


class DocumentGenerator:
    """Enhanced document generator for performance testing."""

    @staticmethod
    def generate_realistic_document(size_bytes: int, doc_type: str = "mixed", seed: int | None = None) -> str:
        """Generate realistic test documents with various characteristics.

        Args:
            size_bytes: Target document size in bytes
            doc_type: Type of document (mixed, technical, narrative, code)
            seed: Random seed for reproducibility

        Returns:
            Generated document text
        """
        import random

        if seed is not None:
            random.seed(seed)

        generator = DocumentGenerator()
        if doc_type == "technical":
            return generator._generate_technical_content(size_bytes)
        if doc_type == "narrative":
            return generator._generate_narrative_content(size_bytes)
        if doc_type == "code":
            return generator._generate_code_content(size_bytes)
        # mixed or default
        return generator._generate_mixed_content(size_bytes)

    def _generate_mixed_content(self, size_bytes: int) -> str:
        """Generate mixed content document."""
        import random

        content = []
        current_size = 0

        section_types = ["technical", "narrative", "list", "code"]

        while current_size < size_bytes:
            section_type = random.choice(section_types)

            if section_type == "technical":
                section = self._generate_technical_section()
            elif section_type == "narrative":
                section = self._generate_narrative_section()
            elif section_type == "list":
                section = self._generate_list_section()
            else:
                section = self._generate_code_section()

            content.append(section)
            current_size += len(section)

        return "\n\n".join(content)[:size_bytes]

    def _generate_technical_section(self) -> str:
        """Generate a technical documentation section."""
        import random

        topics = [
            "API Documentation",
            "System Architecture",
            "Performance Optimization",
            "Security Guidelines",
            "Database Schema",
        ]

        topic = random.choice(topics)

        section = f"## {topic}\n\n"

        # Add paragraphs
        for _ in range(random.randint(2, 4)):
            sentences = []
            for _ in range(random.randint(3, 6)):
                words = [
                    "system",
                    "performance",
                    "architecture",
                    "implementation",
                    "optimize",
                    "configure",
                    "deploy",
                    "integrate",
                    "scale",
                    "monitor",
                    "secure",
                    "authenticate",
                    "validate",
                    "process",
                ]
                sentence = " ".join(random.choices(words, k=random.randint(8, 15)))
                sentences.append(sentence.capitalize() + ".")

            section += " ".join(sentences) + "\n\n"

        return section

    def _generate_narrative_section(self) -> str:
        """Generate narrative text section."""
        import random

        # Simple narrative generator
        subjects = ["The system", "The application", "The service", "The platform"]
        verbs = ["processes", "handles", "manages", "coordinates", "executes"]
        objects = ["requests", "data", "transactions", "operations", "tasks"]

        section = ""
        for _ in range(random.randint(3, 5)):
            sentence = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}. "
            section += sentence

        return section + "\n"

    def _generate_list_section(self) -> str:
        """Generate list section."""
        import random

        list_types = ["Features:", "Requirements:", "Components:", "Benefits:"]
        list_type = random.choice(list_types)

        section = f"### {list_type}\n\n"

        items = [
            "High performance processing",
            "Scalable architecture",
            "Secure authentication",
            "Real-time monitoring",
            "Automated deployment",
            "Comprehensive logging",
            "Error handling",
            "Data validation",
        ]

        for item in random.sample(items, k=random.randint(3, 6)):
            section += f"- {item}\n"

        return section + "\n"

    def _generate_code_section(self) -> str:
        """Generate code block section."""
        import random

        languages = ["python", "javascript", "java", "go"]
        language = random.choice(languages)

        section = f"```{language}\n"

        # Simple code patterns
        if language == "python":
            section += "def process_data(input_data):\n"
            section += "    result = []\n"
            section += "    for item in input_data:\n"
            section += "        if validate(item):\n"
            section += "            result.append(transform(item))\n"
            section += "    return result\n"
        else:
            section += "function processData(inputData) {\n"
            section += "    const result = [];\n"
            section += "    for (const item of inputData) {\n"
            section += "        if (validate(item)) {\n"
            section += "            result.push(transform(item));\n"
            section += "        }\n"
            section += "    }\n"
            section += "    return result;\n"
            section += "}\n"

        section += "```\n"

        return section

    def _generate_technical_content(self, size_bytes: int) -> str:
        """Generate technical content until size is reached."""
        content = []
        current_size = 0

        while current_size < size_bytes:
            section = self._generate_technical_section()
            content.append(section)
            current_size += len(section)

        return "\n\n".join(content)[:size_bytes]

    def _generate_narrative_content(self, size_bytes: int) -> str:
        """Generate narrative content until size is reached."""
        content = []
        current_size = 0

        while current_size < size_bytes:
            section = self._generate_narrative_section()
            content.append(section)
            current_size += len(section)

        return "\n\n".join(content)[:size_bytes]

    def _generate_code_content(self, size_bytes: int) -> str:
        """Generate code content until size is reached."""
        content = []
        current_size = 0

        while current_size < size_bytes:
            section = self._generate_code_section()
            content.append(section)
            current_size += len(section)

        return "\n\n".join(content)[:size_bytes]


def get_git_commit_hash() -> str | None:
    """Get current git commit hash."""
    try:
        import subprocess

        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()[:8]
    except Exception:
        return None


def get_hardware_info() -> dict[str, Any]:
    """Get hardware information for baseline tracking."""
    import platform

    import psutil

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "python_version": platform.python_version(),
    }

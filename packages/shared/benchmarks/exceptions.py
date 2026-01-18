"""Benchmark-specific exceptions.

Exception hierarchy follows the pattern from shared.database.exceptions.
"""


class BenchmarkError(Exception):
    """Base exception for all benchmark errors."""


class BenchmarkMetricError(BenchmarkError):
    """Error during metric calculation.

    Raised when a metric calculation fails due to invalid inputs
    or mathematical issues (e.g., division by zero).
    """

    def __init__(self, metric_name: str, message: str) -> None:
        self.metric_name = metric_name
        super().__init__(f"Error calculating {metric_name}: {message}")


class BenchmarkEvaluationError(BenchmarkError):
    """Error during benchmark evaluation process.

    Raised when the evaluation process fails, such as when
    search operations fail or results cannot be processed.
    """

    def __init__(self, message: str, query_id: int | None = None) -> None:
        self.query_id = query_id
        if query_id is not None:
            super().__init__(f"Evaluation failed for query {query_id}: {message}")
        else:
            super().__init__(f"Evaluation failed: {message}")


class BenchmarkValidationError(BenchmarkError):
    """Error validating benchmark inputs.

    Raised when input validation fails, such as invalid k values,
    missing ground truth, or malformed relevance judgments.
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"Validation error: {message}")


__all__ = [
    "BenchmarkError",
    "BenchmarkMetricError",
    "BenchmarkEvaluationError",
    "BenchmarkValidationError",
]

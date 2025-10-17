# Manual Test Harnesses

These scripts provide ad-hoc diagnostics and performance probes that are **not** meant to run as part of the automated pytest suite. Execute them manually when you need to:

- validate end-to-end streaming or websocket behaviour against a running Semantik stack;
- capture performance numbers for chunking or embedding pipelines;
- exercise maintenance flows that depend on live infrastructure.

## Running A Script

1. Start the required services locally (e.g. `make dev` or `make docker-up`).
2. Activate the same virtualenv used for development (`uv` environment or `source .venv/bin/activate`).
3. Invoke the script directly, for example:
   ```bash
   uv run python manual_tests/validate_streaming_pipeline.py --help
   ```
4. Review the script-specific flags/output before sharing results.

## Available Scripts

- `chunking_benchmarks.py` – chunking throughput benchmark harness.
- `embedding_full_integration_suite.py` – exercises the full embedding workflow.
- `embedding_performance_bench.py` – micro-benchmark for embedding calls.
- `frontend_api_test_suite.py` – legacy API smoke checks driven from the frontend harness.
- `metrics_*_probe.py` – HTTP probes for metric pipelines.
- `search_probe.py` – manual search smoke test.
- `validate_streaming_pipeline.py` – streaming ingestion validation.
- `websocket_cleanup.py` / `websocket_scaling.py` – manual websocket maintenance and load scripts.
- `websocket_performance_probe.py` / `websocket_reindex_probe.py` – high-load websocket diagnostics.

Because `manual_tests/` is listed in `norecursedirs`, pytest will ignore these files automatically.

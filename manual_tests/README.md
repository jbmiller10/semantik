# Manual Test Harnesses

This directory contains exploratory or resource-intensive scripts that are intentionally excluded from automated pytest discovery. Each script is meant for manual validation while running the Semantik stack locally or inside a controlled staging environment.

## Layout

- `embedding_full_integration_suite.py` – drives end-to-end embedding flows against a running vecpipe instance.
- `embedding_performance_bench.py` – measures embedding latency/throughput; requires GPUs or tuned CPU settings.
- `metrics_flow_probe.py`, `metrics_update_probe.py`, `metrics_flow_probe.py` – ad-hoc Prometheus instrumentation probes for chunking/ingestion flows.
- `search_probe.py` – manual search smoke script for troubleshooting vecpipe/search deployments.
- `frontend_api_test_suite.py` – quick manual regression harness for the web UI REST surface.
- `performance/` – performance and benchmark utilities (for example `chunking_benchmarks.py`).
- `test_embedding_oom_handling.py` – stress test for OOM handling scenarios; runs outside pytest to avoid bringing down CI workers.

## Usage Guidelines

1. Ensure `make dev-install` and any service-specific dependencies are installed.
2. Start the stack via `make dev` or `make docker-dev-up` so services and Postgres are available.
3. Run scripts with `uv run python <script>` or the interpreter of your choice; most scripts respect environment variables defined in `.env`/`.env.test`.
4. Never commit changes to these scripts without documenting the intent here; keep them deterministic and side-effect free where possible.
5. Because these utilities are excluded via `pyproject.toml:norecursedirs`, move new manual probes here rather than under `tests/`.

If additional documentation or context is needed for a script, add a short section below describing prerequisites and expected output.

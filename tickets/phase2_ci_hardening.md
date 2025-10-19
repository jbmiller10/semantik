# Phase 2 Ticket: Reinstate CI Quality Gates (Target Window: November 10 – November 21, 2025)

## Background
With security and core features restored, Phase 2 must re-enable Semantik’s “fail-fast” culture. The current GitHub workflow (`.github/workflows/main.yml`) allows `mypy` and `safety` to fail silently, eroding confidence. This ticket focuses solely on CI hardening; UI polish lives in a companion ticket.

## Objectives
1. Update CI pipelines to fail on typing or dependency vulnerabilities.
2. Fix outstanding mypy/type-check errors.
3. Ensure `uv run safety check` passes or documented ignores exist.

## Requirements
### 1. Workflow Updates
- Modify `.github/workflows/main.yml` and any related workflow files to remove `continue-on-error` or conditional logic that skips failure for `mypy`, `ruff`, and `safety` steps.
- Confirm workflow uses the project’s standard commands (`make lint`, `make type-check`, `uv run safety check`) or update to explicit commands per step.

### 2. Type Checking Cleanup
- Run `uv run mypy packages apps` locally; capture all failing modules.
- Resolve type issues, focusing on:
  - New models/services introduced in Phase 1 (collection sharing, resource manager, Qdrant metrics).
  - Existing TODO comments in `packages/webui/services/chunking_error_handler.py`.
  - Third-party stubs; if missing, add `py.typed` markers or minimal `.pyi` files in `stubs/`.
- Ensure incremental re-checks stay fast; consider enabling `mypy.ini` caches.

### 3. Safety Scan Compliance
- Run `uv run safety check` and resolve flagged vulnerabilities:
  - Upgrade dependencies within compatibility boundaries.
  - If an upgrade is impossible, record justification in `safety-policy.toml` with a target removal date.
- Update `pyproject.toml`/`uv.lock` as needed; rerun `uv sync`.

### 4. Documentation & Communication
- Update developer docs (e.g., `docs/quality-assurance.md` or README “Development” section) with the new expectation: developers must run `make format && make lint && make type-check && uv run safety check` before PRs.
- Communicate in `PORTFOLIO_CLEANUP_SPRINT.md` or `NEXTSTEP.md` to note the restored gates.

## Acceptance Criteria
- CI fails whenever mypy or safety detects issues; manual GitHub run shows green after fixes.
- Local `uv run mypy packages apps` returns success with no suppressed warnings beyond intentional ones documented in config.
- `uv run safety check` passes or uses documented ignore entries referencing CVEs/RHAs.
- Documentation outlines the pre-PR checklist including safety/mypy.

## Validation Steps
1. `uv run mypy packages apps`
2. `uv run safety check`
3. `make lint` and `make type-check` (ensures commands still pass)
4. Trigger GitHub workflow (push to branch) and confirm CI status turns red on intentional failures and green after fixes.

## Coordination Notes
- Work closely with the frontend polish ticket to avoid conflicting documentation edits.
- Coordinate with DevOps if dependency upgrades require Docker base image updates.
- Alert the team about new pre-merge expectations via Slack/Docs.

## Out of Scope
- Frontend UX adjustments or pdf.js improvements (handled in companion Phase 2 ticket).
- Performance optimization of CI runtimes beyond necessary caching for mypy.
- Writing new tests besides what’s required to satisfy type/safety fixes.
- Introducing new static analysis tools not already referenced in the project.

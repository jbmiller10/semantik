Title: Clean tracked artifacts and harden ignores

Background
- Tracked noise remains: `__pycache__/` dirs, empty placeholder files (`apps/vite`, `apps/webui-react@0.0.0`, `apps/document-embedding-project@1.0.0`), built artifacts in `package/dist`, and large `node_modules` present in repo.
- Noise inflates diffs and CI time.

Goal
Remove existing artifacts, prevent their return, and optionally codify a hygiene check.

Scope
- Delete tracked caches/placeholders/build artifacts; verify nothing depends on them.
- Update `.gitignore` to cover `__pycache__/`, `package/dist/`, root `node_modules/`, and placeholder files.
- Ensure publishable package builds are generated at release time, not committed.
- Optional: add a lightweight hygiene script/CI check to fail on tracked caches/artifacts.

Out of Scope
- Dependency upgrades or build pipeline redesign.

Suggested Steps
1) Remove tracked noise; confirm clean `git status`.
2) Update `.gitignore` for caches/artifacts/placeholders; keep required assets whitelisted.
3) Verify package build workflow (e.g., `package/`) still works from source without committed dist.
4) Add optional CI/hook script to detect caches/artifacts if chosen.

Acceptance Criteria
- No tracked caches/placeholders/build artifacts remain; `git status` clean after removal.
- `.gitignore` prevents reintroduction of the above paths.
- Package builds succeed from clean repo; optional hygiene check documented or implemented.

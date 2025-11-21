from fastapi import FastAPI
from fastapi.testclient import TestClient
from webui.middleware.csp import CSPMiddleware

DEFAULT_POLICY = (
    "default-src 'self'; "
    "worker-src 'self' blob:; "
    "child-src 'self' blob:; "
    "script-src 'self' blob: 'wasm-unsafe-eval' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self' data:; "
    "connect-src 'self' blob:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "upgrade-insecure-requests"
)

CHUNKING_POLICY = (
    "default-src 'none'; "
    "script-src 'none'; "
    "style-src 'none'; "
    "img-src 'none'; "
    "font-src 'none'; "
    "connect-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)


def _build_app(strict_mode: bool = False) -> TestClient:
    app = FastAPI()
    app.add_middleware(CSPMiddleware, strict_mode=strict_mode)
    return TestClient(app)


def test_csp_middleware_default_policy_allows_embedding_atlas():
    client = _build_app()

    @client.app.get("/")
    async def root():
        return {"status": "ok"}

    response = client.get("/")

    assert response.headers["Content-Security-Policy"] == DEFAULT_POLICY
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_csp_middleware_chunking_policy_remains_strict():
    client = _build_app()

    @client.app.get("/chunking/status")
    async def chunk_status():
        return {"status": "ok"}

    response = client.get("/chunking/status")

    assert response.headers["Content-Security-Policy"] == CHUNKING_POLICY

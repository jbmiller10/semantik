from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.webui.middleware.csp import CSPMiddleware


def test_csp_middleware_adds_headers():
    app = FastAPI()
    app.add_middleware(CSPMiddleware)

    @app.get("/")
    async def root():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/")

    assert response.headers.get("Content-Security-Policy")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"

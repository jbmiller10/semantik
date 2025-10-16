import jwt
import pytest
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from shared.config import settings

from packages.webui.middleware.rate_limit import RateLimitMiddleware


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/whoami")
    async def whoami(request: Request) -> JSONResponse:
        user = getattr(request.state, "user", None)
        return JSONResponse({"user": user})

    return app


@pytest.mark.parametrize(
    ("token_payload", "expected_id"),
    [
        ({"sub": "alice", "user_id": 42}, 42),
        ({"sub": "bob"}, None),
    ],
)
def test_rate_limit_middleware_extracts_user(token_payload: dict[str, object], expected_id: int | None) -> None:
    app = create_app()
    client = TestClient(app)

    token = jwt.encode(token_payload, settings.JWT_SECRET_KEY, algorithm="HS256")
    response = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})

    body = response.json()
    assert body["user"]["username"] == token_payload["sub"]
    if expected_id is not None:
        assert body["user"]["id"] == expected_id
    else:
        assert isinstance(body["user"]["id"], int)


def test_rate_limit_middleware_invalid_token() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/whoami", headers={"Authorization": "Bearer invalid"})
    assert response.status_code == 200
    assert response.json()["user"] is None

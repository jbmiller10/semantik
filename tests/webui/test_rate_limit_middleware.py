from unittest.mock import AsyncMock, Mock, patch

import jwt
import pytest
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from shared.config import settings
from webui.middleware.rate_limit import RateLimitMiddleware


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/whoami")
    async def whoami(request: Request) -> JSONResponse:
        user = getattr(request.state, "user", None)
        return JSONResponse({"user": user})

    return app


def _make_request() -> StarletteRequest:
    return StarletteRequest(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
    )


@pytest.mark.parametrize(
    ("token_payload", "expected_id"),
    [
        ({"sub": "alice", "user_id": 42}, 42),
        ({"sub": "alice", "user_id": "not-an-int"}, None),
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


def test_rate_limit_middleware_jwt_like_but_invalid_token_does_not_set_user() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/whoami", headers={"Authorization": "Bearer header.payload.signature"})
    assert response.status_code == 200
    assert response.json()["user"] is None


def test_rate_limit_middleware_handles_extraction_exceptions_gracefully() -> None:
    app = create_app()
    client = TestClient(app)

    token = jwt.encode({"sub": "alice"}, settings.JWT_SECRET_KEY, algorithm="HS256")
    with patch(
        "webui.middleware.rate_limit.RateLimitMiddleware.get_user_from_token",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        response = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["user"] is None


@pytest.mark.asyncio()
async def test_get_user_from_token_api_key_short_token_returns_none() -> None:
    middleware = RateLimitMiddleware(FastAPI())
    request = _make_request()

    user = await middleware.get_user_from_token(request, "too-short")
    assert user is None


@pytest.mark.asyncio()
async def test_get_user_from_token_api_key_cached_non_dict_returns_none() -> None:
    middleware = RateLimitMiddleware(FastAPI())
    request = _make_request()
    request.state.api_key_auth_checked = True
    request.state.api_key_auth = "not-a-dict"

    user = await middleware.get_user_from_token(request, "smtk_12345678_" + ("x" * 32))
    assert user is None


@pytest.mark.asyncio()
async def test_get_user_from_token_api_key_cached_user_id_and_default_username() -> None:
    middleware = RateLimitMiddleware(FastAPI())
    request = _make_request()
    request.state.api_key_auth_checked = True
    request.state.api_key_auth = {"user_id": 123}

    user = await middleware.get_user_from_token(request, "smtk_12345678_" + ("x" * 32))
    assert user == {"id": 123, "username": "api_key"}


@pytest.mark.asyncio()
async def test_get_user_from_token_api_key_missing_user_id_returns_none() -> None:
    middleware = RateLimitMiddleware(FastAPI())
    request = _make_request()
    request.state.api_key_auth_checked = True
    request.state.api_key_auth = {}

    user = await middleware.get_user_from_token(request, "smtk_12345678_" + ("x" * 32))
    assert user is None


@pytest.mark.asyncio()
async def test_get_user_from_token_api_key_invalid_user_id_returns_none() -> None:
    middleware = RateLimitMiddleware(FastAPI())
    request = _make_request()
    request.state.api_key_auth_checked = True
    request.state.api_key_auth = {"user_id": "not-an-int"}

    user = await middleware.get_user_from_token(request, "smtk_12345678_" + ("x" * 32))
    assert user is None


@pytest.mark.asyncio()
async def test_get_user_from_token_jwt_decode_error_returns_none() -> None:
    middleware = RateLimitMiddleware(FastAPI())
    request = _make_request()

    with patch("webui.middleware.rate_limit.jwt.decode", side_effect=RuntimeError("boom")):
        user = await middleware.get_user_from_token(request, "header.payload.signature")

    assert user is None


@pytest.mark.asyncio()
async def test_get_user_from_token_api_key_verifies_once_and_caches_result() -> None:
    middleware = RateLimitMiddleware(FastAPI())
    request = _make_request()
    token = "smtk_12345678_" + ("x" * 32)

    mock_session = Mock()

    async def fake_get_db_session():
        yield mock_session

    api_key_data = {
        "id": "key-1",
        "name": "test key",
        "user": {"id": 7, "username": "alice"},
    }
    mock_repo = AsyncMock()
    mock_repo.verify_api_key = AsyncMock(return_value=api_key_data)

    with (
        patch("webui.middleware.rate_limit.get_db_session", new=fake_get_db_session),
        patch("webui.middleware.rate_limit.PostgreSQLApiKeyRepository", return_value=mock_repo),
    ):
        first = await middleware.get_user_from_token(request, token)
        second = await middleware.get_user_from_token(request, token)

    assert first == {"id": 7, "username": "alice"}
    assert second == {"id": 7, "username": "alice"}
    mock_repo.verify_api_key.assert_called_once_with(token, update_last_used=True)


@pytest.mark.asyncio()
async def test_get_user_from_token_api_key_verification_exception_returns_none() -> None:
    middleware = RateLimitMiddleware(FastAPI())
    request = _make_request()
    token = "smtk_12345678_" + ("x" * 32)

    mock_session = Mock()

    async def fake_get_db_session():
        yield mock_session

    mock_repo = AsyncMock()
    mock_repo.verify_api_key = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        patch("webui.middleware.rate_limit.get_db_session", new=fake_get_db_session),
        patch("webui.middleware.rate_limit.PostgreSQLApiKeyRepository", return_value=mock_repo),
    ):
        user = await middleware.get_user_from_token(request, token)

    assert user is None


@pytest.mark.asyncio()
async def test_dispatch_with_limiter_disabled_short_circuits() -> None:
    class _Inner:
        def get_window_stats(self, limit, *args):  # noqa: ARG002
            return (0, 1)

    class _Limiter:
        def __init__(self, enabled: bool):
            self.enabled = enabled
            self.limiter = _Inner()

        def _inject_headers(self, response, view_rate_limit):  # noqa: ARG002
            response.headers["X-RateLimit-Injected"] = "1"
            return response

    app = FastAPI()
    app.state.limiter = _Limiter(enabled=False)

    request = StarletteRequest(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "app": app,
        }
    )
    middleware = RateLimitMiddleware(app)
    call_next = AsyncMock(return_value=JSONResponse({"ok": True}))

    with patch("webui.middleware.rate_limit.ensure_limiter_runtime_state", new=Mock()):
        response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_dispatch_with_limiter_exempt_route_skips_checks() -> None:
    class _Inner:
        def get_window_stats(self, limit, *args):  # noqa: ARG002
            return (0, 1)

    class _Limiter:
        def __init__(self):
            self.enabled = True
            self.limiter = _Inner()

        def _inject_headers(self, response, view_rate_limit):  # noqa: ARG002
            response.headers["X-RateLimit-Injected"] = "1"
            return response

    app = FastAPI()
    app.state.limiter = _Limiter()

    request = StarletteRequest(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "app": app,
        }
    )
    middleware = RateLimitMiddleware(app)
    call_next = AsyncMock(return_value=JSONResponse({"ok": True}))

    with (
        patch("webui.middleware.rate_limit.ensure_limiter_runtime_state", new=Mock()),
        patch("webui.middleware.rate_limit._find_route_handler", return_value=object()),
        patch("webui.middleware.rate_limit._should_exempt", return_value=True),
    ):
        response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_dispatch_with_limiter_sets_user_and_injects_headers() -> None:
    class _Inner:
        def get_window_stats(self, limit, *args):  # noqa: ARG002
            return (0, 5)

    class _Limiter:
        def __init__(self):
            self.enabled = True
            self.limiter = _Inner()

        def _inject_headers(self, response, view_rate_limit):  # noqa: ARG002
            response.headers["X-RateLimit-Injected"] = "1"
            return response

    app = FastAPI()
    app.state.limiter = _Limiter()

    request = StarletteRequest(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"authorization", b"Bearer smtk_12345678_" + (b"x" * 32))],
            "app": app,
        }
    )
    middleware = RateLimitMiddleware(app)

    async def call_next(req):
        assert req.state.user == {"id": 1, "username": "alice"}
        return JSONResponse({"ok": True})

    def fake_sync_check_limits(limiter, req, handler, app):  # noqa: ARG001
        req.state.view_rate_limit = ("1/minute", ("user:1",))
        return None, True

    with (
        patch("webui.middleware.rate_limit.ensure_limiter_runtime_state", new=Mock()),
        patch("webui.middleware.rate_limit._find_route_handler", return_value=object()),
        patch("webui.middleware.rate_limit._should_exempt", return_value=False),
        patch("webui.middleware.rate_limit.sync_check_limits", new=fake_sync_check_limits),
        patch.object(middleware, "get_user_from_token", new=AsyncMock(return_value={"id": 1, "username": "alice"})),
    ):
        response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    assert response.headers["X-RateLimit-Injected"] == "1"


@pytest.mark.asyncio()
async def test_dispatch_with_limiter_returns_error_response() -> None:
    class _Inner:
        def get_window_stats(self, limit, *args):  # noqa: ARG002
            return (0, 1)

    class _Limiter:
        def __init__(self):
            self.enabled = True
            self.limiter = _Inner()

        def _inject_headers(self, response, view_rate_limit):  # noqa: ARG002
            response.headers["X-RateLimit-Injected"] = "1"
            return response

    app = FastAPI()
    app.state.limiter = _Limiter()

    request = StarletteRequest(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "app": app,
        }
    )
    middleware = RateLimitMiddleware(app)
    call_next = AsyncMock(return_value=JSONResponse({"ok": True}))

    error_response = JSONResponse({"error": "rate_limit_exceeded"}, status_code=429)

    with (
        patch("webui.middleware.rate_limit.ensure_limiter_runtime_state", new=Mock()),
        patch("webui.middleware.rate_limit._find_route_handler", return_value=object()),
        patch("webui.middleware.rate_limit._should_exempt", return_value=False),
        patch("webui.middleware.rate_limit.sync_check_limits", return_value=(error_response, False)),
    ):
        response = await middleware.dispatch(request, call_next)

    assert response.status_code == 429
    call_next.assert_not_awaited()


@pytest.mark.asyncio()
async def test_dispatch_with_limiter_handles_user_extraction_failure() -> None:
    class _Inner:
        def get_window_stats(self, limit, *args):  # noqa: ARG002
            return (0, 1)

    class _Limiter:
        def __init__(self):
            self.enabled = True
            self.limiter = _Inner()

        def _inject_headers(self, response, view_rate_limit):  # noqa: ARG002
            response.headers["X-RateLimit-Injected"] = "1"
            return response

    app = FastAPI()
    app.state.limiter = _Limiter()

    request = StarletteRequest(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"authorization", b"Bearer smtk_12345678_" + (b"x" * 32))],
            "app": app,
        }
    )
    middleware = RateLimitMiddleware(app)
    call_next = AsyncMock(return_value=JSONResponse({"ok": True}))

    with (
        patch("webui.middleware.rate_limit.ensure_limiter_runtime_state", new=Mock()),
        patch("webui.middleware.rate_limit._find_route_handler", return_value=object()),
        patch("webui.middleware.rate_limit._should_exempt", return_value=False),
        patch("webui.middleware.rate_limit.sync_check_limits", return_value=(None, False)),
        patch.object(middleware, "get_user_from_token", new=AsyncMock(side_effect=RuntimeError("boom"))),
    ):
        response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_awaited_once()

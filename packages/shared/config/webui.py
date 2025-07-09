# shared/config/webui.py

from .base import BaseConfig


class WebuiConfig(BaseConfig):
    """
    WebUI-specific configuration.
    Contains settings specific to the web application and API.
    """

    # JWT Authentication Configuration
    JWT_SECRET_KEY: str = "default-secret-key"  # MUST be overridden in .env
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALGORITHM: str = "HS256"

    # Service Ports
    WEBUI_PORT: int = 8080

    # Service URLs (for internal API calls)
    WEBUI_URL: str = "http://localhost:8080"

    # External service URLs
    SEARCH_API_URL: str = "http://localhost:8000"

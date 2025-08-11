"""
PostgreSQL configuration and connection management.
"""

import logging
import os
from typing import Any
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class PostgresConfig(BaseSettings):
    """PostgreSQL-specific configuration settings."""

    # Database URL - can be set directly or constructed from components
    DATABASE_URL: str | None = Field(default=None, description="PostgreSQL database URL")

    # Individual connection parameters (used if DATABASE_URL not provided)
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_DB: str = Field(default="semantik", description="PostgreSQL database name")
    POSTGRES_USER: str = Field(default="semantik", description="PostgreSQL username")
    POSTGRES_PASSWORD: str = Field(default="", description="PostgreSQL password")

    # Connection pool settings
    DB_POOL_SIZE: int = Field(default=20, description="Connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=40, description="Maximum overflow connections")
    DB_POOL_TIMEOUT: int = Field(default=30, description="Pool timeout in seconds")
    DB_POOL_RECYCLE: int = Field(default=3600, description="Connection recycle time in seconds")
    DB_POOL_PRE_PING: bool = Field(default=True, description="Test connections before use")

    # Query settings
    DB_ECHO: bool = Field(default=False, description="Echo SQL queries (debug mode)")
    DB_ECHO_POOL: bool = Field(default=False, description="Echo connection pool events")
    DB_QUERY_TIMEOUT: int = Field(default=30, description="Query timeout in seconds")

    # Retry settings
    DB_RETRY_LIMIT: int = Field(default=3, description="Number of connection retries")
    DB_RETRY_INTERVAL: float = Field(default=0.5, description="Retry interval in seconds")

    # Chunking settings
    CHUNK_PARTITION_COUNT: int = Field(
        default=100, description="Number of partitions for chunks table (LIST partitioning)"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def construct_database_url(cls, v: str | None) -> str:
        """Construct DATABASE_URL from components if not provided."""
        if v:
            return v

        # Get values from environment or defaults
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ.get("POSTGRES_DB", "semantik")
        user = os.environ.get("POSTGRES_USER", "semantik")
        password = os.environ.get("POSTGRES_PASSWORD", "")

        if not password:
            logger.warning("POSTGRES_PASSWORD not set - using empty password")

        # Construct PostgreSQL URL
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    @property
    def async_database_url(self) -> str:
        """Get async version of database URL for SQLAlchemy."""
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL not configured")

        # Parse the URL and replace scheme with async version
        parsed = urlparse(self.DATABASE_URL)
        async_scheme = "postgresql+asyncpg" if parsed.scheme == "postgresql" else parsed.scheme

        # Reconstruct URL with async scheme
        return self.DATABASE_URL.replace(f"{parsed.scheme}://", f"{async_scheme}://", 1)

    @property
    def sync_database_url(self) -> str:
        """Get sync version of database URL for migrations."""
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL not configured")

        # Parse the URL and ensure it uses sync driver
        parsed = urlparse(self.DATABASE_URL)
        sync_scheme = "postgresql+psycopg2" if parsed.scheme == "postgresql" else parsed.scheme

        # Reconstruct URL with sync scheme
        return self.DATABASE_URL.replace(f"{parsed.scheme}://", f"{sync_scheme}://", 1)

    def get_pool_kwargs(self) -> dict[str, Any]:
        """Get SQLAlchemy connection pool keyword arguments."""
        return {
            "pool_size": self.DB_POOL_SIZE,
            "max_overflow": self.DB_MAX_OVERFLOW,
            "pool_timeout": self.DB_POOL_TIMEOUT,
            "pool_recycle": self.DB_POOL_RECYCLE,
            "pool_pre_ping": self.DB_POOL_PRE_PING,
        }

    def get_connect_args(self) -> dict[str, Any]:
        """Get database-specific connection arguments."""
        parsed = urlparse(self.DATABASE_URL or "")

        if parsed.scheme.startswith("postgresql"):
            # PostgreSQL-specific settings
            connect_args = {
                "server_settings": {
                    "application_name": "semantik",
                    "jit": "off",  # Disable JIT for more predictable performance
                    "statement_timeout": f"{self.DB_QUERY_TIMEOUT * 1000}",  # Convert to milliseconds
                    "lock_timeout": "5000",  # 5 seconds
                    "idle_in_transaction_session_timeout": "60000",  # 60 seconds
                },
            }

            # For asyncpg driver, use 'timeout' instead of 'command_timeout'
            if "asyncpg" in parsed.scheme:
                connect_args["timeout"] = self.DB_QUERY_TIMEOUT  # type: ignore[assignment]
            else:
                connect_args["command_timeout"] = self.DB_QUERY_TIMEOUT  # type: ignore[assignment]

            return connect_args
        return {}


# Create a singleton instance
postgres_config = PostgresConfig()

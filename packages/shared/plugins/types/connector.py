"""Connector plugin base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from shared.dtos.ingestion import IngestedDocument

from ..base import SemanticPlugin


class ConnectorPlugin(SemanticPlugin, ABC):
    """Base class for data source connectors."""

    PLUGIN_TYPE = "connector"

    @abstractmethod
    def __init__(self, config: dict[str, Any]):
        """Initialize connector with configuration."""

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the data source."""

    @abstractmethod
    async def load_documents(self, source_id: str | None = None) -> AsyncIterator[IngestedDocument]:
        """Load documents from the data source."""

    def validate_config(self) -> None:  # noqa: B027
        """Validate configuration. Override to add custom validation."""
        return None

    @classmethod
    @abstractmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        """Return list of config field definitions for UI."""

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        """Return list of secret field definitions for UI."""
        return []

"""Document source connectors package."""

from shared.connectors.base import BaseConnector
from shared.connectors.local import LocalFileConnector

__all__ = ["BaseConnector", "LocalFileConnector"]

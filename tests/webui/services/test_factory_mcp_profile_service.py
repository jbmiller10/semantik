from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webui.services.factory import create_mcp_profile_service, get_mcp_profile_service
from webui.services.mcp_profile_service import MCPProfileService


def test_create_mcp_profile_service_constructs_service() -> None:
    db = MagicMock()
    service = create_mcp_profile_service(db)
    assert service.db_session is db


@pytest.mark.asyncio()
async def test_get_mcp_profile_service_returns_service() -> None:
    db = MagicMock()
    service = await get_mcp_profile_service(db=db)
    assert service.db_session is db


# --------------------------------------------------------------------------
# MCPProfileService Unit Tests
# --------------------------------------------------------------------------


class TestMCPProfileServiceUpdate:
    """Tests for MCP Profile Service update edge cases."""

    @pytest.fixture()
    def mock_db_session(self) -> MagicMock:
        """Create a mock database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture()
    def service(self, mock_db_session: MagicMock) -> MCPProfileService:
        """Create service instance with mocked session."""
        return MCPProfileService(mock_db_session)

    @pytest.mark.asyncio()
    async def test_update_reorders_collections_correctly(self, service: MCPProfileService) -> None:
        """Test that updating collection_ids reorders collections properly."""
        profile_id = "00000000-0000-0000-0000-000000000001"
        owner_id = 1

        # Use valid UUIDs for collection IDs
        coll_id_1 = "00000000-0000-0000-0000-000000000011"
        coll_id_2 = "00000000-0000-0000-0000-000000000012"
        coll_id_3 = "00000000-0000-0000-0000-000000000013"

        # Mock the profile returned by get()
        mock_profile = MagicMock()
        mock_profile.id = profile_id
        mock_profile.owner_id = owner_id
        mock_profile.name = "test-profile"

        # Track associations added
        added_associations = []
        original_add = service.db_session.add

        def track_add(obj: object) -> None:
            if hasattr(obj, "profile_id") and hasattr(obj, "order"):
                added_associations.append({"collection_id": obj.collection_id, "order": obj.order})
            original_add(obj)

        service.db_session.add = track_add

        # Mock service internal methods
        with (
            patch.object(service, "get", new_callable=AsyncMock) as mock_get,
            patch.object(service, "_validate_collection_access", new_callable=AsyncMock) as mock_validate,
            patch.object(service, "_get_with_collections", new_callable=AsyncMock) as mock_reload,
        ):
            mock_get.return_value = mock_profile
            mock_reload.return_value = mock_profile

            # Import the update schema
            from webui.api.v2.mcp_schemas import MCPProfileUpdate

            # Update with new collection order (reversed: 3, 1, 2)
            update_data = MCPProfileUpdate(collection_ids=[coll_id_3, coll_id_1, coll_id_2])

            await service.update(profile_id, update_data, owner_id)

            # Verify collections were validated
            mock_validate.assert_called_once_with([coll_id_3, coll_id_1, coll_id_2], owner_id)

            # Verify associations were created with correct order
            assert len(added_associations) == 3
            assert {"collection_id": coll_id_3, "order": 0} in added_associations
            assert {"collection_id": coll_id_1, "order": 1} in added_associations
            assert {"collection_id": coll_id_2, "order": 2} in added_associations

    @pytest.mark.asyncio()
    async def test_partial_field_updates_work(self, service: MCPProfileService) -> None:
        """Test that partial updates only modify specified fields."""
        profile_id = "profile-456"
        owner_id = 2

        # Mock the profile
        mock_profile = MagicMock()
        mock_profile.id = profile_id
        mock_profile.owner_id = owner_id
        mock_profile.name = "original-name"
        mock_profile.description = "original description"
        mock_profile.enabled = True
        mock_profile.result_count = 10

        with (
            patch.object(service, "get", new_callable=AsyncMock) as mock_get,
            patch.object(service, "_get_with_collections", new_callable=AsyncMock) as mock_reload,
        ):
            mock_get.return_value = mock_profile
            mock_reload.return_value = mock_profile

            from webui.api.v2.mcp_schemas import MCPProfileUpdate

            # Update only description and result_count
            update_data = MCPProfileUpdate(description="new description", result_count=20)

            await service.update(profile_id, update_data, owner_id)

            # Verify only specified fields were updated
            assert mock_profile.description == "new description"
            assert mock_profile.result_count == 20
            # Name should remain unchanged
            assert mock_profile.name == "original-name"
            assert mock_profile.enabled is True


class TestMCPProfileServiceGetConfig:
    """Tests for MCP Profile Service get_config method."""

    @pytest.fixture()
    def mock_db_session(self) -> MagicMock:
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture()
    def service(self, mock_db_session: MagicMock) -> MCPProfileService:
        """Create service instance."""
        return MCPProfileService(mock_db_session)

    @pytest.mark.asyncio()
    async def test_get_config_http_transport(self, service: MCPProfileService) -> None:
        """Test HTTP transport config generation."""
        profile_id = "profile-789"
        owner_id = 3

        mock_profile = MagicMock()
        mock_profile.id = profile_id
        mock_profile.owner_id = owner_id
        mock_profile.name = "my-search-profile"

        with patch.object(service, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_profile

            config = await service.get_config(
                profile_id,
                owner_id,
                webui_url="http://localhost:8080",
                transport="http",
                mcp_http_url="http://mcp-server:9090/mcp",
            )

            assert config.transport == "http"
            assert config.server_name == "semantik-my-search-profile"
            assert config.url == "http://mcp-server:9090/mcp"
            # HTTP transport should NOT have command/args/env
            assert config.command is None
            assert config.args is None
            assert config.env is None

    @pytest.mark.asyncio()
    async def test_get_config_http_transport_default_url(self, service: MCPProfileService) -> None:
        """Test HTTP transport uses default URL when not provided."""
        profile_id = "profile-abc"
        owner_id = 4

        mock_profile = MagicMock()
        mock_profile.id = profile_id
        mock_profile.owner_id = owner_id
        mock_profile.name = "default-url-profile"

        with patch.object(service, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_profile

            config = await service.get_config(
                profile_id,
                owner_id,
                webui_url="http://localhost:8080",
                transport="http",
                # mcp_http_url not provided
            )

            assert config.transport == "http"
            assert config.url == "http://localhost:9090/mcp"  # Default URL

    @pytest.mark.asyncio()
    async def test_get_config_stdio_transport(self, service: MCPProfileService) -> None:
        """Test stdio transport config generation."""
        profile_id = "profile-xyz"
        owner_id = 5

        mock_profile = MagicMock()
        mock_profile.id = profile_id
        mock_profile.owner_id = owner_id
        mock_profile.name = "stdio-profile"

        with patch.object(service, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_profile

            config = await service.get_config(
                profile_id,
                owner_id,
                webui_url="http://localhost:8080",
                transport="stdio",
            )

            assert config.transport == "stdio"
            assert config.server_name == "semantik-stdio-profile"
            assert config.command == "semantik-mcp"
            assert config.args == ["serve", "--profile", "stdio-profile"]
            assert config.env is not None
            assert config.env["SEMANTIK_WEBUI_URL"] == "http://localhost:8080"
            assert config.env["SEMANTIK_AUTH_TOKEN"] == "<your-access-token-or-api-key>"
            # stdio transport should NOT have url
            assert config.url is None

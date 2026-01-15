import pytest
from unittest.mock import AsyncMock, MagicMock
from webui.services.mcp_profile_service import MCPProfileService
from shared.database.repositories.user_preferences_repository import UserPreferencesRepository
from webui.api.v2.mcp_schemas import MCPProfileCreate, MCPProfileUpdate
from webui.api.v2.user_preferences_schemas import UserPreferencesUpdate, SearchPreferences

@pytest.mark.asyncio
async def test_mcp_profile_hyde_setting():
    """Test that MCPProfileService handles hyde_enabled field."""
    db_session = AsyncMock()
    service = MCPProfileService(db_session)
    
    # Mock _get_by_name and _validate_collection_access
    service._get_by_name = AsyncMock(return_value=None)
    service._validate_collection_access = AsyncMock()
    service._get_with_collections = AsyncMock()
    
    data = MCPProfileCreate(
        name="test",
        description="test desc",
        collection_ids=["123e4567-e89b-12d3-a456-426614174000"],
        hyde_enabled=True
    )
    
    await service.create(data, owner_id=1)
    
    # Verify profile was added with hyde_enabled=True
    added_obj = db_session.add.call_args_list[0][0][0]
    assert added_obj.hyde_enabled is True

@pytest.mark.asyncio
async def test_user_preferences_hyde_setting():
    """Test that UserPreferencesRepository handles HyDE fields."""
    db_session = AsyncMock()
    repo = UserPreferencesRepository(db_session)
    
    # Mock get_or_create
    mock_prefs = MagicMock()
    repo.get_or_create = AsyncMock(return_value=mock_prefs)
    
    await repo.update(
        user_id=1,
        hyde_enabled_default=True,
        hyde_llm_tier="high"
    )
    
    assert mock_prefs.hyde_enabled_default is True
    assert mock_prefs.hyde_llm_tier == "high"

@pytest.mark.asyncio
async def test_user_preferences_reset_hyde():
    """Test that reset_search resets HyDE fields."""
    db_session = AsyncMock()
    repo = UserPreferencesRepository(db_session)
    
    mock_prefs = MagicMock()
    repo.get_or_create = AsyncMock(return_value=mock_prefs)
    
    await repo.reset_search(user_id=1)
    
    assert mock_prefs.hyde_enabled_default is False
    assert mock_prefs.hyde_llm_tier == "low"

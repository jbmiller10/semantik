import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from webui.services.search_service import SearchService
from shared.database.models import Collection
from shared.llm.types import LLMQualityTier, LLMResponse
from shared.llm.exceptions import LLMNotConfiguredError

# Mock dependencies
@pytest.fixture
def mock_db_session():
    return AsyncMock()

@pytest.fixture
def mock_collection_repo():
    repo = AsyncMock()
    # Mock get_by_uuid_with_permission_check
    collection = Collection(
        id=1, 
        name="test-coll", 
        vector_store_name="test_coll_vec", 
        embedding_model="test-model",
        quantization="float16",
        status="ready"
    )
    repo.get_by_uuid_with_permission_check.return_value = collection
    return repo

@pytest.fixture
def mock_llm_factory():
    factory = AsyncMock()
    return factory

@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock()
    # Support async context manager
    provider.__aenter__.return_value = provider
    provider.__aexit__.return_value = None
    
    # Mock generate response
    provider.generate.return_value = LLMResponse(
        content="Hypothetical answer",
        model="test-model",
        provider="test-provider",
        input_tokens=10,
        output_tokens=20
    )
    return provider

@pytest.fixture
def mock_prefs_repo():
    repo = AsyncMock()
    # Mock get_or_create to return a mock prefs object
    mock_prefs = MagicMock()
    mock_prefs.hyde_enabled_default = False
    mock_prefs.hyde_llm_tier = "low"
    repo.get_or_create.return_value = mock_prefs
    return repo

@pytest.fixture
def search_service(mock_db_session, mock_collection_repo):
    service = SearchService(mock_db_session, mock_collection_repo)
    # Mock internal methods to avoid real HTTP calls
    service.search_single_collection = AsyncMock(return_value=(
        MagicMock(id=1, name="test-coll"), 
        {"results": [], "warnings": []}, 
        None
    ))
    return service

@pytest.mark.asyncio
async def test_hyde_search_enabled_success(
    search_service, 
    mock_llm_factory, 
    mock_llm_provider,
    mock_prefs_repo
):
    """Test that HyDE generates a hypothetical doc and uses it for search."""
    
    mock_llm_factory.create_provider_for_tier.return_value = mock_llm_provider
    
    with patch("webui.services.search_service.LLMServiceFactory", return_value=mock_llm_factory), \
         patch("webui.services.search_service.UserPreferencesRepository", return_value=mock_prefs_repo):
        
        await search_service.multi_collection_search(
            user_id=1,
            collection_uuids=["123e4567-e89b-12d3-a456-426614174000"],
            query="original query",
            hyde_enabled=True
        )
        
        # Verify LLM was called
        mock_llm_factory.create_provider_for_tier.assert_called_once_with(
            user_id=1,
            quality_tier=LLMQualityTier.LOW
        )
        mock_llm_provider.generate.assert_called_once()
        
        # Verify search_single_collection called with HYPOTHETICAL content
        search_service.search_single_collection.assert_called()
        # Check arguments of the first call
        call_args = search_service.search_single_collection.call_args
        # signature: (collection, query, k, search_params, timeout)
        assert call_args[0][1] == "Hypothetical answer"

@pytest.mark.asyncio
async def test_hyde_search_disabled(
    search_service, 
    mock_llm_factory,
    mock_prefs_repo
):
    """Test that disabled HyDE uses the original query."""
    
    with patch("webui.services.search_service.LLMServiceFactory", return_value=mock_llm_factory), \
         patch("webui.services.search_service.UserPreferencesRepository", return_value=mock_prefs_repo):
        
        await search_service.multi_collection_search(
            user_id=1,
            collection_uuids=["123e4567-e89b-12d3-a456-426614174000"],
            query="original query",
            hyde_enabled=False
        )
        
        # Verify LLM was NOT called
        mock_llm_factory.create_provider_for_tier.assert_not_called()
        
        # Verify search called with ORIGINAL query
        search_service.search_single_collection.assert_called()
        call_args = search_service.search_single_collection.call_args
        assert call_args[0][1] == "original query"

@pytest.mark.asyncio
async def test_hyde_search_fallback_on_error(
    search_service, 
    mock_llm_factory,
    mock_prefs_repo
):
    """Test fallback to original query if LLM is not configured."""
    
    # Mock factory to raise error
    mock_llm_factory.create_provider_for_tier.side_effect = LLMNotConfiguredError(1)
    
    with patch("webui.services.search_service.LLMServiceFactory", return_value=mock_llm_factory), \
         patch("webui.services.search_service.UserPreferencesRepository", return_value=mock_prefs_repo):
        
        await search_service.multi_collection_search(
            user_id=1,
            collection_uuids=["123e4567-e89b-12d3-a456-426614174000"],
            query="original query",
            hyde_enabled=True
        )
        
        # Verify LLM was attempted
        mock_llm_factory.create_provider_for_tier.assert_called_once()
        
        search_service.search_single_collection.assert_called()
        call_args = search_service.search_single_collection.call_args
        assert call_args[0][1] == "original query"

@pytest.mark.asyncio
async def test_hyde_single_search_enabled_success(
    search_service, 
    mock_llm_factory, 
    mock_llm_provider,
    mock_prefs_repo
):
    """Test HyDE for single collection search."""
    
    mock_llm_factory.create_provider_for_tier.return_value = mock_llm_provider
    
    with patch("webui.services.search_service.LLMServiceFactory", return_value=mock_llm_factory), \
         patch("webui.services.search_service.UserPreferencesRepository", return_value=mock_prefs_repo), \
         patch("webui.services.search_service.httpx.AsyncClient") as mock_client:
        
        # Mock httpx response for service.single_collection_search
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [], "warnings": []}
        mock_response.status_code = 200
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        # We need to un-mock search_single_collection to test the real implementation
        search_service.search_single_collection = SearchService.single_collection_search.__get__(search_service, SearchService)
        
        await search_service.single_collection_search(
            user_id=1,
            collection_uuid="123e4567-e89b-12d3-a456-426614174000",
            query="original query",
            hyde_enabled=True
        )
        
        # Verify LLM was called
        mock_llm_factory.create_provider_for_tier.assert_called_once()
        
        # Verify httpx was called with HYPOTHETICAL query
        call_args = mock_client.return_value.__aenter__.return_value.post.call_args
        assert call_args.kwargs["json"]["query"] == "Hypothetical answer"

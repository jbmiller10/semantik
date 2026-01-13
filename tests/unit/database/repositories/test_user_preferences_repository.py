"""Tests for user preferences repository."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.exceptions import DatabaseOperationError, ValidationError
from shared.database.repositories.user_preferences_repository import UserPreferencesRepository


class TestUserPreferencesRepository:
    """Tests for UserPreferencesRepository."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture()
    def repo(self, mock_session):
        """Create repository with mocked session."""
        return UserPreferencesRepository(mock_session)

    # =========================================================================
    # Validation Tests
    # =========================================================================

    def test_validate_search_mode_valid(self, repo):
        """Valid search modes pass validation."""
        repo._validate_search_mode("dense")
        repo._validate_search_mode("sparse")
        repo._validate_search_mode("hybrid")

    def test_validate_search_mode_invalid(self, repo):
        """Invalid search mode raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid search_mode"):
            repo._validate_search_mode("unknown")

    def test_validate_quantization_valid(self, repo):
        """Valid quantization values pass validation."""
        repo._validate_quantization("float32")
        repo._validate_quantization("float16")
        repo._validate_quantization("int8")

    def test_validate_quantization_invalid(self, repo):
        """Invalid quantization raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid quantization"):
            repo._validate_quantization("invalid")

    def test_validate_chunking_strategy_valid(self, repo):
        """Valid chunking strategies pass validation."""
        repo._validate_chunking_strategy("character")
        repo._validate_chunking_strategy("recursive")
        repo._validate_chunking_strategy("markdown")
        repo._validate_chunking_strategy("semantic")

    def test_validate_chunking_strategy_invalid(self, repo):
        """Invalid chunking strategy raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid chunking_strategy"):
            repo._validate_chunking_strategy("invalid")

    def test_validate_sparse_type_valid(self, repo):
        """Valid sparse types pass validation."""
        repo._validate_sparse_type("bm25")
        repo._validate_sparse_type("splade")

    def test_validate_sparse_type_invalid(self, repo):
        """Invalid sparse type raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid sparse_type"):
            repo._validate_sparse_type("invalid")

    def test_validate_range_valid(self, repo):
        """Values within range pass validation."""
        repo._validate_range(10, 5, 50, "test_field")
        repo._validate_range(5, 5, 50, "test_field")  # Boundary
        repo._validate_range(50, 5, 50, "test_field")  # Boundary

    def test_validate_range_invalid(self, repo):
        """Values outside range raise ValidationError."""
        with pytest.raises(ValidationError, match="must be between"):
            repo._validate_range(4, 5, 50, "test_field")
        with pytest.raises(ValidationError, match="must be between"):
            repo._validate_range(51, 5, 50, "test_field")

    def test_validate_hybrid_sparse_consistency_valid(self, repo):
        """Valid hybrid/sparse combinations pass."""
        from shared.database.repositories.user_preferences_repository import UNSET

        # Hybrid disabled, sparse can be anything
        repo._validate_hybrid_sparse_consistency(False, True, False)
        repo._validate_hybrid_sparse_consistency(False, False, False)

        # Hybrid enabled with sparse enabled
        repo._validate_hybrid_sparse_consistency(True, True, False)
        repo._validate_hybrid_sparse_consistency(True, UNSET, True)

    def test_validate_hybrid_sparse_consistency_invalid(self, repo):
        """Hybrid without sparse raises ValidationError."""
        from shared.database.repositories.user_preferences_repository import UNSET

        with pytest.raises(ValidationError, match="Cannot enable hybrid"):
            repo._validate_hybrid_sparse_consistency(True, False, False)

        with pytest.raises(ValidationError, match="Cannot enable hybrid"):
            repo._validate_hybrid_sparse_consistency(True, UNSET, False)

    # =========================================================================
    # CRUD Tests
    # =========================================================================

    async def test_get_by_user_id_found(self, repo, mock_session):
        """Returns preferences when found."""
        mock_prefs = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_prefs
        mock_session.execute.return_value = mock_result

        result = await repo.get_by_user_id(123)

        assert result == mock_prefs

    async def test_get_by_user_id_not_found(self, repo, mock_session):
        """Returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_by_user_id(123)

        assert result is None

    async def test_get_or_create_existing(self, repo, mock_session):
        """Returns existing preferences without creating new one."""
        mock_prefs = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_prefs
        mock_session.execute.return_value = mock_result

        result = await repo.get_or_create(123)

        assert result == mock_prefs
        mock_session.add.assert_not_called()

    async def test_get_or_create_new(self, repo, mock_session):
        """Creates new preferences when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_or_create(123)

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        assert result is not None

    # =========================================================================
    # Update Validation Tests
    # =========================================================================

    async def test_update_validates_search_top_k(self, repo):
        """Update validates search_top_k range."""
        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, search_top_k=4)  # Below minimum

        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, search_top_k=51)  # Above maximum

    async def test_update_validates_search_mode(self, repo):
        """Update validates search_mode value."""
        with pytest.raises(ValidationError, match="Invalid search_mode"):
            await repo.update(123, search_mode="invalid")

    async def test_update_validates_search_rrf_k(self, repo):
        """Update validates search_rrf_k range."""
        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, search_rrf_k=0)

        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, search_rrf_k=101)

    async def test_update_validates_similarity_threshold(self, repo):
        """Update validates similarity_threshold range."""
        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, search_similarity_threshold=-0.1)

        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, search_similarity_threshold=1.1)

    async def test_update_allows_null_similarity_threshold(self, repo, mock_session):
        """Update allows None for similarity_threshold."""
        mock_prefs = MagicMock()
        mock_prefs.default_enable_sparse = False

        with patch.object(repo, "get_or_create", new=AsyncMock(return_value=mock_prefs)):
            await repo.update(123, search_similarity_threshold=None)

        assert mock_prefs.search_similarity_threshold is None

    async def test_update_validates_quantization(self, repo):
        """Update validates quantization value."""
        with pytest.raises(ValidationError, match="Invalid quantization"):
            await repo.update(123, default_quantization="invalid")

    async def test_update_validates_chunking_strategy(self, repo):
        """Update validates chunking_strategy value."""
        with pytest.raises(ValidationError, match="Invalid chunking_strategy"):
            await repo.update(123, default_chunking_strategy="invalid")

    async def test_update_validates_chunk_size(self, repo):
        """Update validates chunk_size range."""
        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, default_chunk_size=255)

        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, default_chunk_size=4097)

    async def test_update_validates_chunk_overlap(self, repo):
        """Update validates chunk_overlap range."""
        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, default_chunk_overlap=-1)

        with pytest.raises(ValidationError, match="must be between"):
            await repo.update(123, default_chunk_overlap=513)

    async def test_update_validates_sparse_type(self, repo):
        """Update validates sparse_type value."""
        with pytest.raises(ValidationError, match="Invalid sparse_type"):
            await repo.update(123, default_sparse_type="invalid")

    async def test_update_validates_hybrid_requires_sparse(self, repo, mock_session):
        """Update validates hybrid requires sparse."""
        mock_prefs = MagicMock()
        mock_prefs.default_enable_sparse = False

        with patch.object(repo, "get_or_create", new=AsyncMock(return_value=mock_prefs)):
            with pytest.raises(ValidationError, match="Cannot enable hybrid"):
                await repo.update(123, default_enable_hybrid=True)

    async def test_update_allows_hybrid_with_sparse(self, repo, mock_session):
        """Update allows enabling hybrid when sparse is also enabled."""
        mock_prefs = MagicMock()
        mock_prefs.default_enable_sparse = False

        with patch.object(repo, "get_or_create", new=AsyncMock(return_value=mock_prefs)):
            await repo.update(123, default_enable_sparse=True, default_enable_hybrid=True)

        assert mock_prefs.default_enable_sparse is True
        assert mock_prefs.default_enable_hybrid is True

    async def test_update_partial_update(self, repo, mock_session):
        """Update only modifies provided fields."""
        mock_prefs = MagicMock()
        mock_prefs.search_top_k = 10
        mock_prefs.search_mode = "dense"
        mock_prefs.default_enable_sparse = False

        with patch.object(repo, "get_or_create", new=AsyncMock(return_value=mock_prefs)):
            await repo.update(123, search_top_k=20)

        # Only search_top_k should be updated
        assert mock_prefs.search_top_k == 20
        mock_session.flush.assert_called_once()

    # =========================================================================
    # Reset Tests
    # =========================================================================

    async def test_reset_search(self, repo, mock_session):
        """reset_search resets all search fields to defaults."""
        mock_prefs = MagicMock()
        mock_prefs.search_top_k = 20
        mock_prefs.search_mode = "hybrid"
        mock_prefs.search_use_reranker = True
        mock_prefs.search_rrf_k = 80
        mock_prefs.search_similarity_threshold = 0.5

        with patch.object(repo, "get_or_create", new=AsyncMock(return_value=mock_prefs)):
            result = await repo.reset_search(123)

        assert result.search_top_k == 10
        assert result.search_mode == "dense"
        assert result.search_use_reranker is False
        assert result.search_rrf_k == 60
        assert result.search_similarity_threshold is None
        mock_session.flush.assert_called_once()

    async def test_reset_collection_defaults(self, repo, mock_session):
        """reset_collection_defaults resets all collection fields to defaults."""
        mock_prefs = MagicMock()
        mock_prefs.default_embedding_model = "custom-model"
        mock_prefs.default_quantization = "scalar"
        mock_prefs.default_chunking_strategy = "markdown"
        mock_prefs.default_chunk_size = 2048
        mock_prefs.default_chunk_overlap = 100
        mock_prefs.default_enable_sparse = True
        mock_prefs.default_sparse_type = "splade"
        mock_prefs.default_enable_hybrid = True

        with patch.object(repo, "get_or_create", new=AsyncMock(return_value=mock_prefs)):
            result = await repo.reset_collection_defaults(123)

        assert result.default_embedding_model is None
        assert result.default_quantization == "float16"
        assert result.default_chunking_strategy == "recursive"
        assert result.default_chunk_size == 1024
        assert result.default_chunk_overlap == 200
        assert result.default_enable_sparse is False
        assert result.default_sparse_type == "bm25"
        assert result.default_enable_hybrid is False
        mock_session.flush.assert_called_once()

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    async def test_get_by_user_id_db_error(self, repo, mock_session):
        """get_by_user_id wraps database errors."""
        mock_session.execute.side_effect = Exception("DB error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.get_by_user_id(123)

        assert "get" in str(exc_info.value)
        assert "UserPreferences" in str(exc_info.value)

    async def test_get_or_create_db_error(self, repo, mock_session):
        """get_or_create wraps database errors on create."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        mock_session.flush.side_effect = Exception("DB error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.get_or_create(123)

        assert "get_or_create" in str(exc_info.value)

    async def test_update_db_error(self, repo, mock_session):
        """update wraps database errors."""
        mock_prefs = MagicMock()
        mock_prefs.default_enable_sparse = False

        with patch.object(repo, "get_or_create", new=AsyncMock(return_value=mock_prefs)):
            mock_session.flush.side_effect = Exception("DB error")

            with pytest.raises(DatabaseOperationError) as exc_info:
                await repo.update(123, search_top_k=20)

            assert "update" in str(exc_info.value)

    async def test_reset_search_db_error(self, repo, mock_session):
        """reset_search wraps database errors."""
        mock_prefs = MagicMock()

        with patch.object(repo, "get_or_create", new=AsyncMock(return_value=mock_prefs)):
            mock_session.flush.side_effect = Exception("DB error")

            with pytest.raises(DatabaseOperationError) as exc_info:
                await repo.reset_search(123)

            assert "reset_search" in str(exc_info.value)

    async def test_reset_collection_defaults_db_error(self, repo, mock_session):
        """reset_collection_defaults wraps database errors."""
        mock_prefs = MagicMock()

        with patch.object(repo, "get_or_create", new=AsyncMock(return_value=mock_prefs)):
            mock_session.flush.side_effect = Exception("DB error")

            with pytest.raises(DatabaseOperationError) as exc_info:
                await repo.reset_collection_defaults(123)

            assert "reset_collection_defaults" in str(exc_info.value)

    # =========================================================================
    # Default Values Tests
    # =========================================================================

    def test_search_defaults_values(self, repo):
        """Verify SEARCH_DEFAULTS contains correct values."""
        assert repo.SEARCH_DEFAULTS["search_top_k"] == 10
        assert repo.SEARCH_DEFAULTS["search_mode"] == "dense"
        assert repo.SEARCH_DEFAULTS["search_use_reranker"] is False
        assert repo.SEARCH_DEFAULTS["search_rrf_k"] == 60
        assert repo.SEARCH_DEFAULTS["search_similarity_threshold"] is None

    def test_collection_defaults_values(self, repo):
        """Verify COLLECTION_DEFAULTS contains correct values."""
        assert repo.COLLECTION_DEFAULTS["default_embedding_model"] is None
        assert repo.COLLECTION_DEFAULTS["default_quantization"] == "float16"
        assert repo.COLLECTION_DEFAULTS["default_chunking_strategy"] == "recursive"
        assert repo.COLLECTION_DEFAULTS["default_chunk_size"] == 1024
        assert repo.COLLECTION_DEFAULTS["default_chunk_overlap"] == 200
        assert repo.COLLECTION_DEFAULTS["default_enable_sparse"] is False
        assert repo.COLLECTION_DEFAULTS["default_sparse_type"] == "bm25"
        assert repo.COLLECTION_DEFAULTS["default_enable_hybrid"] is False

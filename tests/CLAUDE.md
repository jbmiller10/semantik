<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding new test categories or directories
     - Changing test patterns or fixtures
     - Modifying coverage targets or pytest configuration
     - Adding new conftest.py files or shared fixtures
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>Test Infrastructure</name>
  <purpose>Comprehensive pytest-based test suite for backend services</purpose>
  <location>tests/</location>
</component>

<testing-stack>
  <framework>pytest with async support (pytest-asyncio)</framework>
  <coverage-target>>= 80%</coverage-target>
  <coverage-command>pytest --cov=vecpipe --cov=webui --cov=shared --cov-report=html</coverage-command>
  <async-mode>auto (asyncio_mode = "auto" in pyproject.toml)</async-mode>
</testing-stack>

<test-organization>
  <dir path="unit/">
    <purpose>Isolated unit tests - mocked dependencies, no external services</purpose>
    <subdirs>
      - api/ - API endpoint unit tests
      - chunking/ - Chunking strategy unit tests
      - connectors/ - Connector unit tests (git, imap, local)
      - database/repositories/ - Repository unit tests
      - plugins/ - Plugin system unit tests
      - services/ - Service layer unit tests
      - tasks/ - Celery task unit tests
    </subdirs>
    <examples>
      - test_auth.py - Authentication/authorization tests
      - test_collection_repository.py - Repository pattern tests
      - test_memory_governor.py - GPU memory management tests
      - test_search_api.py - Search endpoint tests
      - test_plugin_loader.py - Plugin loading tests
    </examples>
  </dir>

  <dir path="integration/">
    <purpose>Tests requiring real infrastructure (database, Redis, Qdrant)</purpose>
    <subdirs>
      - chunking/ - Chunking integration tests
      - repositories/ - Repository integration tests with real DB
      - services/ - Service integration tests
      - strategies/ - Chunker strategy integration tests
      - web/ - Web layer integration tests
    </subdirs>
    <examples>
      - test_search_api_integration.py - Full search flow tests
      - test_websocket_redis_integration.py - WebSocket + Redis tests
      - test_plugin_lifecycle.py - Plugin lifecycle integration
    </examples>
  </dir>

  <dir path="e2e/">
    <purpose>End-to-end workflow tests requiring running services</purpose>
    <marker>@pytest.mark.e2e</marker>
    <examples>
      - test_collection_deletion_e2e.py - Full deletion workflow
      - test_websocket_integration.py - WebSocket E2E tests
      - test_mcp_flow.py - MCP integration tests
      - test_refactoring_validation.py - Architecture validation
    </examples>
  </dir>

  <dir path="security/">
    <purpose>Security vulnerability tests (OWASP patterns)</purpose>
    <examples>
      - test_path_traversal.py - Path traversal prevention (URL encoding, Unicode, null bytes)
      - test_xss_prevention.py - XSS sanitization tests
      - test_redos_prevention.py - Regex DoS protection tests
    </examples>
  </dir>

  <dir path="performance/">
    <purpose>Performance and load tests</purpose>
    <marker>@pytest.mark.performance</marker>
    <examples>
      - test_chunking_large_documents.py - Large document handling
    </examples>
  </dir>

  <dir path="streaming/">
    <purpose>Streaming chunking tests</purpose>
    <examples>
      - test_basic_functionality.py - Core streaming tests
      - test_memory_pool.py - Memory management tests
      - test_checkpoint_resume.py - Checkpoint/resume tests
      - test_utf8_boundaries.py - UTF-8 boundary handling
    </examples>
  </dir>

  <dir path="application/">
    <purpose>Clean Architecture use case tests (DDD-style)</purpose>
    <examples>
      - test_process_document_use_case.py - Document processing use case
      - test_preview_chunking_use_case.py - Preview chunking use case
      - test_cancel_operation_use_case.py - Operation cancellation
    </examples>
  </dir>

  <dir path="domain/">
    <purpose>Domain entity and value object tests</purpose>
    <examples>
      - test_chunking_strategies.py - Domain chunking strategies
      - test_chunking_operation.py - Operation entity tests
      - test_domain_exceptions.py - Domain exception tests
      - test_value_objects.py - Value object tests
    </examples>
  </dir>

  <dir path="database/">
    <purpose>Database-specific tests (migrations, partitioning)</purpose>
    <examples>
      - test_partitioning.py - Chunk partitioning tests
      - test_migration_100_partitions.py - Migration tests
    </examples>
  </dir>

  <dir path="webui/">
    <purpose>WebUI-specific tests (API endpoints, services)</purpose>
    <subdirs>
      - api/v1/ - V1 API tests
      - api/v2/ - V2 API tests
      - services/ - WebUI service tests
      - tasks/ - WebUI Celery task tests
    </subdirs>
  </dir>

  <dir path="shared/">
    <purpose>Shared package tests</purpose>
    <subdirs>
      - embedding/ - Embedding provider tests
      - database/ - Database utility tests
    </subdirs>
  </dir>
</test-organization>

<fixtures>
  <root-conftest path="conftest.py">
    <description>Main conftest with core fixtures and environment setup</description>
    <environment-setup>
      - Sets TESTING=true, ENV=test, DISABLE_RATE_LIMITING=true
      - Sets PROMETHEUS_DISABLE_SERVER=true, USE_MOCK_EMBEDDINGS=true
      - Loads .env.test or .env with local database safety checks
      - Forces POSTGRES_DB=semantik_test when using .env
    </environment-setup>
    <autouse-fixtures>
      - stub_celery_send_task: Stubs Celery broker for all tests
      - _reset_singletons: Clears Prometheus registries
      - _cleanup_pending_tasks: Cancels lingering asyncio tasks
    </autouse-fixtures>
    <key-fixtures>
      - db_session: Real async PostgreSQL session with rollback
      - test_client: FastAPI TestClient with mocked auth
      - async_client: Async HTTPX client with mocked auth
      - test_user / test_user_db: Test user data and DB entity
      - use_fakeredis: Opt-in fixture for Redis mocking
      - fake_redis_client / real_redis_client: Redis client fixtures
      - collection_factory / document_factory / operation_factory: Entity factories
      - mock_qdrant_client / mock_embedding_service: Service mocks
      - mock_websocket / mock_websocket_manager / mock_scalable_ws_manager: WebSocket mocks
    </key-fixtures>
  </root-conftest>

  <subdirectory-conftest path="database/conftest.py">
    <purpose>Database-specific fixtures with partition setup</purpose>
    <key-fixtures>
      - db_session: Creates partition views and chunks table with triggers
    </key-fixtures>
  </subdirectory-conftest>

  <subdirectory-conftest path="shared/embedding/conftest.py">
    <purpose>Embedding plugin test fixtures</purpose>
    <key-fixtures>
      - clean_registry / empty_registry: Registry state management
      - dummy_definition / dummy_plugin_class: Mock embedding plugins
    </key-fixtures>
  </subdirectory-conftest>

  <subdirectory-conftest path="webui/conftest.py">
    <purpose>WebUI test fixtures</purpose>
    <key-fixtures>
      - _cleanup_pending_tasks: Async task cleanup
      - mock_repositories: Mock repository instances
      - mock_celery_task: Mock Celery task object
    </key-fixtures>
  </subdirectory-conftest>

  <subdirectory-conftest path="webui/api/v2/conftest.py">
    <purpose>V2 API integration test fixtures</purpose>
    <key-fixtures>
      - api_client: AsyncClient with real DB and fakeredis
      - api_client_unauthenticated: AsyncClient without auth override
      - api_client_other_user: AsyncClient for ownership tests
      - api_auth_headers / other_user_auth_headers: JWT auth headers
    </key-fixtures>
  </subdirectory-conftest>

  <subdirectory-conftest path="unit/plugins/testing/conftest.py">
    <purpose>Plugin testing fixtures from shared.plugins.testing</purpose>
    <key-fixtures>
      - mock_chunker / mock_embedding_service / mock_reranker / mock_extractor
      - sample_text / sample_documents / sample_plugin_config
    </key-fixtures>
  </subdirectory-conftest>

  <additional-fixtures path="fixtures/search_reranking_fixtures.py">
    <purpose>Search and reranking test data helpers</purpose>
    <helpers>
      - create_mock_collection(): Mock collection with Qwen model
      - create_search_result(): Search result dict builder
      - create_vecpipe_response(): Mock vecpipe response
      - RERANKING_TEST_SCENARIOS: Predefined test scenarios
      - RERANKER_MODELS: Model configuration test data
    </helpers>
  </additional-fixtures>
</fixtures>

<test-patterns>
  <async-test-pattern>
    @pytest.mark.asyncio
    async def test_something(db_session, test_user_db):
        # Arrange
        collection = await collection_factory(owner_id=test_user_db.id)

        # Act
        result = await service.process(collection.id)

        # Assert
        assert result.status == "completed"
  </async-test-pattern>

  <use-case-test-pattern>
    class TestProcessDocumentUseCase:
        @pytest.fixture
        def mock_unit_of_work(self, mock_operations_repository):
            uow = AsyncMock()
            uow.__aenter__ = AsyncMock(return_value=uow)
            uow.__aexit__ = AsyncMock(return_value=None)
            uow.operations = mock_operations_repository
            uow.commit = AsyncMock()
            uow.rollback = AsyncMock()
            return uow

        @pytest.fixture
        def use_case(self, mock_unit_of_work, mock_document_service):
            return ProcessDocumentUseCase(
                unit_of_work=mock_unit_of_work,
                document_service=mock_document_service,
            )

        @pytest.mark.asyncio
        async def test_successful_processing(self, use_case, valid_request):
            response = await use_case.execute(valid_request)
            assert response.status == OperationStatus.COMPLETED
  </use-case-test-pattern>

  <api-integration-pattern>
    @pytest.mark.asyncio
    async def test_api_endpoint(api_client, api_auth_headers, collection_factory, test_user_db):
        # Create test data
        collection = await collection_factory(owner_id=test_user_db.id)

        # Make request
        response = await api_client.get(
            f"/api/v2/collections/{collection.id}",
            headers=api_auth_headers
        )

        # Verify
        assert response.status_code == 200
  </api-integration-pattern>

  <security-test-pattern>
    class TestPathTraversalSecurity:
        def test_basic_traversal_patterns_blocked(self):
            dangerous_paths = [
                ["../../../etc/passwd"],
                ["%2e%2e%2f%2e%2e%2fetc%2fpasswd"],
            ]
            for paths in dangerous_paths:
                with pytest.raises(ValidationError, match="Invalid file path"):
                    ChunkingSecurityValidator.validate_file_paths(paths)

        def test_legitimate_paths_allowed(self):
            safe_paths = [["documents/file.txt"], ["data/subfolder/document.pdf"]]
            for paths in safe_paths:
                ChunkingSecurityValidator.validate_file_paths(paths)  # No exception
  </security-test-pattern>
</test-patterns>

<running-tests>
  <commands>
    <all>pytest</all>
    <specific-file>pytest tests/unit/test_specific.py</specific-file>
    <specific-test>pytest tests/unit/test_specific.py::test_function</specific-test>
    <by-marker>pytest -m "not e2e"</by-marker>
    <integration-only>pytest -m integration</integration-only>
    <coverage>pytest --cov=packages --cov-report=html</coverage>
    <parallel>pytest -n auto</parallel>
    <verbose>pytest -v</verbose>
  </commands>

  <markers>
    - e2e: End-to-end tests requiring running services
    - integration: Tests requiring real infrastructure
    - performance: Slow performance tests
  </markers>
</running-tests>

<manual-test-harnesses>
  <location>Root of tests/ directory (excluded via norecursedirs)</location>
  <purpose>Exploratory scripts for manual validation with running stack</purpose>
  <scripts>
    - embedding_full_integration_suite.py - End-to-end embedding flows
    - embedding_performance_bench.py - Embedding latency/throughput benchmarks
    - metrics_flow_probe.py / metrics_update_probe.py - Prometheus instrumentation
    - search_probe.py - Search troubleshooting
    - frontend_api_test_suite.py - WebUI REST regression
    - websocket_*.py - WebSocket testing utilities
  </scripts>
  <usage>Run manually with: uv run python tests/script_name.py</usage>
</manual-test-harnesses>

<mocking-strategies>
  <async-mocks>
    Use AsyncMock for all async methods:
    ```python
    mock = AsyncMock()
    mock.execute = AsyncMock(return_value=result)
    ```
  </async-mocks>

  <redis-mocking>
    - use_fakeredis fixture patches redis.from_url and redis.asyncio.from_url
    - Provides both sync and async fake Redis clients
    - Also patches WebSocket manager and service manager Redis imports
  </redis-mocking>

  <database-mocking>
    - Unit tests: Use AsyncMock for session
    - Integration tests: Use db_session fixture with real database + rollback
  </database-mocking>

  <celery-mocking>
    - stub_celery_send_task (autouse): Stubs all Celery task dispatches
    - Access stub via fixture to inspect dispatch calls
  </celery-mocking>

  <factory-fixtures>
    Use factory fixtures for creating test entities:
    ```python
    async def test_with_factories(collection_factory, document_factory, test_user_db):
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)
    ```
  </factory-fixtures>
</mocking-strategies>

<database-testing>
  <isolation>
    - Each test gets its own session with rollback after test
    - db_session fixture creates tables and partition triggers
    - Tests should not rely on data from other tests
  </isolation>

  <partition-setup>
    - conftest.py creates compute_partition_key() function
    - Creates 100 chunk partitions (chunks_part_00 through chunks_part_99)
    - partition_health and partition_distribution views are created
  </partition-setup>

  <skipping-when-unavailable>
    ```python
    try:
        conn = await asyncpg.connect(**conn_params)
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")
    ```
  </skipping-when-unavailable>
</database-testing>

<common-pitfalls>
  <async-cleanup>
    Always use the _cleanup_pending_tasks fixture (autouse in many conftest files)
    to prevent test pollution from dangling tasks.
  </async-cleanup>

  <singleton-reset>
    The _reset_singletons fixture clears Prometheus registries between tests.
    If adding new singletons, ensure they're reset here.
  </singleton-reset>

  <environment-variables>
    conftest.py sets many env vars before imports. If tests need different
    values, use monkeypatch fixture or pytest-env.
  </environment-variables>

  <factory-required-fields>
    collection_factory and operation_factory require owner_id/user_id.
    Don't rely on defaults - always pass explicitly:
    ```python
    await collection_factory(owner_id=test_user_db.id)
    await operation_factory(user_id=test_user_db.id)
    ```
  </factory-required-fields>

  <database-availability>
    Integration tests skip if PostgreSQL is unavailable. Ensure database
    is running: `make docker-postgres-up`
  </database-availability>
</common-pitfalls>

<coverage-requirements>
  <target>80% minimum coverage</target>
  <packages>vecpipe, webui, shared</packages>
  <reports>html, term, xml</reports>
  <ci-enforcement>Coverage is checked in GitHub Actions CI</ci-enforcement>
</coverage-requirements>

<ci-integration>
  <github-actions>.github/workflows/test.yml</github-actions>
  <services>PostgreSQL service container for integration tests</services>
  <parallel>Tests can run in parallel with pytest-xdist</parallel>
</ci-integration>

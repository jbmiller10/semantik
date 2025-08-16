<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Changing test patterns or fixtures
     - Modifying coverage targets
     - Adding new test categories
     - Altering CI configuration
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>Test Infrastructure</name>
  <purpose>Comprehensive testing for backend and frontend</purpose>
  <location>tests/</location>
</component>

<testing-stack>
  <backend>
    <framework>pytest with async support</framework>
    <coverage-target>≥80%</coverage-target>
    <key-fixtures>
      - async_session: Test database session
      - test_client: FastAPI TestClient
      - redis_client: fakeredis for isolation
      - authenticated_client_v2: Async client with mocked auth
      - admin_client_v2: Async client with admin privileges
    </key-fixtures>
  </backend>
  
  <frontend>
    <framework>Vitest + React Testing Library</framework>
    <coverage-target>≥75%</coverage-target>
    <mocking>MSW for API mocking</mocking>
  </frontend>
</testing-stack>

<test-organization>
  <dir path="unit/">
    <purpose>Isolated unit tests</purpose>
    <examples>
      - test_chunking_strategies.py
      - test_collection_repository.py
      - test_auth.py
    </examples>
  </dir>
  
  <dir path="integration/">
    <purpose>Service integration tests</purpose>
    <examples>
      - test_search_api_integration.py
      - test_collection_persistence.py
      - test_websocket_redis_integration.py
    </examples>
  </dir>
  
  <dir path="e2e/">
    <purpose>End-to-end workflow tests</purpose>
    <examples>
      - test_collection_deletion_e2e.py
      - test_websocket_reindex.py
    </examples>
  </dir>
  
  <dir path="security/">
    <purpose>Security vulnerability tests</purpose>
    <examples>
      - test_path_traversal.py (OWASP patterns)
    </examples>
  </dir>
</test-organization>

<authentication-mocking>
  <overview>
    The test suite includes a comprehensive authentication mocking infrastructure that ensures consistent auth behavior across all tests without requiring real authentication or environment configuration.
  </overview>
  
  <key-components>
    <module path="tests/integration/auth_mock.py">
      - TestUser: Dataclass for consistent test user representation
      - AuthMocker: Class for overriding FastAPI auth dependencies
      - create_test_user(): Helper for creating custom test users
      - create_admin_user(): Helper for creating admin users
    </module>
    
    <fixtures>
      - authenticated_client_v2: Async client with default test user auth
      - admin_client_v2: Async client with admin privileges
      - auth_mocker: AuthMocker instance for custom auth scenarios
      - mock_user: TestUser instance for the default test user
      - mock_admin_user: TestUser instance for admin scenarios
    </fixtures>
  </key-components>
  
  <usage-examples>
    <basic-authenticated-test>
      @pytest.mark.asyncio
      async def test_with_auth(authenticated_client_v2):
          """Test endpoint with automatic authentication."""
          response = await authenticated_client_v2.get("/api/v2/collections")
          assert response.status_code == 200
          # User automatically has access to their resources
    </basic-authenticated-test>
    
    <admin-test>
      @pytest.mark.asyncio
      async def test_admin_only(admin_client_v2):
          """Test admin-only functionality."""
          response = await admin_client_v2.get("/api/v2/admin/users")
          assert response.status_code == 200
    </admin-test>
    
    <custom-user-test>
      @pytest.mark.asyncio
      async def test_custom_user(auth_mocker, async_client):
          """Test with a custom user."""
          custom_user = auth_mocker.create_user(
              user_id=42,
              username="custom_user",
              is_active=True
          )
          auth_mocker.set_current_user(custom_user)
          
          # Apply auth to client
          client = auth_mocker.apply_to_client(async_client)
          response = await client.get("/api/v2/collections")
          assert response.status_code == 200
    </custom-user-test>
    
    <multi-user-test>
      @pytest.mark.asyncio
      async def test_access_control(auth_mocker, async_client, async_session):
          """Test access control between users."""
          user1 = auth_mocker.create_user(user_id=1, username="user1")
          user2 = auth_mocker.create_user(user_id=2, username="user2")
          
          # Create resource as user1
          auth_mocker.set_current_user(user1)
          client1 = auth_mocker.apply_to_client(async_client)
          response = await client1.post("/api/v2/collections", json={...})
          collection_id = response.json()["id"]
          
          # Try to access as user2 (should fail)
          auth_mocker.set_current_user(user2)
          client2 = auth_mocker.apply_to_client(async_client)
          response = await client2.get(f"/api/v2/collections/{collection_id}")
          assert response.status_code == 403
    </multi-user-test>
  </usage-examples>
  
  <migration-guide>
    If you have tests using old authentication patterns:
    
    OLD:
    ```python
    async def test_old_pattern(async_client, test_user):
        app.dependency_overrides[get_current_user] = lambda: test_user
        response = await async_client.get("/api/v2/collections")
    ```
    
    NEW:
    ```python
    async def test_new_pattern(authenticated_client_v2):
        response = await authenticated_client_v2.get("/api/v2/collections")
    ```
  </migration-guide>
  
  <important-notes>
    - The auth mocking ensures owner_id consistency between created resources and authenticated users
    - Works regardless of DISABLE_AUTH environment variable setting
    - Automatically adds proper Bearer token headers for HTTPBearer security
    - Thread-safe and async-compatible
    - No real authentication calls are made, keeping tests fast
  </important-notes>
</authentication-mocking>

<test-patterns>
  <backend-pattern>
    @pytest.mark.asyncio
    async def test_something(async_session, authenticated_client_v2):
        # Arrange
        data = create_test_data()
        
        # Act
        response = await authenticated_client_v2.post("/api/v2/endpoint", json=data)
        
        # Assert
        assert response.status_code == 200
  </backend-pattern>
  
  <frontend-pattern>
    it('should handle user interaction', async () => {
      const { user } = renderWithProviders(&lt;Component /&gt;)
      
      await user.click(screen.getByRole('button'))
      
      await waitFor(() => {
        expect(screen.getByText('Success')).toBeInTheDocument()
      })
    })
  </frontend-pattern>
</test-patterns>

<running-tests>
  <backend>
    <all>pytest</all>
    <specific>pytest tests/unit/test_specific.py</specific>
    <coverage>pytest --cov=packages --cov-report=html</coverage>
  </backend>
  
  <frontend>
    <all>npm test</all>
    <watch>npm run test:watch</watch>
    <coverage>npm run test:coverage</coverage>
  </frontend>
</running-tests>

<ci-integration>
  <github-actions>.github/workflows/test.yml</github-actions>
  <parallel-execution>Tests run in parallel by type</parallel-execution>
  <database>PostgreSQL service container</database>
</ci-integration>
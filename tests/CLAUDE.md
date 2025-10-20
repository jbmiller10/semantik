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

<test-patterns>
  <backend-pattern>
    @pytest.mark.asyncio
    async def test_something(async_session, test_client):
        # Arrange
        data = create_test_data()
        
        # Act
        response = await test_client.post("/api/v2/endpoint", json=data)
        
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
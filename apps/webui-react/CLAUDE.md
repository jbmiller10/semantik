<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding/removing/renaming components, hooks, stores, or pages
     - Changing API client methods or endpoints
     - Modifying state management patterns (Zustand/React Query)
     - Altering WebSocket integration
     - Adding new feature modules (plugins, MCP, connectors)
     - Updating testing patterns or configuration
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>React Frontend (webui-react)</name>
  <purpose>React 19 SPA with TypeScript for Semantik's document embedding and semantic search interface</purpose>
  <location>apps/webui-react/</location>
</component>

<tech-stack>
  <core>React 19.1.0, TypeScript 5.8.3 (strict mode), React Router 7</core>
  <state>Zustand 5.0.6 (client state) + React Query 5.81.5 (server state)</state>
  <styling>TailwindCSS 3.4.17, Lucide React icons</styling>
  <build>Vite 7.0.0, outputs to packages/webui/static/</build>
  <testing>Vitest 2.1.9 + React Testing Library + MSW 2.10.4 + Playwright 1.54.1</testing>
</tech-stack>

<design-language>
  <theming>
    Use CSS variables for theme-aware colors (defined in index.css):
    - Backgrounds: --bg-primary, --bg-secondary, --bg-tertiary
    - Text: --text-primary, --text-secondary, --text-muted
    - Borders: --border, --border-subtle
    Dark mode uses Tailwind's class strategy (dark: prefix).
  </theming>
  <color-palette>
    Neutral/white aesthetic - avoid colored accents for interactive states.
    - Selection states: border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10
    - Selected text: text-gray-800 dark:text-white
    - Primary buttons: bg-gray-200 dark:bg-white text-gray-900 dark:text-gray-900
    - Secondary buttons: bg-[var(--bg-tertiary)] border-[var(--border)]
    - Focus rings: focus:ring-gray-400 dark:focus:ring-white (NOT colored rings)
  </color-palette>
  <status-banners>
    Semi-transparent backgrounds for light/dark compatibility:
    - Info: bg-blue-500/10 border-blue-500/30 text-blue-300/text-blue-400
    - Warning: bg-amber-500/10 border-amber-500/20 text-amber-400
    - Error: bg-red-500/10 border-red-500/30 text-red-300/text-red-400
    - Success: bg-green-500/20 text-green-400
  </status-banners>
  <toggle-switches>
    - Active: bg-gray-600 dark:bg-white
    - Inactive: bg-gray-300 dark:bg-gray-600
    - Knob: bg-white, when active in dark mode: bg-gray-800
  </toggle-switches>
  <avoid>
    - signal-* colors (legacy violet/purple palette - removed)
    - Hardcoded light-mode colors (bg-white, text-gray-900 without dark: variants)
    - Colored focus rings (use neutral gray/white instead)
  </avoid>
</design-language>

<architecture>
  <pattern>Component composition with custom hooks, feature-based organization</pattern>
  <state-management>
    <zustand>Client-only state: auth tokens, UI preferences, toasts, chunking config</zustand>
    <react-query>Server state: collections, operations, search results, models</react-query>
    <local>Component-specific UI state (forms, modals)</local>
  </state-management>
  <data-flow>
    Components -> Hooks (useCollections, useSearch) -> API Services -> Backend
    WebSocket -> useOperationsSocket -> React Query cache invalidation
  </data-flow>
</architecture>

<directory-structure>
  <dir path="src/components/">
    <purpose>React components organized by feature</purpose>
    <key-components>
      - Layout.tsx: Main app shell with header, tabs, global modals
      - CollectionsDashboard.tsx: Collection list with cards and operations
      - CollectionDetailsModal.tsx: Full collection details, sources, sync config
      - SearchInterface.tsx / SearchResults.tsx: Semantic search UI
      - ActiveOperationsTab.tsx: Real-time operation monitoring
      - BenchmarksTab.tsx: Search quality benchmarking (datasets, runs, results)
      - CreateCollectionModal.tsx: Collection creation with data source selection
      - AddDataToCollectionModal.tsx: Add sources to existing collections
      - EmbeddingVisualizationTab.tsx: UMAP/PCA embedding projections
      - DocumentViewer.tsx / PdfViewer.tsx: Document content display
    </key-components>
    <subdirs>
      - benchmarks/: Benchmark management UI (datasets, config matrix, results)
        - DatasetUploadModal.tsx: Ground truth dataset upload with JSON parsing
        - ConfigMatrixBuilder.tsx: Search config matrix builder with presets
        - BenchmarkProgress.tsx: Real-time benchmark progress via WebSocket
        - ResultsComparison.tsx: Results table with sorting and metrics display
        - MappingManagementPanel.tsx: Dataset-collection mapping resolution
      - chunking/: Chunking strategy selection and parameter tuning
      - connectors/: Dynamic connector forms (Directory, Git, IMAP)
      - plugins/: Plugin management UI (install, configure, enable)
      - mcp/: MCP profile management (create, edit, delete profiles)
      - settings/: Settings page tabs (Database, Plugins, MCP Profiles, LLM)
        - LLMSettings.tsx: LLM provider/model/quantization configuration per tier
      - search/: Search options and configuration
      - common/: Error boundaries, fallback components
      - pipeline/: DAG-based pipeline builder UI
        - PipelineVisualization.tsx: SVG-based DAG renderer with tier layout
        - PipelineNode.tsx: Node with input/output ports
        - PipelineEdge.tsx: Edge with conditional routing indicators
        - EdgePredicateEditor.tsx: Routing condition editor
        - RoutePreviewPanel.tsx: File routing test panel
        - ConfigurationPanel.tsx: Context-sensitive config panel
        - NodeConfigEditor.tsx: Node plugin configuration
      - wizard/: Collection creation wizard
        - CollectionWizard.tsx: Multi-step wizard (manual or AI-assisted)
        - steps/AnalysisStep.tsx: AI-assisted config with agent streaming
        - steps/BasicsStep.tsx: Name, source selection
        - steps/ConfigureStep.tsx: Manual pipeline configuration
        - steps/ReviewStep.tsx: Final review before creation
      - __tests__/: Component tests (*.test.tsx)
    </subdirs>
  </dir>

  <dir path="src/stores/">
    <purpose>Zustand state stores with persistence</purpose>
    <stores>
      - authStore.ts: JWT tokens, user info, logout (persisted to localStorage)
      - uiStore.ts: Active tab, toasts, modal visibility, document viewer state
      - searchStore.ts: Search params, results, reranking config, validation errors
      - chunkingStore.ts: Chunking strategy selection, preview, comparison, presets
    </stores>
    <pattern>
      Use persist middleware for auth only. Access stores via hooks in components.
      Never access localStorage directly - use authStore.
    </pattern>
  </dir>

  <dir path="src/hooks/">
    <purpose>Custom React hooks for data fetching and WebSocket</purpose>
    <key-hooks>
      - useCollections.ts: Collection CRUD with React Query + optimistic updates
      - useCollectionOperations.ts: Operation queries and cache updates
      - useWebSocket.ts: Generic WebSocket with auto-reconnect, timeout, auth via subprotocol
      - useOperationsSocket.ts: Global WebSocket for real-time operation updates
      - useBenchmarks.ts: Benchmark CRUD, datasets, mappings, results queries
      - useBenchmarkProgress.ts: Real-time benchmark progress via WebSocket
      - usePlugins.ts: Plugin management (install, enable, configure)
      - useMCPProfiles.ts: MCP profile CRUD operations
      - useConnectors.ts: Connector catalog and preview APIs
      - useProjections.ts: Embedding projections with caching
      - useModels.ts: Available embedding models
      - useRerankingAvailability.ts: Check reranker availability
      - useLLMSettings.ts: LLM provider configuration (settings CRUD, model lists)
      - useAssistedFlow.ts: Start assisted-flow sessions (Claude Agent SDK)
      - useAssistedFlowStream.ts: SSE streaming for assisted-flow (text/tool_use/tool_result/done)
      - useDragToConnect.ts: Drag-to-connect state for pipeline builder edges
      - useRoutePreview.ts: File upload and route preview API calls
      - useAvailablePlugins.ts: Plugin listing for pipeline node configuration
      - useTemplates.ts: Pipeline template listing
    </key-hooks>
  </dir>

  <dir path="src/services/api/v2/">
    <purpose>Backend API client layer</purpose>
    <modules>
      - client.ts: Axios instance with auth interceptors, token refresh
      - collections.ts: Collections CRUD, sources, sync, search
      - operations.ts: Operation tracking, WebSocket URLs
      - benchmarks.ts: Benchmark CRUD, datasets, mappings, results
      - plugins.ts: Plugin registry and configuration
      - mcp-profiles.ts: MCP profile management
      - connectors.ts: Connector catalog
      - chunking.ts: Chunking preview and comparison
      - models.ts: Embedding model catalog
      - projections.ts: UMAP/PCA projections
      - documents.ts: Document content retrieval
      - llm.ts: LLM settings and model API
      - assisted-flow.ts: Assisted-flow session start + stream URL (Claude Agent SDK)
      - pipeline.ts: Pipeline route preview, available predicate fields
      - templates.ts: Pipeline template listing
      - types.ts: API request/response types
    </modules>
    <pattern>
      All requests go through apiClient which auto-injects JWT and handles 401 -> refresh.
      Use handleApiError() for consistent error messages.
    </pattern>
  </dir>

  <dir path="src/pages/">
    <purpose>Route-level page components</purpose>
    <pages>
      - LoginPage.tsx: Authentication form
      - HomePage.tsx: Tab-based main view (Collections, Operations, Search)
      - SettingsPage.tsx: Settings tabs (Database, Plugins, MCP Profiles)
      - VerificationPage.tsx: Feature verification (dev only)
    </pages>
  </dir>

  <dir path="src/types/">
    <purpose>TypeScript type definitions</purpose>
    <files>
      - collection.ts: Collection, Operation, Sync, Source types
      - benchmark.ts: Benchmark, Dataset, Mapping, Run, Results types
      - chunking.ts: Chunking strategies, presets, configuration
      - plugin.ts: Plugin registry, config, status types
      - mcp-profile.ts: MCP profile types
      - connector.ts: Connector field definitions
      - projection.ts: Embedding projection types
      - llm.ts: LLM provider, tier, quantization, model types
      - pipeline.ts: PipelineDAG, PipelineNode, PipelineEdge, routing types
      - assisted-flow.ts: Assisted-flow start request/response + streaming event types
      - routePreview.ts: Route preview request/response, path visualization
      - wizard.ts: Wizard step types, mode selection
      - template.ts: Pipeline template types
    </files>
  </dir>

  <dir path="src/tests/">
    <purpose>Test utilities and MSW mocks</purpose>
    <files>
      - setup.ts: Vitest global setup with MSW
      - mocks/handlers.ts: MSW request handlers for all API endpoints
      - mocks/server.ts: MSW server instance
      - utils/test-utils.tsx: Custom render with providers
      - utils/TestWrapper.tsx: QueryClient + Router wrapper
    </files>
  </dir>

  <dir path="e2e/">
    <purpose>Playwright end-to-end tests</purpose>
    <config>playwright.config.ts - runs against localhost:5173</config>
  </dir>
</directory-structure>

<routing>
  <routes>
    /login -> LoginPage (public)
    /verification -> VerificationPage (dev only)
    / -> Layout > HomePage (protected, requires auth)
    /collections/:collectionId -> HomePage with CollectionDetailsModal open
    /settings -> Layout > SettingsPage (protected)
  </routes>
  <protection>ProtectedRoute wrapper checks authStore.token, redirects to /login</protection>
</routing>

<state-patterns>
  <react-query-keys>
    Use collectionKeys factory for consistent cache keys:
    - collectionKeys.all: ['collections']
    - collectionKeys.lists(): ['collections', 'list']
    - collectionKeys.detail(id): ['collections', 'detail', id]
    Other patterns: ['collection-operations', id], ['collection-documents', id]
  </react-query-keys>
  <optimistic-updates>
    useCreateCollection, useUpdateCollection, useDeleteCollection all implement
    optimistic updates with rollback on error. Follow this pattern for new mutations.
  </optimistic-updates>
  <cache-invalidation>
    WebSocket updates trigger cache invalidation via useUpdateCollectionInCache()
    and queryClient.invalidateQueries(). Always invalidate related queries.
  </cache-invalidation>
</state-patterns>

<websocket-integration>
  <global-socket>
    useOperationsSocket() in Layout.tsx listens to /ws/operations for real-time updates.
    Handles operation_started, operation_completed, operation_failed messages.
    Updates React Query cache directly via useUpdateOperationInCache().
  </global-socket>
  <auth-pattern>
    JWT token passed via WebSocket subprotocol header (Sec-WebSocket-Protocol).
    Connection managed by useWebSocket hook with auto-reconnect.
  </auth-pattern>
  <reconnection>
    5 attempts with 3s interval. Only reconnects on abnormal closure (not code 1000).
  </reconnection>
</websocket-integration>

<api-patterns>
  <authentication>
    - Token injection via Axios request interceptor
    - Auto token refresh on 401 (queues requests during refresh)
    - Logout clears auth-storage, sessionStorage, and React Query cache
  </authentication>
  <error-handling>
    - handleApiError() extracts detail from Axios error response
    - Toast notifications for user-facing errors via uiStore.addToast()
    - Error boundaries catch React render errors
  </error-handling>
  <base-url>
    Development: Vite proxy to localhost:8080
    Production: Same origin (static files served by backend)
  </base-url>
</api-patterns>

<connector-system>
  <overview>
    Dynamic UI for data sources (Directory, Git, IMAP, S3, etc.)
    Connector definitions come from /api/v2/connectors catalog.
    Forms render dynamically based on field definitions.
  </overview>
  <adding-new-connectors>
    Backend changes auto-populate the UI, but frontend may need:
    1. Icon mapping in ConnectorTypeSelector.tsx (connectorIcons)
    2. Display order in ConnectorTypeSelector.tsx (displayOrder)
    3. Short description in getShortDescription()
    4. Preview handler if connector supports preview
    5. Source path extraction in getSourcePath()
  </adding-new-connectors>
</connector-system>

<testing>
  <unit-tests>
    Run: npm test (vitest in watch mode)
    Run once: npm run test:ci
    Coverage: npm run test:coverage
    Pattern: *.test.tsx files colocated with components or in __tests__/
  </unit-tests>
  <e2e-tests>
    Run: npm run test:e2e (Playwright)
    Config: playwright.config.ts
    Browsers: Chromium, Firefox, WebKit
    Auto-starts Vite dev server
  </e2e-tests>
  <mocking>
    MSW (Mock Service Worker) for API mocking in tests.
    Handlers in src/tests/mocks/handlers.ts
    Server setup in src/tests/setup.ts
    Custom render wrapper includes QueryClientProvider + Router
  </mocking>
</testing>

<development>
  <commands>
    - Dev server: npm run dev (Vite on localhost:5173, proxies to localhost:8080)
    - Build: npm run build (outputs to packages/webui/static/)
    - Lint: npm run lint (ESLint)
    - Type check: tsc -b (via npm run build)
    - Unit tests: npm test
    - E2E tests: npm run test:e2e
    - Test collections: npm run test:collections (subset)
  </commands>
  <dev-server>
    Vite dev server at localhost:5173
    Proxies /api/* and /ws/* to localhost:8080 (backend)
    Hot module replacement enabled
  </dev-server>
  <build-output>
    Production build goes to packages/webui/static/
    Served by FastAPI backend as static files
    Source maps enabled for debugging
  </build-output>
</development>

<critical-patterns>
  <auth-access>
    ALWAYS use useAuthStore() hook to access auth state.
    NEVER read from localStorage directly.
  </auth-access>
  <query-keys>
    Use collectionKeys factory for collection-related queries.
    Consistent keys enable proper cache invalidation.
  </query-keys>
  <websocket-handling>
    Global socket in Layout.tsx handles operation updates.
    Don't create duplicate WebSocket connections for the same events.
  </websocket-handling>
  <error-boundaries>
    Wrap major UI sections in ErrorBoundary.
    Use ChunkingErrorBoundary for chunking-specific errors.
  </error-boundaries>
  <optimistic-ui>
    Mutations should implement optimistic updates with rollback.
    See useCreateCollection for the pattern.
  </optimistic-ui>
</critical-patterns>

<common-pitfalls>
  <pitfall>Direct localStorage access instead of authStore - breaks SSR, bypasses logout cleanup</pitfall>
  <pitfall>Not handling WebSocket reconnection - use useWebSocket hook</pitfall>
  <pitfall>Missing error boundaries around feature sections</pitfall>
  <pitfall>Forgetting optimistic updates for CRUD mutations</pitfall>
  <pitfall>Using wrong query keys leading to stale cache</pitfall>
  <pitfall>Not invalidating related queries after mutations</pitfall>
  <pitfall>Creating duplicate WebSocket connections</pitfall>
</common-pitfalls>

<performance>
  <optimizations>
    - React.lazy() for route-level code splitting
    - useMemo/useCallback for expensive computations
    - Debounced search input (prevents API spam)
    - Virtual scrolling for large lists (if needed)
    - LRU cache for projection data
    - Console drops in production build
  </optimizations>
  <bundle>
    Single chunk output (manualChunks: undefined)
    Source maps enabled for debugging
  </bundle>
</performance>

<recent-features>
  <pipeline-builder>Visual DAG editor for document processing pipelines with drag-to-connect, conditional routing, and route preview</pipeline-builder>
  <collection-wizard>Multi-step wizard with AI-assisted mode using agent streaming for intelligent pipeline configuration</collection-wizard>
  <benchmarks>Search quality benchmarking with datasets, config matrix, metrics (Precision@K, Recall@K, MRR, nDCG)</benchmarks>
  <llm-settings>LLM provider settings with local/cloud providers, quantization selection, memory display</llm-settings>
  <mcp-profiles>MCP profile management in Settings (Phase 3-4)</mcp-profiles>
  <plugins>Complete plugin system with install/configure UI</plugins>
  <sync>Collection sync (one-time and continuous) with source-level tracking</sync>
  <embedding-viz>UMAP/PCA visualization with WebGPU support</embedding-viz>
  <chunking>Multiple chunking strategies with preview and comparison</chunking>
</recent-features>

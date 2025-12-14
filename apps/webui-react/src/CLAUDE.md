<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding/removing components or stores
     - Changing state management patterns
     - Modifying API client methods
     - Altering WebSocket integration
     - Updating testing patterns
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>React Frontend</name>
  <purpose>React 19 SPA with TypeScript for Semantik UI</purpose>
  <location>apps/webui-react/src/</location>
</component>

<tech-stack>
  <core>React 19.1.0, TypeScript 5.8.3 (strict mode)</core>
  <state>Zustand 5.0.6 + React Query 5.81.5</state>
  <styling>TailwindCSS 3.4.17</styling>
  <build>Vite 7.0.0</build>
  <testing>Vitest + React Testing Library + MSW</testing>
</tech-stack>

<architecture>
  <pattern>Component composition with custom hooks</pattern>
  <state-management>
    - Zustand: Client state (auth, UI)
    - React Query: Server state (collections, search)
    - Local state: Component-specific UI
  </state-management>
  <routing>React Router v7 with protected routes</routing>
</architecture>

<directory-structure>
  <dir path="components/">
    <purpose>React components organized by feature</purpose>
    <subdirs>
      - chunking/: Chunking UI components
      - connectors/: Data source connector UI (Directory, Git, IMAP)
      - __tests__/: Component tests
    </subdirs>
    <key-components>
      - CollectionsDashboard: Main collection view
      - SearchInterface: Semantic search UI
      - OperationProgress: Real-time progress tracking
      - CreateCollectionModal: Collection creation with optional data source
      - AddDataToCollectionModal: Add data sources to existing collections
      - ConnectorTypeSelector: Card-based connector type picker
      - ConnectorForm: Dynamic form rendering based on connector definition
    </key-components>
  </dir>
  
  <dir path="stores/">
    <purpose>Zustand state stores</purpose>
    <stores>
      - authStore: JWT tokens, user state
      - collectionStore: Collection cache
      - searchStore: Search state and history
      - uiStore: UI preferences
      - chunkingStore: Chunking configuration
    </stores>
  </dir>
  
  <dir path="services/api/v2/">
    <purpose>Backend API client</purpose>
    <pattern>Axios with interceptors for auth</pattern>
    <modules>
      - client.ts: Axios configuration
      - collections.ts: Collection API
      - operations.ts: Operation tracking
      - search.ts: Search API
      - connectors.ts: Connector catalog and preview APIs
    </modules>
  </dir>
  
  <dir path="hooks/">
    <purpose>Custom React hooks</purpose>
    <key-hooks>
      - useWebSocket: WebSocket with auto-reconnect
      - useCollections: Collection data fetching
      - useOperationProgress: Real-time progress
      - useConnectorCatalog: Fetch available connector types
      - useGitPreview: Test Git repository connections
      - useImapPreview: Test IMAP server connections
    </key-hooks>
  </dir>
</directory-structure>

<connector-system>
  <overview>
    Dynamic connector UI for adding data sources (Directory, Git, IMAP).
    Connector definitions come from backend API at /api/v2/connectors.
    Forms render dynamically based on field definitions.
  </overview>
  <components>
    - ConnectorTypeSelector: Card-based picker with icons
    - ConnectorForm: Dynamic field rendering with validation
  </components>
  <adding-new-connectors>
    Backend changes auto-populate the UI, but frontend needs:
    1. Icon mapping in ConnectorTypeSelector.tsx (connectorIcons)
    2. Display order in ConnectorTypeSelector.tsx (displayOrder)
    3. Short description in getShortDescription()
    4. Preview handler in handlePreview() if connector supports preview
    5. Source path case in getSourcePath()
  </adding-new-connectors>
</connector-system>

<websocket-integration>
  <client>services/websocket.ts</client>
  <authentication>Send JWT after connection</authentication>
  <reconnection>Exponential backoff with jitter</reconnection>
</websocket-integration>

<api-patterns>
  <authentication>
    - Token injection via Axios interceptor
    - Auto-logout on 401
    - Token refresh on expiry
  </authentication>
  <error-handling>
    - Centralized error extraction
    - Toast notifications for errors
    - Graceful degradation
  </error-handling>
</api-patterns>

<testing>
  <setup>vitest.setup.ts with MSW</setup>
  <mocks>tests/mocks/handlers.ts</mocks>
  <patterns>
    - Render with custom wrapper
    - Mock API responses with MSW
    - Test user interactions
  </patterns>
</testing>

<common-pitfalls>
  <pitfall>Direct localStorage access instead of authStore</pitfall>
  <pitfall>Not handling WebSocket reconnection</pitfall>
  <pitfall>Missing error boundaries</pitfall>
  <pitfall>Forgetting optimistic updates</pitfall>
</common-pitfalls>

<performance>
  <optimizations>
    - Code splitting with lazy()
    - Memo for expensive renders
    - Virtual scrolling for large lists
    - Debounced search input
  </optimizations>
</performance>
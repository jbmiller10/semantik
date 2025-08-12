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
      - __tests__/: Component tests
    </subdirs>
    <key-components>
      - CollectionsDashboard: Main collection view
      - SearchInterface: Semantic search UI
      - OperationProgress: Real-time progress tracking
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
    </modules>
  </dir>
  
  <dir path="hooks/">
    <purpose>Custom React hooks</purpose>
    <key-hooks>
      - useWebSocket: WebSocket with auto-reconnect
      - useCollections: Collection data fetching
      - useOperationProgress: Real-time progress
    </key-hooks>
  </dir>
</directory-structure>

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
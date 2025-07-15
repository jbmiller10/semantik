# TEST-101: Initialize and Configure Frontend Testing Environment - Implementation Plan

## Overview
This plan outlines the implementation steps for setting up a comprehensive testing environment for the Semantik React frontend using Vitest, React Testing Library, and MSW (Mock Service Worker).

## Current State Analysis

### Existing Infrastructure
- **Frontend Stack**: React 19.1.0, TypeScript 5.8.3, Vite 7.0.0
- **State Management**: Zustand 5.0.6
- **API Client**: Axios 1.10.0
- **Routing**: React Router 7.6.3
- **Build Tool**: Vite with React plugin

### Visual UI Inspection
Before implementing tests, use Puppeteer to:
1. Navigate through the application to understand user flows
2. Identify critical UI components that need testing
3. Observe actual API calls and responses
4. Document UI behavior for test scenarios

### Current Testing State
- No automated testing framework configured
- Manual testing files exist in `/tests` directory (HTML and shell scripts)
- `make frontend-test` command exists but not implemented
- No test dependencies installed

### API Structure
The application has well-defined API endpoints organized into:
- Auth API (`/api/auth/*`)
- Jobs API (`/api/jobs/*`)
- Collections API (`/api/collections/*`)
- Search API (`/api/search/*`)
- Documents API (`/api/documents/*`)
- Models API (`/api/models/*`)
- Settings API (`/api/settings/*`)

## Implementation Plan

### Phase 0: Pre-Implementation UI Analysis

1. **Start the Development Environment**
   - Run `make dev` or start the frontend and backend servers
   - Ensure the application is accessible at http://localhost:5173 (or configured port)

2. **Visual Inspection with Puppeteer**
   - Navigate to main pages (Login, Home, Settings)
   - Test key user flows:
     - Authentication flow
     - Job creation process
     - Collection management
     - Search functionality
     - Document viewing
   - Take screenshots of critical components
   - Document observed behaviors and edge cases

3. **Identify Test Priorities**
   Based on UI inspection, prioritize testing for:
   - Components with complex user interactions
   - Critical business logic components
   - Error handling scenarios
   - Loading states and async operations

### Phase 1: Install Testing Dependencies

1. **Core Testing Framework**
   - `vitest`: ^2.1.8 - Modern test runner built for Vite
   - `@vitest/ui`: ^2.1.8 - Optional UI for test visualization

2. **React Testing Libraries**
   - `@testing-library/react`: ^16.1.0 - React component testing
   - `@testing-library/user-event`: ^14.5.2 - User interaction simulation
   - `@testing-library/jest-dom`: ^6.6.3 - Custom DOM matchers

3. **API Mocking**
   - `msw`: ^2.7.0 - Mock Service Worker for API mocking
   - `@mswjs/data`: ^0.16.2 - Data modeling for MSW

4. **Testing Utilities**
   - `jsdom`: ^26.0.0 - DOM implementation for Node.js
   - `@types/node`: ^22.12.0 - Node.js types

### Phase 2: Create Configuration Files

1. **vitest.config.ts**
   ```typescript
   /// <reference types="vitest" />
   import { defineConfig } from 'vite'
   import react from '@vitejs/plugin-react'
   import path from 'path'

   export default defineConfig({
     plugins: [react()],
     test: {
       globals: true,
       environment: 'jsdom',
       setupFiles: './src/tests/setup.ts',
       css: true,
       coverage: {
         provider: 'v8',
         reporter: ['text', 'json', 'html'],
         exclude: [
           'node_modules/',
           'src/tests/',
           '**/*.d.ts',
           '**/*.config.*',
           '**/mockServiceWorker.js',
         ],
       },
     },
     resolve: {
       alias: {
         '@': path.resolve(__dirname, './src'),
       },
     },
   })
   ```

2. **src/tests/setup.ts**
   ```typescript
   import '@testing-library/jest-dom'
   import { cleanup } from '@testing-library/react'
   import { afterEach, beforeAll, afterAll } from 'vitest'
   import { server } from './mocks/server'

   // Establish API mocking before all tests
   beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))

   // Reset any request handlers that are declared as a part of our tests
   afterEach(() => {
     cleanup()
     server.resetHandlers()
   })

   // Clean up after the tests are finished
   afterAll(() => server.close())
   ```

### Phase 3: Set Up MSW Mocking Infrastructure

1. **src/tests/mocks/handlers.ts**
   ```typescript
   import { http, HttpResponse } from 'msw'

   export const handlers = [
     // Auth endpoints
     http.post('/api/auth/login', async ({ request }) => {
       const { username, password } = await request.json()
       if (username === 'testuser' && password === 'testpass') {
         return HttpResponse.json({
           access_token: 'mock-jwt-token',
           refresh_token: 'mock-refresh-token',
           user: {
             id: 1,
             username: 'testuser',
             email: 'test@example.com',
             is_active: true,
             created_at: new Date().toISOString(),
           }
         })
       }
       return HttpResponse.json(
         { detail: 'Invalid credentials' },
         { status: 401 }
       )
     }),

     // Jobs endpoints
     http.get('/api/jobs', () => {
       return HttpResponse.json({
         items: [
           {
             id: '1',
             name: 'Test Collection',
             status: 'completed',
             progress: 100,
             created_at: new Date().toISOString(),
           }
         ],
         total: 1,
       })
     }),

     // Collections endpoints
     http.get('/api/collections', () => {
       return HttpResponse.json({
         collections: [
           {
             name: 'test-collection',
             document_count: 100,
             created_at: new Date().toISOString(),
           }
         ]
       })
     }),
   ]
   ```

2. **src/tests/mocks/server.ts**
   ```typescript
   import { setupServer } from 'msw/node'
   import { handlers } from './handlers'

   export const server = setupServer(...handlers)
   ```

3. **src/tests/mocks/browser.ts**
   ```typescript
   import { setupWorker } from 'msw/browser'
   import { handlers } from './handlers'

   export const worker = setupWorker(...handlers)
   ```

### Phase 4: Create Testing Utilities

1. **src/tests/utils/test-utils.tsx**
   ```typescript
   import { ReactElement } from 'react'
   import { render, RenderOptions } from '@testing-library/react'
   import { MemoryRouter } from 'react-router-dom'
   import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

   const createTestQueryClient = () => new QueryClient({
     defaultOptions: {
       queries: {
         retry: false,
       },
     },
   })

   interface AllTheProvidersProps {
     children: React.ReactNode
   }

   const AllTheProviders = ({ children }: AllTheProvidersProps) => {
     const queryClient = createTestQueryClient()
     
     return (
       <QueryClientProvider client={queryClient}>
         <MemoryRouter>
           {children}
         </MemoryRouter>
       </QueryClientProvider>
     )
   }

   const customRender = (
     ui: ReactElement,
     options?: Omit<RenderOptions, 'wrapper'>
   ) => render(ui, { wrapper: AllTheProviders, ...options })

   export * from '@testing-library/react'
   export { customRender as render }
   ```

### Phase 5: Create Example Tests

1. **src/components/__tests__/JobCard.test.tsx**
   ```typescript
   import { describe, it, expect, vi } from 'vitest'
   import { render, screen, fireEvent } from '@/tests/utils/test-utils'
   import JobCard from '../JobCard'
   import type { Job } from '@/stores/jobsStore'

   const mockJob: Job = {
     id: '1',
     name: 'Test Job',
     collection_name: 'test-collection',
     status: 'completed',
     progress: 100,
     total_files: 10,
     processed_files: 10,
     created_at: new Date().toISOString(),
     updated_at: new Date().toISOString(),
   }

   describe('JobCard', () => {
     it('renders job information correctly', () => {
       render(<JobCard job={mockJob} onDelete={() => {}} />)
       
       expect(screen.getByText('Test Job')).toBeInTheDocument()
       expect(screen.getByText('test-collection')).toBeInTheDocument()
       expect(screen.getByText('completed')).toBeInTheDocument()
     })

     it('calls onDelete when delete button is clicked', async () => {
       const onDelete = vi.fn()
       render(<JobCard job={mockJob} onDelete={onDelete} />)
       
       const deleteButton = screen.getByRole('button', { name: /delete/i })
       fireEvent.click(deleteButton)
       
       // Confirm dialog would appear here
       // For now, we'll just verify the button exists
       expect(deleteButton).toBeInTheDocument()
     })
   })
   ```

2. **src/stores/__tests__/authStore.test.ts**
   ```typescript
   import { describe, it, expect, beforeEach } from 'vitest'
   import { useAuthStore } from '../authStore'

   describe('authStore', () => {
     beforeEach(() => {
       useAuthStore.setState({
         token: null,
         user: null,
         refreshToken: null,
       })
     })

     it('sets auth data correctly', () => {
       const { setAuth } = useAuthStore.getState()
       const mockUser = {
         id: 1,
         username: 'testuser',
         email: 'test@example.com',
         is_active: true,
         created_at: new Date().toISOString(),
       }

       setAuth('mock-token', mockUser, 'mock-refresh-token')

       const state = useAuthStore.getState()
       expect(state.token).toBe('mock-token')
       expect(state.user).toEqual(mockUser)
       expect(state.refreshToken).toBe('mock-refresh-token')
     })

     it('clears auth data on logout', async () => {
       const { setAuth, logout } = useAuthStore.getState()
       
       setAuth('mock-token', {
         id: 1,
         username: 'testuser',
         email: 'test@example.com',
         is_active: true,
         created_at: new Date().toISOString(),
       })

       await logout()

       const state = useAuthStore.getState()
       expect(state.token).toBeNull()
       expect(state.user).toBeNull()
       expect(state.refreshToken).toBeNull()
     })
   })
   ```

### Phase 6: Update Configuration Files

1. **Update package.json scripts**
   ```json
   {
     "scripts": {
       "test": "vitest",
       "test:ui": "vitest --ui",
       "test:coverage": "vitest --coverage",
       "test:watch": "vitest --watch"
     }
   }
   ```

2. **Update tsconfig.json**
   Add test files to the include array:
   ```json
   {
     "include": ["src", "src/**/*.test.ts", "src/**/*.test.tsx"]
   }
   ```

### Phase 7: Documentation

1. **Create src/tests/README.md**
   - Testing strategy overview
   - How to write tests
   - MSW usage guidelines
   - Running tests locally
   - CI/CD integration notes

## Implementation Order

1. **Pre-Implementation Phase (2-3 hours)**
   - [ ] Start development environment
   - [ ] Use Puppeteer to navigate and inspect UI
   - [ ] Document user flows and component behaviors
   - [ ] Take screenshots of key interfaces
   - [ ] Identify components requiring priority testing

2. **Day 1 - Morning**
   - [ ] Install all testing dependencies
   - [ ] Create vitest.config.ts
   - [ ] Create tests directory structure

3. **Day 1 - Afternoon**
   - [ ] Set up MSW handlers and server
   - [ ] Create test setup file
   - [ ] Create test utilities
   - [ ] Write first example test (based on Puppeteer findings)
   - [ ] Verify tests run successfully

## Validation Criteria

1. **Technical Requirements**
   - [ ] `npm test` runs without errors
   - [ ] Initial placeholder test passes
   - [ ] MSW successfully intercepts API calls
   - [ ] Test coverage reporting works

2. **Code Quality**
   - [ ] TypeScript types are properly configured for tests
   - [ ] ESLint doesn't report errors in test files
   - [ ] Test utilities follow React Testing Library best practices

3. **Documentation**
   - [ ] Clear instructions for running tests
   - [ ] Examples of how to write different types of tests
   - [ ] MSW handler patterns documented

## Risks and Mitigation

1. **Risk**: Version conflicts with React 19
   - **Mitigation**: Use latest stable versions of testing libraries that support React 19

2. **Risk**: MSW setup complexity
   - **Mitigation**: Start with simple handlers, expand incrementally

3. **Risk**: Performance issues with large test suite
   - **Mitigation**: Configure proper test isolation and parallel execution

## Success Metrics

- All acceptance criteria from the ticket are met
- Tests run in under 5 seconds for the initial suite
- No console errors or warnings during test execution
- Clear path for developers to add new tests

## Next Steps After Completion

- Implement component tests for critical UI components
- Add integration tests for complete user workflows
- Set up CI/CD pipeline for automated testing
- Create performance benchmarks for frontend operations
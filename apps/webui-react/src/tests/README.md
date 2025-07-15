# Frontend Testing Guide

## Overview
This directory contains the testing infrastructure for the Semantik React frontend, using Vitest, React Testing Library, and MSW (Mock Service Worker).

## Running Tests

```bash
# Run all tests once
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage
```

## Test Structure

```
src/
├── tests/
│   ├── setup.ts          # Global test setup
│   ├── mocks/
│   │   ├── handlers.ts   # MSW request handlers
│   │   ├── server.ts     # MSW server for Node
│   │   └── browser.ts    # MSW worker for browser
│   └── utils/
│       └── test-utils.tsx # Custom render with providers
├── components/
│   └── __tests__/        # Component tests
└── stores/
    └── __tests__/        # Store tests
```

## Writing Tests

### Component Test Example

```typescript
import { render, screen } from '@/tests/utils/test-utils'
import MyComponent from '../MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Expected Text')).toBeInTheDocument()
  })
})
```

### Store Test Example

```typescript
import { useMyStore } from '../myStore'

describe('myStore', () => {
  beforeEach(() => {
    useMyStore.setState({ /* initial state */ })
  })

  it('updates state correctly', () => {
    const { updateValue } = useMyStore.getState()
    updateValue('new value')
    
    expect(useMyStore.getState().value).toBe('new value')
  })
})
```

### API Mocking

MSW handlers are defined in `mocks/handlers.ts`. To override handlers for specific tests:

```typescript
import { server } from '@/tests/mocks/server'
import { http, HttpResponse } from 'msw'

it('handles error response', async () => {
  server.use(
    http.get('/api/endpoint', () => {
      return HttpResponse.json({ error: 'Error' }, { status: 500 })
    })
  )
  
  // Test error handling
})
```

## Best Practices

1. **Use the custom render**: Always import render from `@/tests/utils/test-utils`
2. **Test user behavior**: Focus on what users see and do, not implementation details
3. **Mock at the network level**: Use MSW instead of mocking modules
4. **Reset state between tests**: Use beforeEach to ensure test isolation
5. **Use data-testid sparingly**: Prefer accessible queries (role, label, text)

## Common Testing Patterns

### Testing Async Operations

```typescript
import { waitFor } from '@testing-library/react'

it('loads data', async () => {
  render(<DataComponent />)
  
  // Wait for async operation
  await waitFor(() => {
    expect(screen.getByText('Loaded Data')).toBeInTheDocument()
  })
})
```

### Testing User Interactions

```typescript
import userEvent from '@testing-library/user-event'

it('handles click', async () => {
  const user = userEvent.setup()
  render(<Button onClick={handleClick}>Click me</Button>)
  
  await user.click(screen.getByRole('button'))
  
  expect(handleClick).toHaveBeenCalled()
})
```

### Testing with React Query

The test utils automatically wrap components with QueryClientProvider. For testing queries:

```typescript
it('fetches data', async () => {
  render(<QueryComponent />)
  
  // Initial loading state
  expect(screen.getByText('Loading...')).toBeInTheDocument()
  
  // Wait for data
  await waitFor(() => {
    expect(screen.getByText('Data loaded')).toBeInTheDocument()
  })
})
```

## Debugging Tests

- Use `screen.debug()` to print the current DOM
- Use `screen.logTestingPlaygroundURL()` to get a testing playground link
- Run specific tests: `npm test -- MyComponent.test.tsx`
- Use `test.only()` or `describe.only()` to focus on specific tests
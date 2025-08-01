import { QueryClient } from '@tanstack/react-query'

// Create a new QueryClient for each test to ensure test isolation
export const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      staleTime: Infinity,
    },
    mutations: {
      retry: false,
    },
  },
})
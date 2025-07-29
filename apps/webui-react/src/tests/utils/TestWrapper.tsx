import React from 'react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Toast from '../../components/Toast'

export const TestWrapper = ({ children }: { children: React.ReactNode }) => {
  // Create a new QueryClient for each test to avoid caching issues
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { 
        retry: false,
        cacheTime: 0,
        staleTime: 0,
      },
      mutations: { retry: false }
    }
  })

  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        {children}
        <Toast />
      </MemoryRouter>
    </QueryClientProvider>
  )
}
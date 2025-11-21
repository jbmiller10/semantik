import React from 'react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createTestQueryClient } from './queryClient'

interface AllTheProvidersProps {
  children: React.ReactNode
  initialEntries?: string[]
  queryClient?: QueryClient
}

// Wrapper component that includes all necessary providers
export const AllTheProviders = ({ children, initialEntries = ['/'], queryClient }: AllTheProvidersProps) => {
  const clientRef = React.useRef<QueryClient | null>(null)

  if (!clientRef.current) {
    clientRef.current = queryClient ?? createTestQueryClient()
  }
  
  return (
    <QueryClientProvider client={clientRef.current}>
      <MemoryRouter initialEntries={initialEntries}>
        {children}
      </MemoryRouter>
    </QueryClientProvider>
  )
}

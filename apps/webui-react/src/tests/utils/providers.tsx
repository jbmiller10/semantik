import React from 'react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClientProvider } from '@tanstack/react-query'
import { createTestQueryClient } from './queryClient'

interface AllTheProvidersProps {
  children: React.ReactNode
  initialEntries?: string[]
}

// Wrapper component that includes all necessary providers
export const AllTheProviders = ({ children, initialEntries = ['/'] }: AllTheProvidersProps) => {
  const queryClient = createTestQueryClient()
  
  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={initialEntries}>
        {children}
      </MemoryRouter>
    </QueryClientProvider>
  )
}
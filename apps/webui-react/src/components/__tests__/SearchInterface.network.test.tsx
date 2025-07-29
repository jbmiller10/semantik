import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SearchInterface from '../SearchInterface'
import { renderWithProviders } from '../../tests/utils/test-utils'
import { server } from '../../tests/mocks/server'
import { http, HttpResponse } from 'msw'

describe('SearchInterface - Network Error Handling', () => {
  it('should show error toast when search fails due to network error', async () => {
    // Override the search handler to return a network error
    server.use(
      http.post('/api/v2/search', () => {
        return HttpResponse.error()
      })
    )

    renderWithProviders(<SearchInterface />)

    // Enter search query
    const searchInput = screen.getByPlaceholderText(/Enter your search query/i)
    await userEvent.type(searchInput, 'test query')
    
    // Open collection dropdown
    const collectionDropdown = screen.getByRole('button', { name: /select collections/i })
    await userEvent.click(collectionDropdown)
    
    // Select a collection
    const collection = await screen.findByText('Test Collection 1')
    await userEvent.click(collection)
    
    // Close dropdown by clicking outside
    await userEvent.click(searchInput)
    
    // Submit search
    const searchButton = screen.getByRole('button', { name: /search/i })
    await userEvent.click(searchButton)
    
    // Should show error message
    await waitFor(() => {
      expect(screen.getByText(/search failed/i)).toBeInTheDocument()
    })
  })

  it('should handle server errors gracefully', async () => {
    // Override the search handler to return a 500 error
    server.use(
      http.post('/api/v2/search', () => {
        return HttpResponse.json(
          { detail: 'Internal server error' },
          { status: 500 }
        )
      })
    )

    renderWithProviders(<SearchInterface />)

    // Enter search query
    const searchInput = screen.getByPlaceholderText(/Enter your search query/i)
    await userEvent.type(searchInput, 'test query')
    
    // Open collection dropdown
    const collectionDropdown = screen.getByRole('button', { name: /select collections/i })
    await userEvent.click(collectionDropdown)
    
    // Select a collection
    const collection = await screen.findByText('Test Collection 1')
    await userEvent.click(collection)
    
    // Close dropdown
    await userEvent.click(searchInput)
    
    // Submit search
    const searchButton = screen.getByRole('button', { name: /search/i })
    await userEvent.click(searchButton)
    
    // Should show error message
    await waitFor(() => {
      expect(screen.getByText(/search failed/i)).toBeInTheDocument()
    })
  })

  it('should handle partial failures', async () => {
    // Override the search handler to return partial failure
    server.use(
      http.post('/api/v2/search', () => {
        return HttpResponse.json({
          results: [
            {
              document_id: 'doc_1',
              chunk_id: 'chunk_1',
              score: 0.9,
              text: 'Test result',
              file_path: '/test.txt',
              file_name: 'test.txt',
              collection_id: '123e4567-e89b-12d3-a456-426614174000',
              collection_name: 'Test Collection 1',
            }
          ],
          total_results: 1,
          partial_failure: true,
          failed_collections: [
            {
              collection_id: '456e7890-e89b-12d3-a456-426614174001',
              collection_name: 'Test Collection 2',
              error: 'Vector index corrupted'
            }
          ],
          search_time_ms: 100,
          total_time_ms: 150,
        })
      })
    )

    renderWithProviders(<SearchInterface />)

    // Enter search query
    const searchInput = screen.getByPlaceholderText(/Enter your search query/i)
    await userEvent.type(searchInput, 'test query')
    
    // Open collection dropdown
    const collectionDropdown = screen.getByRole('button', { name: /select collections/i })
    await userEvent.click(collectionDropdown)
    
    // Select both collections
    const collection1 = await screen.findByText('Test Collection 1')
    await userEvent.click(collection1)
    
    const collection2 = await screen.findByText('Test Collection 2')
    await userEvent.click(collection2)
    
    // Close dropdown
    await userEvent.click(searchInput)
    
    // Submit search
    const searchButton = screen.getByRole('button', { name: /search/i })
    await userEvent.click(searchButton)
    
    // Should show warning about partial failure
    await waitFor(() => {
      expect(screen.getByText(/search completed with.*failing/i)).toBeInTheDocument()
    })
  })

  it('should disable search button when no collections selected', () => {
    renderWithProviders(<SearchInterface />)

    const searchButton = screen.getByRole('button', { name: /search/i })
    expect(searchButton).toBeDisabled()
  })

  it('should require search query', async () => {
    renderWithProviders(<SearchInterface />)

    // Open collection dropdown and select a collection
    const collectionDropdown = screen.getByRole('button', { name: /select collections/i })
    await userEvent.click(collectionDropdown)
    
    const collection = await screen.findByText('Test Collection 1')
    await userEvent.click(collection)
    
    // Try to search without query
    const searchButton = screen.getByRole('button', { name: /search/i })
    await userEvent.click(searchButton)
    
    // Should show error about missing query
    await waitFor(() => {
      expect(screen.getByText(/please enter a search query/i)).toBeInTheDocument()
    })
  })
})
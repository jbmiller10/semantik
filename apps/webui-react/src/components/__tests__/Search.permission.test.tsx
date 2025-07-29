import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SearchInterface from '../SearchInterface'
import SearchResults from '../SearchResults'
import { useSearchStore } from '../../stores/searchStore'
import { useCollectionStore } from '../../stores/collectionStore'
import { useUIStore } from '../../stores/uiStore'
import { 
  renderWithErrorHandlers, 
  waitForToast
} from '../../tests/utils/errorTestUtils'
import { 
  documentErrorHandlers
} from '../../tests/mocks/errorHandlers'
import { server } from '../../tests/mocks/server'
import type { SearchResult } from '../../stores/searchStore'

// Mock stores
vi.mock('../../stores/searchStore')
vi.mock('../../stores/collectionStore')
vi.mock('../../stores/uiStore')

describe('Search - Permission Error Handling', () => {
  const mockSearch = vi.fn()
  const mockAddToast = vi.fn()
  const mockSetShowDocumentViewer = vi.fn()
  
  const mockPublicCollection = {
    uuid: 'public-coll',
    name: 'Public Collection',
    status: 'ready',
    is_public: true,
    document_count: 100,
    vector_count: 1000,
    embedding_model: 'test-model'
  }
  
  const mockPrivateCollection = {
    uuid: 'private-coll',
    name: 'Private Collection',
    status: 'ready',
    is_public: false,
    user_id: 999, // Different user
    document_count: 50,
    vector_count: 500,
    embedding_model: 'test-model'
  }

  beforeEach(() => {
    vi.clearAllMocks()
    
    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast,
      setShowDocumentViewer: mockSetShowDocumentViewer,
      showDocumentViewer: null
    } as ReturnType<typeof useUIStore>)
  })

  describe('Collection Search Permissions', () => {
    it('should handle searching in collections without permission', async () => {
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [mockPublicCollection, mockPrivateCollection],
        fetchCollections: vi.fn(),
        loading: false,
        error: null
      } as ReturnType<typeof useCollectionStore>)
      
      vi.mocked(useSearchStore).mockReturnValue({
        searchParams: {
          query: '',
          selectedCollections: [],
          limit: 10,
          searchType: 'hybrid'
        },
        isSearching: false,
        error: null,
        performSearch: mockSearch,
        setSearchParams: vi.fn(),
        results: [],
        totalResults: 0,
        partialFailure: false,
        failedCollections: []
      } as ReturnType<typeof useSearchStore>)
      
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )
      
      // Select both collections
      const multiSelect = screen.getByText(/select collections/i)
      await userEvent.click(multiSelect)
      
      // Should show both collections but indicate private status
      expect(await screen.findByText('Public Collection')).toBeInTheDocument()
      expect(screen.getByText('Private Collection')).toBeInTheDocument()
      
      // Select private collection
      await userEvent.click(screen.getByText('Private Collection'))
      
      // Enter search query
      await userEvent.type(screen.getByPlaceholderText(/search/i), 'test')
      
      // Mock search to return permission error for private collection
      mockSearch.mockImplementation(async () => {
        vi.mocked(useSearchStore).mockReturnValue({
          ...vi.mocked(useSearchStore).mock.results[0].value,
          partialFailure: true,
          failedCollections: [{
            collection_id: 'private-coll',
            collection_name: 'Private Collection',
            error: 'Access denied to collection'
          }],
          results: [] // No results from private collection
        } as ReturnType<typeof useSearchStore>)
      })
      
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      await waitForToast('Search completed with errors', 'warning')
    })

    it('should filter out inaccessible collections from selection', async () => {
      // Collections with different access levels
      const collections = [
        mockPublicCollection,
        { ...mockPrivateCollection, can_access: false },
        {
          uuid: 'shared-coll',
          name: 'Shared Collection',
          status: 'ready',
          is_public: false,
          can_access: true, // Shared with current user
          document_count: 75,
          vector_count: 750,
          embedding_model: 'test-model'
        }
      ]
      
      vi.mocked(useCollectionStore).mockReturnValue({
        collections,
        fetchCollections: vi.fn(),
        loading: false,
        error: null
      } as ReturnType<typeof useCollectionStore>)
      
      vi.mocked(useSearchStore).mockReturnValue({
        searchParams: { selectedCollections: [] },
        isSearching: false,
        error: null,
        performSearch: mockSearch,
        setSearchParams: vi.fn()
      } as Partial<ReturnType<typeof useSearchStore>> as ReturnType<typeof useSearchStore>)
      
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )
      
      const multiSelect = screen.getByText(/select collections/i)
      await userEvent.click(multiSelect)
      
      // Should show accessible collections
      expect(await screen.findByText('Public Collection')).toBeInTheDocument()
      expect(screen.getByText('Shared Collection')).toBeInTheDocument()
      
      // Should not show or disable inaccessible collection
      const privateOption = screen.queryByText('Private Collection')
      if (privateOption) {
        // If shown, should be disabled
        expect(privateOption.closest('button')).toBeDisabled()
      }
    })
  })

  describe('Document Access Permissions', () => {
    it('should handle permission error when viewing document', async () => {
      const searchResults = [
        {
          collection_id: 'coll-1',
          collection_name: 'Test Collection',
          chunk_id: 1,
          content: 'Test content',
          score: 0.9,
          file_path: '/restricted/doc.txt',
          file_name: 'doc.txt'
        }
      ]
      
      vi.mocked(useSearchStore).mockReturnValue({
        results: searchResults,
        totalResults: 1,
        isSearching: false,
        error: null,
        searchTime: 1.0,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: 'test' }
      } as ReturnType<typeof useSearchStore>)
      
      server.use(
        documentErrorHandlers.permissionError()[0]
      )
      
      renderWithErrorHandlers(
        <SearchResults />,
        []
      )
      
      // Click to view document
      const viewButton = screen.getByRole('button', { name: /view.*document/i })
      await userEvent.click(viewButton)
      
      // Mock the document viewer request failing
      await waitFor(() => {
        expect(mockSetShowDocumentViewer).toHaveBeenCalledWith({
          documentPath: '/restricted/doc.txt',
          collectionId: 'coll-1'
        })
      })
      
      await waitForToast('Access denied to document', 'error')
    })

    it('should show different UI for documents user cannot access', async () => {
      const mixedResults = [
        {
          collection_id: 'public-coll',
          collection_name: 'Public Collection',
          chunk_id: 1,
          content: 'Public content',
          score: 0.9,
          file_path: '/public/doc.txt',
          can_access_document: true
        },
        {
          collection_id: 'private-coll',
          collection_name: 'Private Collection',
          chunk_id: 2,
          content: '[Preview not available - insufficient permissions]',
          score: 0.85,
          file_path: '/private/secret.txt',
          can_access_document: false
        }
      ]
      
      vi.mocked(useSearchStore).mockReturnValue({
        results: mixedResults as SearchResult[],
        totalResults: 2,
        isSearching: false,
        error: null,
        searchTime: 1.5,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: 'test' }
      } as ReturnType<typeof useSearchStore>)
      
      renderWithErrorHandlers(
        <SearchResults />,
        []
      )
      
      // Public document should have view button
      const publicResult = screen.getByText('Public content')
      const publicViewButton = publicResult.closest('div')?.querySelector('button[aria-label*="view"]')
      expect(publicViewButton).toBeInTheDocument()
      
      // Private document should show restricted message
      expect(screen.getByText(/preview not available/i)).toBeInTheDocument()
      
      // Private document might not have view button or it's disabled
      const privateResult = screen.getByText(/insufficient permissions/i)
      const privateViewButton = privateResult.closest('div')?.querySelector('button[aria-label*="view"]')
      
      if (privateViewButton) {
        expect(privateViewButton).toBeDisabled()
      }
    })
  })

  describe('Cross-User Search Permissions', () => {
    it('should handle searching across mixed ownership collections', async () => {
      const searchResults = [
        {
          collection_id: 'own-coll',
          collection_name: 'My Collection',
          chunk_id: 1,
          content: 'My content',
          score: 0.95,
          file_path: '/my/doc.txt',
          owner_id: 1 // Current user
        },
        {
          collection_id: 'public-coll',
          collection_name: 'Public Collection',
          chunk_id: 2,
          content: 'Public content',
          score: 0.9,
          file_path: '/public/doc.txt',
          owner_id: 2,
          is_public: true
        }
      ]
      
      const failedPrivate = {
        collection_id: 'other-private',
        collection_name: 'Other User Private',
        error: 'Permission denied'
      }
      
      vi.mocked(useSearchStore).mockReturnValue({
        results: searchResults as SearchResult[],
        totalResults: 2,
        isSearching: false,
        error: null,
        searchTime: 2.0,
        partialFailure: true,
        failedCollections: [failedPrivate],
        searchParams: { query: 'test' }
      } as ReturnType<typeof useSearchStore>)
      
      renderWithErrorHandlers(
        <SearchResults />,
        []
      )
      
      // Should show successful results
      expect(screen.getByText('My content')).toBeInTheDocument()
      expect(screen.getByText('Public content')).toBeInTheDocument()
      
      // Should show permission error for failed collection
      const alert = screen.getByRole('alert')
      expect(alert).toHaveTextContent('Other User Private')
      expect(alert).toHaveTextContent('Permission denied')
    })
  })

  describe('API Key Search Permissions', () => {
    it('should handle API key permission scope limitations', async () => {
      // Simulate using an API key with limited scope
      mockSearch.mockRejectedValue({
        response: {
          status: 403,
          data: { 
            detail: 'API key does not have search permission. Required scope: collections:read' 
          }
        }
      })
      
      vi.mocked(useSearchStore).mockReturnValue({
        searchParams: { 
          query: 'test',
          selectedCollections: ['coll-1']
        },
        isSearching: false,
        error: null,
        performSearch: mockSearch,
        setError: vi.fn()
      } as Partial<ReturnType<typeof useSearchStore>> as ReturnType<typeof useSearchStore>)
      
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [mockPublicCollection],
        fetchCollections: vi.fn()
      } as Partial<ReturnType<typeof useCollectionStore>> as ReturnType<typeof useCollectionStore>)
      
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )
      
      await userEvent.type(screen.getByPlaceholderText(/search/i), 'test')
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      await waitForToast('API key does not have search permission', 'error')
    })

    it('should handle rate-limited API key', async () => {
      mockSearch.mockRejectedValue({
        response: {
          status: 429,
          data: { 
            detail: 'API key rate limit exceeded. 100 requests per hour allowed.' 
          }
        }
      })
      
      vi.mocked(useSearchStore).mockReturnValue({
        searchParams: { query: 'test' },
        isSearching: false,
        error: null,
        performSearch: mockSearch,
        setError: vi.fn()
      } as Partial<ReturnType<typeof useSearchStore>> as ReturnType<typeof useSearchStore>)
      
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [mockPublicCollection],
        fetchCollections: vi.fn()
      } as Partial<ReturnType<typeof useCollectionStore>> as ReturnType<typeof useCollectionStore>)
      
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )
      
      await userEvent.type(screen.getByPlaceholderText(/search/i), 'test')
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      await waitForToast('API key rate limit exceeded', 'error')
      
      // Should show helpful message about rate limits
      expect(mockAddToast).toHaveBeenCalledWith(
        {
    message: expect.stringContaining('100 requests per hour'),
    type: 'error'
  })
    })
  })

  describe('Search Result Filtering', () => {
    it('should automatically filter out results user lost access to', async () => {
      // Initial results
      const initialResults = [
        {
          collection_id: 'coll-1',
          collection_name: 'Collection 1',
          chunk_id: 1,
          content: 'Result 1',
          score: 0.9
        },
        {
          collection_id: 'coll-2',
          collection_name: 'Collection 2',
          chunk_id: 2,
          content: 'Result 2',
          score: 0.85
        }
      ]
      
      vi.mocked(useSearchStore).mockReturnValue({
        results: initialResults as SearchResult[],
        totalResults: 2,
        isSearching: false,
        error: null,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: 'test' }
      } as ReturnType<typeof useSearchStore>)
      
      const { rerender } = renderWithErrorHandlers(
        <SearchResults />,
        []
      )
      
      expect(screen.getByText('Result 1')).toBeInTheDocument()
      expect(screen.getByText('Result 2')).toBeInTheDocument()
      
      // Simulate permission change - user loses access to collection 2
      vi.mocked(useSearchStore).mockReturnValue({
        results: [initialResults[0]] as SearchResult[],
        totalResults: 1,
        isSearching: false,
        error: null,
        partialFailure: true,
        failedCollections: [{
          collection_id: 'coll-2',
          collection_name: 'Collection 2',
          error: 'Access revoked'
        }],
        searchParams: { query: 'test' }
      } as any)
      
      rerender(<SearchResults />)
      
      // Should only show accessible result
      expect(screen.getByText('Result 1')).toBeInTheDocument()
      expect(screen.queryByText('Result 2')).not.toBeInTheDocument()
      
      // Should show permission revoked message
      expect(screen.getByText(/access revoked/i)).toBeInTheDocument()
    })
  })
})
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Routes, Route, useParams } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import ActiveOperationsTab from '../ActiveOperationsTab'
import { operationsV2Api } from '../../services/api/v2/collections'
import { useCollections } from '../../hooks/useCollections'
import { useOperationProgress } from '../../hooks/useOperationProgress'
import { useUIStore } from '../../stores/uiStore'
import { createMockCollectionsQuery } from '../../tests/types/hook-mocks'
import { createMockCollection } from '../../tests/types/test-types'

vi.mock('../../services/api/v2/collections', () => ({
  operationsV2Api: {
    list: vi.fn(),
  },
}))

vi.mock('../../hooks/useCollections')
vi.mock('../../hooks/useOperationProgress', () => ({
  useOperationProgress: vi.fn(() => ({
    isConnected: false,
    readyState: 0,
    sendMessage: vi.fn(),
  })),
}))

function CollectionRoute() {
  const { collectionId } = useParams<{ collectionId: string }>()
  return <div data-testid="collection-route">{collectionId}</div>
}

const renderComponent = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        cacheTime: 0,
        staleTime: 0,
      },
    },
  })

  const user = userEvent.setup()

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route path="/" element={<ActiveOperationsTab />} />
          <Route path="/collections/:collectionId" element={<CollectionRoute />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )

  return { user }
}

describe('ActiveOperationsTab navigation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useUIStore.setState({
      activeTab: 'operations',
      showCollectionDetailsModal: null,
    })

    vi.mocked(useOperationProgress).mockReturnValue({
      isConnected: false,
      readyState: 0,
      sendMessage: vi.fn(),
    })

    vi.mocked(useCollections).mockReturnValue(
      createMockCollectionsQuery({
        data: [
          createMockCollection({ id: 'coll-1', name: 'Collection 1' }),
        ],
      }) as unknown as ReturnType<typeof useCollections>
    )

    vi.mocked(operationsV2Api.list).mockResolvedValue({
      data: [
        {
          id: 'op-1',
          collection_id: 'coll-1',
          type: 'index',
          status: 'processing',
          config: {},
          created_at: new Date().toISOString(),
        },
      ],
    })
  })

  afterEach(() => {
    useUIStore.setState({
      activeTab: 'collections',
      showCollectionDetailsModal: null,
    })
  })

  it('routes to the collection details path when a collection name is clicked', async () => {
    const { user } = renderComponent()

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Collection 1' })).toBeInTheDocument()
    })

    await user.click(screen.getByRole('button', { name: 'Collection 1' }))

    await waitFor(() => {
      expect(screen.getByTestId('collection-route')).toHaveTextContent('coll-1')
    })

    const uiState = useUIStore.getState()
    expect(uiState.activeTab).toBe('collections')
    expect(uiState.showCollectionDetailsModal).toBe('coll-1')
  })
})

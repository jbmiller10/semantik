import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import { shallow } from 'zustand/shallow'
import HomePage from '../HomePage'
import { useUIStore } from '@/stores/uiStore'

vi.mock('@/stores/uiStore')

const mockedUseUIStore = useUIStore as unknown as vi.Mock

const createStoreState = (overrides: Partial<ReturnType<typeof useUIStore>> = {}) => ({
  activeTab: 'search',
  setActiveTab: vi.fn(),
  setShowCollectionDetailsModal: vi.fn(),
  showCollectionDetailsModal: null,
  ...overrides,
})

const mockUIStore = (overrides: Partial<ReturnType<typeof useUIStore>> = {}) => {
  const state = createStoreState(overrides)

  mockedUseUIStore.mockImplementation(
    (
      selector?: (store: typeof state) => unknown,
      equalityFn?: typeof shallow
    ) => (typeof selector === 'function' ? selector(state) : state)
  )

  return state
}

vi.mock('@/components/SearchInterface', () => ({
  default: () => <div data-testid="search-interface">Search Interface</div>,
}))
vi.mock('@/components/CollectionsDashboard', () => ({
  default: () => <div data-testid="collections-dashboard">Collections Dashboard</div>,
}))
vi.mock('@/components/ActiveOperationsTab', () => ({
  default: () => <div data-testid="active-operations-tab">Active Operations Tab</div>,
}))

describe('HomePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders SearchInterface when activeTab is search', () => {
    mockUIStore({ activeTab: 'search' })

    render(<HomePage />)

    expect(screen.getByTestId('search-interface')).toBeInTheDocument()
    expect(screen.queryByTestId('collections-dashboard')).not.toBeInTheDocument()
    expect(screen.queryByTestId('active-operations-tab')).not.toBeInTheDocument()
  })

  it('renders CollectionsDashboard when activeTab is collections', () => {
    mockUIStore({ activeTab: 'collections' })

    render(<HomePage />)

    expect(screen.getByTestId('collections-dashboard')).toBeInTheDocument()
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument()
    expect(screen.queryByTestId('active-operations-tab')).not.toBeInTheDocument()
  })

  it('renders ActiveOperationsTab when activeTab is operations', () => {
    mockUIStore({ activeTab: 'operations' })

    render(<HomePage />)

    expect(screen.getByTestId('active-operations-tab')).toBeInTheDocument()
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument()
    expect(screen.queryByTestId('collections-dashboard')).not.toBeInTheDocument()
  })

  it('renders nothing when activeTab is an unknown value', () => {
    const state = mockUIStore({ activeTab: 'unknown' as never })

    const { container } = render(<HomePage />)

    expect(container.firstChild).toBeNull()
    expect(state.setShowCollectionDetailsModal).not.toHaveBeenCalled()
  })

  it('subscribes to the store using a selector with shallow comparison', () => {
    const state = mockUIStore({ activeTab: 'search' })

    render(<HomePage />)

    expect(mockedUseUIStore).toHaveBeenCalledWith(expect.any(Function), shallow)

    const [[selector]] = mockedUseUIStore.mock.calls as [
      [(store: typeof state) => unknown, typeof shallow]
    ]
    const selected = selector(state)
    expect(selected).toMatchObject({
      activeTab: 'search',
      setActiveTab: state.setActiveTab,
      setShowCollectionDetailsModal: state.setShowCollectionDetailsModal,
    })
  })
})

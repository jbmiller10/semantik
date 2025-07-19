import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import HomePage from '../HomePage'
import { useUIStore } from '@/stores/uiStore'

// Mock the UI store
vi.mock('@/stores/uiStore')

// Mock the components
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
    ;(useUIStore as any).mockImplementation((selector: any) => {
      const state = { activeTab: 'search' }
      return selector ? selector(state) : state
    })

    render(<HomePage />)
    
    expect(screen.getByTestId('search-interface')).toBeInTheDocument()
    expect(screen.queryByTestId('collections-dashboard')).not.toBeInTheDocument()
    expect(screen.queryByTestId('active-operations-tab')).not.toBeInTheDocument()
  })

  it('renders CollectionsDashboard when activeTab is collections', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => {
      const state = { activeTab: 'collections' }
      return selector ? selector(state) : state
    })

    render(<HomePage />)
    
    expect(screen.getByTestId('collections-dashboard')).toBeInTheDocument()
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument()
    expect(screen.queryByTestId('active-operations-tab')).not.toBeInTheDocument()
  })

  it('renders ActiveOperationsTab when activeTab is operations', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => {
      const state = { activeTab: 'operations' }
      return selector ? selector(state) : state
    })

    render(<HomePage />)
    
    expect(screen.getByTestId('active-operations-tab')).toBeInTheDocument()
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument()
    expect(screen.queryByTestId('collections-dashboard')).not.toBeInTheDocument()
  })

  it('renders nothing when activeTab is an unknown value', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => {
      const state = { activeTab: 'unknown' }
      return selector ? selector(state) : state
    })

    const { container } = render(<HomePage />)
    
    // HomePage renders a React Fragment, so when empty, container.firstChild is null
    expect(container.firstChild).toBeNull()
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument()
    expect(screen.queryByTestId('collections-dashboard')).not.toBeInTheDocument()
    expect(screen.queryByTestId('active-operations-tab')).not.toBeInTheDocument()
  })

  it('uses the correct store selector', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => selector({ activeTab: 'search' }))

    render(<HomePage />)
    
    // Verify the component renders correctly with the selector
    expect(screen.getByTestId('search-interface')).toBeInTheDocument()
  })
})
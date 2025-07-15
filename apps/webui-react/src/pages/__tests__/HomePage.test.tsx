import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import HomePage from '../HomePage'
import { useUIStore } from '@/stores/uiStore'

// Mock the UI store
vi.mock('@/stores/uiStore')

// Mock the components
vi.mock('@/components/CreateJobForm', () => ({
  default: () => <div data-testid="create-job-form">Create Job Form</div>,
}))
vi.mock('@/components/JobList', () => ({
  default: () => <div data-testid="job-list">Job List</div>,
}))
vi.mock('@/components/SearchInterface', () => ({
  default: () => <div data-testid="search-interface">Search Interface</div>,
}))
vi.mock('@/components/CollectionList', () => ({
  default: () => <div data-testid="collection-list">Collection List</div>,
}))

describe('HomePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders CreateJobForm when activeTab is create', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => {
      const state = { activeTab: 'create' }
      return selector ? selector(state) : state
    })

    render(<HomePage />)
    
    expect(screen.getByTestId('create-job-form')).toBeInTheDocument()
    expect(screen.queryByTestId('job-list')).not.toBeInTheDocument()
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument()
    expect(screen.queryByTestId('collection-list')).not.toBeInTheDocument()
  })

  it('renders JobList when activeTab is jobs', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => {
      const state = { activeTab: 'jobs' }
      return selector ? selector(state) : state
    })

    render(<HomePage />)
    
    expect(screen.getByTestId('job-list')).toBeInTheDocument()
    expect(screen.queryByTestId('create-job-form')).not.toBeInTheDocument()
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument()
    expect(screen.queryByTestId('collection-list')).not.toBeInTheDocument()
  })

  it('renders SearchInterface when activeTab is search', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => {
      const state = { activeTab: 'search' }
      return selector ? selector(state) : state
    })

    render(<HomePage />)
    
    expect(screen.getByTestId('search-interface')).toBeInTheDocument()
    expect(screen.queryByTestId('create-job-form')).not.toBeInTheDocument()
    expect(screen.queryByTestId('job-list')).not.toBeInTheDocument()
    expect(screen.queryByTestId('collection-list')).not.toBeInTheDocument()
  })

  it('renders CollectionList when activeTab is collections', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => {
      const state = { activeTab: 'collections' }
      return selector ? selector(state) : state
    })

    render(<HomePage />)
    
    expect(screen.getByTestId('collection-list')).toBeInTheDocument()
    expect(screen.queryByTestId('create-job-form')).not.toBeInTheDocument()
    expect(screen.queryByTestId('job-list')).not.toBeInTheDocument()
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument()
  })

  it('renders nothing when activeTab is an unknown value', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => {
      const state = { activeTab: 'unknown' }
      return selector ? selector(state) : state
    })

    const { container } = render(<HomePage />)
    
    expect(container.firstChild).toBeEmptyDOMElement()
    expect(screen.queryByTestId('create-job-form')).not.toBeInTheDocument()
    expect(screen.queryByTestId('job-list')).not.toBeInTheDocument()
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument()
    expect(screen.queryByTestId('collection-list')).not.toBeInTheDocument()
  })

  it('uses the correct store selector', () => {
    ;(useUIStore as any).mockImplementation((selector: any) => selector({ activeTab: 'create' }))

    render(<HomePage />)
    
    // Verify the component renders correctly with the selector
    expect(screen.getByTestId('create-job-form')).toBeInTheDocument()
  })
})
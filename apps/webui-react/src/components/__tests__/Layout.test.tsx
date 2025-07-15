import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import { useNavigate, useLocation } from 'react-router-dom'
import Layout from '../Layout'
import { useAuthStore } from '@/stores/authStore'
import { useUIStore } from '@/stores/uiStore'

// Mock router
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: vi.fn(),
    useLocation: vi.fn(),
    Outlet: () => <div data-testid="outlet">Page Content</div>,
  }
})

// Mock stores
vi.mock('@/stores/authStore')
vi.mock('@/stores/uiStore')

// Mock child components
vi.mock('../Toast', () => ({
  default: () => <div data-testid="toast">Toast</div>,
}))
vi.mock('../DocumentViewerModal', () => ({
  default: () => <div data-testid="document-viewer-modal">DocumentViewerModal</div>,
}))
vi.mock('../JobMetricsModal', () => ({
  default: () => <div data-testid="job-metrics-modal">JobMetricsModal</div>,
}))
vi.mock('../CollectionDetailsModal', () => ({
  default: () => <div data-testid="collection-details-modal">CollectionDetailsModal</div>,
}))

describe('Layout', () => {
  const mockNavigate = vi.fn()
  const mockLogout = vi.fn()
  const mockSetActiveTab = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    
    ;(useNavigate as any).mockReturnValue(mockNavigate)
    ;(useLocation as any).mockReturnValue({
      pathname: '/',
    })
    
    ;(useAuthStore as any).mockReturnValue({
      user: { username: 'testuser' },
      logout: mockLogout,
    })
    
    ;(useUIStore as any).mockReturnValue({
      activeTab: 'create',
      setActiveTab: mockSetActiveTab,
    })
    
    // Mock import.meta.env
    vi.stubGlobal('import', {
      meta: {
        env: {
          DEV: false,
        },
      },
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders header with app title and user info', () => {
    render(<Layout />)
    
    expect(screen.getByText('Semantik')).toBeInTheDocument()
    expect(screen.getByText('Document Embedding Pipeline')).toBeInTheDocument()
    expect(screen.getByText('testuser')).toBeInTheDocument()
  })

  it('renders navigation tabs on home page', () => {
    render(<Layout />)
    
    expect(screen.getByRole('button', { name: 'Create Job' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Jobs' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Search' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Collections' })).toBeInTheDocument()
  })

  it('highlights active tab correctly', () => {
    ;(useUIStore as any).mockReturnValue({
      activeTab: 'search',
      setActiveTab: mockSetActiveTab,
    })
    
    render(<Layout />)
    
    const searchTab = screen.getByRole('button', { name: 'Search' })
    expect(searchTab).toHaveClass('border-blue-500', 'text-blue-600')
    
    const createTab = screen.getByRole('button', { name: 'Create Job' })
    expect(createTab).toHaveClass('border-transparent', 'text-gray-500')
  })

  it('handles tab switching', async () => {
    const user = userEvent.setup()
    
    render(<Layout />)
    
    const jobsTab = screen.getByRole('button', { name: 'Jobs' })
    await user.click(jobsTab)
    
    expect(mockSetActiveTab).toHaveBeenCalledWith('jobs')
  })

  it('renders all modals and toast', () => {
    render(<Layout />)
    
    expect(screen.getByTestId('toast')).toBeInTheDocument()
    expect(screen.getByTestId('document-viewer-modal')).toBeInTheDocument()
    expect(screen.getByTestId('job-metrics-modal')).toBeInTheDocument()
    expect(screen.getByTestId('collection-details-modal')).toBeInTheDocument()
  })

  it('renders outlet for page content', () => {
    render(<Layout />)
    
    expect(screen.getByTestId('outlet')).toBeInTheDocument()
  })

  it('handles logout', async () => {
    const user = userEvent.setup()
    
    render(<Layout />)
    
    const logoutButton = screen.getByRole('button', { name: 'Logout' })
    await user.click(logoutButton)
    
    expect(mockLogout).toHaveBeenCalled()
    await vi.waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/login')
    })
  })

  it('shows settings link on home page', () => {
    render(<Layout />)
    
    const settingsLink = screen.getByRole('link', { name: 'Settings' })
    expect(settingsLink).toBeInTheDocument()
    expect(settingsLink).toHaveAttribute('href', '/settings')
  })

  it('shows back link on settings page', () => {
    ;(useLocation as any).mockReturnValue({
      pathname: '/settings',
    })
    
    render(<Layout />)
    
    const backLink = screen.getByRole('link', { name: '← Back' })
    expect(backLink).toBeInTheDocument()
    expect(backLink).toHaveAttribute('href', '/')
  })

  it('hides navigation tabs on settings page', () => {
    ;(useLocation as any).mockReturnValue({
      pathname: '/settings',
    })
    
    render(<Layout />)
    
    expect(screen.queryByRole('button', { name: 'Create Job' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Jobs' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Search' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Collections' })).not.toBeInTheDocument()
  })

  it('shows verification link in development mode', () => {
    vi.stubGlobal('import', {
      meta: {
        env: {
          DEV: true,
        },
      },
    })
    
    render(<Layout />)
    
    const verificationLink = screen.getByRole('link', { name: 'Verification' })
    expect(verificationLink).toBeInTheDocument()
    expect(verificationLink).toHaveAttribute('href', '/verification')
  })

  it('hides verification link in production mode', () => {
    vi.stubGlobal('import', {
      meta: {
        env: {
          DEV: false,
        },
      },
    })
    
    render(<Layout />)
    
    expect(screen.queryByRole('link', { name: 'Verification' })).not.toBeInTheDocument()
  })

  it('handles all tab clicks correctly', async () => {
    const user = userEvent.setup()
    
    render(<Layout />)
    
    // Test Create Job tab
    await user.click(screen.getByRole('button', { name: 'Create Job' }))
    expect(mockSetActiveTab).toHaveBeenCalledWith('create')
    
    // Test Jobs tab
    await user.click(screen.getByRole('button', { name: 'Jobs' }))
    expect(mockSetActiveTab).toHaveBeenCalledWith('jobs')
    
    // Test Search tab
    await user.click(screen.getByRole('button', { name: 'Search' }))
    expect(mockSetActiveTab).toHaveBeenCalledWith('search')
    
    // Test Collections tab
    await user.click(screen.getByRole('button', { name: 'Collections' }))
    expect(mockSetActiveTab).toHaveBeenCalledWith('collections')
  })

  it('renders with correct layout structure', () => {
    render(<Layout />)
    
    // Check main layout structure
    expect(document.querySelector('.min-h-screen.bg-gray-100')).toBeInTheDocument()
    expect(document.querySelector('header.bg-white.shadow-sm.border-b')).toBeInTheDocument()
    expect(document.querySelector('main.max-w-7xl.mx-auto')).toBeInTheDocument()
  })
})
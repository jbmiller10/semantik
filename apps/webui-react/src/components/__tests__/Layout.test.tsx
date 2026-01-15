import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
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
vi.mock('../CollectionDetailsModal', () => ({
  default: () => <div data-testid="collection-details-modal">CollectionDetailsModal</div>,
}))

describe('Layout', () => {
  const mockNavigate = vi.fn()
  const mockLogout = vi.fn()
  const mockSetActiveTab = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()

      ; (useNavigate as ReturnType<typeof vi.fn>).mockReturnValue(mockNavigate)
      ; (useLocation as ReturnType<typeof vi.fn>).mockReturnValue({
        pathname: '/',
      })

      ; (useAuthStore as ReturnType<typeof vi.fn>).mockReturnValue({
        user: { username: 'testuser' },
        logout: mockLogout,
      })

      ; (useUIStore as ReturnType<typeof vi.fn>).mockReturnValue({
        activeTab: 'collections',
        setActiveTab: mockSetActiveTab,
      })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders header with app title and user info', () => {
    render(<Layout />)

    expect(screen.getByText('Semantik')).toBeInTheDocument()
    expect(screen.getByText('Document Pipeline')).toBeInTheDocument()
    expect(screen.getByText('testuser')).toBeInTheDocument()
  })

  it('renders navigation tabs on home page', () => {
    render(<Layout />)

    expect(screen.getByRole('button', { name: 'Search' })).toBeInTheDocument()
  })

  it('highlights active tab correctly', () => {
    ; (useUIStore as ReturnType<typeof vi.fn>).mockReturnValue({
      activeTab: 'search',
      setActiveTab: mockSetActiveTab,
    })

    render(<Layout />)

    const searchTab = screen.getByRole('button', { name: 'Search' })
    expect(searchTab).toHaveClass('text-white', 'border-signal-500')
  })

  it('handles tab switching', async () => {
    const user = userEvent.setup()

    render(<Layout />)

    const searchTab = screen.getByRole('button', { name: 'Search' })
    await user.click(searchTab)

    expect(mockSetActiveTab).toHaveBeenCalledWith('search')
  })

  it('handles logout click', async () => {
    const user = userEvent.setup()

    render(<Layout />)

    const logoutButton = screen.getByRole('button', { name: /logout/i })
    await user.click(logoutButton)

    expect(mockLogout).toHaveBeenCalled()
    expect(mockNavigate).toHaveBeenCalledWith('/login')
  })

  it('does not show navigation tabs on non-home pages', () => {
    ; (useLocation as ReturnType<typeof vi.fn>).mockReturnValue({
      pathname: '/settings',
    })

    render(<Layout />)

    expect(screen.queryByRole('button', { name: 'Search' })).not.toBeInTheDocument()
  })

  it('renders all modals and toast', () => {
    render(<Layout />)

    expect(screen.getByTestId('toast')).toBeInTheDocument()
    expect(screen.getByTestId('document-viewer-modal')).toBeInTheDocument()
    expect(screen.getByTestId('collection-details-modal')).toBeInTheDocument()
  })

  it('renders outlet for page content', () => {
    render(<Layout />)

    expect(screen.getByTestId('outlet')).toBeInTheDocument()
  })

  it('shows back link on settings page', () => {
    ; (useLocation as ReturnType<typeof vi.fn>).mockReturnValue({
      pathname: '/settings',
    })

    render(<Layout />)

    const backLink = screen.getByText('← Back')
    expect(backLink).toBeInTheDocument()
    expect(backLink).toHaveAttribute('href', '/')
  })

  it('shows settings link on home page', () => {
    render(<Layout />)

    const settingsLink = screen.getByRole('link', { name: 'Settings' })
    expect(settingsLink).toBeInTheDocument()
    expect(settingsLink).toHaveAttribute('href', '/settings')
  })

  it('handles all tab clicks correctly', async () => {
    const user = userEvent.setup()

    render(<Layout />)

    const searchTab = screen.getByRole('button', { name: 'Search' })
    await user.click(searchTab)

    expect(mockSetActiveTab).toHaveBeenLastCalledWith('search')
  })
})
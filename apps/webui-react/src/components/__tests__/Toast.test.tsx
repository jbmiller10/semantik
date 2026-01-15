import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import Toast from '../Toast'
import { useUIStore } from '@/stores/uiStore'

// Mock the UI store
vi.mock('@/stores/uiStore')

describe('Toast', () => {
  const mockRemoveToast = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders nothing when there are no toasts', () => {
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    const { container } = render(<Toast />)
    
    expect(container.firstChild).toBeNull()
  })

  it('renders error toast correctly', () => {
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [
        {
          id: '1',
          type: 'error',
          message: 'Something went wrong!',
        },
      ],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    render(<Toast />)
    
    expect(screen.getByText('Error')).toBeInTheDocument()
    expect(screen.getByText('Something went wrong!')).toBeInTheDocument()
    
    // Check for error styling (red border)
    const toastElement = screen.getByText('Error').closest('.p-4')?.parentElement
    expect(toastElement).toHaveClass('border-l-4')
    expect(toastElement).toHaveClass('border-red-500')
  })

  it('renders success toast correctly', () => {
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [
        {
          id: '2',
          type: 'success',
          message: 'Operation completed successfully!',
        },
      ],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    render(<Toast />)
    
    expect(screen.getByText('Success')).toBeInTheDocument()
    expect(screen.getByText('Operation completed successfully!')).toBeInTheDocument()
    
    // Check for success styling (green border)
    const toastElement = screen.getByText('Success').closest('.p-4')?.parentElement
    expect(toastElement).toHaveClass('border-l-4')
    expect(toastElement).toHaveClass('border-green-500')
  })

  it('renders warning toast correctly', () => {
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [
        {
          id: '3',
          type: 'warning',
          message: 'Please be careful!',
        },
      ],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    render(<Toast />)
    
    expect(screen.getByText('Warning')).toBeInTheDocument()
    expect(screen.getByText('Please be careful!')).toBeInTheDocument()
    
    // Check for warning styling (yellow border)
    const toastElement = screen.getByText('Warning').closest('.p-4')?.parentElement
    expect(toastElement).toHaveClass('border-l-4')
    expect(toastElement).toHaveClass('border-yellow-500')
  })

  it('renders info toast correctly', () => {
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [
        {
          id: '4',
          type: 'info',
          message: 'Here is some information.',
        },
      ],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    render(<Toast />)

    expect(screen.getByText('Info')).toBeInTheDocument()
    expect(screen.getByText('Here is some information.')).toBeInTheDocument()

    // Check for info styling (signal border for dark theme)
    const toastElement = screen.getByText('Info').closest('.p-4')?.parentElement
    expect(toastElement).toHaveClass('border-l-4')
    expect(toastElement).toHaveClass('border-signal-500')
  })

  it('renders multiple toasts', () => {
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [
        {
          id: '1',
          type: 'error',
          message: 'Error message',
        },
        {
          id: '2',
          type: 'success',
          message: 'Success message',
        },
        {
          id: '3',
          type: 'info',
          message: 'Info message',
        },
      ],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    render(<Toast />)
    
    expect(screen.getByText('Error')).toBeInTheDocument()
    expect(screen.getByText('Error message')).toBeInTheDocument()
    expect(screen.getByText('Success')).toBeInTheDocument()
    expect(screen.getByText('Success message')).toBeInTheDocument()
    expect(screen.getByText('Info')).toBeInTheDocument()
    expect(screen.getByText('Info message')).toBeInTheDocument()
  })

  it('calls removeToast when close button is clicked', async () => {
    const user = userEvent.setup()
    
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [
        {
          id: 'toast-1',
          type: 'error',
          message: 'Test error',
        },
      ],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    render(<Toast />)
    
    // Find the close button (the X icon)
    const closeButton = screen.getByRole('button')
    await user.click(closeButton)
    
    expect(mockRemoveToast).toHaveBeenCalledWith('toast-1')
  })

  it('handles multiple close buttons correctly', async () => {
    const user = userEvent.setup()
    
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [
        {
          id: 'toast-1',
          type: 'error',
          message: 'Error 1',
        },
        {
          id: 'toast-2',
          type: 'success',
          message: 'Success 1',
        },
      ],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    render(<Toast />)
    
    // Find all close buttons
    const closeButtons = screen.getAllByRole('button')
    expect(closeButtons).toHaveLength(2)
    
    // Click the second toast's close button
    await user.click(closeButtons[1])
    
    expect(mockRemoveToast).toHaveBeenCalledWith('toast-2')
    expect(mockRemoveToast).not.toHaveBeenCalledWith('toast-1')
  })

  it('renders with correct positioning classes', () => {
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [
        {
          id: '1',
          type: 'info',
          message: 'Test',
        },
      ],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    render(<Toast />)
    
    // Check for fixed positioning at bottom right
    const container = screen.getByText('Info').closest('.fixed')
    expect(container).toHaveClass('fixed', 'bottom-0', 'right-0', 'p-6', 'z-50')
  })

  it('renders toast with all expected classes', () => {
    vi.mocked(useUIStore).mockReturnValue({
      toasts: [
        {
          id: '1',
          type: 'info',
          message: 'Test',
        },
      ],
      removeToast: mockRemoveToast,
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      addToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })

    render(<Toast />)

    const toastElement = screen.getByText('Info').closest('.p-4')?.parentElement
    expect(toastElement).toHaveClass('max-w-sm')
    expect(toastElement).toHaveClass('w-full')
    expect(toastElement).toHaveClass('bg-void-900')
    expect(toastElement).toHaveClass('shadow-lg')
    expect(toastElement).toHaveClass('rounded-lg')
    expect(toastElement).toHaveClass('pointer-events-auto')
    expect(toastElement).toHaveClass('ring-1')
    expect(toastElement).toHaveClass('ring-void-700')
    expect(toastElement).toHaveClass('overflow-hidden')
  })
})
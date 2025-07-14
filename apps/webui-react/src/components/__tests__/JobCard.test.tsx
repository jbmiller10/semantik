import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import JobCard from '../JobCard'
import { useUIStore } from '@/stores/uiStore'
import type { Job } from '@/stores/jobsStore'

// Mock the UI store
vi.mock('@/stores/uiStore', () => ({
  useUIStore: vi.fn(),
}))

// Mock the job progress hook to avoid WebSocket connections in tests
vi.mock('@/hooks/useJobProgress', () => ({
  useJobProgress: vi.fn(),
}))

const mockJob: Job = {
  id: '1',
  name: 'Test Job',
  directory: '/test/path',
  directory_path: '/test/path',
  collection_name: 'test-collection',
  status: 'completed',
  progress: 100,
  total_files: 10,
  total_documents: 10,
  processed_files: 10,
  processed_documents: 10,
  failed_files: 0,
  created_at: '2025-01-14T12:00:00Z',
  updated_at: '2025-01-14T12:30:00Z',
}

describe('JobCard', () => {
  const mockAddToast = vi.fn()
  const mockSetShowJobMetricsModal = vi.fn()
  const mockOnDelete = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    ;(useUIStore as any).mockReturnValue({
      addToast: mockAddToast,
      setShowJobMetricsModal: mockSetShowJobMetricsModal,
    })
  })

  it('renders job information correctly', () => {
    render(<JobCard job={mockJob} onDelete={mockOnDelete} />)
    
    expect(screen.getByText('Test Job')).toBeInTheDocument()
    expect(screen.getByText('test-collection')).toBeInTheDocument()
    expect(screen.getByText(/completed/i)).toBeInTheDocument()
    expect(screen.getByText(/10 files/i)).toBeInTheDocument()
  })

  it('shows progress bar for active jobs', () => {
    const activeJob = { ...mockJob, status: 'processing' as const, progress: 50 }
    render(<JobCard job={activeJob} onDelete={mockOnDelete} />)
    
    const progressBar = screen.getByRole('progressbar')
    expect(progressBar).toBeInTheDocument()
    expect(progressBar).toHaveAttribute('aria-valuenow', '50')
  })

  it('handles delete action with confirmation', async () => {
    const user = userEvent.setup()
    
    // Mock window.confirm
    const mockConfirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    render(<JobCard job={mockJob} onDelete={mockOnDelete} />)
    
    const deleteButton = screen.getByRole('button', { name: /delete/i })
    await user.click(deleteButton)
    
    expect(mockConfirm).toHaveBeenCalledWith('Are you sure you want to delete this job?')
    
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Job deleted successfully',
      })
      expect(mockOnDelete).toHaveBeenCalled()
    })
    
    mockConfirm.mockRestore()
  })

  it('cancels delete when confirmation is rejected', async () => {
    const user = userEvent.setup()
    
    // Mock window.confirm to return false
    const mockConfirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    
    render(<JobCard job={mockJob} onDelete={mockOnDelete} />)
    
    const deleteButton = screen.getByRole('button', { name: /delete/i })
    await user.click(deleteButton)
    
    expect(mockConfirm).toHaveBeenCalled()
    expect(mockOnDelete).not.toHaveBeenCalled()
    expect(mockAddToast).not.toHaveBeenCalled()
    
    mockConfirm.mockRestore()
  })

  it('shows cancel button for active jobs', async () => {
    const activeJob = { ...mockJob, status: 'processing' as const }
    render(<JobCard job={activeJob} onDelete={mockOnDelete} />)
    
    const cancelButton = screen.getByRole('button', { name: /cancel/i })
    expect(cancelButton).toBeInTheDocument()
  })

  it('handles cancel action for active jobs', async () => {
    const user = userEvent.setup()
    const activeJob = { ...mockJob, status: 'processing' as const }
    
    // Mock window.confirm
    const mockConfirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    render(<JobCard job={activeJob} onDelete={mockOnDelete} />)
    
    const cancelButton = screen.getByRole('button', { name: /cancel/i })
    await user.click(cancelButton)
    
    expect(mockConfirm).toHaveBeenCalledWith(
      expect.stringContaining('Are you sure you want to cancel')
    )
    
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'warning',
        message: expect.stringContaining('cancellation requested'),
      })
    })
    
    mockConfirm.mockRestore()
  })

  it('displays error message for failed jobs', () => {
    const failedJob = {
      ...mockJob,
      status: 'failed' as const,
      error: 'Test error message',
    }
    
    render(<JobCard job={failedJob} onDelete={mockOnDelete} />)
    
    expect(screen.getByText(/failed/i)).toBeInTheDocument()
    expect(screen.getByText('Test error message')).toBeInTheDocument()
  })

  it('shows metrics button that opens metrics modal', async () => {
    const user = userEvent.setup()
    
    render(<JobCard job={mockJob} onDelete={mockOnDelete} />)
    
    const metricsButton = screen.getByRole('button', { name: /metrics/i })
    await user.click(metricsButton)
    
    expect(mockSetShowJobMetricsModal).toHaveBeenCalledWith(true)
  })
})
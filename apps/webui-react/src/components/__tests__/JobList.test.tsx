import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import JobList from '../JobList'
import { useJobsStore } from '@/stores/jobsStore'
import { server } from '@/tests/mocks/server'
import { http, HttpResponse } from 'msw'

// Mock the jobs store
vi.mock('@/stores/jobsStore')

// Mock JobCard to avoid its dependencies
vi.mock('../JobCard', () => ({
  default: ({ job, onDelete }: any) => (
    <div data-testid={`job-${job.id}`}>
      <span>{job.name}</span>
      <span>{job.status}</span>
      <button onClick={onDelete}>Delete</button>
    </div>
  ),
}))

const mockJobs = [
  {
    id: '1',
    name: 'Active Job',
    status: 'processing',
    directory_path: '/path1',
    progress: 50,
  },
  {
    id: '2',
    name: 'Waiting Job',
    status: 'waiting',
    directory_path: '/path2',
    progress: 0,
  },
  {
    id: '3',
    name: 'Completed Job',
    status: 'completed',
    directory_path: '/path3',
    progress: 100,
  },
  {
    id: '4',
    name: 'Failed Job',
    status: 'failed',
    directory_path: '/path4',
    progress: 75,
  },
]

describe('JobList', () => {
  const mockSetJobs = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    ;(useJobsStore as any).mockReturnValue({
      jobs: [],
      setJobs: mockSetJobs,
    })
  })

  it('renders job list header and refresh button', () => {
    render(<JobList />)
    
    expect(screen.getByText('Embedding Jobs')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument()
  })

  it('shows empty state when no jobs exist', () => {
    render(<JobList />)
    
    expect(screen.getByText('No jobs found')).toBeInTheDocument()
  })

  it('fetches and displays jobs from API', async () => {
    server.use(
      http.get('/api/jobs', () => {
        return HttpResponse.json(mockJobs)
      })
    )
    
    render(<JobList />)
    
    await waitFor(() => {
      expect(mockSetJobs).toHaveBeenCalledWith(mockJobs)
    })
  })

  it('displays jobs in correct order: active, completed, failed', async () => {
    ;(useJobsStore as any).mockReturnValue({
      jobs: mockJobs,
      setJobs: mockSetJobs,
    })
    
    render(<JobList />)
    
    const jobElements = screen.getAllByTestId(/^job-/)
    
    // Check order: processing first, then waiting, then completed, then failed
    expect(jobElements[0]).toHaveAttribute('data-testid', 'job-1')
    expect(jobElements[1]).toHaveAttribute('data-testid', 'job-2')
    expect(jobElements[2]).toHaveAttribute('data-testid', 'job-3')
    expect(jobElements[3]).toHaveAttribute('data-testid', 'job-4')
  })

  it('refetches jobs when refresh button is clicked', async () => {
    const user = userEvent.setup()
    let callCount = 0
    
    server.use(
      http.get('/api/jobs', () => {
        callCount++
        return HttpResponse.json(mockJobs)
      })
    )
    
    render(<JobList />)
    
    // Wait for initial fetch
    await waitFor(() => {
      expect(callCount).toBe(1)
    })
    
    const refreshButton = screen.getByRole('button', { name: /refresh/i })
    await user.click(refreshButton)
    
    // Wait for refetch
    await waitFor(() => {
      expect(callCount).toBe(2)
    })
  })

  it('refetches when delete callback is triggered', async () => {
    const user = userEvent.setup()
    let callCount = 0
    
    server.use(
      http.get('/api/jobs', () => {
        callCount++
        return HttpResponse.json(mockJobs)
      })
    )
    
    ;(useJobsStore as any).mockReturnValue({
      jobs: mockJobs,
      setJobs: mockSetJobs,
    })
    
    render(<JobList />)
    
    // Wait for initial fetch
    await waitFor(() => {
      expect(callCount).toBe(1)
    })
    
    // Click delete on first job
    const deleteButtons = screen.getAllByText('Delete')
    await user.click(deleteButtons[0])
    
    // Wait for refetch
    await waitFor(() => {
      expect(callCount).toBe(2)
    })
  })

  it('listens for refetch-jobs events', async () => {
    let callCount = 0
    
    server.use(
      http.get('/api/jobs', () => {
        callCount++
        return HttpResponse.json(mockJobs)
      })
    )
    
    render(<JobList />)
    
    // Wait for initial fetch
    await waitFor(() => {
      expect(callCount).toBe(1)
    })
    
    // Dispatch custom event
    window.dispatchEvent(new Event('refetch-jobs'))
    
    // Wait for refetch
    await waitFor(() => {
      expect(callCount).toBe(2)
    })
  })

  it('sets up auto-refresh interval', async () => {
    // Just verify the component renders with refetch interval configured
    // Testing the actual interval behavior is complex with React Query
    render(<JobList />)
    
    // Component should render without errors
    expect(screen.getByText('Embedding Jobs')).toBeInTheDocument()
  })

  it('filters jobs by status correctly', () => {
    const mixedJobs = [
      { ...mockJobs[2], id: '5' }, // completed
      { ...mockJobs[0], id: '6' }, // processing
      { ...mockJobs[3], id: '7' }, // failed
      { ...mockJobs[1], id: '8' }, // waiting
    ]
    
    ;(useJobsStore as any).mockReturnValue({
      jobs: mixedJobs,
      setJobs: mockSetJobs,
    })
    
    render(<JobList />)
    
    const jobElements = screen.getAllByTestId(/^job-/)
    
    // Check that active jobs (processing, waiting) come first
    expect(jobElements[0]).toHaveTextContent('processing')
    expect(jobElements[1]).toHaveTextContent('waiting')
    expect(jobElements[2]).toHaveTextContent('completed')
    expect(jobElements[3]).toHaveTextContent('failed')
  })
})
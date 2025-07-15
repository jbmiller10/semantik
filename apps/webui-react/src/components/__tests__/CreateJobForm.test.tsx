import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import CreateJobForm from '../CreateJobForm'
import { useUIStore } from '@/stores/uiStore'
import { useJobsStore } from '@/stores/jobsStore'
import { useDirectoryScanWebSocket } from '@/hooks/useDirectoryScanWebSocket'
import { server } from '@/tests/mocks/server'
import { http, HttpResponse } from 'msw'

// Mock the stores
vi.mock('@/stores/uiStore')
vi.mock('@/stores/jobsStore')

// Mock the directory scan hook
vi.mock('@/hooks/useDirectoryScanWebSocket')

const mockCompletedJobs = [
  {
    id: '1',
    name: 'Existing Collection',
    collection_name: 'existing-collection',
    status: 'completed',
    total_files: 10,
  },
]

describe('CreateJobForm', () => {
  const mockAddToast = vi.fn()
  const mockSetActiveTab = vi.fn()
  const mockAddJob = vi.fn()
  const mockStartScan = vi.fn()
  const mockReset = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    
    ;(useUIStore as any).mockImplementation((selector) => {
      const state = {
        addToast: mockAddToast,
        setActiveTab: mockSetActiveTab,
      }
      return selector ? selector(state) : state
    })
    
    ;(useJobsStore as any).mockImplementation((selector) => {
      const state = {
        addJob: mockAddJob,
      }
      return selector ? selector(state) : state
    })
    
    ;(useDirectoryScanWebSocket as any).mockReturnValue({
      scanning: false,
      scanResult: null,
      scanProgress: null,
      error: null,
      startScan: mockStartScan,
      reset: mockReset,
    })
    
    // Setup default MSW handlers
    server.use(
      http.get('/api/jobs', () => {
        return HttpResponse.json(mockCompletedJobs)
      }),
      http.get('/api/models', () => {
        return HttpResponse.json({
          models: {
            'Qwen/Qwen3-Embedding-0.6B': {
              name: 'Qwen/Qwen3-Embedding-0.6B',
              size: '600M',
              is_downloaded: true,
            },
          },
        })
      }),
      http.get('/api/jobs/collection-metadata/:collectionName', () => {
        return HttpResponse.json({
          model_name: 'Qwen/Qwen3-Embedding-0.6B',
          chunk_size: 600,
          chunk_overlap: 200,
        })
      })
    )
  })

  it('renders form with basic fields', () => {
    render(<CreateJobForm />)
    
    expect(screen.getByText('Create Embedding Job')).toBeInTheDocument()
    expect(screen.getByText('Directory Path')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /scan/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /create job/i })).toBeInTheDocument()
  })

  it('shows mode toggle between create and append', () => {
    render(<CreateJobForm />)
    
    const createRadio = screen.getByLabelText('Create New Collection')
    const appendRadio = screen.getByLabelText('Add to Existing Collection')
    
    expect(createRadio).toBeChecked()
    expect(appendRadio).not.toBeChecked()
  })

  it('switches to append mode and shows collection dropdown', async () => {
    const user = userEvent.setup()
    
    render(<CreateJobForm />)
    
    // Need to find the radio by its associated text
    const appendRadio = screen.getByRole('radio', { name: /add to existing collection/i })
    await user.click(appendRadio)
    
    await waitFor(() => {
      expect(screen.getByText('Select Collection')).toBeInTheDocument()
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
  })

  it('scans directory when scan button is clicked', async () => {
    const user = userEvent.setup()
    
    render(<CreateJobForm />)
    
    const directoryInput = screen.getByPlaceholderText('/path/to/documents')
    await user.type(directoryInput, '/test/path')
    
    const scanButton = screen.getByRole('button', { name: /scan/i })
    await user.click(scanButton)
    
    expect(mockStartScan).toHaveBeenCalledWith('/test/path')
  })

  it('shows scan results when available', () => {
    const scanResult = {
      total_files: 10,
      valid_files: 8,
      errors: [],
      warnings: [],
      total_size: 1048576,
      file_types: { txt: 5, pdf: 3 },
      files: ['file1.txt', 'file2.pdf'],
    }
    
    ;(useDirectoryScanWebSocket as any).mockReturnValue({
      scanning: false,
      scanResult,
      scanProgress: null,
      error: null,
      startScan: mockStartScan,
      reset: mockReset,
    })
    
    render(<CreateJobForm />)
    
    // Check that the scan complete section is displayed
    expect(screen.getByText('Scan Complete')).toBeInTheDocument()
    // Check that the scan result shows 10 files
    expect(screen.getByText('10')).toBeInTheDocument()
    expect(screen.getByText(/files found/)).toBeInTheDocument()
  })

  it('shows scanning progress', () => {
    ;(useDirectoryScanWebSocket as any).mockReturnValue({
      scanning: true,
      scanResult: null,
      scanProgress: {
        files_scanned: 5,
        total_files: 10,
        current_path: 'file.txt',
        percentage: 50,
      },
      error: null,
      startScan: mockStartScan,
      reset: mockReset,
    })
    
    render(<CreateJobForm />)
    
    // Check for the progress indicator - there are two elements with "Scanning..."
    const scanningElements = screen.getAllByText('Scanning...')
    expect(scanningElements.length).toBeGreaterThan(0)
    expect(screen.getByText(/5 \/ 10 files/)).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()
  })

  it('validates form before submission', async () => {
    const user = userEvent.setup()
    
    render(<CreateJobForm />)
    
    // The submit button should be disabled without scan results
    const createButton = screen.getByRole('button', { name: /create job/i })
    expect(createButton).toBeDisabled()
    
    // Fill in required fields
    const nameInput = screen.getByPlaceholderText('my-documents')
    await user.type(nameInput, 'Test Collection')
    
    const directoryInput = screen.getByPlaceholderText('/path/to/documents')
    await user.type(directoryInput, '/test/path')
    
    // Button should still be disabled without scan results
    expect(createButton).toBeDisabled()
  })

  it('shows error when scanning without directory', async () => {
    const user = userEvent.setup()
    
    render(<CreateJobForm />)
    
    // Try to scan without entering directory
    const scanButton = screen.getByRole('button', { name: /scan/i })
    expect(scanButton).toBeDisabled()
    
    // Enter directory and scan button should be enabled
    const directoryInput = screen.getByPlaceholderText('/path/to/documents')
    await user.type(directoryInput, '/test/path')
    expect(scanButton).not.toBeDisabled()
  })

  it('creates job with form data', async () => {
    const user = userEvent.setup()
    const scanResult = {
      total_files: 10,
      valid_files: 10,
      errors: [],
      warnings: [],
      total_size: 1048576,
      file_types: { txt: 10 },
      files: [],
    }
    
    ;(useDirectoryScanWebSocket as any).mockReturnValue({
      scanning: false,
      scanResult,
      scanProgress: null,
      error: null,
      startScan: mockStartScan,
      reset: mockReset,
    })
    
    server.use(
      http.post('/api/jobs', async ({ request }) => {
        const body = await request.json() as any
        return HttpResponse.json({
          id: '2',
          ...body,
          status: 'pending',
        })
      })
    )
    
    render(<CreateJobForm />)
    
    // Fill in the collection name first
    const nameInput = screen.getByPlaceholderText('my-documents')
    await user.type(nameInput, 'My Collection')
    
    // Directory is already set (the component has the scan result)
    // But we should fill it to match what was scanned
    const directoryInput = screen.getByPlaceholderText('/path/to/documents')
    await user.clear(directoryInput)
    await user.type(directoryInput, '/test/path')
    
    // Submit
    const createButton = screen.getByRole('button', { name: /create job/i })
    await user.click(createButton)
    
    await waitFor(() => {
      expect(mockAddJob).toHaveBeenCalled()
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Job created successfully',
      })
      expect(mockSetActiveTab).toHaveBeenCalledWith('jobs')
    })
  })

  it('shows advanced options when toggled', async () => {
    const user = userEvent.setup()
    
    render(<CreateJobForm />)
    
    const advancedButton = screen.getByText('Show Advanced Options')
    await user.click(advancedButton)
    
    expect(screen.getByLabelText('Embedding Model')).toBeInTheDocument()
    expect(screen.getByLabelText('Chunk Size')).toBeInTheDocument()
    expect(screen.getByLabelText('Chunk Overlap')).toBeInTheDocument()
    expect(screen.getByLabelText('Batch Size')).toBeInTheDocument()
  })

  it('shows warning confirmation dialog', async () => {
    const user = userEvent.setup()
    const scanResult = {
      total_files: 10,
      valid_files: 10,
      errors: [],
      warnings: [
        { type: 'large_file', message: 'Some files are very large' },
      ],
      total_size: 1048576,
      file_types: { txt: 10 },
      files: [],
    }
    
    ;(useDirectoryScanWebSocket as any).mockReturnValue({
      scanning: false,
      scanResult,
      scanProgress: null,
      error: null,
      startScan: mockStartScan,
      reset: mockReset,
    })
    
    // Mock window.confirm before rendering
    const mockConfirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    
    server.use(
      http.post('/api/jobs', async ({ request }) => {
        const body = await request.json() as any
        return HttpResponse.json({
          id: '2',
          ...body,
          status: 'pending',
        })
      })
    )
    
    render(<CreateJobForm />)
    
    // Fill in directory first (required for validation)
    const directoryInput = screen.getByPlaceholderText('/path/to/documents')
    await user.type(directoryInput, '/test/path')
    
    const nameInput = screen.getByPlaceholderText('my-documents')
    await user.type(nameInput, 'My Collection')
    
    const createButton = screen.getByRole('button', { name: /create job/i })
    await user.click(createButton)
    
    await waitFor(() => {
      expect(mockConfirm).toHaveBeenCalledWith(
        expect.stringContaining('Some files are very large')
      )
    })
    
    mockConfirm.mockRestore()
  })

  it('handles append mode submission', async () => {
    const user = userEvent.setup()
    const scanResult = {
      total_files: 5,
      valid_files: 5,
      errors: [],
      warnings: [],
      total_size: 524288,
      file_types: { txt: 5 },
      files: [],
    }
    
    ;(useDirectoryScanWebSocket as any).mockReturnValue({
      scanning: false,
      scanResult,
      scanProgress: null,
      error: null,
      startScan: mockStartScan,
      reset: mockReset,
    })
    
    server.use(
      http.post('/api/jobs/add-to-collection', async () => {
        return HttpResponse.json({
          id: '3',
          collection_name: 'existing-collection',
          status: 'pending',
        })
      })
    )
    
    render(<CreateJobForm />)
    
    // Switch to append mode
    const appendRadio = screen.getByRole('radio', { name: /add to existing collection/i })
    await user.click(appendRadio)
    
    // Wait for dropdown to appear
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    
    // Select a collection - use the actual option value "Existing Collection"
    const collectionSelect = screen.getByRole('combobox')
    await user.selectOptions(collectionSelect, 'Existing Collection')
    
    // Fill in directory
    const directoryInput = screen.getByPlaceholderText('/path/to/documents')
    await user.type(directoryInput, '/test/append/path')
    
    // Submit the form
    const submitButton = screen.getByRole('button', { name: /add documents/i })
    await user.click(submitButton)
    
    // Verify the job was added
    await waitFor(() => {
      expect(mockAddJob).toHaveBeenCalled()
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Adding documents to Existing Collection',
      })
    })
  })
})
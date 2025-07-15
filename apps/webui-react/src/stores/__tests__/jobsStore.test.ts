import { describe, it, expect, beforeEach } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useJobsStore, type Job } from '../jobsStore'

describe('jobsStore', () => {
  const mockJob1: Job = {
    id: '1',
    name: 'Job 1',
    directory: '/path/to/dir1',
    collection_name: 'collection1',
    status: 'pending',
    progress: 0,
    total_files: 100,
    total_documents: 100,
    processed_files: 0,
    processed_documents: 0,
    failed_files: 0,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  }

  const mockJob2: Job = {
    id: '2',
    name: 'Job 2',
    directory: '/path/to/dir2',
    collection_name: 'collection2',
    status: 'processing',
    progress: 50,
    total_files: 200,
    total_documents: 200,
    processed_files: 100,
    processed_documents: 100,
    failed_files: 0,
    created_at: '2024-01-02T00:00:00Z',
    updated_at: '2024-01-02T00:00:00Z',
    current_file: 'file.txt',
  }

  beforeEach(() => {
    // Reset store state before each test
    const { result } = renderHook(() => useJobsStore())
    act(() => {
      result.current.setJobs([])
      result.current.activeJobs.clear()
    })
  })

  describe('jobs management', () => {
    it('starts with empty jobs array', () => {
      const { result } = renderHook(() => useJobsStore())
      expect(result.current.jobs).toEqual([])
    })

    it('sets jobs correctly', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setJobs([mockJob1, mockJob2])
      })

      expect(result.current.jobs).toHaveLength(2)
      expect(result.current.jobs[0]).toEqual(mockJob1)
      expect(result.current.jobs[1]).toEqual(mockJob2)
    })

    it('adds a new job to the beginning of the list', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setJobs([mockJob1])
        result.current.addJob(mockJob2)
      })

      expect(result.current.jobs).toHaveLength(2)
      expect(result.current.jobs[0]).toEqual(mockJob2) // New job at the beginning
      expect(result.current.jobs[1]).toEqual(mockJob1)
    })

    it('updates an existing job', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setJobs([mockJob1, mockJob2])
        result.current.updateJob('1', {
          status: 'completed',
          progress: 100,
          processed_files: 100,
        })
      })

      expect(result.current.jobs[0].status).toBe('completed')
      expect(result.current.jobs[0].progress).toBe(100)
      expect(result.current.jobs[0].processed_files).toBe(100)
      // Other properties should remain unchanged
      expect(result.current.jobs[0].name).toBe('Job 1')
      expect(result.current.jobs[0].total_files).toBe(100)
    })

    it('does not update job if id does not match', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setJobs([mockJob1])
        result.current.updateJob('non-existent', {
          status: 'completed',
        })
      })

      expect(result.current.jobs[0].status).toBe('pending')
    })

    it('removes a job', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setJobs([mockJob1, mockJob2])
        result.current.removeJob('1')
      })

      expect(result.current.jobs).toHaveLength(1)
      expect(result.current.jobs[0]).toEqual(mockJob2)
    })

    it('removes job from activeJobs when removing job', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setJobs([mockJob1, mockJob2])
        result.current.setActiveJob('1', true)
        result.current.setActiveJob('2', true)
      })

      expect(result.current.activeJobs.has('1')).toBe(true)
      expect(result.current.activeJobs.has('2')).toBe(true)

      act(() => {
        result.current.removeJob('1')
      })

      expect(result.current.activeJobs.has('1')).toBe(false)
      expect(result.current.activeJobs.has('2')).toBe(true)
    })
  })

  describe('activeJobs management', () => {
    it('starts with empty activeJobs set', () => {
      const { result } = renderHook(() => useJobsStore())
      expect(result.current.activeJobs.size).toBe(0)
    })

    it('adds job to activeJobs', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setActiveJob('job-1', true)
      })

      expect(result.current.activeJobs.has('job-1')).toBe(true)
      expect(result.current.activeJobs.size).toBe(1)
    })

    it('removes job from activeJobs', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setActiveJob('job-1', true)
        result.current.setActiveJob('job-2', true)
      })

      expect(result.current.activeJobs.size).toBe(2)

      act(() => {
        result.current.setActiveJob('job-1', false)
      })

      expect(result.current.activeJobs.has('job-1')).toBe(false)
      expect(result.current.activeJobs.has('job-2')).toBe(true)
      expect(result.current.activeJobs.size).toBe(1)
    })

    it('handles setting same job multiple times', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setActiveJob('job-1', true)
        result.current.setActiveJob('job-1', true) // Set again
      })

      expect(result.current.activeJobs.size).toBe(1)
      expect(result.current.activeJobs.has('job-1')).toBe(true)
    })

    it('handles removing non-existent job from activeJobs', () => {
      const { result } = renderHook(() => useJobsStore())
      
      act(() => {
        result.current.setActiveJob('job-1', true)
        result.current.setActiveJob('non-existent', false)
      })

      expect(result.current.activeJobs.size).toBe(1)
      expect(result.current.activeJobs.has('job-1')).toBe(true)
    })
  })

  describe('complex scenarios', () => {
    it('handles multiple operations correctly', () => {
      const { result } = renderHook(() => useJobsStore())
      
      // Add initial jobs
      act(() => {
        result.current.setJobs([mockJob1])
        result.current.addJob(mockJob2)
      })

      expect(result.current.jobs).toHaveLength(2)

      // Set some as active
      act(() => {
        result.current.setActiveJob('1', true)
        result.current.setActiveJob('2', true)
      })

      expect(result.current.activeJobs.size).toBe(2)

      // Update a job
      act(() => {
        result.current.updateJob('2', { 
          status: 'completed',
          progress: 100,
          error: undefined,
        })
      })

      expect(result.current.jobs[0].status).toBe('completed')

      // Remove a job
      act(() => {
        result.current.removeJob('1')
      })

      expect(result.current.jobs).toHaveLength(1)
      expect(result.current.activeJobs.size).toBe(1)
      expect(result.current.activeJobs.has('1')).toBe(false)
    })

    it('preserves job order when updating', () => {
      const { result } = renderHook(() => useJobsStore())
      
      const job3: Job = { ...mockJob1, id: '3', name: 'Job 3' }
      
      act(() => {
        result.current.setJobs([mockJob1, mockJob2, job3])
      })

      act(() => {
        result.current.updateJob('2', { status: 'completed' })
      })

      // Order should be preserved
      expect(result.current.jobs[0].id).toBe('1')
      expect(result.current.jobs[1].id).toBe('2')
      expect(result.current.jobs[2].id).toBe('3')
    })

    it('handles job with all optional fields', () => {
      const { result } = renderHook(() => useJobsStore())
      
      const completeJob: Job = {
        ...mockJob1,
        directory_path: '/alternative/path',
        error: 'Some error',
        current_file: 'processing.txt',
        model_name: 'model-123',
        chunk_size: 512,
        chunk_overlap: 64,
        batch_size: 32,
        quantization: 'int8',
        vector_dim: 768,
        instruction: 'Custom instruction',
        metrics: {
          processing_rate: 100,
          estimated_time_remaining: 3600,
          queue_position: 5,
          memory_usage: 2048,
        },
      }

      act(() => {
        result.current.addJob(completeJob)
      })

      expect(result.current.jobs[0]).toEqual(completeJob)
    })
  })

  describe('state isolation', () => {
    it('shares state between multiple hooks', () => {
      const { result: hook1 } = renderHook(() => useJobsStore())
      const { result: hook2 } = renderHook(() => useJobsStore())
      
      act(() => {
        hook1.current.addJob(mockJob1)
      })
      
      // Both hooks should see the same state
      expect(hook1.current.jobs).toHaveLength(1)
      expect(hook2.current.jobs).toHaveLength(1)
      expect(hook2.current.jobs[0]).toEqual(mockJob1)
    })
  })
})
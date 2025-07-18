import { create } from 'zustand';

export interface Job {
  id: string;
  name: string;
  directory: string;
  directory_path?: string;
  collection_name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'waiting' | 'cancelled';
  progress: number;
  total_files: number;
  total_documents: number;
  processed_files: number;
  processed_documents: number;
  failed_files: number;
  created_at: string;
  updated_at: string;
  error?: string;
  current_file?: string;
  model_name?: string;
  chunk_size?: number;
  chunk_overlap?: number;
  batch_size?: number;
  quantization?: string;
  vector_dim?: number;
  instruction?: string;
  metrics?: {
    processing_rate: number;
    estimated_time_remaining: number;
    queue_position?: number;
    memory_usage?: number;
  };
}

interface JobsState {
  jobs: Job[];
  activeJobs: Set<string>;
  setJobs: (jobs: Job[]) => void;
  updateJob: (jobId: string, updates: Partial<Job>) => void;
  addJob: (job: Job) => void;
  removeJob: (jobId: string) => void;
  setActiveJob: (jobId: string, active: boolean) => void;
}

export const useJobsStore = create<JobsState>((set) => ({
  jobs: [],
  activeJobs: new Set(),
  setJobs: (jobs) => set({ jobs }),
  updateJob: (jobId, updates) =>
    set((state) => ({
      jobs: state.jobs.map((job) =>
        job.id === jobId ? { ...job, ...updates } : job
      ),
    })),
  addJob: (job) => set((state) => ({ jobs: [job, ...state.jobs] })),
  removeJob: (jobId) =>
    set((state) => ({
      jobs: state.jobs.filter((job) => job.id !== jobId),
      activeJobs: new Set([...state.activeJobs].filter((id) => id !== jobId)),
    })),
  setActiveJob: (jobId, active) =>
    set((state) => {
      const newActiveJobs = new Set(state.activeJobs);
      if (active) {
        newActiveJobs.add(jobId);
      } else {
        newActiveJobs.delete(jobId);
      }
      return { activeJobs: newActiveJobs };
    }),
}));
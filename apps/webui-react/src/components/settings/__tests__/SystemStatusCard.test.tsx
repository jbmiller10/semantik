import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import SystemStatusCard from '../SystemStatusCard';
import * as useSystemInfoModule from '@/hooks/useSystemInfo';

// Mock the hooks
vi.mock('@/hooks/useSystemInfo', () => ({
  useSystemInfo: vi.fn(),
  useSystemHealth: vi.fn(),
  useSystemStatus: vi.fn(),
}));

// Mock data
const mockSystemInfo = {
  version: '0.8.0',
  environment: 'development',
  python_version: '3.11.0',
  rate_limits: {
    chunking_preview: '10/minute',
    plugin_install: '5/minute',
    llm_test: '3/minute',
  },
};

const mockSystemHealth = {
  postgres: { status: 'healthy', message: 'Connected' },
  redis: { status: 'healthy', message: 'Connected' },
  qdrant: { status: 'healthy', message: 'Connected' },
  vecpipe: { status: 'healthy', message: 'Connected' },
};

const mockSystemStatus = {
  healthy: true,
  reranking_available: true,
  gpu_available: true,
  gpu_memory_mb: 8192,
  cuda_device_name: 'NVIDIA GeForce RTX 4090',
  cuda_device_count: 1,
  available_reranking_models: ['Qwen/Qwen3-Reranker-0.6B'],
};

describe('SystemStatusCard', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    vi.mocked(useSystemInfoModule.useSystemInfo).mockReturnValue({
      data: mockSystemInfo,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useSystemInfoModule.useSystemInfo>);

    vi.mocked(useSystemInfoModule.useSystemHealth).mockReturnValue({
      data: mockSystemHealth,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useSystemInfoModule.useSystemHealth>);

    vi.mocked(useSystemInfoModule.useSystemStatus).mockReturnValue({
      data: mockSystemStatus,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useSystemInfoModule.useSystemStatus>);
  });

  describe('loading state', () => {
    it('shows loading spinner when system info is loading', () => {
      vi.mocked(useSystemInfoModule.useSystemInfo).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      } as unknown as ReturnType<typeof useSystemInfoModule.useSystemInfo>);

      render(<SystemStatusCard />);

      expect(screen.getByText('Loading system information...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when loading fails', () => {
      vi.mocked(useSystemInfoModule.useSystemInfo).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load system info'),
      } as unknown as ReturnType<typeof useSystemInfoModule.useSystemInfo>);

      render(<SystemStatusCard />);

      expect(screen.getByText('Error loading system info')).toBeInTheDocument();
      expect(screen.getByText('Failed to load system info')).toBeInTheDocument();
    });
  });

  describe('system information', () => {
    it('renders the header and description', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('System Information')).toBeInTheDocument();
      expect(screen.getByText(/View system configuration, resource limits, and service health status/)).toBeInTheDocument();
    });

    it('renders the info box', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText(/These settings are read-only and configured via environment variables/)).toBeInTheDocument();
    });

    it('displays application version', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('Version')).toBeInTheDocument();
      expect(screen.getByText('0.8.0')).toBeInTheDocument();
    });

    it('displays environment', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('Environment')).toBeInTheDocument();
      expect(screen.getByText('development')).toBeInTheDocument();
    });

    it('displays Python version', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('Python Version')).toBeInTheDocument();
      expect(screen.getByText('3.11.0')).toBeInTheDocument();
    });
  });

  describe('rate limits', () => {
    it('displays rate limit information', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('Rate Limits')).toBeInTheDocument();
      expect(screen.getByText('Chunking Preview')).toBeInTheDocument();
      expect(screen.getByText('10/minute')).toBeInTheDocument();
      expect(screen.getByText('Plugin Install')).toBeInTheDocument();
      expect(screen.getByText('5/minute')).toBeInTheDocument();
    });
  });

  describe('GPU status', () => {
    it('displays GPU available status', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('GPU Status')).toBeInTheDocument();
      expect(screen.getByText('GPU Available')).toBeInTheDocument();
    });

    it('displays GPU device info when available', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('Device')).toBeInTheDocument();
      expect(screen.getByText('NVIDIA GeForce RTX 4090')).toBeInTheDocument();
      expect(screen.getByText('CUDA Devices')).toBeInTheDocument();
      expect(screen.getByText('1')).toBeInTheDocument();
    });

    it('displays GPU not available when GPU is not available', () => {
      vi.mocked(useSystemInfoModule.useSystemStatus).mockReturnValue({
        data: { ...mockSystemStatus, gpu_available: false },
        isLoading: false,
        error: null,
      } as unknown as ReturnType<typeof useSystemInfoModule.useSystemStatus>);

      render(<SystemStatusCard />);

      expect(screen.getByText('GPU Not Available')).toBeInTheDocument();
    });

    it('displays reranking availability', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('Reranking Available')).toBeInTheDocument();
    });

    it('displays reranking not available when unavailable', () => {
      vi.mocked(useSystemInfoModule.useSystemStatus).mockReturnValue({
        data: { ...mockSystemStatus, reranking_available: false },
        isLoading: false,
        error: null,
      } as unknown as ReturnType<typeof useSystemInfoModule.useSystemStatus>);

      render(<SystemStatusCard />);

      expect(screen.getByText('Reranking Not Available')).toBeInTheDocument();
    });

    it('displays available reranking models', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('Available Models:')).toBeInTheDocument();
      expect(screen.getByText('Qwen/Qwen3-Reranker-0.6B')).toBeInTheDocument();
    });

    it('shows loading state for GPU status', () => {
      vi.mocked(useSystemInfoModule.useSystemStatus).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      } as unknown as ReturnType<typeof useSystemInfoModule.useSystemStatus>);

      render(<SystemStatusCard />);

      expect(screen.getByText('Checking GPU status...')).toBeInTheDocument();
    });
  });

  describe('service health', () => {
    it('displays service health section', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('Service Health')).toBeInTheDocument();
      expect(screen.getByText('Auto-refreshes every 30s')).toBeInTheDocument();
    });

    it('displays all service health cards', () => {
      render(<SystemStatusCard />);

      expect(screen.getByText('PostgreSQL')).toBeInTheDocument();
      expect(screen.getByText('Redis')).toBeInTheDocument();
      expect(screen.getByText('Qdrant')).toBeInTheDocument();
      expect(screen.getByText('VecPipe')).toBeInTheDocument();
    });

    it('displays service messages when available', () => {
      render(<SystemStatusCard />);

      // Should show "Connected" for each healthy service
      expect(screen.getAllByText('Connected')).toHaveLength(4);
    });

    it('shows unhealthy status for failed services', () => {
      vi.mocked(useSystemInfoModule.useSystemHealth).mockReturnValue({
        data: {
          ...mockSystemHealth,
          postgres: { status: 'unhealthy', message: 'Connection failed' },
        },
        isLoading: false,
        error: null,
      } as unknown as ReturnType<typeof useSystemInfoModule.useSystemHealth>);

      render(<SystemStatusCard />);

      expect(screen.getByText('Connection failed')).toBeInTheDocument();
    });
  });
});

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SparseIndexPanel } from '../SparseIndexPanel';
import { TestWrapper } from '../../../tests/utils/TestWrapper';
import type { Collection } from '../../../types/collection';
import type { SparseIndexStatus, SparseReindexProgress } from '../../../types/sparse-index';

// Mock the hooks
const mockEnable = vi.fn();
const mockDisable = vi.fn();
const mockTriggerReindex = vi.fn();

vi.mock('../../../hooks/useSparseIndex', () => ({
  useSparseIndexWithReindex: vi.fn(),
  useSparseReindexProgress: vi.fn(),
}));

// Import mocked hooks after mocking
import {
  useSparseIndexWithReindex,
  useSparseReindexProgress,
} from '../../../hooks/useSparseIndex';

// Mock window.confirm
const mockConfirm = vi.fn();
vi.stubGlobal('confirm', mockConfirm);

// Test data
const mockCollection: Collection = {
  id: 'test-collection-id',
  name: 'Test Collection',
  description: 'Test description',
  owner_id: 1,
  vector_store_name: 'test_vector_store',
  embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
  quantization: 'float32',
  chunk_size: 512,
  chunk_overlap: 50,
  is_public: false,
  status: 'ready',
  document_count: 100,
  vector_count: 500,
  total_size_bytes: 1048576,
  sync_mode: 'one_time',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-02T00:00:00Z',
};

const createMockStatus = (overrides?: Partial<SparseIndexStatus>): SparseIndexStatus => ({
  enabled: false,
  plugin_id: undefined,
  document_count: 0,
  model_config_data: undefined,
  ...overrides,
});

const createMockProgress = (overrides?: Partial<SparseReindexProgress>): SparseReindexProgress => ({
  job_id: 'test-job-id',
  status: 'pending',
  progress: 0,
  ...overrides,
});

describe('SparseIndexPanel', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
    mockConfirm.mockReturnValue(true);

    // Default mock implementations
    vi.mocked(useSparseIndexWithReindex).mockReturnValue({
      status: createMockStatus(),
      isLoading: false,
      isError: false,
      enable: mockEnable,
      isEnabling: false,
      disable: mockDisable,
      isDisabling: false,
      triggerReindex: mockTriggerReindex,
      isReindexing: false,
      reindexJobId: undefined,
    });

    vi.mocked(useSparseReindexProgress).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      refetch: vi.fn(),
    } as ReturnType<typeof useSparseReindexProgress>);
  });

  describe('loading state', () => {
    it('shows loading spinner and message when isLoading is true', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: undefined,
        isLoading: true,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('Loading sparse index status...')).toBeInTheDocument();
      // Check for spinner animation
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error banner when isError is true', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: undefined,
        isLoading: false,
        isError: true,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('Failed to load sparse index status')).toBeInTheDocument();
    });
  });

  describe('disabled state (sparse indexing not enabled)', () => {
    it('shows "Disabled" badge', () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('Disabled')).toBeInTheDocument();
    });

    it('displays explanation text about hybrid search', () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(
        screen.getByText(/Enable sparse indexing to use hybrid search/)
      ).toBeInTheDocument();
    });

    it('renders "Enable Sparse Indexing" button', () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(
        screen.getByRole('button', { name: /enable sparse indexing/i })
      ).toBeInTheDocument();
    });

    it('opens config modal when enable button is clicked', async () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      await user.click(screen.getByRole('button', { name: /enable sparse indexing/i }));

      // Modal should open - check for BM25 plugin option which only appears in the modal
      await waitFor(() => {
        expect(screen.getByText('BM25 (Statistical)')).toBeInTheDocument();
      });
    });

    it('shows Loader2 spinner when isEnabling is true', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus(),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: true,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('Enabling...')).toBeInTheDocument();
    });

    it('disables button when isEnabling is true', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus(),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: true,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      const button = screen.getByRole('button', { name: /enabling/i });
      expect(button).toBeDisabled();
    });
  });

  describe('enabled state', () => {
    beforeEach(() => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
          document_count: 1234,
          model_config_data: { k1: 1.5, b: 0.75 },
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });
    });

    it('shows "Enabled" badge', () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('Enabled')).toBeInTheDocument();
    });

    it('displays plugin name from SPARSE_PLUGIN_INFO', () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('BM25 (Statistical)')).toBeInTheDocument();
    });

    it('displays document/vector count', () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('1,234')).toBeInTheDocument();
    });

    it('shows BM25 parameters section when plugin is bm25-local', () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('BM25 Parameters')).toBeInTheDocument();
      expect(screen.getByText('k1:')).toBeInTheDocument();
      expect(screen.getByText('1.5')).toBeInTheDocument();
      expect(screen.getByText('b:')).toBeInTheDocument();
      expect(screen.getByText('0.75')).toBeInTheDocument();
    });

    it('renders Reindex button', () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByRole('button', { name: /reindex/i })).toBeInTheDocument();
    });

    it('renders Disable button', () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByRole('button', { name: /disable/i })).toBeInTheDocument();
    });
  });

  describe('reindex functionality', () => {
    beforeEach(() => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
          document_count: 100,
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });
    });

    it('calls triggerReindex when Reindex button is clicked', async () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      await user.click(screen.getByRole('button', { name: /^reindex$/i }));

      expect(mockTriggerReindex).toHaveBeenCalledWith('test-collection-id');
    });

    it('shows Loader2 spinner during reindex', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: true,
        reindexJobId: 'job-123',
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('Reindexing...')).toBeInTheDocument();
    });

    it('disables button when isReindexing is true', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: true,
        reindexJobId: 'job-123',
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      const button = screen.getByRole('button', { name: /reindexing/i });
      expect(button).toBeDisabled();
    });
  });

  describe('reindex progress', () => {
    it('shows progress bar when reindex is in progress', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: 'job-123',
      });

      vi.mocked(useSparseReindexProgress).mockReturnValue({
        data: createMockProgress({
          status: 'processing',
          progress: 45,
        }),
        isLoading: false,
        isError: false,
        refetch: vi.fn(),
      } as ReturnType<typeof useSparseReindexProgress>);

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('Reindexing in progress')).toBeInTheDocument();
      expect(screen.getByText('45%')).toBeInTheDocument();
    });

    it('displays current step text when available', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: 'job-123',
      });

      vi.mocked(useSparseReindexProgress).mockReturnValue({
        data: createMockProgress({
          status: 'processing',
          progress: 30,
          current_step: 'Processing document 30 of 100',
        }),
        isLoading: false,
        isError: false,
        refetch: vi.fn(),
      } as ReturnType<typeof useSparseReindexProgress>);

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('Processing document 30 of 100')).toBeInTheDocument();
    });

    it('disables Reindex button during progress', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: 'job-123',
      });

      vi.mocked(useSparseReindexProgress).mockReturnValue({
        data: createMockProgress({
          status: 'processing',
          progress: 50,
        }),
        isLoading: false,
        isError: false,
        refetch: vi.fn(),
      } as ReturnType<typeof useSparseReindexProgress>);

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      const button = screen.getByRole('button', { name: /reindexing/i });
      expect(button).toBeDisabled();
    });
  });

  describe('disable functionality', () => {
    beforeEach(() => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });
    });

    it('shows window.confirm dialog when Disable clicked', async () => {
      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      await user.click(screen.getByRole('button', { name: /disable/i }));

      expect(mockConfirm).toHaveBeenCalledWith(
        'Are you sure you want to disable sparse indexing? This will delete the sparse index.'
      );
    });

    it('calls disable mutation when user confirms', async () => {
      mockConfirm.mockReturnValue(true);

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      await user.click(screen.getByRole('button', { name: /disable/i }));

      expect(mockDisable).toHaveBeenCalledWith('test-collection-id');
    });

    it('does not call disable when user cancels', async () => {
      mockConfirm.mockReturnValue(false);

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      await user.click(screen.getByRole('button', { name: /disable/i }));

      expect(mockDisable).not.toHaveBeenCalled();
    });

    it('shows Loader2 spinner when isDisabling is true', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: true,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      // Check for spinner animation in the disable button
      const disableButton = screen.getByRole('button', { name: /disable/i });
      const spinner = disableButton.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('disables button when isDisabling is true', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'bm25-local',
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: true,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      const button = screen.getByRole('button', { name: /disable/i });
      expect(button).toBeDisabled();
    });
  });

  describe('SPLADE plugin display', () => {
    it('shows SPLADE plugin name when enabled with splade-local', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'splade-local',
          document_count: 500,
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.getByText('SPLADE (Neural)')).toBeInTheDocument();
    });

    it('does not show BM25 Parameters section for SPLADE', () => {
      vi.mocked(useSparseIndexWithReindex).mockReturnValue({
        status: createMockStatus({
          enabled: true,
          plugin_id: 'splade-local',
          document_count: 500,
        }),
        isLoading: false,
        isError: false,
        enable: mockEnable,
        isEnabling: false,
        disable: mockDisable,
        isDisabling: false,
        triggerReindex: mockTriggerReindex,
        isReindexing: false,
        reindexJobId: undefined,
      });

      render(
        <TestWrapper>
          <SparseIndexPanel collection={mockCollection} />
        </TestWrapper>
      );

      expect(screen.queryByText('BM25 Parameters')).not.toBeInTheDocument();
    });
  });
});

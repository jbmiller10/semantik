import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, within } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import ModelCard from '../ModelCard';
import type { CuratedModelResponse } from '@/types/model-manager';
import type { FormattedDownloadProgress, FormattedDeleteProgress } from '@/hooks/useModelManager';

// Factory function for creating mock model data
function createMockModel(overrides: Partial<CuratedModelResponse> = {}): CuratedModelResponse {
  return {
    id: 'Qwen/Qwen3-Embedding-0.6B',
    name: 'Qwen3 Embedding 0.6B',
    description: 'Small efficient embedding model for semantic search',
    model_type: 'embedding',
    memory_mb: { float16: 1200, int8: 600, int4: 400 },
    is_installed: false,
    size_on_disk_mb: null,
    used_by_collections: [],
    active_download_task_id: null,
    active_delete_task_id: null,
    embedding_details: {
      dimension: 1024,
      max_sequence_length: 8192,
      pooling_method: 'mean',
      is_asymmetric: true,
      query_prefix: 'query: ',
      document_prefix: 'document: ',
      default_query_instruction: '',
    },
    llm_details: null,
    ...overrides,
  };
}

// Factory function for download progress
function createMockDownloadProgress(
  overrides: Partial<FormattedDownloadProgress> = {}
): FormattedDownloadProgress {
  return {
    taskId: 'task-123',
    modelId: 'Qwen/Qwen3-Embedding-0.6B',
    status: 'running',
    bytesDownloaded: 500 * 1024 * 1024, // 500 MB
    bytesTotal: 1200 * 1024 * 1024, // 1.2 GB
    error: null,
    updatedAt: Date.now(),
    percentage: 42,
    formattedBytes: '500.0 MB / 1.2 GB',
    ...overrides,
  };
}

// Factory function for delete progress
function createMockDeleteProgress(
  overrides: Partial<FormattedDeleteProgress> = {}
): FormattedDeleteProgress {
  return {
    taskId: 'delete-task-456',
    modelId: 'Qwen/Qwen3-Embedding-0.6B',
    status: 'running',
    error: null,
    updatedAt: Date.now(),
    isDeleting: true,
    isFailed: false,
    ...overrides,
  };
}

describe('ModelCard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('basic rendering', () => {
    it('renders model name and description', () => {
      const model = createMockModel();
      render(<ModelCard model={model} />);

      expect(screen.getByText('Qwen3 Embedding 0.6B')).toBeInTheDocument();
      expect(
        screen.getByText('Small efficient embedding model for semantic search')
      ).toBeInTheDocument();
    });

    it('renders model type badge', () => {
      const model = createMockModel({ model_type: 'embedding' });
      render(<ModelCard model={model} />);

      expect(screen.getByText('Embedding')).toBeInTheDocument();
    });

    it('shows Available status for uninstalled models', () => {
      const model = createMockModel({ is_installed: false });
      render(<ModelCard model={model} />);

      expect(screen.getByText('Available')).toBeInTheDocument();
    });

    it('shows Installed status for installed models', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      render(<ModelCard model={model} />);

      expect(screen.getByText('Installed')).toBeInTheDocument();
    });

    it('shows disk size for installed models', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      render(<ModelCard model={model} />);

      expect(screen.getByText('1.2 GB')).toBeInTheDocument();
    });

    it('shows memory estimate from int8 quantization', () => {
      const model = createMockModel({ memory_mb: { float16: 1200, int8: 600 } });
      render(<ModelCard model={model} />);

      expect(screen.getByText(/~600 MB RAM/)).toBeInTheDocument();
    });

    it('shows collection count for embedding models in use', () => {
      const model = createMockModel({
        model_type: 'embedding',
        is_installed: true,
        used_by_collections: ['Collection A', 'Collection B'],
      });
      render(<ModelCard model={model} />);

      expect(screen.getByText('2 collections')).toBeInTheDocument();
    });
  });

  describe('download button behavior', () => {
    it('shows download button for uninstalled models', () => {
      const model = createMockModel({ is_installed: false });
      const onDownload = vi.fn();
      render(<ModelCard model={model} onDownload={onDownload} />);

      expect(screen.getByRole('button', { name: /download/i })).toBeInTheDocument();
    });

    it('calls onDownload handler when download button is clicked', async () => {
      const user = userEvent.setup();
      const model = createMockModel({ is_installed: false });
      const onDownload = vi.fn();
      render(<ModelCard model={model} onDownload={onDownload} />);

      await user.click(screen.getByRole('button', { name: /download/i }));

      expect(onDownload).toHaveBeenCalledWith('Qwen/Qwen3-Embedding-0.6B');
    });

    it('hides download button for installed models', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const onDownload = vi.fn();
      render(<ModelCard model={model} onDownload={onDownload} />);

      expect(screen.queryByRole('button', { name: /download/i })).not.toBeInTheDocument();
    });

    it('hides download button when download is in progress', () => {
      const model = createMockModel({ is_installed: false });
      const onDownload = vi.fn();
      const downloadProgress = createMockDownloadProgress({ status: 'running' });
      render(
        <ModelCard model={model} onDownload={onDownload} downloadProgress={downloadProgress} />
      );

      expect(screen.queryByRole('button', { name: /download/i })).not.toBeInTheDocument();
    });

    it('disables download button when delete is in progress', () => {
      const model = createMockModel({ is_installed: false, active_delete_task_id: 'delete-123' });
      const onDownload = vi.fn();
      render(<ModelCard model={model} onDownload={onDownload} />);

      const downloadButton = screen.getByRole('button', { name: /download/i });
      expect(downloadButton).toBeDisabled();
      expect(downloadButton).toHaveAttribute('title', 'Delete in progress');
    });
  });

  describe('delete button behavior', () => {
    it('shows delete button for installed models', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const onDelete = vi.fn();
      render(<ModelCard model={model} onDelete={onDelete} />);

      expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
    });

    it('calls onDelete handler when delete button is clicked', async () => {
      const user = userEvent.setup();
      const model = createMockModel({
        is_installed: true,
        size_on_disk_mb: 1200,
        used_by_collections: [],
      });
      const onDelete = vi.fn();
      render(<ModelCard model={model} onDelete={onDelete} />);

      await user.click(screen.getByRole('button', { name: /delete/i }));

      expect(onDelete).toHaveBeenCalledWith('Qwen/Qwen3-Embedding-0.6B');
    });

    it('hides delete button for uninstalled models', () => {
      const model = createMockModel({ is_installed: false });
      const onDelete = vi.fn();
      render(<ModelCard model={model} onDelete={onDelete} />);

      expect(screen.queryByRole('button', { name: /delete/i })).not.toBeInTheDocument();
    });

    it('disables delete button when model is used by collections', () => {
      const model = createMockModel({
        is_installed: true,
        size_on_disk_mb: 1200,
        used_by_collections: ['Collection A'],
      });
      const onDelete = vi.fn();
      render(<ModelCard model={model} onDelete={onDelete} />);

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      expect(deleteButton).toBeDisabled();
      expect(deleteButton).toHaveAttribute('title', 'Model is in use by collections');
    });

    it('disables delete button when download is in progress', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const onDelete = vi.fn();
      const downloadProgress = createMockDownloadProgress({ status: 'running' });
      render(
        <ModelCard model={model} onDelete={onDelete} downloadProgress={downloadProgress} />
      );

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      expect(deleteButton).toBeDisabled();
      expect(deleteButton).toHaveAttribute('title', 'Download in progress');
    });

    it('disables delete button when delete is already in progress', () => {
      const model = createMockModel({
        is_installed: true,
        size_on_disk_mb: 1200,
        active_delete_task_id: 'delete-123',
      });
      const onDelete = vi.fn();
      render(<ModelCard model={model} onDelete={onDelete} />);

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      expect(deleteButton).toBeDisabled();
      expect(deleteButton).toHaveAttribute('title', 'Delete in progress');
    });

    it('disables delete button when delete progress indicates deleting', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const onDelete = vi.fn();
      const deleteProgress = createMockDeleteProgress({ isDeleting: true });
      render(<ModelCard model={model} onDelete={onDelete} deleteProgress={deleteProgress} />);

      const deleteButton = screen.getByRole('button', { name: /delete/i });
      expect(deleteButton).toBeDisabled();
    });
  });

  describe('download progress display', () => {
    it('shows progress bar during download', () => {
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({
        status: 'running',
        percentage: 42,
        formattedBytes: '500.0 MB / 1.2 GB',
      });
      render(<ModelCard model={model} downloadProgress={downloadProgress} />);

      expect(screen.getByText('Downloading...')).toBeInTheDocument();
      expect(screen.getByText('500.0 MB / 1.2 GB')).toBeInTheDocument();
      expect(screen.getByText('42%')).toBeInTheDocument();
    });

    it('shows indeterminate state when total bytes unknown', () => {
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({
        status: 'running',
        bytesTotal: 0,
        percentage: 0,
      });
      render(<ModelCard model={model} downloadProgress={downloadProgress} />);

      // Component shows "Initializing..." twice - in the label and progress area
      const initializingElements = screen.getAllByText('Initializing...');
      expect(initializingElements.length).toBeGreaterThan(0);
    });

    it('shows pending state during download initialization', () => {
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({
        status: 'pending',
        bytesDownloaded: 0,
        bytesTotal: 0,
      });
      render(<ModelCard model={model} downloadProgress={downloadProgress} />);

      // Component shows "Initializing..." twice - in the label and progress area
      const initializingElements = screen.getAllByText('Initializing...');
      expect(initializingElements.length).toBeGreaterThan(0);
    });

    it('shows error state with error message when download fails', () => {
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({
        status: 'failed',
        error: 'Network connection lost',
      });
      render(<ModelCard model={model} downloadProgress={downloadProgress} />);

      expect(screen.getByText('Network connection lost')).toBeInTheDocument();
    });

    it('shows default error message when no error details provided', () => {
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({
        status: 'failed',
        error: null,
      });
      render(<ModelCard model={model} downloadProgress={downloadProgress} />);

      expect(screen.getByText('Download failed')).toBeInTheDocument();
    });

    it('shows retry button on download failure', () => {
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({ status: 'failed' });
      const onRetry = vi.fn();
      render(
        <ModelCard model={model} downloadProgress={downloadProgress} onRetry={onRetry} />
      );

      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });

    it('calls onRetry handler when retry button is clicked', async () => {
      const user = userEvent.setup();
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({ status: 'failed' });
      const onRetry = vi.fn();
      render(
        <ModelCard model={model} downloadProgress={downloadProgress} onRetry={onRetry} />
      );

      await user.click(screen.getByRole('button', { name: /retry/i }));

      expect(onRetry).toHaveBeenCalledWith('Qwen/Qwen3-Embedding-0.6B');
    });

    it('shows dismiss button on download failure', () => {
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({ status: 'failed' });
      const onDismissError = vi.fn();
      render(
        <ModelCard
          model={model}
          downloadProgress={downloadProgress}
          onDismissError={onDismissError}
        />
      );

      expect(screen.getByRole('button', { name: /dismiss/i })).toBeInTheDocument();
    });

    it('calls onDismissError handler when dismiss button is clicked', async () => {
      const user = userEvent.setup();
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({ status: 'failed' });
      const onDismissError = vi.fn();
      render(
        <ModelCard
          model={model}
          downloadProgress={downloadProgress}
          onDismissError={onDismissError}
        />
      );

      await user.click(screen.getByRole('button', { name: /dismiss/i }));

      expect(onDismissError).toHaveBeenCalledWith('Qwen/Qwen3-Embedding-0.6B');
    });
  });

  describe('delete progress display', () => {
    it('shows spinner during delete operation', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const deleteProgress = createMockDeleteProgress({ isDeleting: true, isFailed: false });
      render(<ModelCard model={model} deleteProgress={deleteProgress} />);

      expect(screen.getByText('Deleting model files...')).toBeInTheDocument();
    });

    it('shows error state when delete fails', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const deleteProgress = createMockDeleteProgress({
        isDeleting: false,
        isFailed: true,
        status: 'failed',
        error: 'Permission denied',
      });
      render(<ModelCard model={model} deleteProgress={deleteProgress} />);

      expect(screen.getByText('Permission denied')).toBeInTheDocument();
    });

    it('shows default error message when delete fails without details', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const deleteProgress = createMockDeleteProgress({
        isDeleting: false,
        isFailed: true,
        status: 'failed',
        error: null,
      });
      render(<ModelCard model={model} deleteProgress={deleteProgress} />);

      expect(screen.getByText('Delete failed')).toBeInTheDocument();
    });

    it('shows retry button on delete failure', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const deleteProgress = createMockDeleteProgress({
        isDeleting: false,
        isFailed: true,
        status: 'failed',
      });
      const onRetryDelete = vi.fn();
      render(
        <ModelCard model={model} deleteProgress={deleteProgress} onRetryDelete={onRetryDelete} />
      );

      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });

    it('calls onRetryDelete handler when delete retry button is clicked', async () => {
      const user = userEvent.setup();
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const deleteProgress = createMockDeleteProgress({
        isDeleting: false,
        isFailed: true,
        status: 'failed',
      });
      const onRetryDelete = vi.fn();
      render(
        <ModelCard model={model} deleteProgress={deleteProgress} onRetryDelete={onRetryDelete} />
      );

      await user.click(screen.getByRole('button', { name: /retry/i }));

      expect(onRetryDelete).toHaveBeenCalledWith('Qwen/Qwen3-Embedding-0.6B');
    });

    it('shows dismiss button on delete failure', () => {
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const deleteProgress = createMockDeleteProgress({
        isDeleting: false,
        isFailed: true,
        status: 'failed',
      });
      const onDismissDeleteError = vi.fn();
      render(
        <ModelCard
          model={model}
          deleteProgress={deleteProgress}
          onDismissDeleteError={onDismissDeleteError}
        />
      );

      expect(screen.getByRole('button', { name: /dismiss/i })).toBeInTheDocument();
    });

    it('calls onDismissDeleteError handler when delete dismiss button is clicked', async () => {
      const user = userEvent.setup();
      const model = createMockModel({ is_installed: true, size_on_disk_mb: 1200 });
      const deleteProgress = createMockDeleteProgress({
        isDeleting: false,
        isFailed: true,
        status: 'failed',
      });
      const onDismissDeleteError = vi.fn();
      render(
        <ModelCard
          model={model}
          deleteProgress={deleteProgress}
          onDismissDeleteError={onDismissDeleteError}
        />
      );

      await user.click(screen.getByRole('button', { name: /dismiss/i }));

      expect(onDismissDeleteError).toHaveBeenCalledWith('Qwen/Qwen3-Embedding-0.6B');
    });

    it('shows active delete task warning when no progress tracking', () => {
      const model = createMockModel({
        is_installed: true,
        size_on_disk_mb: 1200,
        active_delete_task_id: 'delete-task-no-progress',
      });
      render(<ModelCard model={model} />);

      expect(screen.getByText('Deletion in progress...')).toBeInTheDocument();
    });
  });

  describe('expanded details', () => {
    it('shows expand/collapse button', () => {
      const model = createMockModel();
      render(<ModelCard model={model} />);

      expect(screen.getByRole('button', { name: /show details/i })).toBeInTheDocument();
    });

    it('expands to show model details when clicked', async () => {
      const user = userEvent.setup();
      const model = createMockModel({
        embedding_details: {
          dimension: 1024,
          max_sequence_length: 8192,
          pooling_method: 'mean',
          is_asymmetric: true,
          query_prefix: 'query: ',
          document_prefix: 'document: ',
          default_query_instruction: '',
        },
      });
      render(<ModelCard model={model} />);

      await user.click(screen.getByRole('button', { name: /show details/i }));

      expect(screen.getByText('Model ID')).toBeInTheDocument();
      expect(screen.getByText('Qwen/Qwen3-Embedding-0.6B')).toBeInTheDocument();
      expect(screen.getByText('Dimension')).toBeInTheDocument();
      expect(screen.getByText('1024')).toBeInTheDocument();
      expect(screen.getByText('Max Sequence')).toBeInTheDocument();
      expect(screen.getByText('8,192')).toBeInTheDocument();
      expect(screen.getByText('Pooling')).toBeInTheDocument();
      expect(screen.getByText('mean')).toBeInTheDocument();
    });

    it('shows memory by quantization when expanded', async () => {
      const user = userEvent.setup();
      const model = createMockModel({
        memory_mb: { float16: 1200, int8: 600, int4: 400 },
      });
      render(<ModelCard model={model} />);

      await user.click(screen.getByRole('button', { name: /show details/i }));

      expect(screen.getByText('Memory by Quantization')).toBeInTheDocument();
      // Check quantization labels exist
      const detailsSection = screen.getByText('Memory by Quantization').closest('div');
      expect(detailsSection).toBeInTheDocument();
    });

    it('shows collection list when expanded for models in use', async () => {
      const user = userEvent.setup();
      const model = createMockModel({
        is_installed: true,
        size_on_disk_mb: 1200,
        used_by_collections: ['My Documents', 'Research Papers'],
      });
      render(<ModelCard model={model} />);

      await user.click(screen.getByRole('button', { name: /show details/i }));

      expect(screen.getByText('Used by Collections')).toBeInTheDocument();
      expect(screen.getByText('My Documents')).toBeInTheDocument();
      expect(screen.getByText('Research Papers')).toBeInTheDocument();
    });

    it('shows LLM-specific details for LLM models', async () => {
      const user = userEvent.setup();
      const model = createMockModel({
        model_type: 'llm',
        embedding_details: null,
        llm_details: { context_window: 32768 },
      });
      render(<ModelCard model={model} />);

      await user.click(screen.getByRole('button', { name: /show details/i }));

      expect(screen.getByText('Context Window')).toBeInTheDocument();
      expect(screen.getByText('32,768 tokens')).toBeInTheDocument();
    });

    it('collapses details when clicked again', async () => {
      const user = userEvent.setup();
      const model = createMockModel();
      render(<ModelCard model={model} />);

      // Expand
      await user.click(screen.getByRole('button', { name: /show details/i }));
      expect(screen.getByText('Model ID')).toBeInTheDocument();

      // Collapse
      await user.click(screen.getByRole('button', { name: /hide details/i }));
      expect(screen.queryByText('Model ID')).not.toBeInTheDocument();
    });
  });

  describe('download progress takes priority over delete progress', () => {
    it('shows download progress instead of delete progress when both are present', () => {
      const model = createMockModel({ is_installed: false });
      const downloadProgress = createMockDownloadProgress({ status: 'running' });
      const deleteProgress = createMockDeleteProgress({ isDeleting: true });
      render(
        <ModelCard
          model={model}
          downloadProgress={downloadProgress}
          deleteProgress={deleteProgress}
        />
      );

      expect(screen.getByText('Downloading...')).toBeInTheDocument();
      expect(screen.queryByText('Deleting model files...')).not.toBeInTheDocument();
    });
  });
});

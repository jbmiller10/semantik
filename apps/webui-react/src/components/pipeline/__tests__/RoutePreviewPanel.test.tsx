import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { RoutePreviewPanel } from '../RoutePreviewPanel';
import * as useRoutePreviewModule from '@/hooks/useRoutePreview';
import type { PipelineDAG } from '@/types/pipeline';
import type { RoutePreviewResponse } from '@/types/routePreview';

// Mock the hook
vi.mock('@/hooks/useRoutePreview');

describe('RoutePreviewPanel', () => {
  const mockPreviewFile = vi.fn();
  const mockClearPreview = vi.fn();
  const mockOnPathHighlight = vi.fn();

  const mockDAG: PipelineDAG = {
    id: 'test-dag',
    version: '1.0',
    nodes: [
      { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
    ],
    edges: [],
  };

  const mockResult: RoutePreviewResponse = {
    file_info: {
      filename: 'test.txt',
      extension: '.txt',
      mime_type: 'text/plain',
      size_bytes: 1024,
      uri: 'file:///test.txt',
    },
    sniff_result: null,
    routing_stages: [],
    path: ['_source', 'parser1'],
    parsed_metadata: null,
    total_duration_ms: 100,
    warnings: [],
  };

  const defaultHookReturn = {
    isLoading: false,
    error: null,
    result: null,
    file: null,
    previewFile: mockPreviewFile,
    clearPreview: mockClearPreview,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue(defaultHookReturn);
  });

  describe('collapsed state', () => {
    it('starts collapsed by default', () => {
      render(<RoutePreviewPanel dag={mockDAG} />);

      // Content should not be visible when collapsed
      expect(
        screen.queryByText('Upload a sample file to see how it would be routed through your pipeline')
      ).not.toBeInTheDocument();
    });

    it('starts expanded when defaultCollapsed is false', () => {
      render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      // Content should be visible when expanded
      expect(
        screen.getByText('Upload a sample file to see how it would be routed through your pipeline')
      ).toBeInTheDocument();
    });

    it('toggles collapsed state on header click', async () => {
      const user = userEvent.setup();
      render(<RoutePreviewPanel dag={mockDAG} />);

      // Click header to expand
      const header = screen.getByText('Test Route');
      await user.click(header);

      // Content should now be visible
      expect(
        screen.getByText('Upload a sample file to see how it would be routed through your pipeline')
      ).toBeInTheDocument();

      // Click again to collapse
      await user.click(header);

      // Content should be hidden
      expect(
        screen.queryByText('Upload a sample file to see how it would be routed through your pipeline')
      ).not.toBeInTheDocument();
    });
  });

  describe('auto-expand on loading', () => {
    it('auto-expands panel when loading starts', () => {
      // Start with collapsed panel and loading = false
      const { rerender } = render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={true} />);

      // Content should not be visible initially
      expect(screen.queryByText('Analyzing routing...')).not.toBeInTheDocument();

      // Update hook to return loading state
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        isLoading: true,
      });

      // Rerender to trigger useEffect
      rerender(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={true} />);

      // Panel should auto-expand and show loading
      expect(screen.getByText('Analyzing routing...')).toBeInTheDocument();
    });
  });

  describe('file selection', () => {
    it('calls previewFile when file is selected', async () => {
      render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      // Find the file input and simulate file selection
      const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });

      Object.defineProperty(fileInput, 'files', { value: [file] });
      fireEvent.change(fileInput);

      await waitFor(() => {
        expect(mockPreviewFile).toHaveBeenCalledWith(file, mockDAG);
      });
    });
  });

  describe('clear functionality', () => {
    it('calls clearPreview and onPathHighlight(null) on clear', async () => {
      const user = userEvent.setup();
      const file = new File(['test'], 'test.txt', { type: 'text/plain' });

      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        file,
        result: mockResult,
      });

      render(
        <RoutePreviewPanel
          dag={mockDAG}
          defaultCollapsed={false}
          onPathHighlight={mockOnPathHighlight}
        />
      );

      // Find and click clear button
      const clearButton = screen.getByTitle('Clear file');
      await user.click(clearButton);

      expect(mockClearPreview).toHaveBeenCalled();
      expect(mockOnPathHighlight).toHaveBeenCalledWith(null);
    });
  });

  describe('error display', () => {
    it('shows error banner when error is present', () => {
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        error: 'Something went wrong with the preview',
      });

      render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      expect(screen.getByText('Something went wrong with the preview')).toBeInTheDocument();
    });

    it('does not show error banner when no error', () => {
      render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      // Error styling should not be present
      const errorContainer = document.querySelector('.bg-red-500\\/10');
      expect(errorContainer).not.toBeInTheDocument();
    });
  });

  describe('result display', () => {
    it('shows node count in header when result present', () => {
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        result: mockResult,
      });

      render(<RoutePreviewPanel dag={mockDAG} />);

      expect(screen.getByText('(2 nodes)')).toBeInTheDocument();
    });

    it('does not show node count when no result', () => {
      render(<RoutePreviewPanel dag={mockDAG} />);

      expect(screen.queryByText(/nodes\)/)).not.toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('shows loading indicator when loading', () => {
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        isLoading: true,
      });

      render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      expect(screen.getByText('Analyzing routing...')).toBeInTheDocument();
    });

    it('shows spinner animation when loading', () => {
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        isLoading: true,
      });

      const { container } = render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      const spinner = container.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });
  });

  describe('path highlight callback', () => {
    it('calls onPathHighlight with path when result has path', () => {
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        result: mockResult,
      });

      render(
        <RoutePreviewPanel
          dag={mockDAG}
          defaultCollapsed={false}
          onPathHighlight={mockOnPathHighlight}
        />
      );

      expect(mockOnPathHighlight).toHaveBeenCalledWith(['_source', 'parser1']);
    });

    it('calls onPathHighlight with null when result has empty path', () => {
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        result: { ...mockResult, path: [] },
      });

      render(
        <RoutePreviewPanel
          dag={mockDAG}
          defaultCollapsed={false}
          onPathHighlight={mockOnPathHighlight}
        />
      );

      expect(mockOnPathHighlight).toHaveBeenCalledWith(null);
    });

    it('calls onPathHighlight with null when no result', () => {
      render(
        <RoutePreviewPanel
          dag={mockDAG}
          defaultCollapsed={false}
          onPathHighlight={mockOnPathHighlight}
        />
      );

      expect(mockOnPathHighlight).toHaveBeenCalledWith(null);
    });
  });

  describe('empty state', () => {
    it('shows empty state message when no file, result, or loading', () => {
      render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      expect(
        screen.getByText('Upload a sample file to see how it would be routed through your pipeline')
      ).toBeInTheDocument();
    });

    it('does not show empty state when file is present', () => {
      const file = new File(['test'], 'test.txt', { type: 'text/plain' });
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        file,
      });

      render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      expect(
        screen.queryByText('Upload a sample file to see how it would be routed through your pipeline')
      ).not.toBeInTheDocument();
    });

    it('does not show empty state when loading', () => {
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        isLoading: true,
      });

      render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      expect(
        screen.queryByText('Upload a sample file to see how it would be routed through your pipeline')
      ).not.toBeInTheDocument();
    });
  });

  describe('SampleFileSelector integration', () => {
    it('passes isLoading to SampleFileSelector', () => {
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        isLoading: true,
      });

      const { container } = render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      // SampleFileSelector should show spinner when loading
      const selectorSpinner = container.querySelector('.animate-spin');
      expect(selectorSpinner).toBeInTheDocument();
    });

    it('passes selectedFile to SampleFileSelector', () => {
      const file = new File(['test'], 'myfile.txt', { type: 'text/plain' });
      vi.mocked(useRoutePreviewModule.useRoutePreview).mockReturnValue({
        ...defaultHookReturn,
        file,
      });

      render(<RoutePreviewPanel dag={mockDAG} defaultCollapsed={false} />);

      // File name should be visible
      expect(screen.getByText('myfile.txt')).toBeInTheDocument();
    });
  });
});

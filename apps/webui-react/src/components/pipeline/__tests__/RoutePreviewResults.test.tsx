import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@/tests/utils/test-utils';
import { RoutePreviewResults } from '../RoutePreviewResults';
import type { RoutePreviewResponse, StageEvaluationResult } from '@/types/routePreview';
import type { PipelineDAG } from '@/types/pipeline';

// Mock child components to focus on RoutePreviewResults logic
vi.mock('../PathVisualization', () => ({
  PathVisualization: ({ path }: { path: string[] }) => (
    <div data-testid="path-visualization">Path: {path.join(' → ')}</div>
  ),
}));

vi.mock('../EdgeEvaluationTree', () => ({
  EdgeEvaluationTree: ({ stages }: { stages: StageEvaluationResult[] }) => (
    <div data-testid="edge-evaluation-tree">{stages.length} stages</div>
  ),
}));

describe('RoutePreviewResults', () => {
  const createMockDAG = (): PipelineDAG => ({
    id: 'test-dag',
    version: '1.0',
    nodes: [
      { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
      { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: {} },
    ],
    edges: [],
  });

  const createMockResult = (overrides?: Partial<RoutePreviewResponse>): RoutePreviewResponse => ({
    file_info: {
      filename: 'test.txt',
      extension: '.txt',
      mime_type: 'text/plain',
      size_bytes: 1024,
      uri: 'file:///test.txt',
    },
    sniff_result: {
      is_code: false,
      is_structured_data: false,
    },
    routing_stages: [
      {
        stage: 'entry',
        from_node: '_source',
        evaluated_edges: [],
        selected_node: 'parser1',
        metadata_snapshot: {},
      },
    ],
    path: ['_source', 'parser1', 'chunker1'],
    parsed_metadata: null,
    total_duration_ms: 150,
    warnings: [],
    ...overrides,
  });

  describe('formatDuration', () => {
    it('displays milliseconds for times under 1 second', () => {
      const result = createMockResult({ total_duration_ms: 150 });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);
      expect(screen.getByText(/150ms/)).toBeInTheDocument();
    });

    it('displays seconds for times at or over 1 second', () => {
      const result = createMockResult({ total_duration_ms: 1500 });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);
      expect(screen.getByText(/1\.50s/)).toBeInTheDocument();
    });

    it('displays whole milliseconds for fractional values', () => {
      const result = createMockResult({ total_duration_ms: 123.456 });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);
      expect(screen.getByText(/123ms/)).toBeInTheDocument();
    });

    it('displays precise seconds for longer times', () => {
      const result = createMockResult({ total_duration_ms: 2345 });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);
      expect(screen.getByText(/2\.35s/)).toBeInTheDocument();
    });
  });

  describe('warnings banner', () => {
    it('does not show warnings banner when no warnings', () => {
      const result = createMockResult({ warnings: [] });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      // Warning icon should not be present
      expect(screen.queryByText('Warning about something')).not.toBeInTheDocument();
    });

    it('shows warnings banner when warnings present', () => {
      const result = createMockResult({
        warnings: ['Parser failed: timeout', 'Metadata extraction incomplete'],
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.getByText('Parser failed: timeout')).toBeInTheDocument();
      expect(screen.getByText('Metadata extraction incomplete')).toBeInTheDocument();
    });

    it('shows single warning correctly', () => {
      const result = createMockResult({
        warnings: ['Something went wrong'],
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });
  });

  describe('route path display', () => {
    it('shows "No route" badge when path has 1 or fewer nodes', () => {
      // path with only _source (no actual route found)
      const result = createMockResult({ path: ['_source'] });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.getByText('No route')).toBeInTheDocument();
    });

    it('shows "No route" badge for empty path', () => {
      const result = createMockResult({ path: [] });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.getByText('No route')).toBeInTheDocument();
    });

    it('shows node count when route found', () => {
      const result = createMockResult({ path: ['_source', 'parser1', 'chunker1'] });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.getByText('3 nodes')).toBeInTheDocument();
    });

    it('shows "No matching route found" message when no route', () => {
      const result = createMockResult({ path: ['_source'] });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.getByText('No matching route found for this file')).toBeInTheDocument();
    });

    it('renders PathVisualization when route found', () => {
      const result = createMockResult({ path: ['_source', 'parser1'] });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.getByTestId('path-visualization')).toBeInTheDocument();
    });
  });

  describe('sniff result section', () => {
    it('shows sniff result section when sniff_result has data', () => {
      const result = createMockResult({
        sniff_result: {
          is_code: true,
          is_structured_data: false,
          language: 'python',
        },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      // Section title should be present
      expect(screen.getByText('Detected Content')).toBeInTheDocument();
    });

    it('does not show sniff result section when sniff_result is null', () => {
      const result = createMockResult({ sniff_result: null });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.queryByText('Detected Content')).not.toBeInTheDocument();
    });

    it('does not show sniff result section when sniff_result is empty object', () => {
      const result = createMockResult({ sniff_result: {} });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.queryByText('Detected Content')).not.toBeInTheDocument();
    });
  });

  describe('parsed metadata section', () => {
    it('shows parsed metadata section when data present', () => {
      const result = createMockResult({
        parsed_metadata: {
          title: 'Test Document',
          page_count: 5,
        },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.getByText('Parser Output')).toBeInTheDocument();
    });

    it('does not show parsed metadata section when null', () => {
      const result = createMockResult({ parsed_metadata: null });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.queryByText('Parser Output')).not.toBeInTheDocument();
    });

    it('does not show parsed metadata section when empty', () => {
      const result = createMockResult({ parsed_metadata: {} });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.queryByText('Parser Output')).not.toBeInTheDocument();
    });
  });

  describe('file information section', () => {
    it('displays file name', () => {
      const result = createMockResult();
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      // File Information section needs to be expanded (defaultExpanded=false)
      const fileInfoButton = screen.getByText('File Information');
      fireEvent.click(fileInfoButton);

      expect(screen.getByText('test.txt')).toBeInTheDocument();
    });

    it('displays file extension', () => {
      const result = createMockResult();
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      const fileInfoButton = screen.getByText('File Information');
      fireEvent.click(fileInfoButton);

      expect(screen.getByText('.txt')).toBeInTheDocument();
    });

    it('displays "none" when extension is null', () => {
      const result = createMockResult({
        file_info: {
          ...createMockResult().file_info,
          extension: null,
        },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      const fileInfoButton = screen.getByText('File Information');
      fireEvent.click(fileInfoButton);

      expect(screen.getByText('none')).toBeInTheDocument();
    });

    it('displays MIME type', () => {
      const result = createMockResult();
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      const fileInfoButton = screen.getByText('File Information');
      fireEvent.click(fileInfoButton);

      expect(screen.getByText('text/plain')).toBeInTheDocument();
    });

    it('displays "unknown" when MIME type is null', () => {
      const result = createMockResult({
        file_info: {
          ...createMockResult().file_info,
          mime_type: null,
        },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      const fileInfoButton = screen.getByText('File Information');
      fireEvent.click(fileInfoButton);

      expect(screen.getByText('unknown')).toBeInTheDocument();
    });

    it('displays formatted file size', () => {
      const result = createMockResult({
        file_info: {
          ...createMockResult().file_info,
          size_bytes: 1024,
        },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      const fileInfoButton = screen.getByText('File Information');
      fireEvent.click(fileInfoButton);

      expect(screen.getByText('1,024 bytes')).toBeInTheDocument();
    });
  });

  describe('Section component', () => {
    it('toggles section expansion on click', () => {
      const result = createMockResult({
        sniff_result: { test_field: 'test_value' },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      // Detected Content section is defaultExpanded=false
      const sectionButton = screen.getByText('Detected Content');

      // Initially collapsed - content not visible
      expect(screen.queryByText('test_value')).not.toBeInTheDocument();

      // Click to expand
      fireEvent.click(sectionButton);
      expect(screen.getByText('test_value')).toBeInTheDocument();

      // Click to collapse
      fireEvent.click(sectionButton);
      expect(screen.queryByText('test_value')).not.toBeInTheDocument();
    });

    it('shows stage count in routing details badge', () => {
      const result = createMockResult({
        routing_stages: [
          {
            stage: 'entry',
            from_node: '_source',
            evaluated_edges: [],
            selected_node: 'parser1',
            metadata_snapshot: {},
          },
          {
            stage: 'parser_out',
            from_node: 'parser1',
            evaluated_edges: [],
            selected_node: 'chunker1',
            metadata_snapshot: {},
          },
        ],
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      expect(screen.getByText('2 stages')).toBeInTheDocument();
    });
  });

  describe('MetadataDisplay', () => {
    it('displays boolean true with green styling', () => {
      const result = createMockResult({
        sniff_result: { is_code: true },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      // Expand the section
      const sectionButton = screen.getByText('Detected Content');
      fireEvent.click(sectionButton);

      const trueValue = screen.getByText('true');
      expect(trueValue).toHaveClass('text-green-400');
    });

    it('displays boolean false with muted styling', () => {
      const result = createMockResult({
        sniff_result: { is_code: false },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      // Expand the section
      const sectionButton = screen.getByText('Detected Content');
      fireEvent.click(sectionButton);

      const falseValue = screen.getByText('false');
      expect(falseValue).toHaveClass('text-[var(--text-muted)]');
    });

    it('displays object values as JSON', () => {
      const result = createMockResult({
        sniff_result: {
          nested: { key: 'value' },
        },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      const sectionButton = screen.getByText('Detected Content');
      fireEvent.click(sectionButton);

      expect(screen.getByText('{"key":"value"}')).toBeInTheDocument();
    });

    it('displays string values as strings', () => {
      const result = createMockResult({
        sniff_result: {
          language: 'python',
        },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      const sectionButton = screen.getByText('Detected Content');
      fireEvent.click(sectionButton);

      expect(screen.getByText('python')).toBeInTheDocument();
    });

    it('displays number values as strings', () => {
      const result = createMockResult({
        parsed_metadata: {
          page_count: 42,
        },
      });
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      const sectionButton = screen.getByText('Parser Output');
      fireEvent.click(sectionButton);

      expect(screen.getByText('42')).toBeInTheDocument();
    });
  });

  describe('routing details section', () => {
    it('renders EdgeEvaluationTree', () => {
      const result = createMockResult();
      render(<RoutePreviewResults result={result} dag={createMockDAG()} />);

      // Expand routing details section
      const sectionButton = screen.getByText('Routing Details');
      fireEvent.click(sectionButton);

      expect(screen.getByTestId('edge-evaluation-tree')).toBeInTheDocument();
    });
  });
});

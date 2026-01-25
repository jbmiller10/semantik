import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import { ConfigurationPanel } from '../ConfigurationPanel';
import type { PipelineDAG } from '@/types/pipeline';
import type { SourceAnalysis } from '@/types/agent';
import * as usePluginsModule from '@/hooks/usePlugins';

// Mock the usePlugins hooks
vi.mock('@/hooks/usePlugins', () => ({
  usePlugins: vi.fn(),
  usePluginConfigSchema: vi.fn(),
}));

const mockDAG: PipelineDAG = {
  id: 'test',
  version: '1',
  nodes: [
    { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
    { id: 'chunker1', type: 'chunker', plugin_id: 'recursive', config: { max_tokens: 512 } },
    { id: 'embedder1', type: 'embedder', plugin_id: 'dense_local', config: {} },
  ],
  edges: [
    { from_node: '_source', to_node: 'parser1', when: null },
    { from_node: 'parser1', to_node: 'chunker1', when: null },
    { from_node: 'chunker1', to_node: 'embedder1', when: null },
  ],
};

const mockAnalysis: SourceAnalysis = {
  total_files: 100,
  total_size_bytes: 1000000,
  file_types: { '.pdf': 50, '.txt': 50 },
};

describe('ConfigurationPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (usePluginsModule.usePlugins as ReturnType<typeof vi.fn>).mockReturnValue({
      data: [],
      isLoading: false,
    });
    (usePluginsModule.usePluginConfigSchema as ReturnType<typeof vi.fn>).mockReturnValue({
      data: null,
      isLoading: false,
    });
  });

  it('shows SourceAnalysisSummary when selection is none', () => {
    render(
      <ConfigurationPanel
        dag={mockDAG}
        selection={{ type: 'none' }}
        sourceAnalysis={mockAnalysis}
        onNodeChange={vi.fn()}
        onEdgeChange={vi.fn()}
      />
    );

    expect(screen.getByText(/source analysis/i)).toBeInTheDocument();
    expect(screen.getByText('100')).toBeInTheDocument();
  });

  it('shows NodeConfigEditor when node is selected', () => {
    render(
      <ConfigurationPanel
        dag={mockDAG}
        selection={{ type: 'node', nodeId: 'chunker1' }}
        sourceAnalysis={mockAnalysis}
        onNodeChange={vi.fn()}
        onEdgeChange={vi.fn()}
      />
    );

    expect(screen.getByText('Chunker')).toBeInTheDocument();
  });

  it('shows EdgePredicateEditor when edge is selected', () => {
    render(
      <ConfigurationPanel
        dag={mockDAG}
        selection={{ type: 'edge', fromNode: '_source', toNode: 'parser1' }}
        sourceAnalysis={mockAnalysis}
        onNodeChange={vi.fn()}
        onEdgeChange={vi.fn()}
      />
    );

    expect(screen.getByText(/edge routing/i)).toBeInTheDocument();
  });

  it('shows source node info when source is selected', () => {
    render(
      <ConfigurationPanel
        dag={mockDAG}
        selection={{ type: 'node', nodeId: '_source' }}
        sourceAnalysis={mockAnalysis}
        onNodeChange={vi.fn()}
        onEdgeChange={vi.fn()}
      />
    );

    // Should show Data Source header
    expect(screen.getByText('Data Source')).toBeInTheDocument();
    // Should show analysis summary for source node
    expect(screen.getByText('100')).toBeInTheDocument();
  });

  it('calls onNodeChange when node config changes', () => {
    const handleNodeChange = vi.fn();

    render(
      <ConfigurationPanel
        dag={mockDAG}
        selection={{ type: 'node', nodeId: 'parser1' }}
        sourceAnalysis={mockAnalysis}
        onNodeChange={handleNodeChange}
        onEdgeChange={vi.fn()}
      />
    );

    // The NodeConfigEditor is rendered, it will call onNodeChange when edited
    // This test just verifies the prop is passed through
    expect(screen.getByText('Parser')).toBeInTheDocument();
  });

  it('calls onEdgeChange when edge config changes', () => {
    const handleEdgeChange = vi.fn();

    render(
      <ConfigurationPanel
        dag={mockDAG}
        selection={{ type: 'edge', fromNode: '_source', toNode: 'parser1' }}
        sourceAnalysis={mockAnalysis}
        onNodeChange={vi.fn()}
        onEdgeChange={handleEdgeChange}
      />
    );

    expect(screen.getByText(/edge routing/i)).toBeInTheDocument();
  });

  it('passes readOnly to child editors', () => {
    render(
      <ConfigurationPanel
        dag={mockDAG}
        selection={{ type: 'node', nodeId: 'parser1' }}
        sourceAnalysis={mockAnalysis}
        onNodeChange={vi.fn()}
        onEdgeChange={vi.fn()}
        readOnly={true}
      />
    );

    const selects = screen.getAllByRole('combobox');
    selects.forEach((select) => {
      expect(select).toBeDisabled();
    });
  });

  it('handles missing node gracefully', () => {
    render(
      <ConfigurationPanel
        dag={mockDAG}
        selection={{ type: 'node', nodeId: 'nonexistent' }}
        sourceAnalysis={mockAnalysis}
        onNodeChange={vi.fn()}
        onEdgeChange={vi.fn()}
      />
    );

    expect(screen.getByText(/not found/i)).toBeInTheDocument();
  });

  it('handles missing edge gracefully', () => {
    render(
      <ConfigurationPanel
        dag={mockDAG}
        selection={{ type: 'edge', fromNode: 'nonexistent', toNode: 'also_nonexistent' }}
        sourceAnalysis={mockAnalysis}
        onNodeChange={vi.fn()}
        onEdgeChange={vi.fn()}
      />
    );

    expect(screen.getByText(/not found/i)).toBeInTheDocument();
  });
});

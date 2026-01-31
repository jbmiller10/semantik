import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { PipelineVisualization } from '../PipelineVisualization';
import type { PipelineDAG } from '@/types/pipeline';

vi.mock('@/hooks/useAvailablePlugins', () => ({
  useAvailablePlugins: (tier: string) => {
    if (tier === 'parser') {
      return { plugins: [{ id: 'text', name: 'Text Parser', description: '' }], isLoading: false, error: null, refetch: vi.fn() };
    }
    if (tier === 'embedder') {
      return { plugins: [{ id: 'dense_local', name: 'Dense Local', description: '' }], isLoading: false, error: null, refetch: vi.fn() };
    }
    return { plugins: [], isLoading: false, error: null, refetch: vi.fn() };
  },
}));

vi.mock('@/utils/dagLayout', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/utils/dagLayout')>();
  return {
    ...actual,
    generateNodeId: () => 'new-node-1',
  };
});

describe('PipelineVisualization add-node flow', () => {
  const originalGetScreenCTM = (SVGSVGElement.prototype as unknown as { getScreenCTM?: unknown }).getScreenCTM;
  const originalCreateSVGPoint = (SVGSVGElement.prototype as unknown as { createSVGPoint?: unknown }).createSVGPoint;

  beforeEach(() => {
    (SVGSVGElement.prototype as unknown as { getScreenCTM: () => null }).getScreenCTM = () => null;
    (SVGSVGElement.prototype as unknown as { createSVGPoint: () => { x: number; y: number; matrixTransform: () => { x: number; y: number } } }).createSVGPoint =
      () => ({ x: 0, y: 0, matrixTransform: () => ({ x: 0, y: 0 }) });
  });

  afterEach(() => {
    if (originalGetScreenCTM) {
      (SVGSVGElement.prototype as unknown as { getScreenCTM: unknown }).getScreenCTM = originalGetScreenCTM;
    } else {
      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
      delete (SVGSVGElement.prototype as unknown as { getScreenCTM?: unknown }).getScreenCTM;
    }

    if (originalCreateSVGPoint) {
      (SVGSVGElement.prototype as unknown as { createSVGPoint: unknown }).createSVGPoint = originalCreateSVGPoint;
    } else {
      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
      delete (SVGSVGElement.prototype as unknown as { createSVGPoint?: unknown }).createSVGPoint;
    }
  });

  it('creates a parser node from the "+" button and connects it from Source', async () => {
    const user = userEvent.setup();
    const onDagChange = vi.fn();
    const onSelectionChange = vi.fn();

    const dag: PipelineDAG = {
      id: 'd',
      version: '1',
      nodes: [{ id: 'parser1', type: 'parser', plugin_id: 'text', config: {} }],
      edges: [{ from_node: '_source', to_node: 'parser1', when: null }],
    };

    render(<PipelineVisualization dag={dag} onDagChange={onDagChange} onSelectionChange={onSelectionChange} />);

    await user.click(screen.getByRole('button', { name: 'Add parser' }));

    await waitFor(() => {
      expect(onDagChange).toHaveBeenCalled();
    });

    expect(onDagChange).toHaveBeenCalledWith({
      ...dag,
      nodes: [...dag.nodes, { id: 'new-node-1', type: 'parser', plugin_id: 'text', config: {} }],
      edges: [...dag.edges, { from_node: '_source', to_node: 'new-node-1', when: null }],
    });
    expect(onSelectionChange).toHaveBeenCalledWith({ type: 'node', nodeId: 'new-node-1' });
  });

  it('shows upstream picker for embedder tier and connects from multiple upstream nodes', async () => {
    const user = userEvent.setup();
    const onDagChange = vi.fn();
    const onSelectionChange = vi.fn();

    const dag: PipelineDAG = {
      id: 'd',
      version: '1',
      nodes: [
        { id: 'chunker1', type: 'chunker', plugin_id: 'semantic', config: {} },
        { id: 'extractor1', type: 'extractor', plugin_id: 'keyword', config: {} },
      ],
      edges: [],
    };

    render(<PipelineVisualization dag={dag} onDagChange={onDagChange} onSelectionChange={onSelectionChange} />);

    await user.click(screen.getByRole('button', { name: 'Add embedder' }));

    await waitFor(() => {
      expect(screen.getByRole('dialog', { name: 'Select upstream nodes' })).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Connect (2)' }));

    expect(onDagChange).toHaveBeenCalledWith({
      ...dag,
      nodes: [
        ...dag.nodes,
        { id: 'new-node-1', type: 'embedder', plugin_id: 'dense_local', config: {} },
      ],
      edges: [
        { from_node: 'extractor1', to_node: 'new-node-1', when: null },
        { from_node: 'chunker1', to_node: 'new-node-1', when: null },
      ],
    });
    expect(onSelectionChange).toHaveBeenCalledWith({ type: 'node', nodeId: 'new-node-1' });
  });
});

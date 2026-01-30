import { describe, it, expect } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import { PathVisualization } from '../PathVisualization';
import type { PipelineDAG } from '@/types/pipeline';

describe('PathVisualization', () => {
  const createMockDAG = (
    nodes: { id: string; type: string; plugin_id: string }[] = []
  ): PipelineDAG => ({
    id: 'test-dag',
    version: '1.0',
    nodes: nodes.map((n) => ({ ...n, config: {} })),
    edges: [],
  });

  describe('empty state', () => {
    it('shows empty state message for empty path', () => {
      render(<PathVisualization path={[]} dag={createMockDAG()} />);
      expect(screen.getByText('No path computed')).toBeInTheDocument();
    });
  });

  describe('getNodeDisplayName', () => {
    it('displays "Source" for _source node', () => {
      const dag = createMockDAG();
      render(<PathVisualization path={['_source']} dag={dag} />);
      expect(screen.getByText('Source')).toBeInTheDocument();
    });

    it('displays plugin_id for found nodes', () => {
      const dag = createMockDAG([
        { id: 'parser1', type: 'parser', plugin_id: 'unstructured' },
      ]);
      render(<PathVisualization path={['parser1']} dag={dag} />);
      expect(screen.getByText('unstructured')).toBeInTheDocument();
    });

    it('displays node id for missing nodes', () => {
      const dag = createMockDAG([]); // No nodes defined
      render(<PathVisualization path={['unknown_node']} dag={dag} />);
      expect(screen.getByText('unknown_node')).toBeInTheDocument();
    });
  });

  describe('getNodeIcon', () => {
    it('renders FileInput icon for _source', () => {
      const dag = createMockDAG();
      render(<PathVisualization path={['_source']} dag={dag} />);

      // Check for the source node with title containing "source"
      const sourceNode = screen.getByTitle(/source/i);
      expect(sourceNode).toBeInTheDocument();
    });

    it('renders Box icon for parser type', () => {
      const dag = createMockDAG([
        { id: 'parser1', type: 'parser', plugin_id: 'text' },
      ]);
      render(<PathVisualization path={['parser1']} dag={dag} />);

      // Check node title includes "parser"
      const parserNode = screen.getByTitle(/parser.*parser1/i);
      expect(parserNode).toBeInTheDocument();
    });

    it('renders Layers icon for chunker type', () => {
      const dag = createMockDAG([
        { id: 'chunker1', type: 'chunker', plugin_id: 'recursive' },
      ]);
      render(<PathVisualization path={['chunker1']} dag={dag} />);

      // Check node title includes "chunker"
      const chunkerNode = screen.getByTitle(/chunker.*chunker1/i);
      expect(chunkerNode).toBeInTheDocument();
    });

    it('renders Sparkles icon for extractor type', () => {
      const dag = createMockDAG([
        { id: 'extractor1', type: 'extractor', plugin_id: 'entities' },
      ]);
      render(<PathVisualization path={['extractor1']} dag={dag} />);

      // Check node title includes "extractor"
      const extractorNode = screen.getByTitle(/extractor.*extractor1/i);
      expect(extractorNode).toBeInTheDocument();
    });

    it('renders Cpu icon for embedder type', () => {
      const dag = createMockDAG([
        { id: 'embedder1', type: 'embedder', plugin_id: 'dense' },
      ]);
      render(<PathVisualization path={['embedder1']} dag={dag} />);

      // Check node title includes "embedder"
      const embedderNode = screen.getByTitle(/embedder.*embedder1/i);
      expect(embedderNode).toBeInTheDocument();
    });

    it('renders default Box icon for unknown type', () => {
      const dag = createMockDAG([
        { id: 'custom1', type: 'unknown_type', plugin_id: 'custom' },
      ]);
      render(<PathVisualization path={['custom1']} dag={dag} />);

      // Should still render the node
      expect(screen.getByText('custom')).toBeInTheDocument();
    });
  });

  describe('path rendering', () => {
    it('renders all nodes in the path', () => {
      const dag = createMockDAG([
        { id: 'parser1', type: 'parser', plugin_id: 'text' },
        { id: 'chunker1', type: 'chunker', plugin_id: 'recursive' },
      ]);
      render(<PathVisualization path={['_source', 'parser1', 'chunker1']} dag={dag} />);

      expect(screen.getByText('Source')).toBeInTheDocument();
      expect(screen.getByText('text')).toBeInTheDocument();
      expect(screen.getByText('recursive')).toBeInTheDocument();
    });

    it('renders chevrons between nodes but not after the last', () => {
      const dag = createMockDAG([
        { id: 'parser1', type: 'parser', plugin_id: 'text' },
        { id: 'chunker1', type: 'chunker', plugin_id: 'recursive' },
      ]);
      const { container } = render(
        <PathVisualization path={['_source', 'parser1', 'chunker1']} dag={dag} />
      );

      // 3 nodes should have 2 chevrons between them
      const chevrons = container.querySelectorAll('.lucide-chevron-right');
      expect(chevrons).toHaveLength(2);
    });

    it('does not render chevron for single node path', () => {
      const dag = createMockDAG();
      const { container } = render(
        <PathVisualization path={['_source']} dag={dag} />
      );

      const chevrons = container.querySelectorAll('.lucide-chevron-right');
      expect(chevrons).toHaveLength(0);
    });

    it('renders two chevrons for three node path', () => {
      const dag = createMockDAG([
        { id: 'p1', type: 'parser', plugin_id: 'a' },
        { id: 'c1', type: 'chunker', plugin_id: 'b' },
      ]);
      const { container } = render(
        <PathVisualization path={['_source', 'p1', 'c1']} dag={dag} />
      );

      const chevrons = container.querySelectorAll('.lucide-chevron-right');
      expect(chevrons).toHaveLength(2);
    });
  });

  describe('node titles', () => {
    it('includes type and node id in title', () => {
      const dag = createMockDAG([
        { id: 'my_parser', type: 'parser', plugin_id: 'text' },
      ]);
      render(<PathVisualization path={['my_parser']} dag={dag} />);

      const node = screen.getByTitle('parser: my_parser');
      expect(node).toBeInTheDocument();
    });

    it('shows "source" type for _source node', () => {
      const dag = createMockDAG();
      render(<PathVisualization path={['_source']} dag={dag} />);

      const node = screen.getByTitle('source: _source');
      expect(node).toBeInTheDocument();
    });

    it('shows "node" for unknown type', () => {
      const dag = createMockDAG([]); // Node not in DAG
      render(<PathVisualization path={['missing_node']} dag={dag} />);

      const node = screen.getByTitle('node: missing_node');
      expect(node).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('applies green styling to path nodes', () => {
      const dag = createMockDAG([
        { id: 'parser1', type: 'parser', plugin_id: 'text' },
      ]);
      const { container } = render(
        <PathVisualization path={['parser1']} dag={dag} />
      );

      // Check for green border class
      const nodeElement = container.querySelector('.border-green-500\\/30');
      expect(nodeElement).toBeInTheDocument();
    });
  });

  describe('parallel paths tree structure', () => {
    it('renders visual branch indicators for parallel paths', () => {
      const dag = createMockDAG([
        { id: 'parser', type: 'parser', plugin_id: 'text' },
        { id: 'chunker', type: 'chunker', plugin_id: 'recursive' },
        { id: 'embedder', type: 'embedder', plugin_id: 'dense' },
        { id: 'extractor', type: 'extractor', plugin_id: 'keyword' },
      ]);

      const parallelPaths = [
        { path_name: 'embedding', nodes: ['_source', 'parser', 'chunker', 'embedder'] },
        { path_name: 'extraction', nodes: ['_source', 'parser', 'chunker', 'extractor'] },
      ];

      const { container } = render(
        <PathVisualization
          path={parallelPaths[0].nodes}
          paths={parallelPaths}
          dag={dag}
        />
      );

      // Should show divergence indicator (path-divergence class)
      const divergenceContainer = container.querySelector('.path-divergence');
      expect(divergenceContainer).toBeInTheDocument();

      // Should show both path names
      expect(screen.getByText('embedding')).toBeInTheDocument();
      expect(screen.getByText('extraction')).toBeInTheDocument();
    });

    it('shows common prefix before divergence point', () => {
      const dag = createMockDAG([
        { id: 'parser', type: 'parser', plugin_id: 'text' },
        { id: 'chunker', type: 'chunker', plugin_id: 'recursive' },
        { id: 'embedder', type: 'embedder', plugin_id: 'dense' },
        { id: 'extractor', type: 'extractor', plugin_id: 'keyword' },
      ]);

      const parallelPaths = [
        { path_name: 'embedding', nodes: ['_source', 'parser', 'chunker', 'embedder'] },
        { path_name: 'extraction', nodes: ['_source', 'parser', 'chunker', 'extractor'] },
      ];

      render(
        <PathVisualization
          path={parallelPaths[0].nodes}
          paths={parallelPaths}
          dag={dag}
        />
      );

      // Common prefix: Source > text > recursive
      // These should appear with gray styling (common prefix)
      expect(screen.getByText('Source')).toBeInTheDocument();
      expect(screen.getByText('text')).toBeInTheDocument();
      expect(screen.getByText('recursive')).toBeInTheDocument();
    });

    it('marks primary path with badge', () => {
      const dag = createMockDAG([
        { id: 'embedder', type: 'embedder', plugin_id: 'dense' },
        { id: 'extractor', type: 'extractor', plugin_id: 'keyword' },
      ]);

      const parallelPaths = [
        { path_name: 'embedding', nodes: ['_source', 'embedder'] },
        { path_name: 'extraction', nodes: ['_source', 'extractor'] },
      ];

      render(
        <PathVisualization
          path={parallelPaths[0].nodes}
          paths={parallelPaths}
          dag={dag}
        />
      );

      // Primary path should have a badge
      expect(screen.getByText('primary')).toBeInTheDocument();
    });
  });
});

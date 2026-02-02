import { describe, it, expect } from 'vitest';
import { render } from '@/tests/utils/test-utils';
import { PipelineVisualization } from '../PipelineVisualization';
import type { PipelineDAG } from '@/types/pipeline';

// Default pipeline (matches defaults.py)
const defaultPipeline: PipelineDAG = {
  id: 'default-v1',
  version: '1',
  nodes: [
    { id: 'unstructured_parser', type: 'parser', plugin_id: 'unstructured', config: { strategy: 'auto' } },
    { id: 'text_parser', type: 'parser', plugin_id: 'text', config: {} },
    { id: 'chunker', type: 'chunker', plugin_id: 'recursive', config: { max_tokens: 512, overlap_tokens: 50 } },
    { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: { model: 'BAAI/bge-base-en-v1.5' } },
  ],
  edges: [
    { from_node: '_source', to_node: 'unstructured_parser', when: { mime_type: ['application/pdf', 'application/vnd.*'] } },
    { from_node: '_source', to_node: 'text_parser', when: null },
    { from_node: 'unstructured_parser', to_node: 'chunker', when: null },
    { from_node: 'text_parser', to_node: 'chunker', when: null },
    { from_node: 'chunker', to_node: 'embedder', when: null },
  ],
};

// Complex pipeline with extractor
const complexPipeline: PipelineDAG = {
  id: 'complex-v1',
  version: '1',
  nodes: [
    { id: 'parser', type: 'parser', plugin_id: 'unstructured', config: {} },
    { id: 'chunker', type: 'chunker', plugin_id: 'semantic', config: { max_tokens: 256 } },
    { id: 'extractor', type: 'extractor', plugin_id: 'keyword_extractor', config: { max_keywords: 10 } },
    { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: {} },
  ],
  edges: [
    { from_node: '_source', to_node: 'parser', when: null },
    { from_node: 'parser', to_node: 'chunker', when: null },
    { from_node: 'chunker', to_node: 'extractor', when: null },
    { from_node: 'extractor', to_node: 'embedder', when: null },
  ],
};

// Minimal pipeline (single path)
const minimalPipeline: PipelineDAG = {
  id: 'minimal-v1',
  version: '1',
  nodes: [
    { id: 'parser', type: 'parser', plugin_id: 'text', config: {} },
    { id: 'embedder', type: 'embedder', plugin_id: 'dense_local', config: {} },
  ],
  edges: [
    { from_node: '_source', to_node: 'parser', when: null },
    { from_node: 'parser', to_node: 'embedder', when: null },
  ],
};

describe('PipelineVisualization Visual Tests', () => {
  it('renders default pipeline correctly', () => {
    const { container } = render(<PipelineVisualization dag={defaultPipeline} />);

    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();

    // Check all nodes are present
    expect(container.querySelectorAll('g[data-node-id]')).toHaveLength(5); // 4 nodes + source

    // Check edges are present
    expect(container.querySelectorAll('g[data-edge-id]')).toHaveLength(5);
  });

  it('renders complex pipeline with extractor', () => {
    const { container } = render(<PipelineVisualization dag={complexPipeline} />);

    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();

    // Check extractor node exists
    const extractorNode = container.querySelector('g[data-node-id="extractor"]');
    expect(extractorNode).toBeInTheDocument();
  });

  it('renders minimal pipeline', () => {
    const { container } = render(<PipelineVisualization dag={minimalPipeline} />);

    // Should have source + 2 nodes
    expect(container.querySelectorAll('g[data-node-id]')).toHaveLength(3);
  });

  it('maintains consistent dimensions', () => {
    const { container: container1 } = render(<PipelineVisualization dag={defaultPipeline} />);
    const { container: container2 } = render(<PipelineVisualization dag={defaultPipeline} />);

    const svg1 = container1.querySelector('svg');
    const svg2 = container2.querySelector('svg');

    // Same DAG should produce same dimensions
    expect(svg1?.getAttribute('width')).toBe(svg2?.getAttribute('width'));
    expect(svg1?.getAttribute('height')).toBe(svg2?.getAttribute('height'));
  });

  it('scales width with horizontal spread within tiers', () => {
    // Pipeline with 3 parsers (horizontal spread in vertical layout)
    const widePipeline: PipelineDAG = {
      id: 'wide',
      version: '1',
      nodes: [
        { id: 'p1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'p2', type: 'parser', plugin_id: 'pdf', config: {} },
        { id: 'p3', type: 'parser', plugin_id: 'office', config: {} },
        { id: 'e', type: 'embedder', plugin_id: 'dense', config: {} },
      ],
      edges: [
        { from_node: '_source', to_node: 'p1', when: { extension: '.txt' } },
        { from_node: '_source', to_node: 'p2', when: { extension: '.pdf' } },
        { from_node: '_source', to_node: 'p3', when: null },
        { from_node: 'p1', to_node: 'e', when: null },
        { from_node: 'p2', to_node: 'e', when: null },
        { from_node: 'p3', to_node: 'e', when: null },
      ],
    };

    const { container: wideContainer } = render(<PipelineVisualization dag={widePipeline} />);
    const { container: defaultContainer } = render(<PipelineVisualization dag={defaultPipeline} />);

    const wideWidth = parseInt(wideContainer.querySelector('svg')?.getAttribute('width') || '0');
    const defaultWidth = parseInt(defaultContainer.querySelector('svg')?.getAttribute('width') || '0');

    // Wide pipeline should be wider (3 parsers vs 2 in parser tier)
    expect(wideWidth).toBeGreaterThan(defaultWidth);
  });
});

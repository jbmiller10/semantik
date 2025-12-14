import { ReactFlow, Controls, Background, MiniMap } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import type { EntityNode, RelationshipEdge } from '@/types/graph';
import { getEntityColor } from '@/types/graph';

/**
 * Test component to verify React Flow installation and type integration.
 * This component displays a simple graph with sample entity nodes.
 */

const sampleNodes: EntityNode[] = [
  {
    id: '1',
    position: { x: 0, y: 0 },
    data: {
      label: 'John Smith',
      entityType: 'PERSON',
      hop: 0,
      entityId: 1,
      confidence: 0.95,
    },
    style: { background: getEntityColor('PERSON'), color: 'white' },
  },
  {
    id: '2',
    position: { x: 200, y: 0 },
    data: {
      label: 'Acme Corp',
      entityType: 'ORG',
      hop: 1,
      entityId: 2,
      confidence: 0.92,
    },
    style: { background: getEntityColor('ORG'), color: 'white' },
  },
  {
    id: '3',
    position: { x: 100, y: 100 },
    data: {
      label: 'New York',
      entityType: 'GPE',
      hop: 1,
      entityId: 3,
      confidence: 0.88,
    },
    style: { background: getEntityColor('GPE'), color: 'white' },
  },
];

const sampleEdges: RelationshipEdge[] = [
  {
    id: 'e1-2',
    source: '1',
    target: '2',
    label: 'WORKS_FOR',
    data: { relationshipType: 'WORKS_FOR', confidence: 0.9 },
  },
  {
    id: 'e1-3',
    source: '1',
    target: '3',
    label: 'LIVES_IN',
    data: { relationshipType: 'LIVES_IN', confidence: 0.8 },
  },
];

export interface GraphTestProps {
  /** Custom class name for the container */
  className?: string;
}

export function GraphTest({ className }: GraphTestProps) {
  return (
    <div className={className} style={{ width: '400px', height: '400px' }}>
      <ReactFlow
        nodes={sampleNodes}
        edges={sampleEdges}
        fitView
        attributionPosition="bottom-left"
      >
        <Controls />
        <MiniMap />
        <Background gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}

export default GraphTest;

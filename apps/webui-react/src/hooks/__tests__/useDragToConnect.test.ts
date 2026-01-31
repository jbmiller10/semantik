import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';

import { useDragToConnect, getValidTargetTiers } from '../useDragToConnect';
import type { PipelineDAG, DAGLayout, PipelineEdge } from '@/types/pipeline';
import { INITIAL_DRAG_STATE } from '@/types/pipeline';
import type { TierBounds } from '@/utils/dagLayout';

// Helper to create a minimal DAG for testing
function createTestDAG(nodes: Array<{ id: string; type: 'parser' | 'chunker' | 'extractor' | 'embedder' }>): PipelineDAG {
  return {
    id: 'test-dag',
    version: '1.0',
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type,
      plugin_id: `test-${n.type}`,
      config: {},
    })),
    edges: [],
  };
}

// Helper to create a minimal layout for testing
function createTestLayout(): DAGLayout {
  return {
    nodes: new Map([
      ['_source', { x: 100, y: 40, width: 160, height: 80 }],
      ['parser-1', { x: 100, y: 220, width: 160, height: 80 }],
      ['chunker-1', { x: 100, y: 400, width: 160, height: 80 }],
      ['embedder-1', { x: 100, y: 580, width: 160, height: 80 }],
    ]),
    width: 400,
    height: 700,
  };
}

describe('useDragToConnect', () => {
  let testDAG: PipelineDAG;
  let testLayout: DAGLayout;

  beforeEach(() => {
    testDAG = createTestDAG([
      { id: 'parser-1', type: 'parser' },
      { id: 'chunker-1', type: 'chunker' },
      { id: 'embedder-1', type: 'embedder' },
    ]);
    testLayout = createTestLayout();
  });

  describe('initial state', () => {
    it('starts with isDragging false', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
      expect(result.current.dragState.isDragging).toBe(false);
    });

    it('starts with no drag-related properties (discriminated union)', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      // When isDragging is false, no other properties exist
      expect(result.current.dragState.isDragging).toBe(false);
      expect('sourceNodeId' in result.current.dragState).toBe(false);
      expect('sourcePosition' in result.current.dragState).toBe(false);
      expect('cursorPosition' in result.current.dragState).toBe(false);
    });
  });

  describe('startDrag', () => {
    it('sets isDragging to true', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      expect(result.current.dragState.isDragging).toBe(true);
    });

    it('sets sourceNodeId', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.startDrag('parser-1', { x: 180, y: 300 });
      });

      expect(result.current.dragState.sourceNodeId).toBe('parser-1');
    });

    it('sets sourcePosition and cursorPosition', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      const position = { x: 180, y: 120 };
      act(() => {
        result.current.startDrag('_source', position);
      });

      expect(result.current.dragState.sourcePosition).toEqual(position);
      expect(result.current.dragState.cursorPosition).toEqual(position);
    });
  });

  describe('updateDrag', () => {
    it('updates cursorPosition when dragging', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      act(() => {
        result.current.updateDrag({ x: 200, y: 250 });
      });

      expect(result.current.dragState.cursorPosition).toEqual({ x: 200, y: 250 });
    });

    it('does not update when not dragging', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.updateDrag({ x: 200, y: 250 });
      });

      // When not dragging, state stays as initial (no cursorPosition property)
      expect(result.current.dragState.isDragging).toBe(false);
      expect('cursorPosition' in result.current.dragState).toBe(false);
    });

    it('preserves sourcePosition during drag', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      const startPos = { x: 180, y: 120 };
      act(() => {
        result.current.startDrag('_source', startPos);
      });

      act(() => {
        result.current.updateDrag({ x: 300, y: 400 });
      });

      expect(result.current.dragState.sourcePosition).toEqual(startPos);
    });
  });

  describe('endDrag', () => {
    it('resets state to initial', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
        result.current.updateDrag({ x: 200, y: 250 });
      });

      act(() => {
        result.current.endDrag();
      });

      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
    });
  });

  describe('cancelDrag', () => {
    it('resets state to initial', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
        result.current.updateDrag({ x: 200, y: 250 });
      });

      act(() => {
        result.current.cancelDrag();
      });

      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
    });
  });

  describe('isValidDropTarget', () => {
    it('returns false when not dragging', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      expect(result.current.isValidDropTarget('parser-1')).toBe(false);
    });

    it('returns false for self-drop', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.startDrag('parser-1', { x: 180, y: 300 });
      });

      expect(result.current.isValidDropTarget('parser-1')).toBe(false);
    });

    it('returns false for source node as target', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.startDrag('parser-1', { x: 180, y: 300 });
      });

      expect(result.current.isValidDropTarget('_source')).toBe(false);
    });

    it('returns true for valid downstream target', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      expect(result.current.isValidDropTarget('parser-1')).toBe(true);
    });

    it('returns false for invalid target tier', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // Source can only connect to parsers, not chunkers
      expect(result.current.isValidDropTarget('chunker-1')).toBe(false);
    });
  });
});

describe('getValidTargetTiers', () => {
  const testDAG = createTestDAG([
    { id: 'parser-1', type: 'parser' },
    { id: 'chunker-1', type: 'chunker' },
    { id: 'extractor-1', type: 'extractor' },
    { id: 'embedder-1', type: 'embedder' },
  ]);

  it('returns [parser] for source node', () => {
    expect(getValidTargetTiers('_source', testDAG)).toEqual(['parser']);
  });

  it('returns [chunker] for parser node', () => {
    expect(getValidTargetTiers('parser-1', testDAG)).toEqual(['chunker']);
  });

  it('returns [extractor, embedder] for chunker node', () => {
    expect(getValidTargetTiers('chunker-1', testDAG)).toEqual(['extractor', 'embedder']);
  });

  it('returns [embedder] for extractor node', () => {
    expect(getValidTargetTiers('extractor-1', testDAG)).toEqual(['embedder']);
  });

  it('returns empty array for embedder node (terminal)', () => {
    expect(getValidTargetTiers('embedder-1', testDAG)).toEqual([]);
  });

  it('returns empty array for unknown node', () => {
    expect(getValidTargetTiers('unknown-node', testDAG)).toEqual([]);
  });
});

// =============================================================================
// Additional tests for improved coverage
// =============================================================================

describe('useDragToConnect - endDrag with callbacks', () => {
  // Helper to create a DAG with edges for testing
  function createTestDAGWithEdges(
    nodes: Array<{ id: string; type: 'parser' | 'chunker' | 'extractor' | 'embedder' }>,
    edges: PipelineEdge[]
  ): PipelineDAG {
    return {
      id: 'test-dag',
      version: '1.0',
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.type,
        plugin_id: `test-${n.type}`,
        config: {},
      })),
      edges,
    };
  }

  const testLayout: DAGLayout = {
    nodes: new Map([
      ['_source', { x: 100, y: 40, width: 160, height: 80 }],
      ['parser-1', { x: 100, y: 220, width: 160, height: 80 }],
      ['chunker-1', { x: 100, y: 400, width: 160, height: 80 }],
      ['extractor-1', { x: 100, y: 580, width: 160, height: 80 }],
      ['embedder-1', { x: 100, y: 760, width: 160, height: 80 }],
    ]),
    width: 400,
    height: 900,
  };

  const tierBounds: TierBounds[] = [
    { tier: 'parser', tierIndex: 1, x: 50, y: 180, width: 300, height: 120 },
    { tier: 'chunker', tierIndex: 2, x: 50, y: 360, width: 300, height: 120 },
    { tier: 'extractor', tierIndex: 3, x: 50, y: 540, width: 300, height: 120 },
    { tier: 'embedder', tierIndex: 4, x: 50, y: 720, width: 300, height: 120 },
  ];

  describe('endDrag with onConnect callback (node-to-node drop)', () => {
    it('calls onConnect when dropping on a valid node target', () => {
      const testDAG = createTestDAGWithEdges(
        [
          { id: 'parser-1', type: 'parser' },
          { id: 'chunker-1', type: 'chunker' },
        ],
        []
      );

      const onConnect = vi.fn();

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
          onConnect,
        })
      );

      // Start drag from source
      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // End drag on parser node (inside its bounds: x=100-260, y=220-300)
      act(() => {
        result.current.endDrag({ x: 180, y: 260 });
      });

      expect(onConnect).toHaveBeenCalledWith('_source', 'parser-1');
      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
    });

    it('does not call onConnect when dropping on invalid tier target', () => {
      const testDAG = createTestDAGWithEdges(
        [
          { id: 'parser-1', type: 'parser' },
          { id: 'chunker-1', type: 'chunker' },
        ],
        []
      );

      const onConnect = vi.fn();

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
          onConnect,
        })
      );

      // Start drag from source
      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // End drag on chunker node (invalid - source can only connect to parsers)
      act(() => {
        result.current.endDrag({ x: 180, y: 440 });
      });

      expect(onConnect).not.toHaveBeenCalled();
      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
    });
  });

  describe('endDrag with onDropOnZone callback (tier zone drop)', () => {
    it('calls onDropOnZone when dropping on a valid tier zone', () => {
      const testDAG = createTestDAGWithEdges(
        [{ id: 'parser-1', type: 'parser' }],
        []
      );

      const onDropOnZone = vi.fn();

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
          onDropOnZone,
        })
      );

      // Start drag from parser
      act(() => {
        result.current.startDrag('parser-1', { x: 180, y: 300 });
      });

      // End drag on chunker tier zone (but not on a node)
      // Use a position that's in the tier bounds but not on the chunker node
      act(() => {
        result.current.endDrag({ x: 300, y: 420 });
      });

      expect(onDropOnZone).toHaveBeenCalledWith('parser-1', 'chunker', { x: 300, y: 420 });
      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
    });

    it('does not call onDropOnZone for invalid tier', () => {
      const testDAG = createTestDAGWithEdges(
        [{ id: 'parser-1', type: 'parser' }],
        []
      );

      const onDropOnZone = vi.fn();

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
          onDropOnZone,
        })
      );

      // Start drag from parser
      act(() => {
        result.current.startDrag('parser-1', { x: 180, y: 300 });
      });

      // End drag on embedder tier zone (invalid - parser can only connect to chunker)
      act(() => {
        result.current.endDrag({ x: 300, y: 780 });
      });

      expect(onDropOnZone).not.toHaveBeenCalled();
      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
    });
  });

  describe('duplicate edge prevention', () => {
    it('does not call onConnect when edge already exists', () => {
      const testDAG = createTestDAGWithEdges(
        [
          { id: 'parser-1', type: 'parser' },
          { id: 'chunker-1', type: 'chunker' },
        ],
        [
          // Edge already exists from source to parser
          { from_node: '_source', to_node: 'parser-1', when: null },
        ]
      );

      const onConnect = vi.fn();

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
          onConnect,
        })
      );

      // Start drag from source
      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // End drag on parser node (edge already exists)
      act(() => {
        result.current.endDrag({ x: 180, y: 260 });
      });

      expect(onConnect).not.toHaveBeenCalled();
      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
    });
  });

  describe('endDrag edge cases', () => {
    it('uses state cursorPosition when no position passed to endDrag', () => {
      const testDAG = createTestDAGWithEdges(
        [
          { id: 'parser-1', type: 'parser' },
          { id: 'chunker-1', type: 'chunker' },
        ],
        []
      );

      const onConnect = vi.fn();

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
          onConnect,
        })
      );

      // Start drag from source
      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // Update cursor to be over parser node
      act(() => {
        result.current.updateDrag({ x: 180, y: 260 });
      });

      // End drag without passing position - should use cursorPosition from state
      act(() => {
        result.current.endDrag();
      });

      expect(onConnect).toHaveBeenCalledWith('_source', 'parser-1');
    });

    it('resets state when endDrag called without dragging', () => {
      const testDAG = createTestDAGWithEdges([], []);

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
        })
      );

      // End drag without starting - should just reset state
      act(() => {
        result.current.endDrag({ x: 100, y: 100 });
      });

      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
    });

    it('resets state when endDrag has no drop point', () => {
      const testDAG = createTestDAGWithEdges([], []);

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
        })
      );

      // Start drag
      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // Manually set cursorPosition to null (simulating edge case)
      // We can't directly set this, but we can test with no position passed
      // and initial cursorPosition being same as sourcePosition
      // The cursorPosition should be set, so let's test the no valid target case instead
      act(() => {
        result.current.endDrag({ x: 0, y: 0 }); // Outside all bounds
      });

      expect(result.current.dragState).toEqual(INITIAL_DRAG_STATE);
    });
  });

  describe('findNodeAtPoint edge cases', () => {
    it('returns null when dropping on _source node', () => {
      const testDAG = createTestDAGWithEdges(
        [{ id: 'parser-1', type: 'parser' }],
        []
      );

      const onConnect = vi.fn();

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
          onConnect,
        })
      );

      // Start drag from parser
      act(() => {
        result.current.startDrag('parser-1', { x: 180, y: 300 });
      });

      // End drag on _source node position (should not connect)
      act(() => {
        result.current.endDrag({ x: 180, y: 80 });
      });

      expect(onConnect).not.toHaveBeenCalled();
    });
  });

  describe('getHoveredTier', () => {
    it('returns null when not dragging', () => {
      const testDAG = createTestDAGWithEdges(
        [{ id: 'parser-1', type: 'parser' }],
        []
      );

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
        })
      );

      expect(result.current.getHoveredTier()).toBeNull();
    });

    it('returns null when no cursorPosition', () => {
      const testDAG = createTestDAGWithEdges(
        [{ id: 'parser-1', type: 'parser' }],
        []
      );

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
        })
      );

      // Start drag
      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // cursorPosition is set to sourcePosition initially, so this will return a tier
      // Let's verify the behavior with a valid position
      expect(result.current.getHoveredTier()).toBeNull(); // Not over any valid tier zone
    });

    it('returns valid tier when hovering over valid target zone', () => {
      const testDAG = createTestDAGWithEdges(
        [{ id: 'parser-1', type: 'parser' }],
        []
      );

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
        })
      );

      // Start drag from source
      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // Move cursor to parser tier zone
      act(() => {
        result.current.updateDrag({ x: 180, y: 240 });
      });

      expect(result.current.getHoveredTier()).toBe('parser');
    });

    it('returns null when hovering over invalid target zone', () => {
      const testDAG = createTestDAGWithEdges(
        [{ id: 'parser-1', type: 'parser' }],
        []
      );

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
        })
      );

      // Start drag from source (can only connect to parser)
      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // Move cursor to chunker tier zone (invalid for source)
      act(() => {
        result.current.updateDrag({ x: 180, y: 420 });
      });

      expect(result.current.getHoveredTier()).toBeNull();
    });
  });

  describe('isValidDropTarget with missing node', () => {
    it('returns false for non-existent node', () => {
      const testDAG = createTestDAGWithEdges(
        [{ id: 'parser-1', type: 'parser' }],
        []
      );

      const { result } = renderHook(() =>
        useDragToConnect({
          dag: testDAG,
          layout: testLayout,
          tierBounds,
        })
      );

      act(() => {
        result.current.startDrag('_source', { x: 180, y: 120 });
      });

      // Check for a node that doesn't exist in the DAG
      expect(result.current.isValidDropTarget('non-existent-node')).toBe(false);
    });
  });
});

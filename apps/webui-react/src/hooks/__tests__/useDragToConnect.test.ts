import { describe, it, expect, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';

import { useDragToConnect, getValidTargetTiers } from '../useDragToConnect';
import type { PipelineDAG, DAGLayout } from '@/types/pipeline';
import { INITIAL_DRAG_STATE } from '@/types/pipeline';

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

    it('starts with null positions', () => {
      const { result } = renderHook(() =>
        useDragToConnect({ dag: testDAG, layout: testLayout })
      );

      expect(result.current.dragState.sourceNodeId).toBeNull();
      expect(result.current.dragState.sourcePosition).toBeNull();
      expect(result.current.dragState.cursorPosition).toBeNull();
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

      expect(result.current.dragState.cursorPosition).toBeNull();
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

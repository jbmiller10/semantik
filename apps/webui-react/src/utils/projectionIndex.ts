import type { DataPoint } from 'embedding-atlas/react';

export type ProjectionIndexInput =
  | DataPoint
  | {
      index?: number;
      rowIndex?: number;
      pointIndex?: number;
      i?: number;
    }
  | number
  | null
  | undefined;

/**
 * Normalise a selection/tooltip value from EmbeddingView into a stable
 * zero-based index into the projection arrays.
 *
 * Embedding Atlas and surrounding code paths may surface:
 * - raw numeric indices
 * - DataPoint objects with an `index` field
 * - legacy shapes that expose `rowIndex`, `pointIndex`, or `i`
 */
export function getProjectionPointIndex(value: ProjectionIndexInput): number | null {
  if (typeof value === 'number') {
    if (Number.isInteger(value) && Number.isFinite(value) && value >= 0) {
      return value;
    }
    return null;
  }

  if (!value || typeof value !== 'object') {
    return null;
  }

  const candidate = value as {
    index?: unknown;
    rowIndex?: unknown;
    pointIndex?: unknown;
    i?: unknown;
  };

  const keys: Array<keyof typeof candidate> = ['index', 'rowIndex', 'pointIndex', 'i'];
  for (const key of keys) {
    const v = candidate[key];
    if (typeof v === 'number' && Number.isInteger(v) && Number.isFinite(v) && v >= 0) {
      return v;
    }
  }

  return null;
}


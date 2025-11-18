import type { DataPoint } from 'embedding-atlas/react';

export type ProjectionIndexInput =
  | DataPoint
  | {
      identifier?: number | bigint | string;
      index?: number;
      rowIndex?: number;
      pointIndex?: number;
      i?: number;
      // Some library shapes may surface the index on a nested `fields` object.
      fields?: {
        index?: number;
        rowIndex?: number;
        pointIndex?: number;
        i?: number;
      };
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
    identifier?: unknown;
    index?: unknown;
    rowIndex?: unknown;
    pointIndex?: unknown;
    i?: unknown;
    fields?: {
      index?: unknown;
      rowIndex?: unknown;
      pointIndex?: unknown;
      i?: unknown;
    };
  };

  // Prefer an explicit numeric identifier when present. In plain EmbeddingView
  // usage, Atlas often uses the row index as the identifier if no id column
  // is configured.
  const identifier = candidate.identifier;
  if (typeof identifier === 'number' && Number.isInteger(identifier) && Number.isFinite(identifier) && identifier >= 0) {
    return identifier;
  }
  if (typeof identifier === 'bigint') {
    const asNumber = Number(identifier);
    if (Number.isSafeInteger(asNumber) && asNumber >= 0) {
      return asNumber;
    }
  }

  const keys: Array<keyof typeof candidate> = ['index', 'rowIndex', 'pointIndex', 'i'];
  for (const key of keys) {
    const v = candidate[key];
    if (typeof v === 'number' && Number.isInteger(v) && Number.isFinite(v) && v >= 0) {
      return v;
    }
  }

  // Fallback for shapes that tuck the index into a nested `fields` object.
  if (candidate.fields && typeof candidate.fields === 'object') {
    const fieldKeys: Array<keyof typeof candidate.fields> = ['index', 'rowIndex', 'pointIndex', 'i'];
    for (const key of fieldKeys) {
      const v = candidate.fields[key];
      if (typeof v === 'number' && Number.isInteger(v) && Number.isFinite(v) && v >= 0) {
        return v;
      }
    }
  }

  return null;
}

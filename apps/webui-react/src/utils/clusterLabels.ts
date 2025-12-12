import type { ProjectionLegendItem } from '../types/projection';

export interface CategoryLabel {
  x: number;
  y: number;
  text: string;
  level?: number;
  priority?: number;
}

interface CreateCategoryLabelsParams {
  x: Float32Array;
  y: Float32Array;
  category: Uint8Array;
  legend?: ProjectionLegendItem[];
  minPoints?: number;
  maxLabels?: number;
}

// Default label heuristics tuned for typical projection sizes
// (tens of thousands to ~200k points). These values strike a
// balance between highlighting meaningful clusters and avoiding
// visual overload in the Embedding Atlas view.
// - DEFAULT_MIN_POINTS: minimum number of points a category
//   must contribute to receive a label. Small, noisy categories
//   below this threshold are omitted.
// - DEFAULT_MAX_LABELS: global cap on rendered labels to
//   prevent the canvas from being saturated with text.
const DEFAULT_MIN_POINTS = 16;
const DEFAULT_MAX_LABELS = 120;

export const DEFAULT_CATEGORY_LABEL_OPTIONS = {
  minPoints: DEFAULT_MIN_POINTS,
  maxLabels: DEFAULT_MAX_LABELS,
} as const;

export function createCategoryLabels({
  x,
  y,
  category,
  legend,
  minPoints = DEFAULT_MIN_POINTS,
  maxLabels = DEFAULT_MAX_LABELS,
}: CreateCategoryLabelsParams): CategoryLabel[] {
  /**
   * Compute cluster-level text labels from 2D projection arrays and backend legend metadata.
   *
   * Inputs:
   * - x, y: Float32Array coordinates of length N (projection output).
   * - category: Uint8Array of length N with category indices for each point (cat.u8.bin).
   * - legend: mapping from category index → { index, label, count } as provided by the backend.
   * - minPoints: minimum number of points required for a category to receive a label.
   * - maxLabels: maximum number of labels to return, sorted by cluster size.
   *
   * Algorithm:
   * 1. Build a legendMap from index → label using the provided legend.
   * 2. Accumulate, for each legend index, the number of points and the sum of x/y coordinates.
   * 3. Filter out categories whose point count is below minPoints.
   * 4. For each remaining category, compute the centroid (mean x/y), and emit a CategoryLabel:
   *    { x, y, text, priority, level } where:
   *      - text: legend label.
   *      - priority: number of contributing points (used for sorting).
   *      - level: reserved for future layering / decluttering (currently fixed to 1).
   * 5. Sort labels by priority (descending) and truncate to maxLabels.
   *
   * Returns:
   * - CategoryLabel[] with centroid positions and display text for up to maxLabels clusters.
   */
  const pointCount = x.length;
  if (pointCount === 0 || y.length !== pointCount || category.length !== pointCount) {
    return [];
  }

  if (!legend || legend.length === 0) {
    return [];
  }

  const legendMap = new Map<number, string>();
  for (const item of legend) {
    if (typeof item.index === 'number' && item.label) {
      legendMap.set(item.index, item.label);
    }
  }

  if (legendMap.size === 0) {
    return [];
  }

  const minPointsThreshold = Math.max(1, Math.floor(minPoints));
  const maxLabelCount = Math.max(1, Math.floor(maxLabels));

  const accumulators = new Map<number, { count: number; sumX: number; sumY: number }>();

  for (let i = 0; i < pointCount; i += 1) {
    const catIndex = category[i];
    const label = legendMap.get(catIndex);
    if (!label) {
      continue;
    }

    const current = accumulators.get(catIndex);
    if (current) {
      current.count += 1;
      current.sumX += x[i];
      current.sumY += y[i];
    } else {
      accumulators.set(catIndex, {
        count: 1,
        sumX: x[i],
        sumY: y[i],
      });
    }
  }

  const labels: CategoryLabel[] = [];

  for (const [catIndex, stats] of accumulators) {
    if (stats.count < minPointsThreshold) {
      continue;
    }

    const labelText = legendMap.get(catIndex);
    if (!labelText) {
      continue;
    }

    labels.push({
      x: stats.sumX / stats.count,
      y: stats.sumY / stats.count,
      text: labelText,
      level: 1,
      priority: stats.count,
    });
  }

  if (labels.length <= maxLabelCount) {
    return labels.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
  }

  return labels
    .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0))
    .slice(0, maxLabelCount);
}

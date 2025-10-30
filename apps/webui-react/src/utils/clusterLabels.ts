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

const DEFAULT_MIN_POINTS = 8;
const DEFAULT_MAX_LABELS = 50;

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


import { describe, expect, it } from 'vitest';
import { createCategoryLabels } from '../clusterLabels';
import type { ProjectionLegendItem } from '../../types/projection';

describe('createCategoryLabels', () => {
  it('computes centroids and sorts by cluster size', () => {
    const x = new Float32Array([0, 1, 2, 10, 11, 12, 20, 21, 22]);
    const y = new Float32Array([0, 0, 0, 5, 5, 5, 10, 10, 10]);
    const category = new Uint8Array([0, 0, 0, 1, 1, 1, 2, 2, 2]);
    const legend: ProjectionLegendItem[] = [
      { index: 0, label: 'Alpha', count: 3 },
      { index: 1, label: 'Beta', count: 3 },
      { index: 2, label: 'Gamma', count: 3 },
    ];

    const labels = createCategoryLabels({
      x,
      y,
      category,
      legend,
      minPoints: 3,
      maxLabels: 5,
    });

    expect(labels).toHaveLength(3);
    expect(labels[0].text).toBe('Alpha');
    expect(labels[0].x).toBeCloseTo(1);
    expect(labels[1].text).toBe('Beta');
    expect(labels[2].text).toBe('Gamma');
  });

  it('filters out clusters below the minimum threshold', () => {
    const x = new Float32Array([0, 1, 10, 11, 50]);
    const y = new Float32Array([0, 0, 5, 5, 12]);
    const category = new Uint8Array([0, 0, 1, 1, 2]);
    const legend: ProjectionLegendItem[] = [
      { index: 0, label: 'Keep', count: 2 },
      { index: 1, label: 'Also Keep', count: 2 },
      { index: 2, label: 'Drop', count: 1 },
    ];

    const labels = createCategoryLabels({
      x,
      y,
      category,
      legend,
      minPoints: 2,
      maxLabels: 5,
    });

    expect(labels).toHaveLength(2);
    expect(labels.every((label) => label.text !== 'Drop')).toBe(true);
  });

  it('enforces maximum label count', () => {
    const categoryCount = 60;
    const pointsPerCategory = 10;
    const totalPoints = categoryCount * pointsPerCategory;
    const x = new Float32Array(totalPoints);
    const y = new Float32Array(totalPoints);
    const category = new Uint8Array(totalPoints);
    const legend: ProjectionLegendItem[] = [];

    for (let i = 0; i < categoryCount; i += 1) {
      legend.push({ index: i, label: `Cluster ${i}`, count: pointsPerCategory });
      for (let j = 0; j < pointsPerCategory; j += 1) {
        const idx = i * pointsPerCategory + j;
        x[idx] = i * 2;
        y[idx] = i * 2;
        category[idx] = i;
      }
    }

    const labels = createCategoryLabels({
      x,
      y,
      category,
      legend,
      minPoints: 5,
      maxLabels: 50,
    });

    expect(labels.length).toBeLessThanOrEqual(50);
  });

  it('returns empty array when lengths mismatch or legend missing', () => {
    const x = new Float32Array([0, 1]);
    const y = new Float32Array([0, 0]);
    const category = new Uint8Array([0]);

    expect(createCategoryLabels({ x, y, category, legend: [], minPoints: 1, maxLabels: 5 })).toEqual([]);
  });
});


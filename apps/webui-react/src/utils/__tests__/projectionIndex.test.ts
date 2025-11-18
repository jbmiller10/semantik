import { describe, expect, it } from 'vitest';
import { getProjectionPointIndex } from '../projectionIndex';

describe('getProjectionPointIndex', () => {
  it('returns index for numeric values', () => {
    expect(getProjectionPointIndex(0)).toBe(0);
    expect(getProjectionPointIndex(5)).toBe(5);
  });

  it('normalises DataPoint-like objects with index fields', () => {
    expect(
      getProjectionPointIndex({
        x: 1,
        y: 2,
        index: 7,
      } as unknown as Parameters<typeof getProjectionPointIndex>[0])
    ).toBe(7);
  });

  it('falls back through rowIndex, pointIndex, and i', () => {
    expect(getProjectionPointIndex({ rowIndex: 3 } as unknown as Parameters<typeof getProjectionPointIndex>[0])).toBe(3);
    expect(getProjectionPointIndex({ pointIndex: 4 } as unknown as Parameters<typeof getProjectionPointIndex>[0])).toBe(4);
    expect(getProjectionPointIndex({ i: 9 } as unknown as Parameters<typeof getProjectionPointIndex>[0])).toBe(9);
  });

  it('prefers index over other fields when multiple are present', () => {
    expect(
      getProjectionPointIndex({
        index: 1,
        rowIndex: 2,
        pointIndex: 3,
        i: 4,
      } as unknown as Parameters<typeof getProjectionPointIndex>[0])
    ).toBe(1);
  });

  it('returns null for negative, non-integer, or invalid values', () => {
    expect(getProjectionPointIndex(-1)).toBeNull();
    expect(getProjectionPointIndex({ index: -2 } as unknown as Parameters<typeof getProjectionPointIndex>[0])).toBeNull();
    expect(getProjectionPointIndex(1.5)).toBeNull();
    expect(getProjectionPointIndex({} as unknown as Parameters<typeof getProjectionPointIndex>[0])).toBeNull();
    expect(getProjectionPointIndex(null)).toBeNull();
    expect(getProjectionPointIndex(undefined)).toBeNull();
  });
});

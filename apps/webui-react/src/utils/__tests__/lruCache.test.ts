import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { LruCache } from '../lruCache';

describe('LruCache', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns values that were set', () => {
    const cache = new LruCache<string, number>({ max: 2, ttlMs: 1_000 });
    cache.set('a', 1);
    expect(cache.get('a')).toBe(1);
  });

  it('evicts least recently used entries when capacity exceeded', () => {
    const cache = new LruCache<string, number>({ max: 2, ttlMs: 5_000 });
    cache.set('a', 1);
    cache.set('b', 2);
    // Access `a` so `b` becomes least recently used.
    expect(cache.get('a')).toBe(1);
    cache.set('c', 3);

    expect(cache.get('a')).toBe(1);
    expect(cache.get('b')).toBeUndefined();
    expect(cache.get('c')).toBe(3);
  });

  it('expires entries after TTL', () => {
    const cache = new LruCache<string, number>({ max: 5, ttlMs: 1_000 });
    cache.set('token', 42);
    expect(cache.get('token')).toBe(42);

    vi.advanceTimersByTime(1_001);
    expect(cache.get('token')).toBeUndefined();
  });

  it('reports current size', () => {
    const cache = new LruCache<string, number>({ max: 3, ttlMs: 5_000 });
    cache.set('a', 1);
    cache.set('b', 2);
    cache.set('c', 3);
    expect(cache.size).toBe(3);

    cache.set('d', 4); // evicts the oldest entry
    expect(cache.size).toBe(3);
  });
});


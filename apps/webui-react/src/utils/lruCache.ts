type LruEntry<V> = {
  value: V;
  expiresAt: number | null;
};

export interface LruCacheOptions {
  /** Maximum number of entries retained at once. */
  max: number;
  /** Optional TTL in milliseconds for each entry. */
  ttlMs?: number;
}

/**
 * Simple TTL-aware LRU cache backed by Map insertion order.
 * Evicts the least recently used entry when capacity is exceeded.
 */
export class LruCache<K, V> {
  private readonly store = new Map<K, LruEntry<V>>();

  private readonly max: number;

  private readonly ttlMs: number | null;

  constructor(options: LruCacheOptions) {
    if (!options || typeof options.max !== 'number' || options.max <= 0) {
      throw new Error('LruCache requires a positive `max` option');
    }
    this.max = Math.floor(options.max);
    this.ttlMs = typeof options.ttlMs === 'number' && options.ttlMs > 0 ? options.ttlMs : null;
  }

  get size(): number {
    return this.store.size;
  }

  get(key: K): V | undefined {
    const existing = this.store.get(key);
    if (!existing) {
      return undefined;
    }

    if (this.isExpired(existing)) {
      this.store.delete(key);
      return undefined;
    }

    // Re-insert to update iteration order (most recently used moves to end).
    this.store.delete(key);
    this.store.set(key, existing);
    return existing.value;
  }

  set(key: K, value: V): void {
    if (this.store.has(key)) {
      this.store.delete(key);
    } else if (this.store.size >= this.max) {
      this.evictLeastRecentlyUsed();
    }

    const expiresAt = this.ttlMs === null ? null : Date.now() + this.ttlMs;
    this.store.set(key, { value, expiresAt });
  }

  delete(key: K): void {
    this.store.delete(key);
  }

  clear(): void {
    this.store.clear();
  }

  private isExpired(entry: LruEntry<V>): boolean {
    if (entry.expiresAt === null) {
      return false;
    }
    return entry.expiresAt <= Date.now();
  }

  private evictLeastRecentlyUsed(): void {
    const iterator = this.store.keys();
    const oldestKey = iterator.next().value;
    if (oldestKey !== undefined) {
      this.store.delete(oldestKey);
    }
  }
}


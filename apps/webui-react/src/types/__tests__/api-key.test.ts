import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  getApiKeyStatus,
  formatRelativeTime,
  truncateKeyId,
  type ApiKeyResponse,
} from '../api-key';

describe('API Key utility functions', () => {
  describe('getApiKeyStatus', () => {
    const baseKey: ApiKeyResponse = {
      id: 'key-1',
      name: 'test-key',
      is_active: true,
      permissions: null,
      last_used_at: null,
      expires_at: null,
      created_at: '2025-01-01T00:00:00Z',
    };

    it('returns "active" for active non-expired key', () => {
      const key = { ...baseKey, is_active: true, expires_at: '2099-01-01T00:00:00Z' };
      expect(getApiKeyStatus(key)).toBe('active');
    });

    it('returns "active" for active key with no expiration', () => {
      const key = { ...baseKey, is_active: true, expires_at: null };
      expect(getApiKeyStatus(key)).toBe('active');
    });

    it('returns "disabled" for inactive key regardless of expiration', () => {
      const key = { ...baseKey, is_active: false, expires_at: '2099-01-01T00:00:00Z' };
      expect(getApiKeyStatus(key)).toBe('disabled');
    });

    it('returns "disabled" for inactive key with no expiration', () => {
      const key = { ...baseKey, is_active: false, expires_at: null };
      expect(getApiKeyStatus(key)).toBe('disabled');
    });

    it('returns "expired" for active but expired key', () => {
      const key = { ...baseKey, is_active: true, expires_at: '2020-01-01T00:00:00Z' };
      expect(getApiKeyStatus(key)).toBe('expired');
    });

    it('returns "disabled" for inactive expired key (inactive takes precedence)', () => {
      // When a key is both inactive and expired, disabled takes precedence
      // because the check for is_active comes first
      const key = { ...baseKey, is_active: false, expires_at: '2020-01-01T00:00:00Z' };
      expect(getApiKeyStatus(key)).toBe('disabled');
    });
  });

  describe('formatRelativeTime', () => {
    // Use fake timers to control "now"
    beforeEach(() => {
      vi.useFakeTimers();
      // Set current time to 2025-06-15T12:00:00Z
      vi.setSystemTime(new Date('2025-06-15T12:00:00Z'));
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('returns "Never" for null input', () => {
      expect(formatRelativeTime(null)).toBe('Never');
    });

    it('returns "Just now" for same time', () => {
      expect(formatRelativeTime('2025-06-15T12:00:00Z')).toBe('Just now');
    });

    // Past dates
    it('returns "X minutes ago" for recent past', () => {
      expect(formatRelativeTime('2025-06-15T11:55:00Z')).toBe('5 minutes ago');
    });

    it('returns "1 minute ago" for singular minute', () => {
      expect(formatRelativeTime('2025-06-15T11:59:00Z')).toBe('1 minute ago');
    });

    it('returns "X hours ago" for past hours', () => {
      expect(formatRelativeTime('2025-06-15T09:00:00Z')).toBe('3 hours ago');
    });

    it('returns "1 hour ago" for singular hour', () => {
      expect(formatRelativeTime('2025-06-15T11:00:00Z')).toBe('1 hour ago');
    });

    it('returns "Yesterday" for one day ago', () => {
      expect(formatRelativeTime('2025-06-14T12:00:00Z')).toBe('Yesterday');
    });

    it('returns "X days ago" for recent past days', () => {
      expect(formatRelativeTime('2025-06-10T12:00:00Z')).toBe('5 days ago');
    });

    it('returns "X months ago" for past months', () => {
      expect(formatRelativeTime('2025-03-15T12:00:00Z')).toBe('3 months ago');
    });

    it('returns "1 month ago" for singular month', () => {
      expect(formatRelativeTime('2025-05-15T12:00:00Z')).toBe('1 month ago');
    });

    it('returns "X years ago" for past years', () => {
      expect(formatRelativeTime('2023-06-15T12:00:00Z')).toBe('2 years ago');
    });

    it('returns "1 year ago" for singular year', () => {
      expect(formatRelativeTime('2024-06-15T12:00:00Z')).toBe('1 year ago');
    });

    // Future dates
    it('returns "in X minutes" for near future', () => {
      expect(formatRelativeTime('2025-06-15T12:05:00Z')).toBe('in 5 minutes');
    });

    it('returns "in 1 minute" for singular minute', () => {
      expect(formatRelativeTime('2025-06-15T12:01:00Z')).toBe('in 1 minute');
    });

    it('returns "in X hours" for future hours', () => {
      expect(formatRelativeTime('2025-06-15T15:00:00Z')).toBe('in 3 hours');
    });

    it('returns "in 1 hour" for singular hour', () => {
      expect(formatRelativeTime('2025-06-15T13:00:00Z')).toBe('in 1 hour');
    });

    it('returns "Tomorrow" for one day ahead', () => {
      expect(formatRelativeTime('2025-06-16T12:00:00Z')).toBe('Tomorrow');
    });

    it('returns "in X days" for future days', () => {
      expect(formatRelativeTime('2025-06-20T12:00:00Z')).toBe('in 5 days');
    });

    it('returns "in X months" for future months', () => {
      expect(formatRelativeTime('2025-09-15T12:00:00Z')).toBe('in 3 months');
    });

    it('returns "in 1 month" for singular month', () => {
      expect(formatRelativeTime('2025-07-15T12:00:00Z')).toBe('in 1 month');
    });

    it('returns "in X years" for future years', () => {
      expect(formatRelativeTime('2027-06-15T12:00:00Z')).toBe('in 2 years');
    });

    it('returns "in 1 year" for singular year', () => {
      expect(formatRelativeTime('2026-06-15T12:00:00Z')).toBe('in 1 year');
    });
  });

  describe('truncateKeyId', () => {
    it('formats as smtk_{first8}...', () => {
      expect(truncateKeyId('a1b2c3d4e5f6g7h8i9j0k1l2')).toBe('smtk_a1b2c3d4...');
    });

    it('handles UUID format', () => {
      expect(truncateKeyId('123e4567-e89b-12d3-a456-426614174000')).toBe('smtk_123e4567...');
    });

    it('handles short strings (edge case)', () => {
      expect(truncateKeyId('abc')).toBe('smtk_abc...');
    });

    it('handles exactly 8 character string', () => {
      expect(truncateKeyId('12345678')).toBe('smtk_12345678...');
    });
  });
});

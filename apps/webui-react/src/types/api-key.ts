/**
 * TypeScript types for API key management.
 * Matches backend schemas from packages/webui/api/v2/api_key_schemas.py
 */

/**
 * Request schema for creating an API key.
 */
export interface ApiKeyCreate {
  name: string;
  expires_in_days?: number | null;
}

/**
 * Response schema for API key details (excludes raw key and hash).
 */
export interface ApiKeyResponse {
  id: string;
  name: string;
  is_active: boolean;
  permissions: Record<string, unknown> | null;
  last_used_at: string | null;
  expires_at: string | null;
  created_at: string;
}

/**
 * Response schema for newly created API key (includes raw key once).
 */
export interface ApiKeyCreateResponse extends ApiKeyResponse {
  /** The full API key (only shown once at creation) */
  api_key: string;
}

/**
 * Response schema for listing API keys.
 */
export interface ApiKeyListResponse {
  api_keys: ApiKeyResponse[];
  total: number;
}

/**
 * Request schema for updating an API key (soft revoke/reactivate).
 */
export interface ApiKeyUpdate {
  is_active: boolean;
}

/**
 * API key status derived from is_active and expires_at.
 */
export type ApiKeyStatus = 'active' | 'disabled' | 'expired';

/**
 * Determine the status of an API key based on its properties.
 */
export function getApiKeyStatus(key: ApiKeyResponse): ApiKeyStatus {
  if (!key.is_active) {
    return 'disabled';
  }
  if (key.expires_at && new Date(key.expires_at) < new Date()) {
    return 'expired';
  }
  return 'active';
}

/**
 * Format a date string as relative time (e.g., "2 days ago", "in 30 days").
 */
export function formatRelativeTime(dateStr: string | null): string {
  if (!dateStr) {
    return 'Never';
  }

  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = date.getTime() - now.getTime();
  const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) {
    const diffHours = Math.round(diffMs / (1000 * 60 * 60));
    if (diffHours === 0) {
      const diffMins = Math.round(diffMs / (1000 * 60));
      if (diffMins === 0) {
        return 'Just now';
      }
      if (diffMins > 0) {
        return `in ${diffMins} minute${diffMins === 1 ? '' : 's'}`;
      }
      return `${Math.abs(diffMins)} minute${Math.abs(diffMins) === 1 ? '' : 's'} ago`;
    }
    if (diffHours > 0) {
      return `in ${diffHours} hour${diffHours === 1 ? '' : 's'}`;
    }
    return `${Math.abs(diffHours)} hour${Math.abs(diffHours) === 1 ? '' : 's'} ago`;
  }

  if (diffDays > 0) {
    if (diffDays === 1) return 'Tomorrow';
    if (diffDays < 30) return `in ${diffDays} days`;
    if (diffDays < 365) {
      const months = Math.round(diffDays / 30);
      return `in ${months} month${months === 1 ? '' : 's'}`;
    }
    const years = Math.round(diffDays / 365);
    return `in ${years} year${years === 1 ? '' : 's'}`;
  }

  const absDays = Math.abs(diffDays);
  if (absDays === 1) return 'Yesterday';
  if (absDays < 30) return `${absDays} days ago`;
  if (absDays < 365) {
    const months = Math.round(absDays / 30);
    return `${months} month${months === 1 ? '' : 's'} ago`;
  }
  const years = Math.round(absDays / 365);
  return `${years} year${years === 1 ? '' : 's'} ago`;
}

/**
 * Truncate a key ID to a displayable format.
 * Format: smtk_{first 8 chars}...
 */
export function truncateKeyId(fullId: string): string {
  return `smtk_${fullId.substring(0, 8)}...`;
}

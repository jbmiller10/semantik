/**
 * Connector type definitions for the dynamic connector form UI.
 * These types match the backend connector catalog schema.
 */

/**
 * Option for select/multiselect fields
 */
export interface FieldOption {
  value: string;
  label: string;
}

/**
 * Conditional visibility for fields
 */
export interface ShowWhen {
  field: string;
  equals: string | string[];
}

/**
 * Definition for a form field in the connector schema
 */
export interface FieldDefinition {
  name: string;
  type: 'text' | 'number' | 'select' | 'multiselect' | 'textarea' | 'boolean' | 'glob_list';
  label: string;
  description?: string;
  required: boolean;
  default?: unknown;
  placeholder?: string;
  options?: FieldOption[];
  show_when?: ShowWhen;
  min?: number;
  max?: number;
  step?: number;
}

/**
 * Definition for a secret field (passwords, tokens, keys)
 */
export interface SecretDefinition {
  name: string;
  label: string;
  description?: string;
  required: boolean;
  show_when?: ShowWhen;
  is_multiline?: boolean;
}

/**
 * Complete definition for a connector type
 */
export interface ConnectorDefinition {
  name: string;
  description: string;
  icon: string;
  fields: FieldDefinition[];
  secrets: SecretDefinition[];
  supports_sync: boolean;
  preview_endpoint?: string;
}

/**
 * Catalog of all available connector types
 */
export type ConnectorCatalog = Record<string, ConnectorDefinition>;

/**
 * Request to preview/validate a Git repository connection
 */
export interface GitPreviewRequest {
  repo_url: string;
  ref?: string;
  auth_method?: 'none' | 'https_token' | 'ssh_key';
  token?: string;
  ssh_key?: string;
  ssh_passphrase?: string;
  include_globs?: string[];
  exclude_globs?: string[];
}

/**
 * Response from Git preview endpoint
 */
export interface GitPreviewResponse {
  valid: boolean;
  repo_url: string;
  ref: string;
  refs_found: string[];
  error?: string;
}

/**
 * Request to preview/validate an IMAP connection
 */
export interface ImapPreviewRequest {
  host: string;
  port?: number;
  use_ssl?: boolean;
  username: string;
  password: string;
  mailboxes?: string[];
}

/**
 * Response from IMAP preview endpoint
 */
export interface ImapPreviewResponse {
  valid: boolean;
  host: string;
  username: string;
  mailboxes_found: string[];
  error?: string;
}

/**
 * Union type for preview responses
 */
export type PreviewResponse = GitPreviewResponse | ImapPreviewResponse;

/**
 * Helper to check if a field should be shown based on show_when condition
 */
export function shouldShowField(
  field: { show_when?: ShowWhen },
  allValues: Record<string, unknown>
): boolean {
  if (!field.show_when) return true;

  const { field: depField, equals } = field.show_when;
  const currentValue = allValues[depField];

  if (Array.isArray(equals)) {
    return equals.includes(currentValue as string);
  }

  return currentValue === equals;
}

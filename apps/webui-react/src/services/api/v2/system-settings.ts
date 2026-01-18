/**
 * System Settings API client.
 * Implements endpoints for admin-only system configuration.
 */
import apiClient from './client';
import type {
  SystemSettingsResponse,
  SystemSettingsUpdateRequest,
  SystemSettingsUpdateResponse,
  EffectiveSettingsResponse,
  DefaultSettingsResponse,
} from '../../../types/system-settings';

export const systemSettingsApi = {
  /**
   * Get all system settings with metadata.
   * Admin-only endpoint.
   */
  getSettings: () =>
    apiClient.get<SystemSettingsResponse>('/api/v2/system-settings'),

  /**
   * Get effective settings (resolved DB value or env fallback).
   * Admin-only endpoint.
   */
  getEffective: () =>
    apiClient.get<EffectiveSettingsResponse>('/api/v2/system-settings/effective'),

  /**
   * Update system settings.
   * Admin-only endpoint. Only send settings you want to change.
   * @param data Settings to update
   */
  updateSettings: (data: SystemSettingsUpdateRequest) =>
    apiClient.patch<SystemSettingsUpdateResponse>('/api/v2/system-settings', data),

  /**
   * Get default values for all settings.
   * Admin-only endpoint.
   */
  getDefaults: () =>
    apiClient.get<DefaultSettingsResponse>('/api/v2/system-settings/defaults'),
};

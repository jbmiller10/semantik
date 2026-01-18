import { describe, it, expect } from 'vitest'
import { AxiosError } from 'axios'

import { apiKeysApi } from '../api-keys'

describe('apiKeysApi (MSW integration)', () => {
  it('lists API keys', async () => {
    const response = await apiKeysApi.list()
    expect(response.data.total).toBeGreaterThan(0)
    expect(response.data.api_keys.length).toBe(response.data.total)
  })

  it('gets a specific API key', async () => {
    const response = await apiKeysApi.get('key-123')
    expect(response.data.id).toBe('key-123')
    expect(response.data.name).toBe('Test Key')
  })

  it('creates API key with expiration', async () => {
    const response = await apiKeysApi.create({ name: 'new-key', expires_in_days: 30 })
    expect(response.status).toBe(201)
    expect(response.data.api_key).toMatch(/^smtk_/)
    expect(response.data.expires_at).not.toBeNull()
  })

  it('creates API key without expiration', async () => {
    const response = await apiKeysApi.create({ name: 'never-expires', expires_in_days: null })
    expect(response.status).toBe(201)
    expect(response.data.expires_at).toBeNull()
  })

  it('surfaces duplicate-name errors (409)', async () => {
    try {
      await apiKeysApi.create({ name: 'duplicate-name' })
      throw new Error('Expected request to fail')
    } catch (error) {
      expect(error).toBeInstanceOf(AxiosError)
      expect((error as AxiosError).response?.status).toBe(409)
    }
  })

  it('surfaces limit-reached errors (400)', async () => {
    try {
      await apiKeysApi.create({ name: 'limit-reached' })
      throw new Error('Expected request to fail')
    } catch (error) {
      expect(error).toBeInstanceOf(AxiosError)
      expect((error as AxiosError).response?.status).toBe(400)
    }
  })

  it('updates API key status', async () => {
    const response = await apiKeysApi.update('key-1-uuid', { is_active: false })
    expect(response.data.id).toBe('key-1-uuid')
    expect(response.data.is_active).toBe(false)
  })
})


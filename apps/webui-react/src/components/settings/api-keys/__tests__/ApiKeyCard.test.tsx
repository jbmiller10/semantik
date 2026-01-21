import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import userEvent from '@testing-library/user-event'
import { render, screen, act, waitFor } from '@/tests/utils/test-utils'

import ApiKeyCard from '../ApiKeyCard'
import type { ApiKeyResponse } from '@/types/api-key'
import { useRevokeApiKey } from '../../../../hooks/useApiKeys'

vi.mock('../../../../hooks/useApiKeys', () => ({
  useRevokeApiKey: vi.fn(),
}))

const baseKey: ApiKeyResponse = {
  id: 'key-1-uuid',
  name: 'Development Key',
  is_active: true,
  permissions: null,
  last_used_at: null,
  expires_at: '2999-01-01T00:00:00Z',
  created_at: '2025-01-01T00:00:00Z',
}

describe('ApiKeyCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders active key with revoke button', () => {
    vi.mocked(useRevokeApiKey).mockReturnValue({
      mutateAsync: vi.fn().mockResolvedValue({ ...baseKey }),
    } as unknown as ReturnType<typeof useRevokeApiKey>)

    render(<ApiKeyCard apiKey={baseKey} />)

    expect(screen.getByText('Development Key')).toBeInTheDocument()
    expect(screen.getByText('Active')).toBeInTheDocument()
    expect(screen.getByText('smtk_key-1-uu...')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Revoke' })).toBeInTheDocument()
  })

  it('calls revoke mutation and shows toggling state', async () => {
    const user = userEvent.setup()
    let resolveMutation: (() => void) | undefined
    const mutationPromise = new Promise<void>((resolve) => {
      resolveMutation = resolve
    })
    const mutateAsync = vi.fn(() => mutationPromise)

    vi.mocked(useRevokeApiKey).mockReturnValue({
      mutateAsync,
    } as unknown as ReturnType<typeof useRevokeApiKey>)

    render(<ApiKeyCard apiKey={baseKey} />)

    const button = screen.getByRole('button', { name: 'Revoke' })
    await user.click(button)

    expect(mutateAsync).toHaveBeenCalledWith({
      keyId: baseKey.id,
      isActive: false,
      keyName: baseKey.name,
    })
    expect(button).toBeDisabled()
    expect(button).toHaveTextContent('...')

    await act(async () => {
      resolveMutation?.()
      await mutationPromise
    })

    await waitFor(() => {
      expect(button).not.toBeDisabled()
    })
  })

  it('renders disabled key with reactivate button', async () => {
    const user = userEvent.setup()
    const disabledKey = { ...baseKey, is_active: false, name: 'Revoked Key' }
    const mutateAsync = vi.fn().mockResolvedValue({ ...disabledKey, is_active: true })

    vi.mocked(useRevokeApiKey).mockReturnValue({
      mutateAsync,
    } as unknown as ReturnType<typeof useRevokeApiKey>)

    render(<ApiKeyCard apiKey={disabledKey} />)

    expect(screen.getByText('Disabled')).toBeInTheDocument()
    const button = screen.getByRole('button', { name: 'Reactivate' })
    await user.click(button)

    expect(mutateAsync).toHaveBeenCalledWith({
      keyId: disabledKey.id,
      isActive: true,
      keyName: disabledKey.name,
    })
  })

  it('hides toggle button when key is expired', () => {
    vi.mocked(useRevokeApiKey).mockReturnValue({
      mutateAsync: vi.fn(),
    } as unknown as ReturnType<typeof useRevokeApiKey>)

    const expiredKey: ApiKeyResponse = {
      ...baseKey,
      name: 'Expired Key',
      expires_at: '2000-01-01T00:00:00Z',
      is_active: true,
    }

    render(<ApiKeyCard apiKey={expiredKey} />)

    expect(screen.getByText('Expired')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Revoke' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Reactivate' })).not.toBeInTheDocument()
  })

  it('logs error when toggle fails and resets toggling state', async () => {
    const user = userEvent.setup()
    vi.mocked(useRevokeApiKey).mockReturnValue({
      mutateAsync: vi.fn().mockRejectedValue(new Error('Update failed')),
    } as unknown as ReturnType<typeof useRevokeApiKey>)

    render(<ApiKeyCard apiKey={baseKey} />)

    const button = screen.getByRole('button', { name: 'Revoke' })
    await user.click(button)

    await waitFor(() => {
      expect(vi.mocked(console.error)).toHaveBeenCalled()
      expect(button).not.toBeDisabled()
    })
  })
})


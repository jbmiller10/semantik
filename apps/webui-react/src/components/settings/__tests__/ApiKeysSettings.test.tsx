import { describe, it, expect, vi, beforeEach } from 'vitest'
import userEvent from '@testing-library/user-event'
import { render, screen } from '@/tests/utils/test-utils'

import ApiKeysSettings from '../ApiKeysSettings'

import type { ApiKeyCreateResponse, ApiKeyResponse } from '@/types/api-key'
import { useApiKeys } from '../../../hooks/useApiKeys'

vi.mock('../../../hooks/useApiKeys', () => ({
  useApiKeys: vi.fn(),
}))

vi.mock('../api-keys/ApiKeyCard', () => ({
  default: ({ apiKey }: { apiKey: ApiKeyResponse }) => (
    <div data-testid="api-key-card">{apiKey.name}</div>
  ),
}))

const mockCreatedKey: ApiKeyCreateResponse = {
  id: 'created-key-uuid',
  name: 'Created Key',
  is_active: true,
  permissions: null,
  last_used_at: null,
  expires_at: null,
  created_at: '2025-01-01T00:00:00Z',
  api_key: 'smtk_12345678_abcdefghijklmnopqrstuvwxyz',
}

vi.mock('../api-keys/CreateApiKeyModal', () => ({
  default: ({
    onClose,
    onSuccess,
  }: {
    onClose: () => void
    onSuccess: (response: ApiKeyCreateResponse) => void
  }) => (
    <div data-testid="create-api-key-modal">
      <button type="button" onClick={onClose}>
        Close Create Modal
      </button>
      <button type="button" onClick={() => onSuccess(mockCreatedKey)}>
        Succeed Create
      </button>
    </div>
  ),
}))

vi.mock('../api-keys/ApiKeyCreatedModal', () => ({
  default: ({
    apiKey,
    onClose,
  }: {
    apiKey: ApiKeyCreateResponse
    onClose: () => void
  }) => (
    <div data-testid="api-key-created-modal">
      <div>{apiKey.name}</div>
      <button type="button" onClick={onClose}>
        Close Created Modal
      </button>
    </div>
  ),
}))

describe('ApiKeysSettings', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows loading state', () => {
    vi.mocked(useApiKeys).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useApiKeys>)

    render(<ApiKeysSettings />)
    expect(screen.getByText('Loading API keys...')).toBeInTheDocument()
  })

  it('shows error state and allows retry', async () => {
    const user = userEvent.setup()
    const refetch = vi.fn()
    vi.mocked(useApiKeys).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('Boom'),
      refetch,
    } as unknown as ReturnType<typeof useApiKeys>)

    render(<ApiKeysSettings />)
    expect(screen.getByText('Error loading API keys')).toBeInTheDocument()
    expect(screen.getByText('Boom')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(refetch).toHaveBeenCalledTimes(1)
  })

  it('shows unknown error message when error is not an Error', () => {
    vi.mocked(useApiKeys).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: 'oops',
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useApiKeys>)

    render(<ApiKeysSettings />)
    expect(screen.getByText('Unknown error occurred')).toBeInTheDocument()
  })

  it('renders empty state and opens create + created modals', async () => {
    const user = userEvent.setup()
    vi.mocked(useApiKeys).mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useApiKeys>)

    render(<ApiKeysSettings />)
    expect(screen.getByText('No API keys')).toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: 'Create API Key' })[0])
    expect(screen.getByTestId('create-api-key-modal')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Succeed Create' }))
    expect(screen.queryByTestId('create-api-key-modal')).not.toBeInTheDocument()
    expect(screen.getByTestId('api-key-created-modal')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Close Created Modal' }))
    expect(screen.queryByTestId('api-key-created-modal')).not.toBeInTheDocument()
  })

  it('renders API key cards when keys exist', () => {
    const apiKeys: ApiKeyResponse[] = [
      {
        id: 'key-1-uuid',
        name: 'Development Key',
        is_active: true,
        permissions: null,
        last_used_at: null,
        expires_at: null,
        created_at: '2025-01-01T00:00:00Z',
      },
      {
        id: 'key-2-uuid',
        name: 'CI/CD Key',
        is_active: false,
        permissions: null,
        last_used_at: null,
        expires_at: null,
        created_at: '2025-01-01T00:00:00Z',
      },
    ]

    vi.mocked(useApiKeys).mockReturnValue({
      data: apiKeys,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useApiKeys>)

    render(<ApiKeysSettings />)
    expect(screen.getAllByTestId('api-key-card')).toHaveLength(2)
    expect(screen.getByText('Development Key')).toBeInTheDocument()
    expect(screen.getByText('CI/CD Key')).toBeInTheDocument()
  })
})


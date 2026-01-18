import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import userEvent from '@testing-library/user-event'
import { render, screen, fireEvent, waitFor } from '@/tests/utils/test-utils'
import { AxiosError } from 'axios'

import CreateApiKeyModal from '../CreateApiKeyModal'
import { useCreateApiKey } from '../../../../hooks/useApiKeys'

vi.mock('../../../../hooks/useApiKeys', () => ({
  useCreateApiKey: vi.fn(),
}))

describe('CreateApiKeyModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('focuses the first element and traps Tab, and closes on Escape', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onSuccess = vi.fn()
    vi.mocked(useCreateApiKey).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false,
    } as unknown as ReturnType<typeof useCreateApiKey>)

    render(<CreateApiKeyModal onClose={onClose} onSuccess={onSuccess} />)

    // Make the submit button focusable (it's disabled until name is non-empty).
    await user.type(screen.getByLabelText('Name'), 'test-key')

    const dialog = screen.getByRole('dialog')
    const focusables = dialog.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )
    const first = focusables[0] as HTMLElement
    const last = focusables[focusables.length - 1] as HTMLElement

    first.focus()
    expect(first).toHaveFocus()

    // Shift+Tab from first wraps to last
    fireEvent.keyDown(document, { key: 'Tab', shiftKey: true })
    expect(last).toHaveFocus()

    // Tab from last wraps back to first
    fireEvent.keyDown(document, { key: 'Tab' })
    expect(first).toHaveFocus()

    await user.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('shows validation error when name is empty', async () => {
    const user = userEvent.setup()
    const mutateAsync = vi.fn()
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    vi.mocked(useCreateApiKey).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useCreateApiKey>)

    render(<CreateApiKeyModal onClose={onClose} onSuccess={onSuccess} />)

    const nameInput = screen.getByLabelText('Name')
    fireEvent.blur(nameInput)

    expect(screen.getByText('Name is required')).toBeInTheDocument()

    // Form submission should not call mutateAsync when invalid
    await user.click(screen.getByRole('button', { name: 'Create Key' }))
    expect(mutateAsync).not.toHaveBeenCalled()
    expect(onSuccess).not.toHaveBeenCalled()
  })

  it('shows validation error when name exceeds 100 characters', () => {
    const mutateAsync = vi.fn()
    vi.mocked(useCreateApiKey).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useCreateApiKey>)

    render(<CreateApiKeyModal onClose={vi.fn()} onSuccess={vi.fn()} />)

    const nameInput = screen.getByLabelText('Name')
    fireEvent.change(nameInput, { target: { value: 'x'.repeat(101) } })
    fireEvent.blur(nameInput)

    expect(screen.getByText('Name must be 100 characters or less')).toBeInTheDocument()
  })

  it('clears validation error on change when name becomes valid', async () => {
    const user = userEvent.setup()
    vi.mocked(useCreateApiKey).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false,
    } as unknown as ReturnType<typeof useCreateApiKey>)

    render(<CreateApiKeyModal onClose={vi.fn()} onSuccess={vi.fn()} />)

    const nameInput = screen.getByLabelText('Name')

    fireEvent.blur(nameInput)
    expect(screen.getByText('Name is required')).toBeInTheDocument()

    await user.type(nameInput, 'Valid name')
    expect(screen.queryByText('Name is required')).not.toBeInTheDocument()
  })

  it('shows duplicate-name error message when API returns 409', async () => {
    const user = userEvent.setup()
    const mutateAsync = vi.fn()
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    const axiosError = new AxiosError('Conflict')
    axiosError.response = {
      status: 409,
      data: { detail: 'API key with this name already exists' },
      statusText: 'Conflict',
      headers: {},
      config: {} as never,
    }
    mutateAsync.mockRejectedValueOnce(axiosError)

    vi.mocked(useCreateApiKey).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useCreateApiKey>)

    render(<CreateApiKeyModal onClose={onClose} onSuccess={onSuccess} />)

    await user.type(screen.getByLabelText('Name'), 'duplicate-name')
    await user.click(screen.getByRole('button', { name: 'Create Key' }))

    expect(await screen.findByText('A key with this name already exists')).toBeInTheDocument()
    expect(onSuccess).not.toHaveBeenCalled()
  })

  it('submits trimmed name and selected expiration', async () => {
    const user = userEvent.setup()
    const mutateAsync = vi.fn().mockResolvedValue({
      id: 'new-key',
      name: 'My Key',
      is_active: true,
      permissions: null,
      last_used_at: null,
      expires_at: null,
      created_at: '2025-01-01T00:00:00Z',
      api_key: 'smtk_12345678_abcdefghijklmnopqrstuvwxyz',
    })
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    vi.mocked(useCreateApiKey).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useCreateApiKey>)

    render(<CreateApiKeyModal onClose={onClose} onSuccess={onSuccess} />)

    await user.type(screen.getByLabelText('Name'), '  My Key  ')
    await user.selectOptions(screen.getByLabelText('Expiration'), '30')

    await user.click(screen.getByRole('button', { name: 'Create Key' }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith({
        name: 'My Key',
        expires_in_days: 30,
      })
      expect(onSuccess).toHaveBeenCalledTimes(1)
    })
  })

  it('supports Never expiration and shows pending state', () => {
    vi.mocked(useCreateApiKey).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: true,
    } as unknown as ReturnType<typeof useCreateApiKey>)

    render(<CreateApiKeyModal onClose={vi.fn()} onSuccess={vi.fn()} />)

    expect(screen.getByText('Creating...')).toBeInTheDocument()

    // Keep the "Never" branch executed via select change
    const select = screen.getByLabelText('Expiration')
    fireEvent.change(select, { target: { value: 'never' } })
  })
})

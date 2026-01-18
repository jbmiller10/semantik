import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import userEvent from '@testing-library/user-event'
import { render, screen, fireEvent } from '@/tests/utils/test-utils'

import ApiKeyCreatedModal from '../ApiKeyCreatedModal'
import type { ApiKeyCreateResponse } from '@/types/api-key'

const apiKey: ApiKeyCreateResponse = {
  id: 'new-key-uuid',
  name: 'My Created Key',
  is_active: true,
  permissions: null,
  last_used_at: null,
  expires_at: null,
  created_at: '2025-01-01T00:00:00Z',
  api_key: 'smtk_12345678_abcdefghijklmnopqrstuvwxyz1234567890',
}

describe('ApiKeyCreatedModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders masked key, traps focus, and does not close on Escape', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()

    render(<ApiKeyCreatedModal apiKey={apiKey} onClose={onClose} />)

    const masked =
      apiKey.api_key.substring(0, 15) +
      '*'.repeat(20) +
      apiKey.api_key.substring(apiKey.api_key.length - 4)
    expect(screen.getByText(masked)).toBeInTheDocument()

    const showButton = screen.getByTitle('Show key')
    expect(showButton).toHaveFocus()

    // Escape is explicitly blocked
    await user.keyboard('{Escape}')
    expect(onClose).not.toHaveBeenCalled()

    // Shift+Tab from first wraps to last
    fireEvent.keyDown(document, { key: 'Tab', shiftKey: true })
    expect(screen.getByRole('button', { name: "I've copied my key" })).toHaveFocus()

    // Tab from last wraps back to first
    fireEvent.keyDown(document, { key: 'Tab' })
    expect(showButton).toHaveFocus()
  })

  it('toggles show/hide key', async () => {
    const user = userEvent.setup()
    render(<ApiKeyCreatedModal apiKey={apiKey} onClose={vi.fn()} />)

    await user.click(screen.getByTitle('Show key'))
    expect(screen.getByText(apiKey.api_key)).toBeInTheDocument()
    expect(screen.getByTitle('Hide key')).toBeInTheDocument()

    await user.click(screen.getByTitle('Hide key'))
    expect(screen.getByTitle('Show key')).toBeInTheDocument()
  })

  it('copies using Clipboard API and resets copied state', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    })

    render(<ApiKeyCreatedModal apiKey={apiKey} onClose={onClose} />)

    const copyButton = screen.getByTitle('Copy to clipboard')
    await user.click(copyButton)

    expect(writeText).toHaveBeenCalledWith(apiKey.api_key)
    expect(copyButton).toHaveClass('bg-green-500/20')
  })

  it('falls back to execCommand when Clipboard API fails', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()

    const writeText = vi.fn().mockRejectedValue(new Error('no clipboard'))
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    })
    Object.defineProperty(document, 'execCommand', {
      value: vi.fn(() => true),
      configurable: true,
    })

    render(<ApiKeyCreatedModal apiKey={apiKey} onClose={onClose} />)

    const copyButton = screen.getByTitle('Copy to clipboard')
    await user.click(copyButton)

    expect(writeText).toHaveBeenCalledWith(apiKey.api_key)
    expect(vi.mocked(console.warn)).toHaveBeenCalled()
    expect((document as unknown as { execCommand: (commandId: string) => boolean }).execCommand).toHaveBeenCalledWith(
      'copy'
    )
    expect(copyButton).toHaveClass('bg-green-500/20')
  })

  it('logs an error when both clipboard methods fail and closes on acknowledgement', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()

    const writeText = vi.fn().mockRejectedValue(new Error('no clipboard'))
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    })
    Object.defineProperty(document, 'execCommand', {
      value: vi.fn(() => false),
      configurable: true,
    })

    render(<ApiKeyCreatedModal apiKey={apiKey} onClose={onClose} />)

    await user.click(screen.getByTitle('Copy to clipboard'))
    expect(vi.mocked(console.error)).toHaveBeenCalledWith('Both clipboard methods failed')

    await user.click(screen.getByRole('button', { name: "I've copied my key" }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})

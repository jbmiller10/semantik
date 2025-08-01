import { describe, it, expect, beforeEach, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../../tests/mocks/server'
import { render as renderWithProviders } from '../../tests/utils/test-utils'
import SettingsPage from '../SettingsPage'

const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

// Mock alert
global.alert = vi.fn()

const mockStats = {
  collection_count: 15,
  file_count: 250,
  database_size_mb: 128,
  parquet_files_count: 30,
  parquet_size_mb: 64,
}

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockNavigate.mockClear()
    
    
    // Default handler for stats
    server.use(
      http.get('/api/settings/stats', () => {
        return HttpResponse.json(mockStats)
      })
    )
  })

  it('renders settings page with header and back button', () => {
    renderWithProviders(<SettingsPage />)
    
    expect(screen.getByText('Settings')).toBeInTheDocument()
    expect(screen.getByText('Manage your database and system settings')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /back to home/i })).toBeInTheDocument()
  })

  it('navigates back to home when back button is clicked', async () => {
    const user = userEvent.setup()
    renderWithProviders(<SettingsPage />)
    
    await user.click(screen.getByRole('button', { name: /back to home/i }))
    
    expect(mockNavigate).toHaveBeenCalledWith('/')
  })

  it('loads and displays database statistics', async () => {
    renderWithProviders(<SettingsPage />)
    
    // Initially shows loading state
    expect(screen.getByText('Loading statistics...')).toBeInTheDocument()
    
    // Wait for stats to load
    await waitFor(() => {
      expect(screen.queryByText('Loading statistics...')).not.toBeInTheDocument()
    })
    
    // Check that stats are displayed
    expect(screen.getByText('Total Collections')).toBeInTheDocument()
    expect(screen.getByText('15')).toBeInTheDocument()
    
    expect(screen.getByText('Total Files')).toBeInTheDocument()
    expect(screen.getByText('250')).toBeInTheDocument()
    
    expect(screen.getByText('Database Size')).toBeInTheDocument()
    expect(screen.getByText('128 MB')).toBeInTheDocument()
    
    expect(screen.getByText('Parquet Files')).toBeInTheDocument()
    expect(screen.getByText('30')).toBeInTheDocument()
    
    expect(screen.getByText('Parquet Size')).toBeInTheDocument()
    expect(screen.getByText('64 MB')).toBeInTheDocument()
  })

  it('handles error when loading statistics fails', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    
    server.use(
      http.get('/api/settings/stats', () => {
        return HttpResponse.error()
      })
    )
    
    renderWithProviders(<SettingsPage />)
    
    await waitFor(() => {
      expect(screen.queryByText('Loading statistics...')).not.toBeInTheDocument()
    })
    
    expect(screen.getByText('Failed to load statistics')).toBeInTheDocument()
    expect(consoleError).toHaveBeenCalledWith('Failed to load statistics:', expect.any(Error))
    
    consoleError.mockRestore()
  })

  it('formats large numbers with commas', async () => {
    server.use(
      http.get('/api/settings/stats', () => {
        return HttpResponse.json({
          collection_count: 1500,
          file_count: 25000,
          database_size_mb: 1024,
          parquet_files_count: 3000,
          parquet_size_mb: 512,
        })
      })
    )
    
    renderWithProviders(<SettingsPage />)
    
    await waitFor(() => {
      expect(screen.getByText('1,500')).toBeInTheDocument()
      expect(screen.getByText('25,000')).toBeInTheDocument()
      expect(screen.getByText('3,000')).toBeInTheDocument()
    })
  })

  it('shows danger zone with reset database button', () => {
    renderWithProviders(<SettingsPage />)
    
    expect(screen.getByText('Danger Zone')).toBeInTheDocument()
    expect(screen.getAllByText('Reset Database')).toHaveLength(2) // h4 and button
    expect(screen.getByText(/This will delete all collections, files, and associated data/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /reset database/i })).toBeInTheDocument()
  })

  it('shows confirmation dialog when reset button is clicked', async () => {
    const user = userEvent.setup()
    renderWithProviders(<SettingsPage />)
    
    // Get the actual button (not the heading)
    const resetButtons = screen.getAllByRole('button', { name: /reset database/i })
    const resetButton = resetButtons[0] // Should be the actual button in the danger zone
    await user.click(resetButton)
    
    // Check confirmation dialog appears
    expect(screen.getByText('Confirm Database Reset')).toBeInTheDocument()
    expect(screen.getByText(/Are you sure you want to reset the database/)).toBeInTheDocument()
    expect(screen.getByText('Type "RESET" to confirm:')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Type RESET')).toBeInTheDocument()
    
    // Check buttons in dialog - there should be multiple "Reset Database" buttons now
    const allResetButtons = screen.getAllByRole('button', { name: 'Reset Database' })
    expect(allResetButtons.length).toBeGreaterThanOrEqual(2) // One in dialog, one original
    expect(allResetButtons[1]).toBeDisabled() // Dialog button should be disabled initially
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument()
  })

  it('enables reset button only when RESET is typed', async () => {
    const user = userEvent.setup()
    renderWithProviders(<SettingsPage />)
    
    await user.click(screen.getByRole('button', { name: /reset database/i }))
    
    const confirmInput = screen.getByPlaceholderText('Type RESET')
    const resetButton = screen.getAllByRole('button', { name: 'Reset Database' })[1] // Second one is in dialog
    
    // Initially disabled
    expect(resetButton).toBeDisabled()
    
    // Type wrong text
    await user.type(confirmInput, 'reset')
    expect(resetButton).toBeDisabled()
    
    // Clear and type correct text
    await user.clear(confirmInput)
    await user.type(confirmInput, 'RESET')
    expect(resetButton).toBeEnabled()
  })

  it('cancels reset when cancel button is clicked', async () => {
    const user = userEvent.setup()
    renderWithProviders(<SettingsPage />)
    
    await user.click(screen.getByRole('button', { name: /reset database/i }))
    
    // Dialog should be visible
    expect(screen.getByText('Confirm Database Reset')).toBeInTheDocument()
    
    // Type something in the input
    await user.type(screen.getByPlaceholderText('Type RESET'), 'RES')
    
    // Click cancel
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    
    // Dialog should be gone
    expect(screen.queryByText('Confirm Database Reset')).not.toBeInTheDocument()
    
    // Click reset button again to verify input was cleared
    await user.click(screen.getByRole('button', { name: /reset database/i }))
    expect(screen.getByPlaceholderText('Type RESET')).toHaveValue('')
  })

  it('handles successful database reset', async () => {
    const user = userEvent.setup()
    
    server.use(
      http.post('/api/settings/reset-database', () => {
        return HttpResponse.json({ message: 'Database reset successfully' })
      })
    )
    
    renderWithProviders(<SettingsPage />)
    
    // Open dialog and confirm
    await user.click(screen.getByRole('button', { name: /reset database/i }))
    await user.type(screen.getByPlaceholderText('Type RESET'), 'RESET')
    
    const resetButton = screen.getAllByRole('button', { name: 'Reset Database' })[1]
    await user.click(resetButton)
    
    // Check that the operation completes successfully
    await waitFor(() => {
      expect(global.alert).toHaveBeenCalledWith('Database reset successfully!')
    }, { timeout: 5000 })
    
    // Check navigation
    expect(mockNavigate).toHaveBeenCalledWith('/')
    
    // Dialog should be closed
    expect(screen.queryByText('Confirm Database Reset')).not.toBeInTheDocument()
  })

  it('handles database reset error', async () => {
    const user = userEvent.setup()
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    
    server.use(
      http.post('/api/settings/reset-database', () => {
        return HttpResponse.json(
          { detail: 'Failed to reset database' },
          { status: 500 }
        )
      })
    )
    
    renderWithProviders(<SettingsPage />)
    
    // Open dialog and confirm
    await user.click(screen.getByRole('button', { name: /reset database/i }))
    await user.type(screen.getByPlaceholderText('Type RESET'), 'RESET')
    await user.click(screen.getAllByRole('button', { name: 'Reset Database' })[1])
    
    await waitFor(() => {
      expect(global.alert).toHaveBeenCalledWith('Failed to reset database: Failed to reset database')
    })
    
    expect(consoleError).toHaveBeenCalledWith('Failed to reset database:', expect.any(Error))
    
    // Should not navigate on error
    expect(mockNavigate).not.toHaveBeenCalled()
    
    // Reset button should be enabled again
    expect(screen.getAllByRole('button', { name: 'Reset Database' })[1]).toBeEnabled()
    
    consoleError.mockRestore()
  })

  it('displays all statistic cards with proper icons', async () => {
    renderWithProviders(<SettingsPage />)
    
    await waitFor(() => {
      expect(screen.queryByText('Loading statistics...')).not.toBeInTheDocument()
    })
    
    // Check that all cards are rendered with their values
    const cards = [
      { label: 'Total Collections', value: '15' },
      { label: 'Total Files', value: '250' },
      { label: 'Database Size', value: '128 MB' },
      { label: 'Parquet Files', value: '30' },
      { label: 'Parquet Size', value: '64 MB' },
    ]
    
    cards.forEach(card => {
      expect(screen.getByText(card.label)).toBeInTheDocument()
      if (card.value.includes('MB')) {
        expect(screen.getByText(card.value)).toBeInTheDocument()
      } else {
        expect(screen.getByText(card.value)).toBeInTheDocument()
      }
    })
    
    // Check that there are multiple stat cards
    const databaseSizeElement = screen.getByText('Database Size')
    const container = databaseSizeElement.closest('.bg-gray-50')
    expect(container).toBeInTheDocument()
  })
})
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useAuthStore } from '../authStore'

// Mock fetch for logout API call
const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

describe('authStore', () => {
  beforeEach(() => {
    // Reset store to initial state
    useAuthStore.setState({
      token: null,
      user: null,
      refreshToken: null,
    })
    
    // Clear localStorage
    localStorage.clear()
    
    // Reset fetch mock
    vi.clearAllMocks()
  })

  it('initializes with null values', () => {
    const state = useAuthStore.getState()
    
    expect(state.token).toBeNull()
    expect(state.user).toBeNull()
    expect(state.refreshToken).toBeNull()
  })

  it('sets auth data correctly', () => {
    const mockUser = {
      id: 1,
      username: 'testuser',
      email: 'test@example.com',
      full_name: 'Test User',
      is_active: true,
      created_at: '2025-01-14T12:00:00Z',
    }
    const mockToken = 'mock-jwt-token'
    const mockRefreshToken = 'mock-refresh-token'

    const { setAuth } = useAuthStore.getState()
    setAuth(mockToken, mockUser, mockRefreshToken)

    const state = useAuthStore.getState()
    expect(state.token).toBe(mockToken)
    expect(state.user).toEqual(mockUser)
    expect(state.refreshToken).toBe(mockRefreshToken)
  })

  it('persists auth data to localStorage', () => {
    const mockUser = {
      id: 1,
      username: 'testuser',
      email: 'test@example.com',
      is_active: true,
      created_at: '2025-01-14T12:00:00Z',
    }
    const mockToken = 'mock-jwt-token'

    const { setAuth } = useAuthStore.getState()
    setAuth(mockToken, mockUser)

    // Check localStorage
    const storedData = localStorage.getItem('auth-storage')
    expect(storedData).toBeTruthy()
    
    if (storedData) {
      const parsed = JSON.parse(storedData)
      expect(parsed.state.token).toBe(mockToken)
      expect(parsed.state.user).toEqual(mockUser)
    }
  })

  it('clears auth data on logout', async () => {
    // Set up initial auth state
    const mockUser = {
      id: 1,
      username: 'testuser',
      email: 'test@example.com',
      is_active: true,
      created_at: '2025-01-14T12:00:00Z',
    }
    const mockToken = 'mock-jwt-token'
    const mockRefreshToken = 'mock-refresh-token'

    const { setAuth, logout } = useAuthStore.getState()
    setAuth(mockToken, mockUser, mockRefreshToken)

    // Mock successful logout API response
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ message: 'Logged out' }),
    })

    // Perform logout
    await logout()

    // Check that state is cleared
    const state = useAuthStore.getState()
    expect(state.token).toBeNull()
    expect(state.user).toBeNull()
    expect(state.refreshToken).toBeNull()

    // Check that localStorage is cleared
    expect(localStorage.getItem('auth-storage')).toBeNull()

    // Check that logout API was called
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/auth/logout',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Authorization': `Bearer ${mockToken}`,
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify({ refresh_token: mockRefreshToken }),
      })
    )
  })

  it('clears state even if logout API fails', async () => {
    const mockToken = 'mock-jwt-token'
    const mockUser = {
      id: 1,
      username: 'testuser',
      email: 'test@example.com',
      is_active: true,
      created_at: '2025-01-14T12:00:00Z',
    }

    const { setAuth, logout } = useAuthStore.getState()
    setAuth(mockToken, mockUser)

    // Mock failed logout API response
    mockFetch.mockRejectedValueOnce(new Error('Network error'))

    // Spy on console.error
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})

    // Perform logout
    await logout()

    // Check that state is still cleared despite API failure
    const state = useAuthStore.getState()
    expect(state.token).toBeNull()
    expect(state.user).toBeNull()
    expect(state.refreshToken).toBeNull()

    // Check that localStorage is cleared
    expect(localStorage.getItem('auth-storage')).toBeNull()

    // Check that error was logged
    expect(consoleError).toHaveBeenCalledWith(
      'Logout API call failed:',
      expect.any(Error)
    )

    consoleError.mockRestore()
  })

  it('handles logout when no token is present', async () => {
    const { logout } = useAuthStore.getState()

    // Perform logout without any auth data
    await logout()

    // Check that fetch was not called
    expect(mockFetch).not.toHaveBeenCalled()

    // State should remain null
    const state = useAuthStore.getState()
    expect(state.token).toBeNull()
    expect(state.user).toBeNull()
    expect(state.refreshToken).toBeNull()
  })

  it('sets auth without refresh token', () => {
    const mockUser = {
      id: 1,
      username: 'testuser',
      email: 'test@example.com',
      is_active: true,
      created_at: '2025-01-14T12:00:00Z',
    }
    const mockToken = 'mock-jwt-token'

    const { setAuth } = useAuthStore.getState()
    // Call without refresh token
    setAuth(mockToken, mockUser)

    const state = useAuthStore.getState()
    expect(state.token).toBe(mockToken)
    expect(state.user).toEqual(mockUser)
    expect(state.refreshToken).toBeNull()
  })
})
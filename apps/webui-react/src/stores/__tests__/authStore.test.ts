import { describe, it, expect, beforeEach, vi } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from '../../tests/mocks/server'
import { useAuthStore } from '../authStore'

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
    
    // Reset all mocks
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

    // Track if logout API was called
    let logoutCalled = false
    server.use(
      http.post('/api/auth/logout', () => {
        logoutCalled = true
        return HttpResponse.json({ message: 'Logged out successfully' })
      })
    )

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
    expect(logoutCalled).toBe(true)
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
    server.use(
      http.post('/api/auth/logout', () => {
        return HttpResponse.error()
      })
    )

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

    // Track if logout API was called
    let logoutCalled = false
    server.use(
      http.post('/api/auth/logout', () => {
        logoutCalled = true
        return HttpResponse.json({ message: 'Logged out successfully' })
      })
    )

    // Perform logout without any auth data
    await logout()

    // Check that API was not called (no token)
    expect(logoutCalled).toBe(false)

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
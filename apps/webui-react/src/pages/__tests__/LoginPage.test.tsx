import { describe, it, expect, beforeEach, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../../tests/mocks/server'
import { render as renderWithProviders } from '@/tests/utils/test-utils'
import LoginPage from '../LoginPage'
import { useAuthStore } from '../../stores/authStore'
import { useUIStore } from '../../stores/uiStore'

const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockNavigate.mockClear()
    
    // Reset stores
    useAuthStore.setState({
      token: null,
      user: null,
      refreshToken: null,
    })
    useUIStore.setState({
      toasts: [],
    })
  })

  it('renders login form by default', () => {
    renderWithProviders(<LoginPage />)
    
    expect(screen.getByText('Sign in to Semantik')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Username')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Password')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeInTheDocument()
    expect(screen.getByText("Don't have an account? Register")).toBeInTheDocument()
    
    // Email and full name fields should not be visible in login mode
    expect(screen.queryByPlaceholderText('Email address')).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText('Full Name (optional)')).not.toBeInTheDocument()
  })

  it('toggles to registration form', async () => {
    const user = userEvent.setup()
    renderWithProviders(<LoginPage />)
    
    const toggleButton = screen.getByText("Don't have an account? Register")
    await user.click(toggleButton)
    
    expect(screen.getByText('Create a Semantik account')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Email address')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Full Name (optional)')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Register' })).toBeInTheDocument()
    expect(screen.getByText('Already have an account? Sign in')).toBeInTheDocument()
  })

  it('handles successful login', async () => {
    const user = userEvent.setup()
    
    const mockUser = {
      id: 1,
      username: 'testuser',
      email: 'test@example.com',
      full_name: 'Test User',
      is_active: true,
      created_at: new Date().toISOString(),
    }
    
    server.use(
      http.post('/api/auth/login', async ({ request }) => {
        const body = await request.json() as any
        if (body.username === 'testuser' && body.password === 'testpass') {
          return HttpResponse.json({
            access_token: 'mock-jwt-token',
            refresh_token: 'mock-refresh-token',
            user: mockUser,
          })
        }
        return HttpResponse.json(
          { detail: 'Invalid credentials' },
          { status: 401 }
        )
      }),
      http.get('/api/auth/me', () => {
        return HttpResponse.json(mockUser)
      })
    )
    
    renderWithProviders(<LoginPage />)
    
    // Fill in login form
    await user.type(screen.getByPlaceholderText('Username'), 'testuser')
    await user.type(screen.getByPlaceholderText('Password'), 'testpass')
    
    // Submit form
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    
    // Check loading state
    expect(screen.getByRole('button', { name: 'Processing...' })).toBeDisabled()
    
    // Wait for navigation
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/')
    })
    
    // Check that auth was set
    const authState = useAuthStore.getState()
    expect(authState.token).toBe('mock-jwt-token')
    expect(authState.user).toEqual(mockUser)
    expect(authState.refreshToken).toBe('mock-refresh-token')
    
    // Check toast notification
    const uiState = useUIStore.getState()
    expect(uiState.toasts).toHaveLength(1)
    expect(uiState.toasts[0]).toMatchObject({
      type: 'success',
      message: 'Logged in successfully',
    })
  })

  it('handles login error', async () => {
    const user = userEvent.setup()
    
    server.use(
      http.post('/api/auth/login', () => {
        return HttpResponse.json(
          { detail: 'Invalid username or password' },
          { status: 401 }
        )
      })
    )
    
    renderWithProviders(<LoginPage />)
    
    await user.type(screen.getByPlaceholderText('Username'), 'wronguser')
    await user.type(screen.getByPlaceholderText('Password'), 'wrongpass')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Sign in' })).toBeEnabled()
    })
    
    // Check that navigation didn't happen
    expect(mockNavigate).not.toHaveBeenCalled()
    
    // Check error toast
    const uiState = useUIStore.getState()
    expect(uiState.toasts).toHaveLength(1)
    expect(uiState.toasts[0]).toMatchObject({
      type: 'error',
      message: 'Invalid username or password',
    })
  })

  it('handles successful registration', async () => {
    const user = userEvent.setup()
    
    server.use(
      http.post('/api/auth/register', async ({ request }) => {
        const body = await request.json() as any
        return HttpResponse.json({
          id: 2,
          username: body.username,
          email: body.email,
          full_name: body.full_name,
          is_active: true,
          created_at: new Date().toISOString(),
        })
      })
    )
    
    renderWithProviders(<LoginPage />)
    
    // Switch to registration mode
    await user.click(screen.getByText("Don't have an account? Register"))
    
    // Fill in registration form
    await user.type(screen.getByPlaceholderText('Username'), 'newuser')
    await user.type(screen.getByPlaceholderText('Email address'), 'new@example.com')
    await user.type(screen.getByPlaceholderText('Full Name (optional)'), 'New User')
    await user.type(screen.getByPlaceholderText('Password'), 'newpass')
    
    // Submit form
    await user.click(screen.getByRole('button', { name: 'Register' }))
    
    await waitFor(() => {
      // Should switch back to login mode
      expect(screen.getByText('Sign in to Semantik')).toBeInTheDocument()
    })
    
    // Username should be preserved, password should be cleared
    expect(screen.getByPlaceholderText('Username')).toHaveValue('newuser')
    expect(screen.getByPlaceholderText('Password')).toHaveValue('')
    
    // Check success toast
    const uiState = useUIStore.getState()
    expect(uiState.toasts).toHaveLength(1)
    expect(uiState.toasts[0]).toMatchObject({
      type: 'success',
      message: 'Registration successful! Please log in with your credentials.',
    })
  })

  it('handles registration error', async () => {
    const user = userEvent.setup()
    
    server.use(
      http.post('/api/auth/register', () => {
        return HttpResponse.json(
          { detail: 'Username already exists' },
          { status: 400 }
        )
      })
    )
    
    renderWithProviders(<LoginPage />)
    
    // Switch to registration mode
    await user.click(screen.getByText("Don't have an account? Register"))
    
    // Fill in registration form
    await user.type(screen.getByPlaceholderText('Username'), 'existinguser')
    await user.type(screen.getByPlaceholderText('Email address'), 'existing@example.com')
    await user.type(screen.getByPlaceholderText('Password'), 'password')
    
    // Submit form
    await user.click(screen.getByRole('button', { name: 'Register' }))
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Register' })).toBeEnabled()
    })
    
    // Should still be in registration mode
    expect(screen.getByText('Create a Semantik account')).toBeInTheDocument()
    
    // Check error toast
    const uiState = useUIStore.getState()
    expect(uiState.toasts).toHaveLength(1)
    expect(uiState.toasts[0]).toMatchObject({
      type: 'error',
      message: 'Username already exists',
    })
  })

  it('disables form during submission', async () => {
    const user = userEvent.setup()
    
    let resolveLogin: ((value: any) => void) | null = null
    const loginPromise = new Promise((resolve) => {
      resolveLogin = resolve
    })
    
    server.use(
      http.post('/api/auth/login', async () => {
        await loginPromise
        return HttpResponse.json({
          access_token: 'mock-jwt-token',
          refresh_token: 'mock-refresh-token',
        })
      })
    )
    
    renderWithProviders(<LoginPage />)
    
    await user.type(screen.getByPlaceholderText('Username'), 'testuser')
    await user.type(screen.getByPlaceholderText('Password'), 'testpass')
    
    // Start submission
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    
    // Button should be disabled and show loading text
    const submitButton = screen.getByRole('button', { name: 'Processing...' })
    expect(submitButton).toBeDisabled()
    
    // Resolve the promise to complete the login
    resolveLogin!({})
  })

  it('handles network error gracefully', async () => {
    const user = userEvent.setup()
    
    server.use(
      http.post('/api/auth/login', () => {
        return HttpResponse.error()
      })
    )
    
    renderWithProviders(<LoginPage />)
    
    await user.type(screen.getByPlaceholderText('Username'), 'testuser')
    await user.type(screen.getByPlaceholderText('Password'), 'testpass')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Sign in' })).toBeEnabled()
    })
    
    // Check generic error toast
    const uiState = useUIStore.getState()
    expect(uiState.toasts).toHaveLength(1)
    expect(uiState.toasts[0]).toMatchObject({
      type: 'error',
      message: 'Authentication failed',
    })
  })
})
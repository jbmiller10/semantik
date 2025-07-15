import { describe, it, expect } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from './mocks/server'

describe('MSW Handler Demo', () => {
  it('demonstrates that MSW is intercepting API calls', async () => {
    // This test verifies that our MSW setup is working correctly
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'testuser', password: 'testpass' }),
    })

    const data = await response.json()
    
    expect(response.status).toBe(200)
    expect(data.access_token).toBe('mock-jwt-token')
    expect(data.user.username).toBe('testuser')
  })

  it('handles failed authentication', async () => {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'wrong', password: 'wrong' }),
    })

    const data = await response.json()
    
    expect(response.status).toBe(401)
    expect(data.detail).toBe('Invalid credentials')
  })

  it('can override handlers for specific tests', async () => {
    // Override the default handler for this test
    server.use(
      http.post('/api/auth/login', () => {
        return HttpResponse.json(
          { detail: 'Server error' },
          { status: 500 }
        )
      })
    )

    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'testuser', password: 'testpass' }),
    })

    expect(response.status).toBe(500)
    const data = await response.json()
    expect(data.detail).toBe('Server error')
  })
})
import { describe, it, expect } from 'vitest'

describe('Basic Test Setup', () => {
  it('should run a simple test', () => {
    expect(1 + 1).toBe(2)
  })

  it('should handle async operations', async () => {
    const promise = Promise.resolve('test')
    const result = await promise
    expect(result).toBe('test')
  })
})
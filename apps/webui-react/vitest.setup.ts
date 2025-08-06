import '@testing-library/jest-dom';
import { vi, beforeAll, afterEach, afterAll } from 'vitest';
import { server } from './src/tests/mocks/server';

// Suppress console errors in tests by default
const originalError = console.error;
const originalWarn = console.warn;

beforeAll(() => {
  // Start MSW server before all tests
  server.listen({ onUnhandledRequest: 'error' });
  
  // Mock console.error and console.warn to reduce noise in test output
  console.error = vi.fn();
  console.warn = vi.fn();
});

// Reset handlers after each test
afterEach(() => {
  server.resetHandlers();
  // Clear mock calls but keep the mocks in place
  vi.mocked(console.error).mockClear();
  vi.mocked(console.warn).mockClear();
});

// Clean up after all tests
afterAll(() => {
  server.close();
  // Restore original console methods
  console.error = originalError;
  console.warn = originalWarn;
});

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  disconnect: vi.fn(),
  observe: vi.fn(),
  unobserve: vi.fn(),
}));

// Mock scrollTo
window.scrollTo = vi.fn();

// Mock scrollIntoView
HTMLElement.prototype.scrollIntoView = vi.fn();

// Clean up after each test
afterEach(() => {
  vi.clearAllMocks();
});
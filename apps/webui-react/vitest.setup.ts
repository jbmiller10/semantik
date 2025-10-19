import '@testing-library/jest-dom';
import { vi, beforeAll, afterEach, afterAll } from 'vitest';
import { server } from './src/tests/mocks/server';

// Suppress console errors in tests by default
const originalError = console.error;
const originalWarn = console.warn;

// Track timers created during tests so we can clean them up explicitly
const activeTimeouts = new Set<ReturnType<typeof setTimeout>>();
const activeIntervals = new Set<ReturnType<typeof setInterval>>();

const originalSetTimeout = global.setTimeout;
const originalClearTimeout = global.clearTimeout;
const originalSetInterval = global.setInterval;
const originalClearInterval = global.clearInterval;

global.setTimeout = ((handler: TimerHandler, timeout?: number, ...args: unknown[]) => {
  const id = originalSetTimeout(handler, timeout, ...args);
  activeTimeouts.add(id);
  return id;
}) as typeof setTimeout;
window.setTimeout = global.setTimeout;

global.clearTimeout = ((id?: number | NodeJS.Timeout) => {
  if (id !== undefined && id !== null) {
    activeTimeouts.delete(id as ReturnType<typeof setTimeout>);
  }
  return originalClearTimeout(id as ReturnType<typeof setTimeout>);
}) as typeof clearTimeout;
window.clearTimeout = global.clearTimeout;

global.setInterval = ((handler: TimerHandler, timeout?: number, ...args: unknown[]) => {
  const id = originalSetInterval(handler, timeout, ...args);
  activeIntervals.add(id);
  return id;
}) as typeof setInterval;
window.setInterval = global.setInterval;

global.clearInterval = ((id?: number | NodeJS.Timeout) => {
  if (id !== undefined && id !== null) {
    activeIntervals.delete(id as ReturnType<typeof setInterval>);
  }
  return originalClearInterval(id as ReturnType<typeof setInterval>);
}) as typeof clearInterval;
window.clearInterval = global.clearInterval;

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

  // Ensure all timers created during the test are cleared
  activeTimeouts.forEach((id) => originalClearTimeout(id));
  activeTimeouts.clear();
  activeIntervals.forEach((id) => originalClearInterval(id));
  activeIntervals.clear();
});

// Clean up after all tests
afterAll(() => {
  server.close();
  // Restore original console methods
  console.error = originalError;
  console.warn = originalWarn;

  global.setTimeout = originalSetTimeout;
  global.clearTimeout = originalClearTimeout;
  global.setInterval = originalSetInterval;
  global.clearInterval = originalClearInterval;
  window.setTimeout = originalSetTimeout;
  window.clearTimeout = originalClearTimeout;
  window.setInterval = originalSetInterval;
  window.clearInterval = originalClearInterval;
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

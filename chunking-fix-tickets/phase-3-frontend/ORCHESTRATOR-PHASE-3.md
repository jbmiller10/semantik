# ORCHESTRATOR PHASE 3: Frontend Integration Fixes

## Phase Overview
**Priority**: CRITICAL - User-facing functionality broken  
**Total Duration**: 13 hours  
**Risk Level**: HIGH - Features completely non-functional with mocks  
**Success Gate**: Real-time chunking works end-to-end in browser

## Context
The frontend appears functional but is entirely mocked. No real API calls, no WebSocket integration, missing error boundaries, and accessibility violations. This phase connects the frontend to the real backend and ensures a production-ready user experience.

## Execution Strategy

### Pre-Flight Checklist
- [ ] Phase 2 backend fully operational
- [ ] API endpoints tested and documented
- [ ] WebSocket server running
- [ ] Test user accounts created
- [ ] Browser testing tools ready

### Ticket Execution Order

#### Stage 1: API Integration (4 hours)
**Ticket**: FE-001 - Replace Mock API Calls with Real Implementation

**Critical Actions**:
1. Create ChunkingAPIClient with proper types
2. Remove ALL mock implementations
3. Implement authentication flow
4. Add retry logic and error handling
5. Update store to use real API

**Validation**:
```typescript
// Test real API integration
describe('API Integration', () => {
  it('should fetch real strategies from backend', async () => {
    const strategies = await chunkingApi.getStrategies();
    expect(strategies).toHaveLength(6);
    expect(strategies[0].id).toBeTruthy();
    expect(strategies[0].name).toBeTruthy();
  });

  it('should handle auth properly', async () => {
    localStorage.setItem('auth_token', 'test-token');
    const response = await chunkingApi.previewChunking({...});
    expect(response.chunks).toBeDefined();
  });

  it('should show real preview results', async () => {
    renderWithProviders(<ChunkingPreviewPanel />);
    fireEvent.click(screen.getByText('Preview'));
    
    await waitFor(() => {
      expect(screen.getByText(/Chunk 1/)).toBeInTheDocument();
    });
  });
});
```

**Browser Validation**:
```javascript
// Run in browser console
(async () => {
  const store = window.__ZUSTAND_STORE__;
  await store.getState().fetchStrategies();
  const strategies = store.getState().strategies;
  console.assert(strategies.length === 6, 'Should have 6 strategies');
  console.assert(!strategies[0].mock, 'Should not be mock data');
  console.log('✓ API integration validated');
})();
```

#### Stage 2: WebSocket Implementation (4 hours)
**Ticket**: FE-002 - Implement WebSocket Integration

**Critical Actions**:
1. Create WebSocketService with reconnection
2. Implement useChunkingWebSocket hook
3. Update components for real-time updates
4. Add connection status indicators
5. Implement graceful degradation

**Validation**:
```typescript
// WebSocket integration test
it('should receive real-time updates', async () => {
  const { result } = renderHook(() => useChunkingWebSocket());
  
  act(() => {
    result.current.connect('test-operation');
  });
  
  // Wait for connection
  await waitFor(() => {
    expect(result.current.connectionState).toBe('connected');
  });
  
  // Simulate backend sending progress
  mockWS.send({
    type: 'progress',
    data: { progress: 50, current_chunk: 5, total_chunks: 10 }
  });
  
  expect(result.current.progress.progress).toBe(50);
});
```

**Live Test**:
```javascript
// Test WebSocket in browser
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'chunking:test',
    data: {}
  }));
};
ws.onmessage = (e) => {
  console.log('Real-time update:', JSON.parse(e.data));
};
```

#### Stage 3: Error Boundaries (2 hours)
**Ticket**: FE-003 - Add Error Boundaries

**Parallel Execution**: Can run parallel with FE-004

**Critical Actions**:
1. Create ChunkingErrorBoundary component
2. Wrap all major components
3. Implement fallback UI
4. Add error reporting
5. Provide recovery actions

**Validation**:
```typescript
// Error boundary test
it('should catch and display errors gracefully', () => {
  const ThrowError = () => {
    throw new Error('Test error');
  };
  
  render(
    <ChunkingErrorBoundary>
      <ThrowError />
    </ChunkingErrorBoundary>
  );
  
  expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
  expect(screen.getByText('Retry')).toBeInTheDocument();
});

// Test in browser
(() => {
  // Intentionally cause error
  window.__BREAK_CHUNKING__ = true;
  document.querySelector('[data-testid="preview-button"]').click();
  
  // Should see error UI, not white screen
  const errorUI = document.querySelector('[data-testid="error-boundary"]');
  console.assert(errorUI, 'Error boundary should catch error');
})();
```

#### Stage 4: Accessibility Fixes (3 hours)
**Ticket**: FE-004 - Fix Accessibility Issues

**Parallel Execution**: Can run parallel with FE-003

**Critical Actions**:
1. Add ARIA labels to all interactive elements
2. Implement keyboard navigation
3. Add screen reader announcements
4. Fix focus management
5. Ensure WCAG AA compliance

**Validation**:
```typescript
// Accessibility tests
import { axe } from '@axe-core/react';

it('should have no accessibility violations', async () => {
  const { container } = render(<ChunkingStrategySelector />);
  const results = await axe(container);
  expect(results.violations).toHaveLength(0);
});

it('should be keyboard navigable', () => {
  render(<ChunkingParameterTuner />);
  const slider = screen.getByRole('slider');
  
  slider.focus();
  fireEvent.keyDown(slider, { key: 'ArrowRight' });
  expect(slider.getAttribute('aria-valuenow')).toBe('51');
  
  fireEvent.keyDown(slider, { key: 'Home' });
  expect(slider.getAttribute('aria-valuenow')).toBe('0');
});
```

**Manual Testing**:
```javascript
// Keyboard navigation test
document.querySelector('[role="slider"]').focus();
// Press arrow keys - should change value
// Press Tab - should move to next element
// All interactive elements should be reachable

// Screen reader test (NVDA/JAWS)
// Should announce:
// - "Chunk size slider, 500, adjustable"
// - "Processing chunk 5 of 10"
// - "Chunking completed successfully"
```

### Integration Testing Protocol

#### End-to-End User Flow
```typescript
describe('Complete Chunking Workflow', () => {
  it('should work end-to-end with real backend', async () => {
    // 1. Login
    await login('test@example.com', 'password');
    
    // 2. Upload document
    const file = new File(['content'], 'test.txt');
    await uploadDocument(file);
    
    // 3. Select strategy
    fireEvent.click(screen.getByText('Semantic'));
    
    // 4. Adjust parameters
    const slider = screen.getByLabelText('Chunk Size');
    fireEvent.change(slider, { target: { value: 1000 } });
    
    // 5. Preview with WebSocket updates
    fireEvent.click(screen.getByText('Preview'));
    
    // 6. Watch real-time progress
    await waitFor(() => {
      expect(screen.getByText('100%')).toBeInTheDocument();
    });
    
    // 7. View results
    expect(screen.getAllByTestId('chunk-card')).toHaveLength(10);
    
    // 8. Apply chunking
    fireEvent.click(screen.getByText('Apply'));
    
    // 9. Confirm success
    await waitFor(() => {
      expect(screen.getByText('Chunking completed')).toBeInTheDocument();
    });
  });
});
```

### Browser Compatibility Matrix

Test on these browsers:
- Chrome 90+ ✓
- Firefox 88+ ✓
- Safari 14+ ✓
- Edge 90+ ✓
- Mobile Safari (iOS 14+) ✓
- Chrome Mobile (Android 10+) ✓

```javascript
// Browser feature detection
const features = {
  websocket: 'WebSocket' in window,
  indexeddb: 'indexedDB' in window,
  serviceworker: 'serviceWorker' in navigator,
  intersectionobserver: 'IntersectionObserver' in window
};

console.table(features);
// All should be true for full functionality
```

### Performance Requirements

#### Metrics to Achieve
```javascript
// Performance measurement
const measurePerformance = () => {
  const metrics = {
    firstContentfulPaint: performance.getEntriesByType('paint')[0].startTime,
    timeToInteractive: performance.timing.domInteractive - performance.timing.navigationStart,
    apiResponseTime: performance.getEntriesByType('resource')
      .filter(e => e.name.includes('/api/'))
      .map(e => e.duration),
    memoryUsage: performance.memory?.usedJSHeapSize / 1048576, // MB
  };
  
  console.table(metrics);
  
  // Targets
  console.assert(metrics.firstContentfulPaint < 1500, 'FCP should be < 1.5s');
  console.assert(metrics.timeToInteractive < 3000, 'TTI should be < 3s');
  console.assert(Math.max(...metrics.apiResponseTime) < 2000, 'API should be < 2s');
  console.assert(metrics.memoryUsage < 100, 'Memory should be < 100MB');
};
```

### State Management Validation

```javascript
// Validate Zustand store
const validateStore = () => {
  const store = window.__ZUSTAND_STORE__;
  const state = store.getState();
  
  // Check no mock data
  console.assert(!state.strategies.some(s => s.mock), 'No mock strategies');
  console.assert(!state.mockApi, 'Mock API removed');
  
  // Check WebSocket integration
  console.assert(state.websocket, 'WebSocket initialized');
  console.assert(state.connectionState, 'Connection state tracked');
  
  // Check error handling
  store.setState({ error: new Error('Test') });
  console.assert(document.querySelector('[data-testid="error-display"]'), 'Error displayed');
  
  console.log('✓ State management validated');
};
```

### Monitoring During Deployment

```javascript
// Real-time monitoring script
const monitor = setInterval(() => {
  const metrics = {
    wsConnections: performance.getEntriesByType('resource')
      .filter(e => e.name.includes('ws://')).length,
    apiErrors: performance.getEntriesByType('resource')
      .filter(e => e.name.includes('/api/') && e.responseStatus >= 400).length,
    memoryTrend: performance.memory?.usedJSHeapSize,
    activeComponents: document.querySelectorAll('[data-component]').length
  };
  
  console.log(new Date().toISOString(), metrics);
  
  // Alert on issues
  if (metrics.apiErrors > 5) {
    console.error('High API error rate!');
  }
  if (metrics.memoryTrend > 200 * 1048576) {
    console.error('Memory usage too high!');
  }
}, 5000);
```

### Success Criteria

#### Functional Requirements
- [ ] All mock calls replaced with real API
- [ ] WebSocket updates working in real-time
- [ ] Error boundaries preventing crashes
- [ ] Full keyboard navigation working
- [ ] Screen reader compatible

#### Performance Metrics
- [ ] Initial load < 3 seconds
- [ ] API responses < 2 seconds
- [ ] WebSocket latency < 100ms
- [ ] Memory usage < 100MB
- [ ] 60 FPS during animations

#### User Experience
- [ ] Progress visible during operations
- [ ] Errors shown with recovery options
- [ ] Connection status always visible
- [ ] Smooth animations and transitions
- [ ] Works on mobile devices

### Rollback Strategy

```javascript
// Feature flags for gradual rollout
const features = {
  useRealAPI: localStorage.getItem('feature_real_api') === 'true',
  useWebSocket: localStorage.getItem('feature_websocket') === 'true',
  useErrorBoundaries: localStorage.getItem('feature_error_boundaries') === 'true'
};

// Quick rollback function
window.rollbackToMocks = () => {
  localStorage.setItem('feature_real_api', 'false');
  localStorage.setItem('feature_websocket', 'false');
  location.reload();
};
```

### Post-Phase Validation

#### User Acceptance Testing
1. Real user performs complete workflow
2. Test on slow network (3G simulation)
3. Test with large documents (10MB+)
4. Test concurrent users
5. Test error recovery scenarios

#### Automated E2E Suite
```bash
# Run Playwright tests
npm run test:e2e -- --grep "chunking"

# Expected output:
# ✓ Complete chunking workflow (15s)
# ✓ Error recovery (5s)
# ✓ WebSocket reconnection (8s)
# ✓ Accessibility compliance (3s)
# ✓ Mobile responsiveness (10s)
```

### Handoff to Phase 4

#### Deliverables
1. Fully functional UI with real backend
2. Real-time updates via WebSocket
3. Comprehensive error handling
4. WCAG AA compliant
5. Performance optimized

#### Documentation Updates
```markdown
## User Guide Updates
- Remove "Preview Only" warnings
- Add WebSocket connection troubleshooting
- Document keyboard shortcuts
- Add accessibility features section
```

## Notes for Orchestrating Agent

**Critical Success Factors**:
1. NO mock data remaining
2. WebSocket MUST work reliably
3. Accessibility is NOT optional
4. Performance targets are requirements
5. Error handling must be comprehensive

**Testing Priority**:
1. Real API calls working
2. WebSocket real-time updates
3. Error scenarios handled
4. Keyboard navigation complete
5. Mobile functionality

**Common Issues**:
- CORS errors → Check backend configuration
- WebSocket drops → Implement heartbeat
- Memory leaks → Check event listener cleanup
- Slow API → Verify backend caching
- Missing ARIA → Use axe-core for validation

This phase makes the application real. Users will immediately notice if anything is still mocked or broken. Test thoroughly with real users before considering complete.
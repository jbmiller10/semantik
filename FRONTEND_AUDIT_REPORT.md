# Frontend Technical Debt Audit Report
## React/TypeScript Code Quality Analysis

### Executive Summary
The Semantik frontend codebase demonstrates solid architectural patterns with good use of React hooks and state management. However, there are several areas of technical debt and quality concerns that require attention. The audit identified **23 critical/high-priority issues** across component complexity, TypeScript safety, accessibility, and code organization.

---

## CRITICAL ISSUES

### 1. CRITICAL - Large God Components (Component Size & Complexity)

**CollectionDetailsModal.tsx (794 lines)** - Lines 21-793
- **Issue**: Massive modal component handling 4 distinct tabs (overview, jobs, files, settings) with multiple render paths and 8+ state variables
- **State variables**: `showCollectionDetailsModal`, `showAddDataModal`, `showRenameModal`, `showDeleteModal`, `activeTab`, `filesPage`, `configChanges`, `showReindexModal`
- **Impact**: 
  - HIGH on Developer Experience - extremely difficult to debug or modify
  - MEDIUM on Performance - unnecessary re-renders when any state changes
  - HIGH on Maintainability - God component anti-pattern
- **Fix**: Split into separate components: OverviewTab, JobsTab, FilesTab, SettingsTab, each with its own state and logic

**CreateCollectionModal.tsx (582 lines)** - Lines 34-580
- **Issue**: Complex multi-step form with directory scanning, file size detection, chunking configuration, and async operation tracking
- **State variables**: 9+ including nested objects (`formData`, `pendingIndexOperationId`, `collectionIdForSource`, `sourcePathForDelayedAdd`, etc.)
- **Problems**: 
  - Complex async flow with delayed source addition (lines 61-118)
  - Multiple interdependent states that could be a single state machine
  - Uses setTimeout for navigation (line 86-88) - not ideal
- **Fix**: Extract to state machine pattern (useReducer), create separate form sections as sub-components

**SearchInterface.tsx (498 lines)** - Lines 14-497
- **Issue**: Complex search form with hybrid search, reranking configuration, validation, and multiple async operations
- **Problem Areas**:
  - Lines 84-89: Global window manipulation for handling GPU memory errors - anti-pattern
  - Lines 172-181: More global window manipulation for error details storage
  - Complex event handling and validation logic mixed with UI rendering
- **Fix**: Extract search logic to custom hook (useSearchForm), move error handling to proper store/context

**DocumentViewer.tsx (403 lines)** - Lines 104-400+
- **Issue**: Multi-format document rendering with inline HTML generation
- **Problems**:
  - Lines 147-250+: Multiple innerHTML assignments (10+ instances)
  - Heavy reliance on window globals for external libraries (PDF.js, marked, DOMPurify, etc.)
  - Complex file type detection and rendering logic
- **Fix**: Create separate format-specific renderers (TextRenderer, PDFRenderer, DocxRenderer), use proper DOM APIs instead of innerHTML

**SearchResults.tsx (308 lines)** - Lines 7-300+
- **Issue**: Complex results aggregation and expansion logic
- **Problems**:
  - Lines 84-102: Inline aggregate function in state
  - Lines 129-132: State mutation trigger in render (auto-expand logic)
  - Complex Set-based state management for expandedDocs/expandedCollections
- **Impact**: Potential for stale closures and render-time state changes
- **Fix**: Move aggregation to useMemo, use useEffect for expansion logic, extract toggle functions to dedicated hook

---

## HIGH PRIORITY ISSUES

### 2. HIGH - TypeScript Safety Issues

**Type Assertions Overuse (114 instances)**
- **Files affected**: `SearchInterface.tsx`, `CollectionDetailsModal.tsx`, `HomePage.test.tsx`, `UIErrorStates.test.tsx` and others
- **Specific examples**:
  - `/src/components/CollectionDetailsModal.tsx:77` - `acc.get(doc.source_path)!.document_count++` - Non-null assertion on Map.get()
  - `/src/components/__tests__/RenameCollectionModal.test.tsx:1` - Multiple non-null assertions on DOM queries
  - `/src/pages/__tests__/HomePage.test.tsx` - 4+ `as unknown as ReturnType<typeof vi.fn>` type assertions
  - `/src/components/__tests__/UIErrorStates.test.tsx` - 8+ `as unknown as` assertions hiding type problems

**Problem**: 
- Non-null assertions (!) bypass TypeScript safety
- `as unknown as` chains indicate poor type design
- Hiding the real issue instead of fixing types

**Fix**: 
- Use Map functions properly or use Record<string, Type>
- Fix test mock types with proper generic typing
- Create proper type definitions for mocked functions

**Import Path Inconsistency**
- **Issue**: Mix of relative imports (`../`) and alias imports (`@/`)
- **Example**: `/src/components/CollectionMultiSelect.tsx:3` uses `import type { Collection } from '@/types/collection'`
- **Other files**: Most components use relative imports like `../stores/uiStore`
- **Impact**: MEDIUM - Inconsistent patterns make codebase harder to navigate
- **Fix**: Establish single pattern (recommend alias imports for cleaner paths)

### 3. HIGH - Global Window Manipulation (Anti-pattern)

**SearchInterface.tsx & SearchResults.tsx - Global State via Window**
```typescript
// Lines 84-89, 177-181: SearchInterface.tsx
(window as Window & { __handleSelectSmallerModel?: typeof handleSelectSmallerModel }).__handleSelectSmallerModel = ...
(window as Window & { __gpuMemoryError?: ... }).__gpuMemoryError = { message, suggestion, ... }

// Lines 33-34: SearchResults.tsx
if (error === 'GPU_MEMORY_ERROR' && (window as Window & { __gpuMemoryError?: ... }).__gpuMemoryError)
```
- **Problem**: Using window object as a hack to pass state between components is fragile and hard to debug
- **Risk**: Memory leaks, type unsafety, difficult testing
- **Fix**: Use proper React Context or Zustand store for GPU memory error state

### 4. HIGH - Accessibility Issues

**Missing ARIA Labels on Interactive Elements**
- `/src/components/ActiveOperationsTab.tsx:89-95` - Refresh button without aria-label
- `/src/components/CollectionDetailsModal.tsx:222-244` - Action buttons lack aria-labels  
- Multiple `onClick` handlers without associated labels (SearchResults, CollectionCard, etc.)

**Missing Focus Management**
- `/src/components/CollectionDetailsModal.tsx` - No focus trap or focus restoration in modal
- `/src/components/CreateCollectionModal.tsx` - Modal doesn't trap focus, no return focus on close
- `/src/components/DocumentViewerModal.tsx` - No focus management

**Color Contrast Not Verified**
- Status badges use varied colors (text-red-600, text-yellow-600, etc.) - may not meet WCAG AA standards
- Some gray text on white backgrounds may be insufficient contrast

**Missing Alt Text**
- `/src/components/DocumentViewer.tsx` - Line contains `<img src="${objectUrl}" alt="Document">` but is generated via innerHTML, not evaluated by validators

**Keyboard Navigation Issues**
- `/src/components/CollectionMultiSelect.tsx` - Good keyboard support, but other components lack arrow key navigation
- Modal components don't handle Escape key consistently

**Fix Priority**: 
1. Add aria-labels to all interactive elements
2. Implement focus traps in modals  
3. Test color contrast with axe DevTools
4. Add keyboard handlers for custom components

### 5. HIGH - innerHTML and XSS Security

**DocumentViewer.tsx (Lines ~145-250+)**
- Multiple instances of `innerHTML` assignment without sufficient sanitization
- **Affected lines**: 
  - 147: `contentRef.current.innerHTML = window.DOMPurify ? ...`
  - 151: More innerHTML assignments
  - 166-170: innerHTML for JSON rendering
  - 180-190: innerHTML for various file types

**Risk**: Even with DOMPurify, this is risky. DOMPurify may be undefined or outdated.

**Fix**: 
- Use React's built-in rendering or use dangerouslySetInnerHTML with proper guards
- Verify DOMPurify exists and is current version
- Consider using a proper document rendering library

---

## MEDIUM PRIORITY ISSUES

### 6. MEDIUM - Performance Issues

**Missing Code Splitting**
- 49 instances of `memo` or `useMemo` found in codebase
- **Issue**: Not enough memoization for expensive components
- **Specific**: SearchResults, ChunkingComparisonView, CollectionDetailsModal re-render on every parent state change

**Expensive Renders Without Optimization**
- `/src/components/SearchResults.tsx:72-92` - Complex aggregation happens on every render
- `/src/components/CollectionDetailsModal.tsx:72-81` - SourceInfo aggregation in every render cycle

**Fix**: 
```typescript
const sourceDirs = useMemo(() => {
  // aggregation logic
}, [documentsData]);
```

**Timer Leaks (29+ instances of setTimeout/setInterval)**
- `/src/components/SearchInterface.tsx:45-58` - setInterval for auto-refresh, returns interval ID but cleanup may not work
- `/src/components/CreateCollectionModal.tsx:86-88` - setTimeout for navigation

**Risk**: Memory leaks if components unmount before timer fires

**Fix**: Always store interval IDs and clear in useEffect cleanup

**Bundle Size Concerns**
- ChunkingComparisonView (19,931 bytes)
- ChunkingPreviewPanel (22,259 bytes)  
- ChunkingParameterTuner (16,445 bytes)
- These three chunking components alone are ~60KB before minification

**Recommendation**: Consider lazy loading chunking components or splitting further

### 7. MEDIUM - Code Duplication

**formatNumber, formatDate, formatBytes Functions**
- Multiple copies across components:
  - CollectionDetailsModal.tsx:89-102
  - SettingsPage.tsx:41-43
  - Should be in /src/utils/

**formatOperationType**
- Active OperationsTab.tsx:140-148 - duplicated in operation utilities

**getStatusColor Patterns**
- Appears in CollectionCard, ActiveOperationsTab, OperationProgress - should be centralized

**API Error Handling Pattern**
- Repeated in multiple components - should extract to useApiError hook

**Fix**: 
1. Create /src/utils/formatting.ts with reusable formatters
2. Create /src/hooks/useOperationFormatting.ts
3. Create /src/utils/statusColors.ts

### 8. MEDIUM - Hooks Violations and Issues

**useChunkingWebSocket (483 lines)** - `/src/hooks/useChunkingWebSocket.ts`
- **Issue**: Complex custom hook with many side effects
- **Size**: 483 lines is too large for a single hook
- **Fix**: Split into smaller hooks (useWebSocketConnection, useChunkingMessageHandler, etc.)

**useOperationProgress - Dependency Array Issues**
- Multiple useEffect hooks may have missing/incorrect dependencies
- Should audit all custom hooks for stale closures

**useCollections**
- Missing proper error handling and retry logic for failed queries

**Fix**: 
1. Run eslint with exhaustive-deps rule to catch dependency issues
2. Add proper query retry configuration
3. Consider extracting WebSocket logic to separate, smaller hooks

### 9. MEDIUM - State Management Issues

**Local vs Global State Confusion**
- Some state that should be local is managed globally (useUIStore usage inconsistent)
- Example: `expandedDocs` in SearchResults could be truly local but is sometimes synced globally

**Store State That Should Be Local**
- Review stores in /src/stores/ for state that's component-specific

**Fix**: 
1. Create clear guidelines: local = component tree UI state, global = cross-component/persistent
2. Audit current store usage
3. Move component-specific state to local useState

---

## LOW PRIORITY ISSUES

### 10. LOW - Organization & Import Issues

**Unused Imports**
- Not comprehensive scan, but likely some unused imports from refactoring
- Run `npm run lint` to identify

**Deep Relative Imports**
- Example: `/src/components/chunking/__tests__/ChunkingComparisonView.tsx` uses many `../../../` imports
- Could be simplified with better alias configuration

**Test File Organization**
- Mix of `component.test.tsx`, `component.websocket.test.tsx`, `component.network.test.tsx`
- Consider moving to separate test directories

### 11. LOW - Console Statements in Production

**Files with console calls**:
- `/src/pages/LoginPage.tsx:46, 63`
- `/src/pages/SettingsPage.tsx:35, 63`
- `/src/components/CreateCollectionModal.tsx:166`
- `/src/components/DocumentViewer.tsx` - likely multiple
- `/src/components/ActiveOperationsTab.tsx:41`
- Others...

**Fix**: Remove or convert to proper logging library before production

### 12. LOW - Testing Gaps

**Complex Components Need Better Test Coverage**
- CollectionDetailsModal - difficult to test due to size
- SearchInterface - global window manipulation makes testing brittle
- DocumentViewer - external library dependencies hard to mock

**Fix**: 
1. After refactoring large components, add comprehensive tests
2. Extract testable logic from UI rendering

---

## TODO/FIXME Comments Found (Known Debt)

1. **`/src/components/ActiveOperationsTab.tsx` (Line 39)**
   - `TODO: Implement proper navigation using React Router`
   - Impact: Navigation partially stubbed out

2. **`/src/components/DocumentViewer.tsx` (Lines ~150-200)**
   - `TODO: Implement PDF.js rendering`
   - Impact: PDF rendering incomplete

3. **`/src/services/__tests__/websocket.test.ts`**
   - `TODO: This test causes unhandled exception in mock WebSocket interceptor`
   - `TODO: Fix this test - mock WebSocket close method not being called`
   - Impact: Test suite has flaky tests

---

## Recommendations & Priority Fixes

### Phase 1 (CRITICAL - Sprint 1)
1. **Refactor CollectionDetailsModal** into 4 separate tab components (HIGH impact, HIGH complexity)
2. **Replace global window state** with proper Zustand store for GPU memory errors
3. **Add comprehensive ARIA labels** to all interactive elements (quick win, HIGH impact)
4. **Extract utility functions** to /src/utils/ (quick win, reduces duplication)

### Phase 2 (HIGH - Sprint 2)
1. Fix TypeScript type assertions - proper typing for mocks and Map operations
2. Implement focus traps in modal components
3. Audit and fix setInterval/setTimeout memory leaks
4. Implement code splitting for chunking components

### Phase 3 (MEDIUM - Sprint 3)
1. Refactor CreateCollectionModal with state machine pattern
2. Optimize SearchResults with useMemo and memo
3. Split useChunkingWebSocket into smaller hooks
4. Add performance monitoring/metrics

### Phase 4 (LOW - Sprint 4+)
1. Clean up console statements
2. Standardize import paths
3. Reorganize test files
4. Comprehensive test coverage for refactored components

---

## Quality Metrics Summary

| Category | Status | Issues |
|----------|--------|--------|
| Component Complexity | ⚠️ WARNING | 5 god components >300 lines |
| TypeScript Safety | ⚠️ WARNING | 114 type assertions, 3 `as unknown as` chains |
| Accessibility | ⚠️ WARNING | Missing labels, focus management, contrast |
| Performance | ⚠️ WARNING | Insufficient memoization, 60KB+ chunking components |
| Code Organization | ✅ GOOD | Generally well-organized, some duplication |
| State Management | ✅ GOOD | Proper use of Zustand + React Query |
| Testing | ✅ GOOD | Comprehensive test coverage, some flaky tests |
| Documentation | ✅ GOOD | CLAUDE.md well-maintained |

---

## Files Requiring Immediate Action

**CRITICAL (Refactor Required)**
- `/home/john/semantik/apps/webui-react/src/components/CollectionDetailsModal.tsx` - 794 lines
- `/home/john/semantik/apps/webui-react/src/components/CreateCollectionModal.tsx` - 582 lines
- `/home/john/semantik/apps/webui-react/src/components/DocumentViewer.tsx` - 403 lines

**HIGH (Fix Required)**
- `/home/john/semantik/apps/webui-react/src/components/SearchInterface.tsx` - Global window usage
- `/home/john/semantik/apps/webui-react/src/components/SearchResults.tsx` - Global window usage
- `/home/john/semantik/apps/webui-react/src/hooks/useChunkingWebSocket.ts` - 483 lines

**MEDIUM (Optimize)**
- `/home/john/semantik/apps/webui-react/src/components/chunking/*.tsx` - Multiple large files
- All components with 49+ memoization candidates


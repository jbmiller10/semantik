# React Refactor Verification Plan

This document provides a comprehensive testing checklist to verify that all functionality from the vanilla JS implementation has been preserved in the React refactor.

## 1. Authentication Flow Testing

### Login Process
- [ ] Navigate to `/login` when not authenticated
- [ ] Enter valid credentials and verify login works
- [ ] Verify token is stored in localStorage
- [ ] Verify automatic redirect to home page after login
- [ ] Check that username appears in header after login

### Registration Process
- [ ] Switch to registration mode on login page
- [ ] Create new account with unique email
- [ ] Verify automatic login after registration
- [ ] Check tokens are stored properly

### Session Management
- [ ] Refresh page and verify session persists
- [ ] Click logout and verify redirect to login
- [ ] Verify tokens are cleared from localStorage on logout
- [ ] Test accessing protected routes when logged out (should redirect)

### Token Handling
- [ ] Make API calls and verify Bearer token is included in headers
- [ ] Test expired token handling (401 response should logout user)
- [ ] Verify refresh token functionality if implemented

## 2. Job Creation Testing

### Directory Scanning
- [ ] Enter directory path in Create Job form
- [ ] Click "Scan" button and verify WebSocket connection opens
- [ ] Verify real-time scan progress appears
- [ ] Check that scan results show:
  - Total files found
  - Total size in human-readable format
  - List of files (expandable/collapsible)
- [ ] Test scanning non-existent directory (should show error)
- [ ] Test scanning empty directory
- [ ] Test scanning directory with various file types

### Job Creation Form
- [ ] Verify cannot create job without scanning first
- [ ] Enter collection name
- [ ] Verify job creation button becomes enabled after scan
- [ ] Click create job and verify:
  - Job is created successfully
  - Toast notification appears
  - Redirects to Jobs tab
  - Form is reset

### Advanced Options (if implemented)
- [ ] Test custom model selection
- [ ] Modify chunk_size, chunk_overlap
- [ ] Test batch_size parameter
- [ ] Test quantization options
- [ ] Add custom instruction text

## 3. Job Management Testing

### Job List Display
- [ ] Navigate to Jobs tab
- [ ] Verify jobs are grouped by status (Active, Completed, Failed)
- [ ] Check job card displays:
  - Collection name
  - Directory path
  - Status badge with correct color
  - Creation date/time
  - Progress bar (for active jobs)
  - Delete button

### Real-time Updates
- [ ] Create a new job and switch to Jobs tab
- [ ] Verify WebSocket connection established for active job
- [ ] Check real-time progress updates:
  - Progress percentage increases
  - Processed/total documents updates
  - Processing rate (docs/s) shown
  - ETA updates
- [ ] Verify status changes from processing → completed
- [ ] Check toast notification appears on completion

### Job Actions
- [ ] Test delete job functionality:
  - Click delete button
  - Confirm deletion dialog
  - Verify job removed from list
- [ ] Test job metrics modal (if available):
  - Click metrics button on active job
  - Verify detailed metrics display
  - Check resource usage graphs

### Error Handling
- [ ] Test job failure scenarios:
  - Verify failed status displayed
  - Check error message shown
  - Confirm job can be deleted

## 4. Search Functionality Testing

### Basic Search
- [ ] Navigate to Search tab
- [ ] Select a collection from dropdown
- [ ] Enter search query
- [ ] Click Search button
- [ ] Verify results appear grouped by document
- [ ] Check each result shows:
  - File name and path
  - Chunk content preview
  - Chunk number (e.g., "Chunk 3 of 10")
  - Relevance score
  - View button

### Advanced Search Options
- [ ] Click "Show advanced options"
- [ ] Test modifying number of results (top_k)
- [ ] Test score threshold filtering
- [ ] Switch to Hybrid Search mode
- [ ] Test hybrid alpha slider (0-1 range)
- [ ] Verify search type changes behavior

### Search Results Interaction
- [ ] Click on a search result
- [ ] Verify DocumentViewer opens
- [ ] Check that search query is highlighted in viewer
- [ ] Test navigation between highlights
- [ ] Close viewer and return to results

### Error Cases
- [ ] Search without selecting collection
- [ ] Search with empty query
- [ ] Search in non-existent collection
- [ ] Verify appropriate error messages

## 5. Document Viewer Testing

### File Type Support
- [ ] Test viewing PDF files:
  - Verify PDF renders
  - Check page navigation works
  - Test zoom functionality
- [ ] Test viewing DOCX files
- [ ] Test viewing TXT files
- [ ] Test viewing Markdown files
- [ ] Test viewing HTML files
- [ ] Test viewing email (EML) files
- [ ] Test unsupported file types (should show error)

### Viewer Features
- [ ] Verify search term highlighting
- [ ] Test highlight navigation (previous/next)
- [ ] Check highlight counter (e.g., "2 / 5")
- [ ] Test download button functionality
- [ ] Verify close button works
- [ ] Check modal backdrop closes viewer

### PDF-Specific Features
- [ ] Test page navigation controls
- [ ] Verify current page indicator
- [ ] Check total pages displayed
- [ ] Test keyboard navigation (if implemented)

## 6. WebSocket Connection Testing

### Job Progress WebSocket
- [ ] Monitor browser DevTools for WebSocket connections
- [ ] Verify connection established when job is active
- [ ] Check messages received match expected format
- [ ] Test connection closes when job completes
- [ ] Verify auto-reconnection on disconnect

### Scan WebSocket
- [ ] Monitor scan WebSocket during directory scanning
- [ ] Verify real-time file count updates
- [ ] Check scan completion message
- [ ] Test error message handling

## 7. UI/UX Features Testing

### Navigation
- [ ] Test tab switching (Create, Jobs, Search)
- [ ] Verify active tab highlighting
- [ ] Check tab content changes without page reload
- [ ] Test browser back/forward with React Router

### Toast Notifications
- [ ] Verify success toasts (green) appear for:
  - Successful login
  - Job creation
  - Job completion
- [ ] Verify error toasts (red) appear for:
  - Failed operations
  - API errors
- [ ] Check toast auto-dismissal
- [ ] Test manual toast dismissal

### Loading States
- [ ] Check spinning indicators during:
  - Login/registration
  - Directory scanning
  - Job creation
  - Search operations
- [ ] Verify buttons disabled during operations
- [ ] Test loading text changes (e.g., "Scanning...")

### Responsive Design
- [ ] Test on mobile viewport
- [ ] Check tablet viewport
- [ ] Verify desktop layout
- [ ] Test component overflow handling

## 8. Error Handling Testing

### API Errors
- [ ] Test with backend server stopped
- [ ] Verify user-friendly error messages
- [ ] Check errors don't crash the app
- [ ] Test recovery after errors

### Validation Errors
- [ ] Submit forms with missing required fields
- [ ] Test invalid input formats
- [ ] Verify inline validation messages

### Network Issues
- [ ] Test with slow network (throttling)
- [ ] Simulate network disconnection
- [ ] Verify graceful degradation

## 9. Performance Testing

### Initial Load
- [ ] Measure time to interactive
- [ ] Check bundle size is reasonable
- [ ] Verify no console errors on load

### Runtime Performance
- [ ] Test with many jobs in list
- [ ] Search with large result sets
- [ ] Monitor memory usage over time
- [ ] Check for memory leaks with repeated operations

## 10. State Management Verification

### Zustand Stores
- [ ] Use React DevTools to inspect:
  - Auth store (user, token)
  - Jobs store (jobs list, active jobs)
  - Search store (results, params)
  - UI store (toasts, modals)
- [ ] Verify state updates correctly
- [ ] Check state persistence where expected

## 11. API Integration Testing

### Endpoint Coverage
- [ ] Verify all endpoints from original app are called:
  - Auth endpoints
  - Job management endpoints
  - Search endpoints
  - Document endpoints
- [ ] Check request formats match original
- [ ] Verify response handling

## 12. Browser Compatibility

- [ ] Test in Chrome
- [ ] Test in Firefox
- [ ] Test in Safari
- [ ] Test in Edge
- [ ] Verify no browser-specific issues

## Testing Execution Checklist

1. **Setup**
   - [ ] Start all backend services
   - [ ] Ensure test data is available
   - [ ] Open browser DevTools

2. **Systematic Testing**
   - [ ] Complete each section above
   - [ ] Document any issues found
   - [ ] Take screenshots of key features

3. **Regression Testing**
   - [ ] Compare behavior with original app
   - [ ] Verify no features are missing
   - [ ] Check performance is comparable

4. **Edge Cases**
   - [ ] Test with minimal data
   - [ ] Test with large datasets
   - [ ] Test concurrent operations

## Issues Found

Document any issues discovered during testing:

1. Issue: [Description]
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Priority: High/Medium/Low

## Sign-off

- [ ] All core functionality verified working
- [ ] No critical bugs found
- [ ] Performance acceptable
- [ ] User experience matches or exceeds original
- [ ] Ready for production deployment

Testing completed by: ________________
Date: ________________
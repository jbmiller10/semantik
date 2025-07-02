# API Endpoint Comparison: Vanilla JS vs React Implementation

This document provides a comprehensive comparison of API endpoint usage between the vanilla JavaScript and React implementations of the VecPipe web UI.

## Summary

- **Total Unique Endpoints**: 23
- **Endpoints Used by Both**: 15
- **Vanilla JS Only**: 7
- **React Only**: 1
- **Backend Endpoints Not Used by Either**: 0

## Detailed Comparison Table

| Endpoint Path | HTTP Method | Used in Vanilla JS | Used in React | Purpose/Functionality | Differences in Usage |
|--------------|-------------|-------------------|---------------|----------------------|---------------------|
| `/api/auth/login` | POST | ✅ | ✅ | User authentication | Both use similarly |
| `/api/auth/register` | POST | ❌ | ✅ | User registration | React has registration UI, vanilla JS redirects to login |
| `/api/auth/me` | GET | ✅ | ✅ | Get current user info | Both use for auth verification |
| `/api/auth/logout` | POST | ✅ | ❌ | User logout | React uses client-side logout only |
| `/api/auth/refresh` | POST | ❌ | ❌ | Refresh access token | Neither implementation uses this |
| `/api/jobs` | GET | ✅ | ✅ | List all jobs | Both use for job listing |
| `/api/jobs` | POST | ✅ | ✅ | Create new job | Both use for job creation |
| `/api/jobs/new-id` | GET | ✅ | ❌ | Get new job ID | Vanilla JS pre-generates ID for WebSocket |
| `/api/jobs/{job_id}` | GET | ✅ | ❌ | Get specific job details | Vanilla JS uses for metrics modal |
| `/api/jobs/{job_id}` | DELETE | ❌ | ✅ | Delete a job | Only React has delete functionality |
| `/api/jobs/{job_id}/cancel` | POST | ✅ | ❌ | Cancel running job | Only vanilla JS has cancel button |
| `/api/jobs/collections-status` | GET | ✅ | ❌ | Check collection status | Vanilla JS caches this for UI indicators |
| `/api/jobs/{job_id}/collection-exists` | GET | ❌ | ❌ | Check if collection exists | Backend endpoint not used by either |
| `/api/models` | GET | ✅ | ✅ | List available models | Both use for model selection |
| `/api/search` | POST | ✅ | ✅ | Vector search | Both implement search functionality |
| `/api/hybrid_search` | POST | ✅ | ❌ | Hybrid search | Only vanilla JS has hybrid search |
| `/api/search/collections` | GET | ❌ | ✅ | List collections for search | React uses this, vanilla gets from jobs |
| `/api/documents/{job_id}/{doc_id}` | GET | ✅ | ✅ | Get document content | Both use for document viewer |
| `/api/documents/{job_id}/{doc_id}/info` | GET | ❌ | ✅ | Get document metadata | React gets info first, vanilla doesn't |
| `/api/documents/temp-images/{session_id}/{filename}` | GET | ❌ | ❌ | Serve temporary images | Backend endpoint not used |
| `/api/scan-directory` | POST | ❌ | ❌ | Scan directory (non-WS) | Both use WebSocket instead |
| `/api/metrics` | GET | ✅ | ❌ | Get Prometheus metrics | Vanilla JS shows detailed metrics |
| `/api/settings/reset-database` | POST | ❌ | ❌ | Reset database | Neither has this in UI |
| `/api/settings/stats` | GET | ❌ | ❌ | Get system stats | Neither uses this endpoint |

## WebSocket Endpoints

| WebSocket Path | Used in Vanilla JS | Used in React | Purpose |
|----------------|-------------------|---------------|---------|
| `/ws/{job_id}` | ✅ | ✅ | Job progress updates |
| `/ws/scan/{scan_id}` | ✅ | ✅ | Directory scan progress |

## Key Differences

### 1. **Authentication Flow**
- **Vanilla JS**: Login only, stores tokens in localStorage, has logout endpoint
- **React**: Full registration flow, uses Zustand store for auth state, client-side logout

### 2. **Job Management**
- **Vanilla JS**: 
  - Pre-generates job IDs for WebSocket connection
  - Has cancel job functionality
  - Shows collection status with caching
  - Displays detailed metrics modal
- **React**: 
  - Simpler job creation flow
  - Has delete job functionality
  - No cancel or metrics features yet

### 3. **Search Implementation**
- **Vanilla JS**: 
  - Supports both vector and hybrid search
  - Gets collections from jobs list
  - More search options (keyword modes)
- **React**: 
  - Vector search only currently
  - Dedicated collections endpoint
  - Cleaner search parameter management

### 4. **Document Viewer**
- **Vanilla JS**: Directly fetches document
- **React**: Fetches document info first, then content

### 5. **Missing Features in React**
- Hybrid search (`/api/hybrid_search`)
- Job cancellation (`/api/jobs/{job_id}/cancel`)
- Metrics display (`/api/metrics`, `/api/jobs/{job_id}`)
- Collection status indicators (`/api/jobs/collections-status`)
- Logout endpoint (`/api/auth/logout`)

### 6. **Missing Features in Vanilla JS**
- User registration (`/api/auth/register`)
- Job deletion (`/api/jobs/{job_id}`)
- Dedicated collections list (`/api/search/collections`)
- Document info endpoint (`/api/documents/{job_id}/{doc_id}/info`)

## Recommendations

1. **Implement in React**:
   - Hybrid search functionality
   - Job cancellation feature
   - Metrics/monitoring modal
   - Collection status caching
   - Proper logout with API call

2. **Consider Adding**:
   - Settings page using `/api/settings/*` endpoints
   - System stats display
   - Token refresh mechanism

3. **Optimize**:
   - Implement caching strategy like vanilla JS for collection status
   - Pre-generate job IDs in React for better WebSocket handling
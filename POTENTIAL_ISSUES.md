# Potential Issues, Bugs, and Architectural Concerns

This document contains a comprehensive list of potential issues, bugs, and architectural concerns discovered during the codebase audit of Project Semantik.

## 1. Documentation Discrepancies

### API Documentation (API_REFERENCE.md)
- **Issue**: The documented search response format doesn't match the actual implementation
  - **Documented**: Results have "id" and "metadata" structure
  - **Actual**: Results return "path", "chunk_id", "doc_id", "content" directly
- **Issue**: Cross-encoder reranking parameters are documented but implementation details are missing
- **Issue**: Model status endpoint documentation doesn't match actual response structure

### Database Schema (DATABASE_ARCH.md)
- **Issue**: Missing columns in documentation:
  - jobs table: `parent_job_id`, `mode`, `user_id` (marked as "future" but already implemented)
  - files table: `content_hash` column not documented
- **Issue**: Migration strategy documentation is incomplete

### Search System Documentation
- **Issue**: ModelManager component with lazy loading not documented
- **Issue**: Cross-encoder reranking with automatic model selection not properly documented
- **Issue**: Reranker model mapping and fallback logic not documented

## 2. Architectural Concerns

### Package Separation Violation
- **Issue**: `embedding_service.py` is shared between vecpipe and webui packages, violating the documented architectural separation
- **Risk**: This creates tight coupling between supposedly independent packages
- **Location**: Both packages import from webui.embedding_service

### Import Dependencies
- **Issue**: vecpipe package imports from webui package (embedding_service), breaking the independence principle
- **Issue**: This makes vecpipe not truly "headless" as documented

### Authentication Implementation
- **Issue**: JWT authentication is fully implemented but documented as partial or future work
- **Issue**: User isolation for collections is not implemented despite database support

## 3. Code Quality Issues

### Error Handling
- **Issue**: Some API endpoints lack proper error handling for edge cases
- **Issue**: Inconsistent error message formats between different endpoints

### Type Hints
- **Issue**: Some functions lack proper type hints despite the mandate for full type annotation
- **Issue**: Optional types not consistently used (Union[str, None] vs Optional[str])

### Resource Management
- **Issue**: GPU memory management could fail silently in some edge cases
- **Issue**: No maximum limit on concurrent model loads could lead to OOM

## 4. Security Concerns

### Path Validation
- **Issue**: Directory traversal prevention is mentioned but implementation varies across endpoints
- **Issue**: File path validation is inconsistent between scan and processing operations

### Authentication Gaps
- **Issue**: Some WebSocket endpoints may not properly validate authentication
- **Issue**: Rate limiting is implemented but not consistently applied to all endpoints

### Configuration Security
- **Issue**: JWT secret key validation is minimal (just checks if set)
- **Issue**: No rotation mechanism for refresh tokens

## 5. Performance Issues

### Database Queries
- **Issue**: Some queries could benefit from additional indexes
- **Issue**: N+1 query patterns in job listing with file counts

### Memory Management
- **Issue**: Large file processing could cause memory spikes
- **Issue**: No streaming support for very large documents

### Batch Processing
- **Issue**: Fixed batch sizes don't adapt to available memory
- **Issue**: No backpressure mechanism for overwhelmed systems

## 6. Missing Features vs Documentation

### Documented but Not Implemented
- **Issue**: User-specific collections mentioned but not implemented
- **Issue**: Incremental updates for changed files not fully implemented
- **Issue**: Audit logging mentioned but not present

### Implemented but Not Documented
- **Issue**: Rate limiting via SlowAPI
- **Issue**: WebSocket real-time updates
- **Issue**: Collection metadata management
- **Issue**: Model lazy loading and automatic unloading

## 7. Testing Gaps

### Missing Tests
- **Issue**: No unit tests for ModelManager
- **Issue**: No integration tests for the full pipeline
- **Issue**: WebSocket functionality lacks tests
- **Issue**: No tests for core vecpipe modules:
  - `extract_chunks.py` (document parsing/chunking)
  - `search_api.py` (core search API)
  - `hybrid_search.py` (hybrid search implementation)
- **Issue**: No tests for critical webui components:
  - `embedding_service.py` (embedding generation)
  - Job management endpoints
  - Collection management logic
- **Issue**: No automated React component tests (no test runner configured)

### Test Quality Issues
- **Issue**: Some test files appear to be debug scripts rather than proper tests (e.g., `test_search.py`)
- **Issue**: Inconsistent test patterns - mix of unit and integration tests without clear separation
- **Issue**: Tests depend on environment variables without proper test environment setup

### Test Documentation
- **Issue**: Test README files exist but are not comprehensive
- **Issue**: No documentation on how to run tests in different modes (GPU/CPU/Mock)
- **Issue**: No testing guidelines or standards documented
- **Issue**: No test data fixtures or factories documented

### Frontend Testing Gaps
- **Issue**: Only manual HTML-based test files, no automated test suite
- **Issue**: No test runner configured in package.json (Jest, Vitest, etc.)
- **Issue**: No component-level testing
- **Issue**: No integration tests for frontend-backend interactions

## 8. Configuration Issues

### Environment Variables
- **Issue**: Some environment variables have defaults that might not be suitable for production
- **Issue**: No validation for environment variable values (e.g., port ranges)

### Docker Configuration
- **Issue**: Docker setup is referenced in code but not in main architecture documentation
- **Issue**: Volume mounts and permissions could cause issues

## 9. UI/UX Concerns

### Frontend State Management
- **Issue**: Some state updates could cause unnecessary re-renders
- **Issue**: Error states not consistently handled across components

### API Response Consistency
- **Issue**: Different endpoints return errors in different formats
- **Issue**: Pagination is implemented partially but not consistently

### Frontend Audit Findings (Puppeteer)
- **Issue**: Search functionality may not provide feedback when no results are found
  - **Observation**: Search for "database architecture" returned no visible results or error messages
  - **Impact**: Users may not know if search is working or if no results were found
- **Issue**: Collections page shows job count but this feature isn't documented
- **Issue**: Settings page shows Parquet files statistics but these are always 0 (feature may be incomplete)
- **Issue**: No visual loading indicators during search operations
- **Issue**: Cross-encoder reranking option is present in UI but its impact is not clearly explained to users

## 10. Scalability Concerns

### Single Instance Limitations
- **Issue**: SQLite limits concurrent writes
- **Issue**: No horizontal scaling strategy documented
- **Issue**: Job processing is sequential within a job

### Resource Bottlenecks
- **Issue**: Model loading/unloading could become a bottleneck
- **Issue**: No connection pooling for Qdrant in some cases

## 11. Maintenance Issues

### Code Duplication
- **Issue**: Similar retry logic implemented in multiple places
- **Issue**: Database connection code duplicated

### Configuration Sprawl
- **Issue**: Configuration is spread across multiple files
- **Issue**: Some hardcoded values should be configurable

## 12. Deployment Concerns

### Service Dependencies
- **Issue**: No health check endpoints for dependent services
- **Issue**: Service startup order not enforced

### Monitoring Gaps
- **Issue**: Metrics are collected but no alerting mentioned
- **Issue**: No log aggregation strategy

## Recommendations

1. **Immediate Actions**:
   - Update documentation to match implementation
   - Fix architectural violations (embedding_service location)
   - Add missing error handling
   - Implement proper path validation

2. **Short-term Improvements**:
   - Add comprehensive tests
   - Implement user isolation
   - Improve error consistency
   - Document all features

3. **Long-term Enhancements**:
   - Design horizontal scaling strategy
   - Implement audit logging
   - Add connection pooling
   - Create deployment guides

## Next Steps

1. Prioritize issues by severity and impact
2. Create GitHub issues for tracking
3. Update documentation incrementally
4. Add tests for critical paths
5. Refactor shared code to maintain architectural boundaries
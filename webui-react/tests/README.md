# VecPipe WebUI Test Suite

This directory contains comprehensive test scenarios to verify feature parity between the vanilla JS and React implementations.

## Test Files

### 1. **websocket-tests.html**
Interactive browser-based tests for WebSocket functionality:
- Job progress WebSocket connections
- Directory scan WebSocket connections
- Connection reliability tests
- Message handling tests

**Usage:**
```bash
# Open in browser
open tests/websocket-tests.html
# Or serve locally
python -m http.server 8080
# Then navigate to http://localhost:8080/tests/websocket-tests.html
```

### 2. **configuration-tests.html**
Tests for configuration and hardcoded values:
- Dynamic URL construction
- Port detection
- API endpoint configuration
- Environment detection
- Security checks

**Usage:**
```bash
# Open in browser
open tests/configuration-tests.html
```

### 3. **document-viewer-tests.html**
Tests for document viewing functionality:
- Library loading (PDF.js, Mammoth.js, etc.)
- File type support verification
- Document preview functionality
- Error handling for unsupported formats

**Usage:**
```bash
# Open in browser
open tests/document-viewer-tests.html
```

### 4. **search-tests.html**
Comprehensive search functionality tests:
- Vector search
- Hybrid search detection
- Search filters
- Performance testing
- Search parameter validation

**Usage:**
```bash
# Open in browser
open tests/search-tests.html
```

### 5. **api-test-commands.sh**
Shell script with curl commands for API testing:
- All API endpoints
- WebSocket connections (requires wscat)
- Performance tests
- Error handling

**Usage:**
```bash
# Basic usage
./tests/api-test-commands.sh

# With custom API URL
API_BASE=http://localhost:8000 ./tests/api-test-commands.sh

# With authentication
AUTH_TOKEN=your-token-here ./tests/api-test-commands.sh
```

### 6. **api_test_suite.py**
Comprehensive Python test suite using asyncio:
- All API endpoints
- WebSocket testing
- Performance benchmarks
- Feature availability detection

**Requirements:**
```bash
pip install aiohttp websockets
```

**Usage:**
```bash
# Run all tests
python tests/api_test_suite.py

# Custom URL
python tests/api_test_suite.py --url http://localhost:8000

# With authentication
python tests/api_test_suite.py --token your-token-here

# Run specific test
python tests/api_test_suite.py --test vector_search
```

## Critical Features to Verify

### 1. WebSocket Functionality
- **Job Progress Updates**: Real-time job status updates via WebSocket
- **Directory Scanning**: Real-time file discovery during scans
- **Connection Stability**: Automatic reconnection handling
- **Message Format**: Consistent message structure

### 2. Search Capabilities
- **Vector Search**: Semantic search using embeddings
- **Hybrid Search**: Combined vector + keyword search (if available)
- **Search Filters**: File type, date range, size filters
- **Performance**: Sub-second response times

### 3. Document Viewing
- **Supported Formats**: PDF, DOC/DOCX, TXT, MD, CSV, JSON, XML, Images
- **Library Loading**: CDN-based loading of rendering libraries
- **Error Handling**: Graceful fallback for unsupported formats

### 4. Configuration
- **Dynamic URLs**: No hardcoded localhost references
- **Port Flexibility**: Works on any port
- **Environment Adaptation**: Development vs production detection

## Running All Tests

### Quick Test
```bash
# 1. Start the API server
cd /path/to/webui
python main.py

# 2. In another terminal, run shell tests
cd webui-react
./tests/api-test-commands.sh

# 3. Run Python test suite
python tests/api_test_suite.py

# 4. Open HTML tests in browser
python -m http.server 8080
# Navigate to http://localhost:8080/tests/
```

### Comprehensive Test
```bash
# 1. Run Python test suite with full output
python tests/api_test_suite.py 2>&1 | tee test-results.log

# 2. Check specific features
python tests/api_test_suite.py --test hybrid_search
python tests/api_test_suite.py --test job_websocket

# 3. Manual browser testing
# Open each HTML file and run all tests
```

## Expected Results

### ✅ Must Have (Feature Parity)
- Vector search working
- Job creation and management
- Directory scanning
- Basic document preview
- Settings management

### 🟡 Should Have
- WebSocket connections for real-time updates
- Search filters
- Performance under 1 second for searches
- Proper error handling

### 🔵 Nice to Have
- Hybrid search capability
- Advanced search filters
- Batch operations
- Metrics endpoints

## Troubleshooting

### WebSocket Connection Failed
- Check if the API server supports WebSocket endpoints
- Verify WebSocket URLs are correctly constructed
- Check for proxy/firewall issues

### Search Not Working
- Verify embeddings are properly generated
- Check if vector database is initialized
- Test with simple queries first

### Document Preview Issues
- Verify CDN access for libraries
- Check CORS settings
- Test with different file formats

## Notes

- These tests are designed to identify feature gaps, not to break the system
- Some features may not be implemented yet, which is expected
- Focus on critical features needed for feature parity
- Document any discovered issues for the development team
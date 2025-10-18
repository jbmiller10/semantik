# Configuration & Environment Management Audit Report

## Executive Summary
The semantik project has a **reasonably well-structured configuration system** using Pydantic BaseSettings with environment variables, but contains several issues that could impact deployment, scaling, and production safety.

**Key Risk Areas:**
- Hardcoded localhost addresses in development code that could leak into production
- Missing configuration validation for required variables in some services
- Type conversion inconsistencies (raw os.getenv() with int() conversions)
- Development-only features (DISABLE_AUTH) not properly gated
- Vite proxy configuration hardcoded to localhost

---

## 1. HARDCODED VALUES - HIGH PRIORITY

### 1.1 Backend Hardcoded localhost References

#### File: `/home/john/semantik/packages/shared/config/base.py` (Line 22-23)
```python
QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333
```
**Issue:** Default Qdrant host hardcoded to localhost - breaks Docker networking
**Impact:** Production deployments outside localhost will fail silently
**Fix:** Keep as defaults (acceptable for local dev), but ensure docker-compose.yml provides QDRANT_HOST=qdrant

#### File: `/home/john/semantik/packages/shared/config/webui.py` (Lines 26-30)
```python
WEBUI_URL: str = "http://localhost:8080"
WEBUI_INTERNAL_HOST: str = "localhost"
SEARCH_API_URL: str = "http://localhost:8000"
REDIS_URL: str = "redis://localhost:6379/0"
CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
```
**Issue:** Multiple hardcoded localhost URLs mixed with comments
**Impact:** 
- SEARCH_API_URL breaks in Docker (should be http://vecpipe:8000)
- REDIS_URL breaks in Docker (should be redis://redis:6379/0)
- CORS_ORIGINS needs update for production
**Fix:** Ensure environment variables override these defaults in docker-compose.yml ✓ (docker-compose already does this)

#### File: `/home/john/semantik/packages/shared/config/vecpipe.py` (Line 25)
```python
SEARCH_API_URL: str = "http://localhost:8000"
```
**Issue:** Duplicate SEARCH_API_URL default, same as webui
**Impact:** Redundant configuration
**Fix:** Reference from shared base or consolidate

#### File: `/home/john/semantik/packages/webui/websocket/scalable_manager.py` (Lines 53, 360)
```python
def __init__(self, redis_url: str = "redis://localhost:6379/2", ...):
scalable_ws_manager = ScalableWebSocketManager(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/2"))
```
**Issue:** Hardcoded localhost in default parameter AND fallback
**Impact:** If REDIS_URL env var missing, WebSocket manager fails to connect
**Fix:** Use explicit Redis URL from settings, not os.getenv fallback

#### File: `/home/john/semantik/packages/webui/websocket_manager.py` (Line 67)
```python
redis_url = getattr(_settings, "REDIS_URL", "redis://localhost:6379/0")
```
**Issue:** Another hardcoded localhost fallback
**Impact:** Inconsistent Redis DB (this uses DB 0, scalable_manager uses DB 2)
**Fix:** Standardize Redis URL handling

#### File: `/home/john/semantik/packages/webui/celery_app.py` (Lines 115-116)
```python
broker_url = os.getenv(broker_url_env, "redis://localhost:6379/0")
backend_url = os.getenv(backend_url_env, "redis://localhost:6379/0")
```
**Issue:** Direct os.getenv with localhost fallback, not using shared settings
**Impact:** Uses different Redis configuration than other services
**Fix:** Use settings.REDIS_URL instead of os.getenv

#### File: `/home/john/semantik/packages/webui/api/health.py` (Lines 34, 122)
```python
search_api_url = os.getenv("SEARCH_API_URL", "http://vecpipe:8000")
search_api_url = os.getenv("SEARCH_API_URL", "http://vecpipe:8000")
```
**Issue:** health.py correctly uses "http://vecpipe:8000" but duplicates the code
**Impact:** Inconsistent with SEARCH_API_URL in settings (http://localhost:8000)
**Fix:** Use settings.SEARCH_API_URL instead of os.getenv

#### File: `/home/john/semantik/packages/webui/api/metrics.py` (Line 69)
```python
response = await client.get(f"http://localhost:{METRICS_PORT}/metrics")
```
**Issue:** Hardcoded localhost for metrics endpoint
**Impact:** Cannot fetch metrics from remote instances
**Fix:** Make metrics host configurable

### 1.2 Frontend Hardcoded localhost References

#### File: `/home/john/semantik/apps/webui-react/vite.config.ts` (Lines 21-28)
```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8080',
      changeOrigin: true,
    },
    '/ws': {
      target: 'ws://localhost:8080',
      ws: true,
      changeOrigin: true,
    },
  },
},
```
**Issue:** Development proxy hardcoded to localhost:8080
**Impact:** Only works for local development, not configurable for different host:port
**Fix:** Use environment variables (VITE_API_URL or import.meta.env)

#### File: `/home/john/semantik/apps/webui-react/playwright.config.ts`
```typescript
baseURL: 'http://localhost:5173',
url: 'http://localhost:5173',
```
**Issue:** E2E test configuration hardcoded
**Impact:** Tests only work against localhost:5173
**Fix:** Use environment variables for test URLs

### 1.3 Summary: Hardcoded Values

| Location | Value | Environment | Severity |
|----------|-------|-------------|----------|
| Multiple config defaults | localhost | DEV fallback | MEDIUM |
| health.py | vecpipe:8000 | CORRECT | OK |
| metrics.py | localhost | PROD BREAK | HIGH |
| vite.config.ts | localhost:8080 | DEV ONLY | MEDIUM |
| celery_app.py | Direct os.getenv | INCONSISTENT | MEDIUM |
| scalable_manager.py | DB 2 fallback | CONFLICT | MEDIUM |

---

## 2. ENVIRONMENT VARIABLES VALIDATION - HIGH PRIORITY

### 2.1 Missing Validation for Required Variables

#### Problem: No startup validation for POSTGRES_PASSWORD in production
**File:** docker-compose.yml (Line 48)
```yaml
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-CHANGE_THIS_TO_A_STRONG_PASSWORD}
```
**Issue:** Default placeholder password in docker-compose.yml
**Impact:** If env var not set, database uses insecure default password
**Fix:** Add validation at startup to reject insecure defaults in production

#### Problem: JWT_SECRET_KEY validation only in WebuiConfig, not checked during app startup
**File:** `/home/john/semantik/packages/shared/config/webui.py` (Lines 52-86)
**Status:** ✓ GOOD - Production check exists, but only during config initialization

**Current Code:**
```python
if self.ENVIRONMENT == "production":
    if self.JWT_SECRET_KEY == "default-secret-key" or not self.JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY must be explicitly set via environment variable in production.")
```

**Issue:** Only checks for literal "default-secret-key", not empty/whitespace values
**Fix:** Already handled by checking `not self.JWT_SECRET_KEY`

#### Problem: INTERNAL_API_KEY validation only raises error in production
**File:** `/home/john/semantik/packages/shared/config/internal_api_key.py` (Lines 74-90)
**Status:** ✓ GOOD - Proper error handling

### 2.2 Type Conversion Issues

#### File: `/home/john/semantik/packages/webui/config/rate_limits.py` (Lines 16-30)
```python
PREVIEW_LIMIT = int(os.getenv("CHUNKING_PREVIEW_RATE_LIMIT", "10"))
COMPARE_LIMIT = int(os.getenv("CHUNKING_COMPARE_RATE_LIMIT", "5"))
PROCESS_LIMIT = int(os.getenv("CHUNKING_PROCESS_RATE_LIMIT", "20"))
READ_LIMIT = int(os.getenv("CHUNKING_READ_RATE_LIMIT", "60"))
ANALYTICS_LIMIT = int(os.getenv("CHUNKING_ANALYTICS_RATE_LIMIT", "30"))
CIRCUIT_BREAKER_FAILURES = int(os.getenv("CIRCUIT_BREAKER_FAILURES", "5"))
CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
```

**Issue:** Multiple problems:
1. `int(os.getenv(...))` will raise ValueError if env var is non-numeric string
2. No error handling for invalid input
3. Using raw os.getenv instead of Pydantic BaseSettings
4. No documentation for these variables

**Impact:** Invalid rate limit env vars crash the application
**Fix:** Use Pydantic Field() with validators instead:
```python
PREVIEW_LIMIT: int = Field(default=10, description="...")
```

#### Problem: Similar issue in celery_app.py (Line 104)
```python
testing_mode = _is_truthy(os.getenv("TESTING"))
```
**Status:** ✓ GOOD - Uses custom _is_truthy function with proper handling

### 2.3 Missing Environment Variable Documentation

**Missing Documentation:**
- `ENVIRONMENT` variable options (development vs production)
- `CHUNKING_*_RATE_LIMIT` variables not in .env.example
- `CIRCUIT_BREAKER_*` variables not in .env.example
- `DISABLE_AUTH` variable (development-only flag)
- `USE_CHUNKING_ORCHESTRATOR` feature flag
- `DISABLE_RATE_LIMITING` variable
- `RATE_LIMIT_BYPASS_TOKEN` variable
- `MODEL_UNLOAD_AFTER_SECONDS`
- `ENABLE_ADAPTIVE_BATCH_SIZE`, `MIN_BATCH_SIZE`, `MAX_BATCH_SIZE`, `BATCH_SIZE_SAFETY_MARGIN`

**Fix:** Add all variables to .env.example with descriptions

---

## 3. CONFIGURATION PATTERN INCONSISTENCIES - MEDIUM PRIORITY

### 3.1 Multiple Configuration Loading Approaches

**Problem:** Code uses three different patterns for configuration:

1. **Pydantic BaseSettings** (recommended, used by config/)
   ```python
   from shared.config import settings
   settings.QDRANT_HOST
   ```

2. **Direct os.getenv()** (used in health.py, rate_limiter.py)
   ```python
   os.getenv("SEARCH_API_URL", "http://vecpipe:8000")
   ```

3. **Class attributes with os.getenv()** (used in rate_limits.py)
   ```python
   PREVIEW_LIMIT = int(os.getenv("CHUNKING_PREVIEW_RATE_LIMIT", "10"))
   ```

**Impact:** 
- Inconsistent configuration sources
- Hard to track all configuration
- Duplication of defaults
- Different validation levels

**Fix:** Consolidate to use Pydantic BaseSettings only

### 3.2 Redis URL Configuration Drift

**Found three different Redis URLs:**

| File | Variable | URL | DB |
|------|----------|-----|-----|
| webui.py | REDIS_URL | redis://localhost:6379/0 | 0 |
| scalable_manager.py | redis_url param | redis://localhost:6379/2 | 2 |
| rate_limits.py | REDIS_URL | redis://redis:6379/1 | 1 |
| websocket_manager.py | from settings | redis://localhost:6379/0 | 0 |

**Issue:** Different Redis database numbers for same functionality
**Impact:** WebSocket state (DB 2) doesn't match rate limit state (DB 1)
**Fix:** Standardize all to use single REDIS_URL from settings

### 3.3 Service URL Configuration Split

**Problem:** SEARCH_API_URL defined in TWO places:
- `webui.py`: defaults to "http://localhost:8000"
- `vecpipe.py`: defaults to "http://localhost:8000"

**Impact:** Redundancy, potential for drift

---

## 4. DEVELOPMENT CODE IN PRODUCTION - MEDIUM PRIORITY

### 4.1 DISABLE_AUTH Flag
**File:** `/home/john/semantik/packages/shared/config/webui.py` (Line 19)
```python
DISABLE_AUTH: bool = False  # Set to True for development only
```
**File:** `/home/john/semantik/packages/webui/auth.py`
```python
if settings.DISABLE_AUTH:
    # Skip JWT validation
```
**Issue:** Development bypass in production-enabled code
**Impact:** If accidentally enabled in production, completely bypasses auth
**Fix:** Move to development config, assert False in production config

### 4.2 Feature Flags Without Proper Gating
**File:** `/home/john/semantik/packages/shared/config/webui.py` (Line 42)
```python
USE_CHUNKING_ORCHESTRATOR: bool = False
```
**Issue:** Feature flag defaults to False, but code assumes it might be True
**Impact:** Incomplete feature rollout mechanism
**Fix:** Add deprecation timeline, clear feature roadmap documentation

### 4.3 Debug Logging Configuration
**File:** `/home/john/semantik/packages/webui/celery_app.py` (Line 113)
```python
logger.debug("Initializing Celery in testing mode with in-memory transports.")
```
**Status:** ✓ GOOD - Uses debug level, not info

---

## 5. DATABASE CONNECTION CONFIGURATION - MEDIUM PRIORITY

### 5.1 PostgreSQL Configuration Details

**File:** `/home/john/semantik/packages/shared/config/postgres.py`

**Good:**
- ✓ Pool size and timeout settings configurable
- ✓ Retry logic with exponential backoff
- ✓ Connection validation (pool_pre_ping = True)
- ✓ Proper async driver selection

**Issues:**
1. **Default empty password:**
   ```python
   POSTGRES_PASSWORD: str = Field(default="", description="PostgreSQL password")
   ```
   **Impact:** Empty password accepted by default (insecure for production)
   **Fix:** Require password in production

2. **Hardcoded timeouts in connect_args:**
   ```python
   "lock_timeout": "5000",      # 5 seconds
   "idle_in_transaction_session_timeout": "60000",  # 60 seconds
   ```
   **Impact:** Not configurable per deployment
   **Fix:** Add to configuration

3. **Magic number for chunks partition count:**
   ```python
   CHUNK_PARTITION_COUNT: int = Field(default=100, ...)
   ```
   **Impact:** Changing this requires database recreation
   **Fix:** Document immutability and provide migration path

---

## 6. CONFIGURATION VALIDATION AT STARTUP

### 6.1 Validation Checklist

| Check | Status | Location |
|-------|--------|----------|
| JWT_SECRET_KEY required in production | ✓ YES | webui.py:52-58 |
| INTERNAL_API_KEY required in production | ✓ YES | internal_api_key.py:74-79 |
| DATABASE_URL validation | ✓ YES | postgres.py:73-96 |
| POSTGRES_PASSWORD validation | ✗ NO | postgres.py:27 |
| Redis connectivity validation | ✗ NO | No startup check |
| Qdrant connectivity validation | ✓ PARTIAL | health.py:97-113 |
| Required env vars documented | ✗ NO | .env.example incomplete |

### 6.2 Missing Startup Validations

**Problem:** Services don't validate all required configuration at startup
**Impact:** Configuration errors discovered at runtime when first used

**Recommended Startup Checks:**
```python
# In webui/main.py startup
async def validate_configuration():
    # Check Redis connectivity
    # Check database connectivity
    # Validate all required env vars
    # Check service URLs are reachable (optional)
```

---

## 7. FRONTEND CONFIGURATION - MEDIUM PRIORITY

### 7.1 Vite Configuration
**File:** `/home/john/semantik/apps/webui-react/vite.config.ts`

**Issues:**
1. **Hardcoded proxy target:**
   ```typescript
   target: 'http://localhost:8080'
   ```
   **Impact:** Development proxy only works with localhost
   **Fix:** Use environment variable:
   ```typescript
   target: process.env.VITE_API_HOST || 'http://localhost:8080'
   ```

2. **No environment variable support**
   **Impact:** Cannot customize API URLs without rebuilding

### 7.2 Frontend API Client
**File:** `/home/john/semantik/apps/webui-react/src/services/api/v2/client.ts`

**Status:** ✓ GOOD - Uses baseURL: '', which means relative URLs
**Impact:** Works with any API server

---

## 8. DOCKER CONFIGURATION - LOW PRIORITY (Well-Handled)

### 8.1 docker-compose.yml Analysis

**Good Practices:**
- ✓ Service names correct for inter-container communication
- ✓ Environment variables properly passed
- ✓ Health checks configured
- ✓ Resource limits defined
- ✓ Security options (no-new-privileges, cap_drop)

**Issues:**
1. **Default placeholders:**
   ```yaml
   - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-CHANGE_THIS_TO_A_STRONG_PASSWORD}
   ```
   **Fix:** Better to fail explicitly if not set

2. **Qdrant port hardcoded:**
   ```yaml
   - QDRANT_PORT=6333
   ```
   **Status:** OK (standard Qdrant port, not expected to change)

---

## 9. CONFIGURATION SUMMARY BY SEVERITY

### CRITICAL
- [ ] None identified

### HIGH
1. **Hardcoded localhost in metrics.py** - Breaks metrics collection
2. **Type conversion without error handling in rate_limits.py** - Can crash app
3. **Missing POSTGRES_PASSWORD validation** - Can cause insecure deployments
4. **Vite proxy hardcoded to localhost** - Breaks development proxy

### MEDIUM
1. **Celery using direct os.getenv() instead of settings** - Inconsistent config
2. **Multiple Redis URLs with different DB numbers** - State sync issues
3. **WebSocket manager falling back to localhost:6379/2** - Docker will fail
4. **DISABLE_AUTH flag production-accessible** - Security risk
5. **Rate limit env vars not in .env.example** - Undocumented
6. **Health endpoint using os.getenv() instead of settings** - Inconsistent

### LOW
1. **Duplicate SEARCH_API_URL in vecpipe.py** - Code duplication
2. **Hardcoded lock_timeout in PostgreSQL** - Not production-tunable
3. **Feature flags without deprecation path** - Code maintenance

---

## 10. RECOMMENDED FIXES (Priority Order)

### Phase 1: Critical Path (Fix Before Production)
1. **Fix metrics.py hardcoded localhost:**
   ```python
   METRICS_HOST = settings.WEBUI_INTERNAL_HOST or "0.0.0.0"
   response = await client.get(f"http://{METRICS_HOST}:{METRICS_PORT}/metrics")
   ```

2. **Fix type conversions in rate_limits.py:**
   ```python
   # Migrate to Pydantic BaseSettings approach
   class RateLimitConfig(BaseSettings):
       PREVIEW_LIMIT: int = Field(default=10)
       # ... etc
   ```

3. **Fix celery_app.py Redis configuration:**
   ```python
   from shared.config import settings
   broker_url = settings.REDIS_URL
   backend_url = settings.REDIS_URL
   ```

### Phase 2: Configuration Consistency (Before 1.0 Release)
1. Consolidate all configuration to use Pydantic BaseSettings
2. Remove all direct os.getenv() calls (except TESTING flag)
3. Standardize Redis URL usage (one URL, clear DB allocation)
4. Document all environment variables in .env.example

### Phase 3: Production Hardening (Ongoing)
1. Add startup validation for all required configuration
2. Add Redis connectivity check at startup
3. Implement feature flag deprecation timeline
4. Move DISABLE_AUTH to separate development config
5. Add configuration validation tests

---

## 11. CONFIGURATION FILES CHECKLIST

| File | Status | Issues | Severity |
|------|--------|--------|----------|
| .env | ✓ Present | Missing rate limit vars | LOW |
| .env.example | MISSING | N/A | N/A |
| docker-compose.yml | ✓ Good | Default placeholders | LOW |
| Dockerfile | ✓ Good | None | - |
| vite.config.ts | ✓ Works | Hardcoded localhost | MEDIUM |
| playwright.config.ts | ✓ Works | Hardcoded localhost | LOW |
| packages/shared/config/*.py | ✓ Good | See detailed section | MEDIUM |
| packages/webui/config/*.py | ✗ Needs work | Type conversions | HIGH |

---

## 12. ENVIRONMENT VARIABLE REFERENCE

**All Found Variables:**

```
ENVIRONMENT                          # development|production (default: development)
JWT_SECRET_KEY                       # REQUIRED in production
ACCESS_TOKEN_EXPIRE_MINUTES          # Token expiry (default: 1440)
DISABLE_AUTH                         # Development only (default: false)
USE_CHUNKING_ORCHESTRATOR            # Feature flag (default: false)

POSTGRES_HOST                        # Database host (default: localhost)
POSTGRES_PORT                        # Database port (default: 5432)
POSTGRES_DB                          # Database name (default: semantik)
POSTGRES_USER                        # Database user (default: semantik)
POSTGRES_PASSWORD                    # Database password (REQUIRED in prod)
DATABASE_URL                         # Full connection string (overrides above)
DB_POOL_SIZE                         # Connection pool (default: 20)
DB_MAX_OVERFLOW                      # Overflow connections (default: 40)
DB_POOL_TIMEOUT                      # Pool timeout (default: 30s)
DB_POOL_RECYCLE                      # Connection recycle (default: 3600s)
DB_POOL_PRE_PING                     # Test connections (default: true)
DB_ECHO                              # Log SQL queries (default: false)
DB_ECHO_POOL                         # Log pool events (default: false)
DB_QUERY_TIMEOUT                     # Query timeout (default: 30s)
DB_RETRY_LIMIT                       # Connection retries (default: 3)
DB_RETRY_INTERVAL                    # Retry interval (default: 0.5s)

QDRANT_HOST                          # Vector DB host (default: localhost)
QDRANT_PORT                          # Vector DB port (default: 6333)
DEFAULT_COLLECTION                   # Default collection name
DEFAULT_EMBEDDING_MODEL              # Model to use
DEFAULT_QUANTIZATION                 # float32|float16|int8
USE_MOCK_EMBEDDINGS                  # Mock mode for testing (default: false)

REDIS_URL                            # Redis connection URL
CELERY_BROKER_URL                    # Celery broker
CELERY_RESULT_BACKEND                # Celery result backend
CELERY_TEST_BROKER_URL              # Test broker
CELERY_TEST_RESULT_BACKEND          # Test backend

SEARCH_API_URL                       # vecpipe service URL
WEBUI_URL                            # WebUI service URL
WEBUI_INTERNAL_HOST                  # Internal hostname
WEBUI_PORT                           # WebUI port (default: 8080)
SEARCH_API_PORT                      # Search API port (default: 8000)
WEBUI_METRICS_PORT                   # Metrics port (default: 9092)
METRICS_PORT                         # (same as above)

CORS_ORIGINS                         # Allowed CORS origins
INTERNAL_API_KEY                     # Internal API authentication

CHUNKING_PREVIEW_RATE_LIMIT          # (UNDOCUMENTED, default: 10/min)
CHUNKING_COMPARE_RATE_LIMIT          # (UNDOCUMENTED, default: 5/min)
CHUNKING_PROCESS_RATE_LIMIT          # (UNDOCUMENTED, default: 20/hour)
CHUNKING_READ_RATE_LIMIT             # (UNDOCUMENTED, default: 60/min)
CHUNKING_ANALYTICS_RATE_LIMIT        # (UNDOCUMENTED, default: 30/min)
RATE_LIMIT_BYPASS_TOKEN              # Admin bypass for rate limits
CIRCUIT_BREAKER_FAILURES             # (UNDOCUMENTED, default: 5)
CIRCUIT_BREAKER_TIMEOUT              # (UNDOCUMENTED, default: 60s)
DISABLE_RATE_LIMITING                # (UNDOCUMENTED, for testing)

CUDA_VISIBLE_DEVICES                 # GPU selection
HF_HOME                              # HuggingFace cache directory
HF_HUB_OFFLINE                       # Offline mode (default: false)
HF_CACHE_DIR                         # Model storage path
MODEL_MAX_MEMORY_GB                  # Memory limit for models

LOG_LEVEL                            # Logging level (default: INFO)
ENVIRONMENT                          # development|production
TESTING                              # Testing mode flag
DOCUMENT_PATH                        # Path to document files
WEBUI_WORKERS                        # Worker process count

HEALTH_CHECK_TIMEOUT                 # Health check timeout (5.0s)
MODEL_UNLOAD_AFTER_SECONDS          # Model cache timeout (300s)
ENABLE_ADAPTIVE_BATCH_SIZE           # Dynamic batch sizing (true)
MIN_BATCH_SIZE                       # Min batch size (1)
MAX_BATCH_SIZE                       # Max batch size (256)
BATCH_SIZE_SAFETY_MARGIN             # OOM prevention margin (0.2)
```

---

## 13. ACTION ITEMS

- [ ] Create .env.example with all variables documented
- [ ] Fix metrics.py localhost reference
- [ ] Migrate rate_limits.py to Pydantic BaseSettings
- [ ] Consolidate celery_app.py to use settings
- [ ] Fix WebSocket manager Redis URL
- [ ] Update health.py to use settings
- [ ] Add startup configuration validation
- [ ] Add Redis connectivity check at startup
- [ ] Document POSTGRES_PASSWORD requirement in production
- [ ] Create environment setup guide for production deployment
- [ ] Add configuration validation tests
- [ ] Document all feature flags with deprecation timelines


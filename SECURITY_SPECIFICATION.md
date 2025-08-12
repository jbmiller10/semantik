# Semantik Security Specification

## Executive Summary

This document provides a comprehensive security analysis of the Semantik codebase, documenting all security measures, patterns, and implementations. Semantik implements defense-in-depth security with multiple layers of protection including authentication, authorization, input validation, rate limiting, and secure infrastructure configuration.

## Table of Contents

1. [Authentication System](#1-authentication-system)
2. [Authorization Patterns](#2-authorization-patterns)
3. [Input Validation and Sanitization](#3-input-validation-and-sanitization)
4. [API Security](#4-api-security)
5. [Data Security](#5-data-security)
6. [Infrastructure Security](#6-infrastructure-security)
7. [Security Best Practices](#7-security-best-practices)
8. [Security Recommendations](#8-security-recommendations)

---

## 1. Authentication System

### 1.1 JWT Token Management

**Implementation Location**: `packages/webui/auth.py`

#### Token Generation
- **Access Tokens**: Short-lived tokens (24 hours default)
- **Refresh Tokens**: Long-lived tokens (30 days)
- **Algorithm**: HS256 (HMAC with SHA-256)
- **Secret Key Management**: 
  - Production: Requires explicit environment variable
  - Development: Auto-generates and persists in `.jwt_secret` file

```python
# Token creation with proper expiration
def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM)
```

#### Token Verification
- Validates token signature
- Checks token type (access vs refresh)
- Verifies expiration time
- Returns username or None for invalid tokens

### 1.2 Password Security

**Implementation**: Using `passlib` with bcrypt hashing

```python
from shared.database import pwd_context

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
```

**Password Requirements**:
- Minimum 8 characters (enforced via Pydantic validator)
- Bcrypt with default cost factor (12 rounds)
- Never stored in plain text

### 1.3 Session Management

#### Refresh Token Storage
- Stored as hashed values in PostgreSQL
- Includes expiration timestamp
- Can be revoked (soft delete with `is_revoked` flag)
- Associated with user_id via foreign key

#### WebSocket Authentication
- Special handler for WebSocket connections
- Token passed as query parameter or first message
- Falls back to development user when `DISABLE_AUTH=true`

```python
async def get_current_user_websocket(token: str | None) -> dict[str, Any]:
    if not token and settings.DISABLE_AUTH:
        return development_user
    username = verify_token(token, "access")
    # Verify user exists and is active
```

### 1.4 Authentication Endpoints

**Location**: `packages/webui/api/auth.py`

- `/register`: User registration with automatic superuser for first user
- `/login`: Returns access and refresh tokens
- `/refresh`: Exchange refresh token for new access token
- `/logout`: Revokes refresh token
- `/me`: Get current user information

---

## 2. Authorization Patterns

### 2.1 Role-Based Access Control (RBAC)

**User Roles**:
- **Regular User**: Default permissions
- **Superuser**: Administrative privileges
  - First registered user automatically becomes superuser
  - Can reset database
  - Access to admin-only endpoints

**Implementation**:
```python
# Check for superuser status
if not current_user.get("is_superuser", False):
    raise HTTPException(status_code=403, detail="Admin access required")
```

### 2.2 Collection-Level Permissions

**Permission Types** (`PermissionTypeEnum`):
- `owner`: Full control
- `read`: Read-only access
- `write`: Read and write access
- `admin`: Administrative access

**Permission Checking**:
```python
# Repository method for permission verification
async def get_by_uuid_with_permission_check(
    collection_uuid: str, 
    user_id: int
) -> Collection
```

### 2.3 API Key Permissions

**Features**:
- API keys can have custom permission dictionaries
- Expiration dates supported
- Keys stored as hashes
- Can be revoked without deletion

```python
async def create_api_key(
    user_id: str,
    name: str,
    permissions: dict[str, Any] | None = None
) -> dict[str, Any]
```

### 2.4 Document Access Control

- Documents inherit collection permissions
- Access logged for audit trail
- Path traversal prevention in document serving

---

## 3. Input Validation and Sanitization

### 3.1 Pydantic Validation

**Location**: Various schema files in `packages/webui/api/`

**Validators Used**:
- `field_validator`: Custom validation logic
- Type hints for automatic validation
- Min/max length constraints
- Enum validation for restricted choices

**Examples**:
```python
@field_validator("username")
def validate_username(cls, v: str) -> str:
    if len(v) < 3:
        raise ValueError("Username must be at least 3 characters")
    if not all(c.isalnum() or c == "_" for c in v):
        raise ValueError("Username must contain only alphanumeric characters")
    return v
```

### 3.2 SQL Injection Prevention

**Approach**: Parameterized queries using SQLAlchemy ORM

```python
# Safe query construction
stmt = select(User).where(User.username == username)
result = await db.execute(stmt)

# Never using raw SQL strings
# All queries use SQLAlchemy's query builder
```

### 3.3 Path Traversal Prevention

**Implementation in** `packages/webui/services/chunking_security.py`:

```python
def validate_file_paths(file_paths: list[str], base_dir: str | None = None):
    for file_path in file_paths:
        # Normalize and resolve path
        normalized_path = os.path.normpath(file_path)
        
        # Check for path traversal attempts
        if ".." in normalized_path:
            raise SecurityError("Path traversal detected")
        
        # Ensure path is within base directory
        if base_dir:
            real_path = os.path.realpath(file_path)
            real_base = os.path.realpath(base_dir)
            if not real_path.startswith(real_base):
                raise SecurityError("Path outside allowed directory")
```

### 3.4 Error Message Sanitization

**PII Removal from Errors**:
```python
def _sanitize_error_message(error_msg: str) -> str:
    # Replace user home paths
    sanitized = re.sub(r"/home/[^/]+", "/home/~", error_msg)
    sanitized = re.sub(r"/Users/[^/]+", "/Users/~", sanitized)
    
    # Redact email addresses
    sanitized = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[email]", sanitized)
    
    return sanitized
```

### 3.5 Content Validation

**Chunking Input Validation** (`ChunkingInputValidator`):
- Null byte detection
- Content size limits
- Character encoding validation
- File type validation
- Metadata sanitization

---

## 4. API Security

### 4.1 Rate Limiting

**Implementation**: Using `slowapi` with Redis backend

**Configuration** (`packages/webui/config/rate_limits.py`):
```python
class RateLimitConfig:
    PREVIEW_LIMIT = 10  # per minute
    COMPARE_LIMIT = 5   # per minute
    PROCESS_LIMIT = 20  # per hour
    READ_LIMIT = 60     # per minute
    ANALYTICS_LIMIT = 30 # per minute
    DEFAULT_LIMIT = "100/minute"
```

**Features**:
- Per-user rate limiting (falls back to IP)
- Admin bypass token support
- Circuit breaker pattern for cascading failures
- Custom rate limits per endpoint
- Rate limit headers in responses

**Rate Limit Headers**:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1640995200
Retry-After: 60
```

### 4.2 CORS Configuration

**Implementation** (`packages/webui/main.py`):

```python
def _validate_cors_origins(origins: list[str]) -> list[str]:
    """Validate CORS origins and return only valid URLs."""
    valid_origins = []
    for origin in origins:
        # Reject wildcards in production
        if origin == "*" and settings.ENVIRONMENT == "production":
            logger.error(f"Rejecting wildcard origin in production")
            continue
        
        # Validate URL format
        parsed = urlparse(origin)
        if parsed.scheme and parsed.netloc:
            valid_origins.append(origin)
    return valid_origins
```

**CORS Middleware Settings**:
- `allow_credentials`: True
- `allow_methods`: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
- `allow_headers`: ["*"]
- Origins validated at startup

### 4.3 Request/Response Validation

**Request Validation**:
- Pydantic models for all request bodies
- Query parameter validation
- Header validation for internal APIs
- File upload restrictions

**Response Validation**:
- Consistent error response format
- HTTP status codes properly mapped
- Sensitive data excluded from responses

### 4.4 Internal API Security

**Internal API Key**:
- Required for service-to-service communication
- Auto-generated in development
- Must be explicitly set in production
- Passed via `X-Internal-API-Key` header

```python
def verify_internal_api_key(x_internal_api_key: str = Header(...)):
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid internal API key")
```

---

## 5. Data Security

### 5.1 Sensitive Data Handling

**Password Storage**:
- Never stored in plain text
- Bcrypt hashing with salt
- Password field excluded from user responses

**Token Storage**:
- Refresh tokens hashed before storage
- JWT secrets stored securely
- API keys hashed in database

### 5.2 Audit Logging

**Implementation** (`CollectionAuditLog` model):

```python
class CollectionAuditLog(Base):
    __tablename__ = "collection_audit_log"
    
    id = Column(Integer, primary_key=True)
    collection_id = Column(String, ForeignKey("collections.id"))
    operation_id = Column(UUID, ForeignKey("operations.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String, nullable=False)
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Logged Actions**:
- Collection creation/deletion
- Source additions/removals
- Reindexing operations
- Permission changes
- Failed operations

**Privacy Protection**:
- All audit details sanitized for PII
- File paths anonymized
- Email addresses redacted

### 5.3 Environment Variables

**Secure Defaults**:
```python
# Production enforcement
if self.ENVIRONMENT == "production":
    if self.JWT_SECRET_KEY == "default-secret-key":
        raise ValueError("JWT_SECRET_KEY must be set in production")
```

**Sensitive Variables**:
- `JWT_SECRET_KEY`: JWT signing key
- `POSTGRES_PASSWORD`: Database password
- `INTERNAL_API_KEY`: Service communication key
- `RATE_LIMIT_BYPASS_TOKEN`: Admin bypass token

### 5.4 Redis Security

**Usage**:
- WebSocket pub/sub
- Rate limiting storage
- Chunking cache
- Operation status tracking

**Security Measures**:
- No sensitive data in Redis
- TTL on all keys to prevent memory leaks
- Separate databases for different purposes

---

## 6. Infrastructure Security

### 6.1 Docker Security

**Security Options** (docker-compose.yml):

```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE  # Only what's needed
```

**Resource Limits**:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      memory: 2G
```

### 6.2 Network Security

**Service Isolation**:
- Services communicate via Docker network
- External ports exposed only when necessary
- Health checks for all services

**Port Configuration**:
- PostgreSQL: 5432 (database)
- Redis: 6379 (cache/broker)
- Qdrant: 6333 (vector DB)
- WebUI: 8080 (main application)
- VecPipe: 8000 (search API)

### 6.3 Volume Security

**Persistent Volumes**:
- `postgres_data`: Database storage
- `qdrant_storage`: Vector embeddings
- `redis_data`: Cache persistence
- Model cache: HuggingFace models

**Permissions**:
- Volumes owned by service users
- Read-only mounts where possible
- No world-writable directories

### 6.4 Secret Management

**Development**:
- Auto-generated secrets saved to files
- `.jwt_secret` with 0600 permissions
- Git-ignored secret files

**Production Requirements**:
- All secrets via environment variables
- No hardcoded credentials
- Strong password requirements
- Rotation procedures documented

---

## 7. Security Best Practices

### 7.1 Secure Coding Practices

1. **Never Trust User Input**
   - All inputs validated
   - Parameterized queries only
   - Path traversal prevention

2. **Principle of Least Privilege**
   - Minimal Docker capabilities
   - Role-based permissions
   - Service accounts limited

3. **Defense in Depth**
   - Multiple validation layers
   - Rate limiting + circuit breakers
   - Audit logging throughout

4. **Fail Securely**
   - Generic error messages
   - Sanitized error details
   - Graceful degradation

### 7.2 Security Headers

**Implemented Headers**:
- CORS headers (restrictive)
- Rate limit headers
- Content-Type validation

**Recommended Additional Headers**:
- Content-Security-Policy
- X-Frame-Options
- X-Content-Type-Options
- Strict-Transport-Security

### 7.3 Monitoring and Alerting

**Current Monitoring**:
- Audit logs for all operations
- Error tracking with correlation IDs
- Rate limit violations logged
- Circuit breaker events

**Logging Security**:
- PII sanitization in logs
- Structured logging format
- Log rotation configured
- Sensitive data never logged

---

## 8. Security Recommendations

### 8.1 High Priority

1. **HTTPS/TLS Implementation**
   - Currently no TLS termination
   - Recommend nginx/traefik reverse proxy
   - Let's Encrypt for certificates

2. **Database Encryption**
   - Implement encryption at rest
   - Use SSL for database connections
   - Encrypt sensitive columns

3. **API Rate Limiting Enhancement**
   - Implement distributed rate limiting
   - Add IP-based blocking for abusers
   - Implement CAPTCHA for repeated failures

### 8.2 Medium Priority

1. **Security Headers**
   - Add CSP headers
   - Implement HSTS
   - Add X-Frame-Options

2. **Authentication Enhancements**
   - Multi-factor authentication (MFA)
   - OAuth2/OIDC support
   - Session timeout policies

3. **Monitoring Improvements**
   - Implement SIEM integration
   - Add security event alerting
   - Create security dashboards

### 8.3 Low Priority

1. **Compliance Features**
   - GDPR data export/deletion
   - Audit log retention policies
   - Privacy policy enforcement

2. **Advanced Security**
   - Web Application Firewall (WAF)
   - Intrusion Detection System (IDS)
   - Penetration testing schedule

---

## Conclusion

Semantik implements comprehensive security measures across all layers of the application. The system follows security best practices including:

- Strong authentication with JWT tokens
- Role-based authorization with granular permissions
- Comprehensive input validation and sanitization
- Rate limiting and abuse prevention
- Secure infrastructure configuration
- Audit logging with privacy protection

While the current implementation provides solid security, the recommended enhancements would further strengthen the system's security posture, particularly the addition of TLS/HTTPS and database encryption.

## Security Contact

For security issues or questions, please refer to the project's security policy or contact the maintainers through the appropriate channels.

---

*Document Version: 1.0*  
*Last Updated: 2025-01-12*  
*Classification: Public*
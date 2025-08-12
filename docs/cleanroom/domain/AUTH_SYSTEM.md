# AUTH_SYSTEM - Authentication & Authorization Component

## 1. Component Overview

The AUTH_SYSTEM component provides comprehensive authentication and authorization services for the Semantik application. It implements a multi-layered security model with JWT-based authentication for users and hash-based authentication for API keys.

### Core Responsibilities
- User authentication via username/password
- JWT token generation and validation (access + refresh tokens)
- API key management for programmatic access
- Session management and token refresh mechanisms
- Permission model for resource access control
- Password security using bcrypt hashing
- Protected route enforcement in frontend and backend

### Key Technologies
- **Backend**: FastAPI, SQLAlchemy, JWT (PyJWT), Passlib (bcrypt)
- **Frontend**: React, Zustand (state management), Axios (API client)
- **Database**: PostgreSQL with timezone-aware datetime columns
- **Security**: bcrypt for password hashing, SHA-256 for token/API key hashing

## 2. Architecture & Design Patterns

### Authentication Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   React App     │────▶│   Auth API       │────▶│  PostgreSQL     │
│  (authStore)    │◀────│  (/api/auth)     │◀────│  (Users, Tokens)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         │                       ▼                         │
         │              ┌──────────────────┐              │
         └─────────────▶│  Auth Service    │◀─────────────┘
                        │  (JWT & bcrypt)  │
                        └──────────────────┘
```

### Token Flow Architecture

1. **Access Token**: Short-lived (24 hours default), used for API authentication
2. **Refresh Token**: Long-lived (30 days), used to obtain new access tokens
3. **API Keys**: Permanent until revoked, hash-stored for programmatic access

### Repository Pattern Implementation

```python
AuthRepository (Abstract) ──▶ PostgreSQLAuthRepository (Concrete)
UserRepository (Abstract) ──▶ PostgreSQLUserRepository (Concrete)
ApiKeyRepository (Abstract) ──▶ PostgreSQLApiKeyRepository (Concrete)
```

## 3. Key Interfaces & Contracts

### Authentication API Endpoints

```python
# packages/webui/api/auth.py

POST /api/auth/register
  Request: UserCreate {username, email, password, full_name?}
  Response: User {id, username, email, full_name, is_active, created_at}

POST /api/auth/login
  Request: UserLogin {username, password}
  Response: Token {access_token, refresh_token, token_type="bearer"}

POST /api/auth/refresh
  Request: {refresh_token: string}
  Response: Token {access_token, refresh_token, token_type="bearer"}

POST /api/auth/logout
  Request: {refresh_token?: string}
  Response: {message: "Logged out successfully"}

GET /api/auth/me
  Headers: Authorization: Bearer <access_token>
  Response: User {id, username, email, full_name, is_active, created_at}
```

### User Model Interface

```python
# packages/shared/database/models.py

class User(Base):
    id: int (primary key, autoincrement)
    username: str (unique, indexed)
    email: str (unique, indexed)
    full_name: str | None
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    created_at: DateTime(timezone=True)
    updated_at: DateTime(timezone=True)
    last_login: DateTime(timezone=True) | None
```

### JWT Token Structure

```python
# Access Token Payload
{
    "sub": "username",      # Subject - the username
    "type": "access",       # Token type identifier
    "exp": 1234567890       # Expiration timestamp
}

# Refresh Token Payload
{
    "sub": "username",      # Subject - the username
    "type": "refresh",      # Token type identifier
    "exp": 1234567890       # Expiration timestamp
}
```

## 4. Data Flow & Dependencies

### Login Flow

```python
1. User submits credentials to POST /api/auth/login
2. UserRepository.get_user_by_username() retrieves user
3. verify_password() checks bcrypt hash
4. AuthRepository.update_user_last_login() updates timestamp
5. create_access_token() generates JWT with 24h expiry
6. create_refresh_token() generates JWT with 30d expiry
7. AuthRepository.save_refresh_token() stores hashed refresh token
8. Frontend receives tokens and stores in Zustand + localStorage
9. Frontend fetches user details via GET /api/auth/me
10. All subsequent API calls include Authorization: Bearer <token>
```

### Token Validation Flow

```python
1. Request arrives with Authorization: Bearer <token>
2. get_current_user() dependency extracts token
3. verify_token() decodes JWT and validates:
   - Signature using JWT_SECRET_KEY
   - Expiration timestamp
   - Token type matches expected ("access")
4. UserRepository.get_user_by_username() retrieves user
5. Validates user.is_active status
6. Returns user dict for request context
```

### Refresh Token Flow

```python
1. Client sends refresh_token to POST /api/auth/refresh
2. AuthRepository.verify_refresh_token() validates:
   - Token hash exists in database
   - Token not revoked
   - Token not expired
   - Associated user is active
3. Generate new access_token and refresh_token
4. AuthRepository.revoke_refresh_token() invalidates old token
5. AuthRepository.save_refresh_token() stores new token hash
6. Return new tokens to client
```

## 5. Critical Implementation Details

### Password Hashing Implementation

```python
# packages/webui/auth.py

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """Generate bcrypt hash with automatic salt"""
    return str(pwd_context.hash(password))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash"""
    return bool(pwd_context.verify(plain_password, hashed_password))
```

### JWT Token Generation

```python
# packages/webui/auth.py

def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return str(jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM))
```

### Frontend Auth State Management

```typescript
// apps/webui-react/src/stores/authStore.ts

interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: User | null;
  setAuth: (token: string, user: User, refreshToken?: string) => void;
  logout: () => Promise<void>;
}

// Persisted to localStorage with key "auth-storage"
export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // ... implementation
    }),
    { name: 'auth-storage' }
  )
);
```

### API Client Token Injection

```typescript
// apps/webui-react/src/services/api/v2/client.ts

apiClient.interceptors.request.use(
  (config) => {
    const state = useAuthStore.getState();
    const token = state.token;
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  }
);
```

## 6. Security Considerations

### Token Security

1. **JWT Secret Key Management**
   - Production requires explicit JWT_SECRET_KEY environment variable
   - Development auto-generates and persists to `.jwt_secret` file (0600 permissions)
   - Minimum recommended: 32 bytes (256 bits) of entropy
   - Algorithm: HS256 (HMAC with SHA-256)

2. **Token Storage**
   - Refresh tokens hashed with SHA-256 before database storage
   - Original tokens never stored in plaintext
   - Token hash used for lookups and validation

3. **Token Expiration**
   - Access tokens: 24 hours (configurable via ACCESS_TOKEN_EXPIRE_MINUTES)
   - Refresh tokens: 30 days (hardcoded REFRESH_TOKEN_EXPIRE_DAYS)
   - Expired tokens automatically cleaned up via cleanup_expired_tokens()

### Password Security

1. **Bcrypt Configuration**
   - Automatic salt generation per password
   - Cost factor adjusts automatically based on hardware
   - Deprecated schemes handled gracefully

2. **Password Requirements**
   - Minimum 8 characters (validated in UserCreate model)
   - No maximum length restriction
   - No complexity requirements (rely on length)

### Session Security

1. **Token Revocation**
   - Refresh tokens can be revoked individually
   - All user tokens revoked on logout
   - Revoked tokens remain in database (marked as revoked)

2. **User Status Checks**
   - is_active flag checked on every authentication
   - Inactive users cannot authenticate or use existing tokens

3. **CORS Configuration**
   - Configurable origins via CORS_ORIGINS environment variable
   - Default: localhost:5173 for development

## 7. Testing Requirements

### Unit Tests

```python
# tests/unit/test_auth.py
- test_password_hashing()
- test_jwt_token_creation()
- test_jwt_token_verification()
- test_token_expiration()

# tests/unit/test_auth_repository.py
- test_save_refresh_token()
- test_verify_refresh_token()
- test_revoke_refresh_token()
- test_cleanup_expired_tokens()
```

### Integration Tests

```python
# tests/integration/test_auth_api.py
- test_user_registration_success()
- test_user_registration_duplicate()
- test_login_success()
- test_login_invalid_credentials()
- test_refresh_token_success()
- test_logout()
- test_get_current_user()
```

### Frontend Tests

```typescript
// Test protected routes
- Routes redirect to /login when unauthenticated
- Token included in API requests
- Auto-logout on 401 responses
- Token persistence across page refreshes
```

## 8. Common Pitfalls & Best Practices

### Common Pitfalls

1. **Not checking token type**
   ```python
   # BAD: Accepting any valid JWT
   username = verify_token(token)
   
   # GOOD: Validate token type
   username = verify_token(token, token_type="access")
   ```

2. **Storing plaintext tokens**
   ```python
   # BAD: Storing refresh token directly
   await save_token(user_id, refresh_token)
   
   # GOOD: Hash before storage
   token_hash = pwd_context.hash(refresh_token)
   await save_token(user_id, token_hash)
   ```

3. **Not handling token refresh properly**
   ```typescript
   // BAD: Using expired tokens indefinitely
   if (token) { config.headers.Authorization = `Bearer ${token}`; }
   
   // GOOD: Implement token refresh logic
   // See apiClient response interceptor for 401 handling
   ```

### Best Practices

1. **Always use dependency injection for auth**
   ```python
   @router.get("/protected")
   async def protected_endpoint(
       current_user: dict[str, Any] = Depends(get_current_user)
   ):
       # User is guaranteed to be authenticated
   ```

2. **Validate user permissions for resources**
   ```python
   async def get_collection_for_user(
       collection_uuid: str,
       current_user: dict[str, Any] = Depends(get_current_user),
       db: AsyncSession = Depends(get_db)
   ):
       # Verify user owns or has access to collection
   ```

3. **Use timezone-aware datetimes**
   ```python
   from datetime import UTC, datetime
   
   expires_at = datetime.now(UTC) + timedelta(days=30)
   ```

4. **Handle auth errors gracefully in frontend**
   ```typescript
   try {
     const response = await authApi.login(credentials);
     // Handle success
   } catch (error) {
     addToast({
       type: 'error',
       message: getErrorMessage(error),
     });
   }
   ```

## 9. Configuration & Environment

### Required Environment Variables

```bash
# Production (REQUIRED)
JWT_SECRET_KEY=<32-byte-hex-string>  # Generate: openssl rand -hex 32
ENVIRONMENT=production

# Optional
ACCESS_TOKEN_EXPIRE_MINUTES=1440  # Default: 24 hours
ALGORITHM=HS256                   # Default: HS256
DISABLE_AUTH=false                 # NEVER true in production
CORS_ORIGINS=https://yourdomain.com
```

### Development Configuration

```python
# packages/shared/config/webui.py

class WebuiConfig(BaseConfig):
    JWT_SECRET_KEY: str = "default-secret-key"  # Auto-generated in dev
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALGORITHM: str = "HS256"
    DISABLE_AUTH: bool = False  # Set True for testing without auth
```

### JWT Secret Key Management

1. **Production**: Must set JWT_SECRET_KEY environment variable
2. **Development**: Auto-generates and saves to `data/.jwt_secret`
3. **File permissions**: 0600 (owner read/write only)
4. **Key rotation**: Update JWT_SECRET_KEY and restart services

## 10. Integration Points

### Frontend Integration

1. **Protected Routes** (`apps/webui-react/src/App.tsx`)
   ```typescript
   function ProtectedRoute({ children }: { children: React.ReactNode }) {
     const token = useAuthStore((state) => state.token);
     
     if (!token) {
       return <Navigate to="/login" replace />;
     }
     
     return <>{children}</>;
   }
   ```

2. **API Client Integration** (`apps/webui-react/src/services/api/v2/client.ts`)
   - Request interceptor adds Authorization header
   - Response interceptor handles 401 errors

### Backend Integration

1. **FastAPI Dependencies** (`packages/webui/dependencies.py`)
   ```python
   async def get_current_user(
       credentials: HTTPAuthorizationCredentials | None = Depends(security)
   ) -> dict[str, Any]:
       # Returns authenticated user or raises HTTPException
   ```

2. **WebSocket Authentication** (`packages/webui/auth.py`)
   ```python
   async def get_current_user_websocket(token: str | None) -> dict[str, Any]:
       # Special handler for WebSocket connections
       # Token passed as query parameter or first message
   ```

3. **Internal API Authentication** (`packages/webui/api/internal.py`)
   ```python
   def verify_internal_api_key(
       x_internal_api_key: Annotated[str | None, Header()] = None
   ) -> None:
       # Validates internal service-to-service API key
   ```

### Database Integration

1. **User Management**: PostgreSQLUserRepository
2. **Token Storage**: PostgreSQLAuthRepository  
3. **API Key Management**: PostgreSQLApiKeyRepository
4. **Permission Model**: CollectionPermission table

### Service Integration

1. **Collection Access Control**
   - Collections owned by users (owner_id foreign key)
   - Permission checks in CollectionRepository
   - API endpoints validate user access

2. **Operation Tracking**
   - Operations linked to users (user_id foreign key)
   - Audit logs track user actions

3. **API Key Permissions**
   - API keys can have limited permissions
   - Stored as JSON in permissions column
   - Validated during request processing

## Critical Files Reference

### Backend Authentication
- `packages/webui/auth.py` - Core authentication logic
- `packages/webui/api/auth.py` - Authentication API endpoints
- `packages/webui/repositories/postgres/user_repository.py` - User data access
- `packages/webui/repositories/postgres/auth_repository.py` - Token management
- `packages/webui/repositories/postgres/api_key_repository.py` - API key management
- `packages/webui/dependencies.py` - Auth dependency injection
- `packages/shared/database/models.py` - User, RefreshToken, ApiKey models
- `packages/shared/config/webui.py` - JWT configuration

### Frontend Authentication
- `apps/webui-react/src/stores/authStore.ts` - Auth state management
- `apps/webui-react/src/pages/LoginPage.tsx` - Login/register UI
- `apps/webui-react/src/services/api/v2/client.ts` - API client with auth
- `apps/webui-react/src/services/api/v2/auth.ts` - Auth API service
- `apps/webui-react/src/App.tsx` - Protected route implementation

### Testing
- `tests/integration/test_auth_api.py` - API integration tests
- `tests/unit/test_auth.py` - Unit tests for auth functions
- `tests/unit/test_auth_repository.py` - Repository unit tests

## Security Checklist

- [ ] JWT_SECRET_KEY set in production environment
- [ ] JWT_SECRET_KEY has sufficient entropy (32+ bytes)
- [ ] DISABLE_AUTH is False in production
- [ ] CORS_ORIGINS configured for production domain
- [ ] Database connections use SSL in production
- [ ] Refresh tokens are hashed before storage
- [ ] Password minimum length enforced (8 chars)
- [ ] Token expiration times are reasonable
- [ ] Failed login attempts are logged
- [ ] User sessions can be revoked
- [ ] API keys have appropriate permissions
- [ ] Internal API key is secured
- [ ] Frontend stores tokens securely (httpOnly cookies preferred)
- [ ] 401 responses trigger re-authentication
- [ ] Token refresh mechanism implemented
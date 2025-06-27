#!/usr/bin/env python3
"""
Authentication system for the Document Embedding Web UI
Provides JWT-based authentication with user management
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import logging

from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "/var/embeddings/webui.db"
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        # Check if username contains only alphanumeric characters and underscores
        if not all(c.isalnum() or c == '_' for c in v):
            raise ValueError('Username must contain only alphanumeric characters and underscores')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: str
    last_login: Optional[str] = None

# Database functions
def init_auth_tables():
    """Initialize authentication tables in the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  full_name TEXT,
                  hashed_password TEXT NOT NULL,
                  is_active BOOLEAN DEFAULT 1,
                  created_at TEXT NOT NULL,
                  last_login TEXT)''')
    
    # Refresh tokens table (for token revocation)
    c.execute('''CREATE TABLE IF NOT EXISTS refresh_tokens
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  token_hash TEXT UNIQUE NOT NULL,
                  expires_at TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  is_revoked BOOLEAN DEFAULT 0,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    
    # Create indices
    c.execute('''CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_refresh_tokens_hash ON refresh_tokens(token_hash)''')
    
    conn.commit()
    conn.close()
    logger.info("Authentication tables initialized")

# Password hashing functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

# Token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT refresh token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, token_type: str = "access") -> Optional[str]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type_claim: str = payload.get("type")
        
        if username is None or token_type_claim != token_type:
            return None
        return username
    except JWTError:
        return None

# User database operations
def get_user(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    return dict(user) if user else None

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    
    return dict(user) if user else None

def create_user(user_data: UserCreate) -> Dict[str, Any]:
    """Create a new user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if user already exists
    c.execute("SELECT id FROM users WHERE username = ? OR email = ?", 
              (user_data.username, user_data.email))
    if c.fetchone():
        conn.close()
        raise ValueError("User with this username or email already exists")
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    created_at = datetime.utcnow().isoformat()
    
    c.execute("""INSERT INTO users (username, email, full_name, hashed_password, created_at)
                 VALUES (?, ?, ?, ?, ?)""",
              (user_data.username, user_data.email, user_data.full_name, 
               hashed_password, created_at))
    
    user_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "is_active": True,
        "created_at": created_at,
        "last_login": None
    }

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    
    # Update last login
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET last_login = ? WHERE id = ?",
              (datetime.utcnow().isoformat(), user["id"]))
    conn.commit()
    conn.close()
    
    return user

def save_refresh_token(user_id: int, token: str, expires_at: datetime):
    """Save refresh token to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Hash the token for storage
    token_hash = pwd_context.hash(token)
    
    c.execute("""INSERT INTO refresh_tokens (user_id, token_hash, expires_at, created_at)
                 VALUES (?, ?, ?, ?)""",
              (user_id, token_hash, expires_at.isoformat(), datetime.utcnow().isoformat()))
    
    conn.commit()
    conn.close()

def verify_refresh_token(token: str) -> Optional[int]:
    """Verify refresh token and return user_id if valid"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get all non-revoked, non-expired tokens
    c.execute("""SELECT * FROM refresh_tokens 
                 WHERE is_revoked = 0 AND expires_at > ?""",
              (datetime.utcnow().isoformat(),))
    
    tokens = c.fetchall()
    conn.close()
    
    # Check each token
    for token_row in tokens:
        if pwd_context.verify(token, token_row["token_hash"]):
            return token_row["user_id"]
    
    return None

def revoke_refresh_token(token: str):
    """Revoke a refresh token"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Find and revoke the token
    c.execute("""UPDATE refresh_tokens SET is_revoked = 1 
                 WHERE token_hash IN (SELECT token_hash FROM refresh_tokens WHERE is_revoked = 0)""")
    
    conn.commit()
    conn.close()

# FastAPI dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    token = credentials.credentials
    username = verify_token(token, "access")
    
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_user(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user

# Optional: Create admin dependency
async def get_current_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current admin user (for future use)"""
    # For now, all authenticated users have full access
    # In the future, you could add an is_admin field to the users table
    return current_user

# Initialize auth tables when module is imported
init_auth_tables()
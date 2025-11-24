# app/auth/jwt.py
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union
from jose import jwt, JWTError
import bcrypt as _bcrypt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from uuid import UUID
import secrets

from app.core.config import get_settings

# Try to import Redis, fall back to mock if not available
try:
    from app.auth.redis import add_to_blacklist, is_blacklisted
except (ImportError, ModuleNotFoundError):
    # Use mock Redis for testing/development
    from app.auth.redis_mock import add_to_blacklist, is_blacklisted

from app.schemas.token import TokenType
from app.database import get_db
from sqlalchemy.orm import Session
from app.models.user import User

settings = get_settings()

# Password hashing - use bcrypt directly to avoid passlib compatibility issues
import bcrypt as _bcrypt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash using bcrypt directly."""
    try:
        password_bytes = plain_password.encode('utf-8')
        # Handle both string and bytes for hashed password
        if isinstance(hashed_password, str):
            hashed_bytes = hashed_password.encode('utf-8')
        else:
            hashed_bytes = hashed_password
        return _bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        # Log the error but return False rather than crashing
        print(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt directly, handling the 72-byte limit."""
    # Convert to bytes and truncate to 72 bytes if necessary
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    
    # Generate salt and hash
    salt = _bcrypt.gensalt(rounds=12)
    hashed = _bcrypt.hashpw(password_bytes, salt)
    
    # Return as string
    return hashed.decode('utf-8')

def create_token(
    user_id: Union[str, UUID],
    token_type: TokenType,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT token (access or refresh).
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        if token_type == TokenType.ACCESS:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                days=settings.REFRESH_TOKEN_EXPIRE_DAYS
            )

    if isinstance(user_id, UUID):
        user_id = str(user_id)

    to_encode = {
        "sub": user_id,
        "type": token_type.value,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_hex(16)
    }

    secret = (
        settings.JWT_SECRET_KEY 
        if token_type == TokenType.ACCESS 
        else settings.JWT_REFRESH_SECRET_KEY
    )

    try:
        return jwt.encode(to_encode, secret, algorithm=settings.ALGORITHM)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create token: {str(e)}"
        )

async def decode_token(
    token: str,
    token_type: TokenType,
    verify_exp: bool = True
) -> dict[str, Any]:
    """
    Decode and verify a JWT token.
    """
    try:
        secret = (
            settings.JWT_SECRET_KEY 
            if token_type == TokenType.ACCESS 
            else settings.JWT_REFRESH_SECRET_KEY
        )
        
        payload = jwt.decode(
            token,
            secret,
            algorithms=[settings.ALGORITHM],
            options={"verify_exp": verify_exp}
        )
        
        if payload.get("type") != token_type.value:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        if await is_blacklisted(payload["jti"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get current user from access token.
    Returns the actual User model instance.
    """
    try:
        payload = await decode_token(token, TokenType.ACCESS)
        user_id = payload["sub"]
        
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
            
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
            
        return user
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
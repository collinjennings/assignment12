# app/auth/redis_mock.py
"""
Mock Redis implementation for testing/development when Redis is not available.
This allows the app to run without Redis by providing no-op implementations.
"""

async def add_to_blacklist(token_jti: str, expires_in: int = None) -> bool:
    """Mock implementation - does nothing but returns success."""
    return True

async def is_blacklisted(token_jti: str) -> bool:
    """Mock implementation - always returns False (token not blacklisted)."""
    return False

async def get_redis_connection():
    """Mock implementation - returns None."""
    return None
"""
Authentication & Authorization Middleware for FastAPI using Supabase JWT.
Validates Bearer tokens and checks user roles ('user' vs 'admin').
"""
import os
import logging
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Header
from db.supabase_client import db_service

logger = logging.getLogger("auth_middleware")


async def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validates Supabase JWT token string and returns user info dict.
    Returns None if token is invalid or user not found.
    """
    if not token or not token.strip():
        return None

    if db_service.client:
        try:
            user_response = db_service.client.auth.get_user(token)
            if user_response and user_response.user:
                u = user_response.user
                profile = await db_service.get_user_profile(u.id)
                return {
                    "id": u.id,
                    "email": u.email,
                    "role": profile.get("role", "user") if profile else "user",
                }
        except Exception as e:
            logger.warning(f"Invalid auth token: {e}")
            return None

    # Fallback mockup user if Supabase client is not connected in local dev mode
    if not db_service.is_connected:
        return {"id": "dev-user-id", "email": "dev@local.host", "role": "admin"}

    return None


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """
    Extracts and validates Supabase JWT token from Authorization header.
    Returns user dict if valid token provided; returns None if optional token missing.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.split(" ")[1].strip()
    return await verify_token(token)


async def require_authenticated_user(user: Optional[Dict[str, Any]] = Depends(get_optional_user)) -> Dict[str, Any]:
    """Requires a valid authenticated user."""
    if not user:
        # If Supabase client isn't configured, allow anonymous fallback user in local dev mode
        if not db_service.is_connected:
            return {"id": "dev-user-id", "email": "dev@local.host", "role": "admin"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin_user(user: Dict[str, Any] = Depends(require_authenticated_user)) -> Dict[str, Any]:
    """Requires Admin role."""
    if user.get("role") != "admin" and db_service.is_connected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user

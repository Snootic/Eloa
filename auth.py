"""Offline-capable JWT validation for service-to-service calls.

Keycloak is the issuer, but this module only needs a cached JWKS document;
tokens remain valid while Keycloak is unreachable.
"""
from __future__ import annotations

import hmac
import os
from typing import Any, Iterable

import jwt
from jwt import PyJWKClient


class AuthError(Exception):
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


ISSUER = os.getenv("AUTH_ISSUER", "")
JWKS_URL = os.getenv("AUTH_JWKS_URL", "")
CLIENTS = {
    item.strip()
    for item in os.getenv("AUTH_CLIENTS", "gatekeeper,voice-llm,voice-client").split(",")
    if item.strip()
}
STATIC_TOKEN = os.getenv("AUTH_STATIC_TOKEN", "")
ALLOW_STATIC = _env_bool("AUTH_ALLOW_STATIC", False)
ENABLED = bool(ISSUER and JWKS_URL)

_jwks: PyJWKClient | None = None
if ENABLED:
    _jwks = PyJWKClient(JWKS_URL, cache_keys=True, lifespan=300)


def enabled() -> bool:
    return ENABLED


def _token_from_header(authorization: str | None) -> str:
    if not authorization:
        raise AuthError("missing authorization")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthError("invalid authorization scheme")
    return token


def check(authorization: str | None, query_token: str | None = None) -> None:
    """Validate a bearer JWT. Falls back to the static token during rollout."""
    if not ENABLED:
        if STATIC_TOKEN and not hmac.compare_digest(
            authorization or "", f"Bearer {STATIC_TOKEN}"
        ):
            raise AuthError("unauthorized")
        return

    token = ""
    try:
        token = _token_from_header(authorization)
    except AuthError:
        token = query_token or ""
    if ALLOW_STATIC and STATIC_TOKEN:
        if authorization and hmac.compare_digest(
            authorization, f"Bearer {STATIC_TOKEN}"
        ):
            return
        if query_token and hmac.compare_digest(query_token, STATIC_TOKEN):
            return
    if token:
        try:
            assert _jwks is not None
            signing_key = _jwks.get_signing_key_from_jwt(token)
            claims: dict[str, Any] = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=ISSUER,
                options={
                    "verify_aud": False,
                    "require": ["exp", "iat", "iss", "azp"],
                },
            )
        except Exception as exc:
            raise AuthError("invalid token") from exc
        client = str(claims.get("azp") or "")
        if client not in CLIENTS:
            raise AuthError("client not allowed")
        return

    raise AuthError("unauthorized")


def authorization(authorization: str | None, query_token: str | None = None) -> None:
    check(authorization, query_token)


def allowed_clients() -> Iterable[str]:
    return CLIENTS

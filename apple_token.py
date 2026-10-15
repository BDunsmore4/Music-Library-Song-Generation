"""
Apple Music (MusicKit) developer-token minting.

MusicKit JS in the browser needs a short-lived **developer token**: an ES256 JWT
signed with the MusicKit private key (.p8) from your Apple Developer account. The
private key must never reach the browser, so the backend mints the token on
request. The user's own *Music-User-Token* is obtained client-side by MusicKit
after they authorise; we never see their Apple credentials.

Env:
  APPLE_TEAM_ID       your 10-char Apple Developer Team ID
  APPLE_KEY_ID        the 10-char Key ID of the MusicKit key
  APPLE_PRIVATE_KEY   the contents of the .p8 file (PEM, newlines preserved),
                      or APPLE_PRIVATE_KEY_PATH pointing at the .p8 on disk
"""

import os
import time

import jwt  # PyJWT, with the `cryptography` extra for ES256

TEAM_ID = os.environ.get("APPLE_TEAM_ID", "")
KEY_ID  = os.environ.get("APPLE_KEY_ID", "")

# Tokens are valid up to 6 months; we mint shorter ones and cache in-process.
TOKEN_TTL_SECONDS = 60 * 60 * 12  # 12h
_cache = {"token": None, "exp": 0}


def _load_private_key():
    raw = os.environ.get("APPLE_PRIVATE_KEY", "")
    if raw:
        # Env vars often arrive with literal "\n" instead of real newlines.
        return raw.replace("\\n", "\n")
    path = os.environ.get("APPLE_PRIVATE_KEY_PATH", "")
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


def is_configured():
    return bool(TEAM_ID and KEY_ID and _load_private_key())


def get_developer_token():
    """Return a cached, valid MusicKit developer token, minting one if needed."""
    now = int(time.time())
    if _cache["token"] and _cache["exp"] - 60 > now:
        return _cache["token"]

    key = _load_private_key()
    if not (TEAM_ID and KEY_ID and key):
        raise RuntimeError("Apple MusicKit is not configured (missing APPLE_* env vars)")

    exp = now + TOKEN_TTL_SECONDS
    token = jwt.encode(
        {"iss": TEAM_ID, "iat": now, "exp": exp},
        key,
        algorithm="ES256",
        headers={"alg": "ES256", "kid": KEY_ID},
    )
    _cache.update(token=token, exp=exp)
    return token
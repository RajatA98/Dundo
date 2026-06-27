"""HMAC-signed opaque context token for the /narrative endpoint.

Codex round-2 review Q3 pushed back on the in-memory cache approach: a TTL
cache breaks across HF restarts, multiple workers, and page refreshes. This
module is the replacement — a stateless signed token.

`/neighbors` issues a token containing the full NarrativeContext payload for
every neighbor it returns. The token is HMAC-signed with a server secret so
the backend can later verify that the client didn't tamper with it. On
`/narrative`, the backend verifies the signature + expiry + model/catalog
hashes, looks up the requested trackId inside the token payload, and rebuilds
NarrativeContext server-side from the verified claims.

This means: zero server-side state, survives restarts, scales horizontally,
and the client can't inflate the prompt with garbage to drive up cost.

Token format:
    base64url(json_payload) + "." + hex(hmac_sha256(payload_bytes, secret))

Payload shape (sorted keys for stability):
    {
      "queryFingerprint": "<sha256 of upload bytes>",
      "modelSha": "<MuQ-MuLan model sha>",
      "catalogSha": "<sha256 of manifest.json bytes>",
      "expiresAt": <unix seconds>,
      "acrcloudCoverSongId": <dict | null>,
      "neighbors": {
        "<trackId>": {full per-neighbor context fields},
        ...
      }
    }
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_TTL_SECONDS = 1800  # 30 minutes — long enough for a UI session, short
                            # enough that stale tokens after a redeploy expire
                            # on their own without operator intervention.


class TokenError(Exception):
    """Raised when a token is malformed, tampered, expired, or stale."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class VerifiedToken:
    queryFingerprint: str
    modelSha: str
    catalogSha: str
    expiresAt: int
    acrcloudCoverSongId: dict | None
    neighbors: dict[str, dict]


def _hmac_key() -> bytes:
    raw = os.getenv("CONTEXT_TOKEN_HMAC_KEY", "").strip()
    if not raw:
        raise TokenError("hmac-key-missing")
    return raw.encode("utf-8")


def issue(
    *,
    query_fingerprint: str,
    model_sha: str,
    catalog_sha: str,
    neighbors: dict[str, dict],
    acrcloud_cover_song_id: dict | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    now: int | None = None,
) -> str:
    """Sign a context token. Returns the encoded `<payload>.<sig>` string.

    `neighbors` is a dict keyed by trackId; each value is the per-neighbor
    fragment of NarrativeContext (title, artist, queryWindow, matchWindow,
    rawCosine, criteria). Top-level acrcloud signal is global per query.
    """
    secret = _hmac_key()
    now_ts = int(now if now is not None else time.time())
    payload = {
        "queryFingerprint": query_fingerprint,
        "modelSha": model_sha,
        "catalogSha": catalog_sha,
        "expiresAt": now_ts + ttl_seconds,
        "acrcloudCoverSongId": acrcloud_cover_song_id,
        "neighbors": neighbors,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body_b64 = base64.urlsafe_b64encode(body).rstrip(b"=").decode("ascii")
    sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return f"{body_b64}.{sig}"


def verify(
    token: str,
    *,
    expected_model_sha: str,
    expected_catalog_sha: str,
    now: int | None = None,
) -> VerifiedToken:
    """Verify signature + expiry + model/catalog hashes; return decoded payload.

    Raises TokenError with a specific code on every failure path:
      - hmac-key-missing       — server isn't configured for tokens
      - malformed              — token wasn't `<body>.<sig>` shape
      - invalid-signature      — HMAC mismatch (tampered or wrong secret)
      - token-expired          — past the embedded expiresAt
      - stale-model            — modelSha doesn't match current load
      - stale-catalog          — catalogSha doesn't match current load
    """
    secret = _hmac_key()

    if not isinstance(token, str) or token.count(".") != 1:
        raise TokenError("malformed")
    body_b64, sig = token.split(".", 1)
    try:
        pad = "=" * (-len(body_b64) % 4)
        body = base64.urlsafe_b64decode(body_b64 + pad)
    except Exception as exc:
        raise TokenError("malformed") from exc

    expected_sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, sig):
        raise TokenError("invalid-signature")

    try:
        payload = json.loads(body)
    except Exception as exc:
        raise TokenError("malformed") from exc

    required = {"queryFingerprint", "modelSha", "catalogSha", "expiresAt", "neighbors"}
    if not isinstance(payload, dict) or not required.issubset(payload.keys()):
        raise TokenError("malformed")

    now_ts = int(now if now is not None else time.time())
    if now_ts > int(payload["expiresAt"]):
        raise TokenError("token-expired")

    if payload["modelSha"] != expected_model_sha:
        raise TokenError("stale-model")
    if payload["catalogSha"] != expected_catalog_sha:
        raise TokenError("stale-catalog")

    return VerifiedToken(
        queryFingerprint=str(payload["queryFingerprint"]),
        modelSha=str(payload["modelSha"]),
        catalogSha=str(payload["catalogSha"]),
        expiresAt=int(payload["expiresAt"]),
        acrcloudCoverSongId=payload.get("acrcloudCoverSongId"),
        neighbors=dict(payload["neighbors"]),
    )


def is_configured() -> bool:
    """True if CONTEXT_TOKEN_HMAC_KEY is set. /neighbors uses this to decide
    whether to attach a token at all."""
    return bool(os.getenv("CONTEXT_TOKEN_HMAC_KEY", "").strip())


def neighbor_context_fragment(
    *,
    track_id: str,
    title: str,
    artist: str | None,
    query_window: tuple[float, float],
    match_window: tuple[float, float],
    raw_cosine: float,
    criteria: list[dict[str, Any]] | None,
    evidence_shared: list[dict[str, Any]] | None = None,
) -> dict:
    """Build the per-neighbor dict the token embeds. Matches the
    NarrativeContext shape Codex's rag_narrative module expects, minus the
    global queryFingerprint + acrcloudCoverSongId (those live at top level)."""
    return {
        "trackId": track_id,
        "title": title,
        "artist": artist,
        "queryWindow": [float(query_window[0]), float(query_window[1])],
        "matchWindow": [float(match_window[0]), float(match_window[1])],
        "rawCosine": float(raw_cosine),
        "criteria": criteria if criteria is not None else [],
        # Evidence Layer: the gated shared descriptors (genre/instrument/mood) for grounding
        # the narrative even when MIR criteria are absent.
        "evidenceShared": evidence_shared if evidence_shared is not None else [],
    }

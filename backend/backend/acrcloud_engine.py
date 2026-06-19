"""ACRCloud commercial second-opinion engine — Cover Song ID + AI Music Detector.

Phase 5 module. Owns:
  1. ACRCloud Identification API auth (HMAC-SHA1 signed multipart POST).
  2. Two independent signal calls per upload:
       - call_cover_song_id(audio_bytes) → normalized Cover Song ID payload.
       - call_ai_music_detector(audio_bytes) → normalized AI Music Detector payload.
  3. The combined `AcrCloudResponse` dataclass that `/neighbors` returns inline.
  4. The feature-flag gate — `ENABLE_ACRCLOUD` env var. When false, both signals
     return `status="disabled"` and the network is never touched.

Per LOCKED_DECISIONS (Q10 + the ACRCloud section):

  - Both signals stay independent — never collapsed into a single verdict.
  - Cover Song ID asks "does this resemble a known composition?"
    AI Music Detector asks "is this AI-generated, likely from Suno/Udio/etc?"
    Different questions, different rows on the ReportCard.
  - Credentials are server-side only. Never returned to the frontend.
  - Trial-gated. Live calls happen during the ACRCloud 14-day free trial;
    after expiration the flag flips to false and cached responses preserve
    the example-chip experience.
  - The normalized adapter shape is fixed (see `to_response_dict` below).

References (Codex: read these before implementing the API auth):
  - https://docs.acrcloud.com/reference/identification-api
  - https://docs.acrcloud.com/reference/console-api/file-scanning/metadata/ai-music-detection
  - https://docs.acrcloud.com/faq/ai-music-detection
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx


# ---------------------------------------------------------------------------
# Configuration — all credentials read from env at process start
# ---------------------------------------------------------------------------

# Master gate. When false, no network calls happen; both signals return
# `status="disabled"`. Default is `false` so a misconfigured deploy never
# accidentally bills the user.
ENABLE_ACRCLOUD: bool = os.getenv("ENABLE_ACRCLOUD", "false").lower() == "true"

# Identification API region host — `identify-us-west-2`, `identify-eu-west-1`,
# `identify-ap-southeast-1`, etc. Defaults to us-west-2 because the trial
# project ships there by default.
ACRCLOUD_HOST: str = os.getenv("ACRCLOUD_HOST", "identify-us-west-2.acrcloud.com")
ACRCLOUD_ACCESS_KEY: str = os.getenv("ACRCLOUD_ACCESS_KEY", "")
ACRCLOUD_ACCESS_SECRET: str = os.getenv("ACRCLOUD_ACCESS_SECRET", "")

# AI Music Detector lives behind a separate console-API endpoint; the
# bearer-token path is documented at the URL above. Codex will fill in.
ACRCLOUD_AI_DETECTOR_URL: str = os.getenv(
    "ACRCLOUD_AI_DETECTOR_URL",
    "",  # set in env once the actual endpoint is confirmed from ACRCloud docs
)
ACRCLOUD_AI_DETECTOR_BEARER: str = os.getenv("ACRCLOUD_AI_DETECTOR_BEARER", "")

# Single HTTP timeout for both APIs. 10s is generous; the live Identification
# API typically returns in 1–3 s.
ACRCLOUD_TIMEOUT_S: float = float(os.getenv("ACRCLOUD_TIMEOUT_S", "10"))


# ---------------------------------------------------------------------------
# Normalized response shapes — LOCKED in LOCKED_DECISIONS Q10
# ---------------------------------------------------------------------------

# Both signals share the same status enum so the frontend handles them uniformly.
Status = Literal["match", "no_match", "timeout", "quota_exceeded", "disabled"]


@dataclass
class CoverSongPayload:
    """One ACRCloud Cover Song ID result, normalized for the frontend.

    `score_semantics` is locked to the string `"acrcloud_music_confidence_70_100"`
    so the frontend can interpret the numeric range without guessing. ACRCloud's
    music-recognition score range is documented as 70–100 for confident matches.
    """

    status: Status
    title: str | None = None
    artist: str | None = None
    score: float | None = None
    score_semantics: str = "acrcloud_music_confidence_70_100"
    external_ids: dict = field(default_factory=dict)


@dataclass
class AiMusicPayload:
    """One ACRCloud AI Music Detector result, normalized for the frontend.

    `likely_source` echoes ACRCloud's documented enum: `suno`, `udio`,
    `sonauto`, `mureka`, `riffusion`, etc., or None when human.

    `ai_probability` is the 0–100 range from ACRCloud's response.
    `verdict` is the categorical winner — `ai_generated`, `human`, `no_vocals`.
    """

    status: Status
    verdict: Literal["ai_generated", "human", "no_vocals"] | None = None
    ai_probability: float | None = None
    likely_source: str | None = None
    score_semantics: str = "acrcloud_ai_probability"


@dataclass
class AcrCloudResponse:
    """Combined ACRCloud response — both signals together. This is what
    `/neighbors` includes under the `acrcloud` key.
    """

    provider: str = "acrcloud"
    cover_song_id: CoverSongPayload | None = None
    ai_music_detector: AiMusicPayload | None = None


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return True if the master gate + Identification-API credentials are set.

    The master gate covers the Identification API (used by Cover Song ID).
    `is_enabled() == True` requires all three of:
      - ENABLE_ACRCLOUD == "true"
      - ACRCLOUD_ACCESS_KEY non-empty
      - ACRCLOUD_ACCESS_SECRET non-empty

    The **AI Music Detector** env vars (`ACRCLOUD_AI_DETECTOR_URL`,
    `ACRCLOUD_AI_DETECTOR_BEARER`) are NOT part of the master gate. They're a
    per-signal gate inside `call_ai_music_detector` — missing them returns
    `status="disabled"` for that one signal while Cover Song ID still runs.
    This split lets a deploy ship Cover Song ID alone if the AI Music Detector
    is unavailable, without disabling the whole ACRCloud feature.
    """
    flag = os.getenv("ENABLE_ACRCLOUD", "false").lower() == "true"
    access_key = os.getenv("ACRCLOUD_ACCESS_KEY", "")
    access_secret = os.getenv("ACRCLOUD_ACCESS_SECRET", "")
    return bool(flag and access_key and access_secret)


def disabled_response() -> AcrCloudResponse:
    """Build the all-disabled response used pre-call when the flag/creds are off.

    Used by `/neighbors` to inline a uniform "Signal unavailable in public demo"
    payload that the frontend's AcrCloudRow component already handles via the
    `status="disabled"` branch.
    """
    return AcrCloudResponse(
        cover_song_id=CoverSongPayload(status="disabled"),
        ai_music_detector=AiMusicPayload(status="disabled"),
    )


def call_for_query(audio_bytes: bytes) -> AcrCloudResponse:
    """Call BOTH ACRCloud signals for a single query upload.

    Args:
        audio_bytes: the audio bytes from the upload, already validated by
            api._validate_upload. Caller pre-truncates to a sensible window
            (LOCKED_DECISIONS says ≤15 s / ≤1 MB is the ACRCloud-recommended
            size; the api.py integration passes the first 15 s of the decode).

    Returns:
        An AcrCloudResponse with both `cover_song_id` and `ai_music_detector`
        populated. Failures degrade gracefully — a timeout on one signal does
        NOT cancel the other. If `is_enabled()` returns False, returns
        `disabled_response()` without any network call.

    Implementation note: the two signals can be called concurrently with
    `httpx.AsyncClient` and `asyncio.gather`; or sequentially with
    `httpx.Client` if you'd rather keep the existing sync request path.
    Pick whichever fits the api.py integration cleanly. Both signals' results
    must land in the same response dict.
    """
    if not is_enabled():
        return disabled_response()

    try:
        cover = call_cover_song_id(audio_bytes)
    except Exception:
        cover = CoverSongPayload(status="timeout")

    try:
        ai = call_ai_music_detector(audio_bytes)
    except Exception:
        ai = AiMusicPayload(status="timeout")

    return AcrCloudResponse(cover_song_id=cover, ai_music_detector=ai)


def call_cover_song_id(audio_bytes: bytes) -> CoverSongPayload:
    """Call ACRCloud Identification API and normalize the response.

    Args:
        audio_bytes: validated audio (≤1 MB / ~15 s preferred per ACRCloud).

    Returns:
        CoverSongPayload with status in {"match", "no_match", "timeout",
        "quota_exceeded"}. Maps ACRCloud's response shape to the normalized
        adapter:
          - 200 + status.code == 0 + metadata.music[0] populated → "match"
            with title / artist / score / external_ids
          - 200 + status.code != 0 (e.g. 1001 "No result") → "no_match"
          - HTTPStatusError 4xx with quota-related body → "quota_exceeded"
          - httpx.TimeoutException → "timeout"

    The signing protocol:
        string_to_sign = "POST\\n/v1/identify\\n{access_key}\\naudio\\n1\\n{timestamp}"
        signature      = base64(hmac_sha1(access_secret, string_to_sign))

    See the docs link in the module docstring for the exact field encoding.
    """
    access_key = os.getenv("ACRCLOUD_ACCESS_KEY", ACRCLOUD_ACCESS_KEY)
    access_secret = os.getenv("ACRCLOUD_ACCESS_SECRET", ACRCLOUD_ACCESS_SECRET)
    host = os.getenv("ACRCLOUD_HOST", ACRCLOUD_HOST)
    timestamp = str(int(time.time()))
    string_to_sign = f"POST\n/v1/identify\n{access_key}\naudio\n1\n{timestamp}"
    signature = base64.b64encode(
        hmac.new(
            access_secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha1,
        ).digest()
    ).decode("ascii")

    data = {
        "access_key": access_key,
        "data_type": "audio",
        "signature_version": "1",
        "signature": signature,
        "timestamp": timestamp,
        "sample_bytes": str(len(audio_bytes)),
    }
    files = {"sample": ("sample.wav", io.BytesIO(audio_bytes), "audio/wav")}
    try:
        with httpx.Client(timeout=ACRCLOUD_TIMEOUT_S) as client:
            response = client.post(f"https://{host}/v1/identify", data=data, files=files)
            response.raise_for_status()
    except httpx.TimeoutException:
        return CoverSongPayload(status="timeout")
    except httpx.HTTPStatusError as exc:
        return CoverSongPayload(status="quota_exceeded" if _looks_quota_limited(exc.response) else "no_match")

    payload = response.json()
    if payload.get("status", {}).get("code") != 0:
        return CoverSongPayload(status="no_match")
    music = payload.get("metadata", {}).get("music") or []
    if not music:
        return CoverSongPayload(status="no_match")
    first = music[0]
    artists = first.get("artists") or []
    artist = artists[0].get("name") if artists and isinstance(artists[0], dict) else None
    return CoverSongPayload(
        status="match",
        title=first.get("title"),
        artist=artist,
        score=first.get("score"),
        external_ids=first.get("external_ids") or {},
    )


def call_ai_music_detector(audio_bytes: bytes) -> AiMusicPayload:
    """Call ACRCloud AI Music Detector and normalize the response.

    Args:
        audio_bytes: validated audio (≤1 MB / ~15 s preferred).

    Returns:
        AiMusicPayload with status in {"match", "no_match", "timeout",
        "quota_exceeded"}. Maps the ACRCloud response to the normalized adapter.
        The verdict + likely_source + ai_probability fields are documented at
        https://docs.acrcloud.com/reference/console-api/file-scanning/metadata/ai-music-detection.

    Implementation note: AI Music Detector is a console-API endpoint (not the
    Identification API), so auth uses a bearer token (`ACRCLOUD_AI_DETECTOR_BEARER`)
    rather than the HMAC-SHA1 signature. Confirm the exact path against the
    ACRCloud docs before sending.

    Per-signal gate: when `ACRCLOUD_AI_DETECTOR_URL` or
    `ACRCLOUD_AI_DETECTOR_BEARER` is empty, return `status="disabled"` without
    a network call. This is independent of `is_enabled()` — it lets Cover Song
    ID ship even when AI Music Detector isn't configured.
    """
    url = os.getenv("ACRCLOUD_AI_DETECTOR_URL", ACRCLOUD_AI_DETECTOR_URL)
    bearer = os.getenv("ACRCLOUD_AI_DETECTOR_BEARER", ACRCLOUD_AI_DETECTOR_BEARER)
    if not url or not bearer:
        return AiMusicPayload(status="disabled")

    headers = {"Authorization": f"Bearer {bearer}"}
    files = {"file": ("sample.wav", io.BytesIO(audio_bytes), "audio/wav")}
    try:
        with httpx.Client(timeout=ACRCLOUD_TIMEOUT_S) as client:
            response = client.post(url, headers=headers, files=files)
            response.raise_for_status()
    except httpx.TimeoutException:
        return AiMusicPayload(status="timeout")
    except httpx.HTTPStatusError as exc:
        return AiMusicPayload(status="quota_exceeded" if _looks_quota_limited(exc.response) else "no_match")

    payload = response.json()
    verdict = payload.get("prediction") or payload.get("verdict")
    if verdict not in {"ai_generated", "human", "no_vocals"}:
        return AiMusicPayload(status="no_match")
    return AiMusicPayload(
        status="match",
        verdict=verdict,
        ai_probability=payload.get("ai_probability"),
        likely_source=payload.get("likely_source"),
    )


def to_response_dict(resp: AcrCloudResponse) -> dict:
    """Convert AcrCloudResponse → the locked JSON shape `/neighbors` returns.

    Locked structure (LOCKED_DECISIONS Q10):

        {
          "provider": "acrcloud",
          "coverSongId": {
            "status": "match|no_match|timeout|quota_exceeded|disabled",
            "title": "...",
            "artist": "...",
            "score": 88,
            "scoreSemantics": "acrcloud_music_confidence_70_100",
            "externalIds": {}
          },
          "aiMusicDetector": {
            "status": "match|no_match|timeout|quota_exceeded|disabled",
            "verdict": "ai_generated|human|no_vocals",
            "ai_probability": 0,
            "likely_source": "suno|udio|sonauto|mureka|riffusion|null",
            "scoreSemantics": "acrcloud_ai_probability"
          }
        }

    Note the camelCase keys at the wire — Python snake_case stays internal.
    """
    cover = resp.cover_song_id or CoverSongPayload(status="disabled")
    ai = resp.ai_music_detector or AiMusicPayload(status="disabled")
    return {
        "provider": resp.provider,
        "coverSongId": {
            "status": cover.status,
            "title": cover.title,
            "artist": cover.artist,
            "score": cover.score,
            "scoreSemantics": cover.score_semantics,
            "externalIds": cover.external_ids,
        },
        "aiMusicDetector": {
            "status": ai.status,
            "verdict": ai.verdict,
            "ai_probability": ai.ai_probability,
            "likely_source": ai.likely_source,
            "scoreSemantics": ai.score_semantics,
        },
    }


def _looks_quota_limited(response: httpx.Response) -> bool:
    if response.status_code == 429:
        return True
    body = response.text.lower()
    return "quota" in body or "rate limit" in body or "too many requests" in body


# ---------------------------------------------------------------------------
# Pre-cached responses for example chips (Phase 5 minimal; expand in Phase 6)
# ---------------------------------------------------------------------------


def load_cached_for_example(example_id: str) -> AcrCloudResponse | None:
    """Look up a pre-computed ACRCloud response for an example-chip track.

    Pre-caching plan: during the ACRCloud trial window, a one-time script
    (`backend/backend/scripts/cache_acrcloud_examples.py`, Phase 5.5) calls
    both signals for each example track and writes the normalized payloads
    into `quality-scorer/public/corpus/examples.json` so the demo continues
    showing real ACRCloud results after the trial expires.

    For Phase 5, this stub returns None — the api.py integration falls back
    to the live call (or `disabled_response`) when no cache exists.

    Args:
        example_id: the example chip's `id` field, e.g. "ex_suno_pop_001".

    Returns:
        An AcrCloudResponse if a cached payload exists in `examples.json`,
        else None. The api.py code path prefers the cache when available,
        only making a live call when it isn't.
    """
    # TODO(codex): implement after Phase 5.5 produces cached entries.
    # For now, return None — Phase 5 integration treats any example as live-call.
    return None

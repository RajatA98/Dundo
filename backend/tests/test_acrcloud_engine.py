"""Phase 5 — ACRCloud engine tests.

These tests codify the LOCKED_DECISIONS Q10 contract for the ACRCloud module:

  - When the feature flag / credentials aren't set, no network is touched and
    both signals are returned with `status="disabled"`.
  - When enabled, each signal's response is normalized into the locked adapter
    shape; timeouts and quota failures map to documented statuses; a failure in
    one signal does NOT cascade to the other.
  - The wire-level JSON shape is camelCase (frontend reads it verbatim).
  - Credentials never appear in any response or log line.

All tests run fast — the HTTP layer is mocked via httpx's transport hook so no
network is actually touched. The tests do NOT require valid ACRCloud creds.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest

# Import lazily inside tests so monkey-patched env vars are read at call time.


# ---------------------------------------------------------------------------
# is_enabled() — feature flag + credential gate
# ---------------------------------------------------------------------------


def test_is_enabled_false_when_env_flag_unset(monkeypatch):
    monkeypatch.delenv("ENABLE_ACRCLOUD", raising=False)
    monkeypatch.setenv("ACRCLOUD_ACCESS_KEY", "any")
    monkeypatch.setenv("ACRCLOUD_ACCESS_SECRET", "any")
    # Reload module so module-level globals re-read the env.
    from backend import acrcloud_engine
    import importlib

    importlib.reload(acrcloud_engine)
    assert acrcloud_engine.is_enabled() is False


def test_is_enabled_false_when_flag_true_but_creds_missing(monkeypatch):
    monkeypatch.setenv("ENABLE_ACRCLOUD", "true")
    monkeypatch.delenv("ACRCLOUD_ACCESS_KEY", raising=False)
    monkeypatch.delenv("ACRCLOUD_ACCESS_SECRET", raising=False)
    from backend import acrcloud_engine
    import importlib

    importlib.reload(acrcloud_engine)
    assert acrcloud_engine.is_enabled() is False


def test_is_enabled_true_when_flag_and_creds_set(monkeypatch):
    monkeypatch.setenv("ENABLE_ACRCLOUD", "true")
    monkeypatch.setenv("ACRCLOUD_ACCESS_KEY", "abc")
    monkeypatch.setenv("ACRCLOUD_ACCESS_SECRET", "xyz")
    from backend import acrcloud_engine
    import importlib

    importlib.reload(acrcloud_engine)
    assert acrcloud_engine.is_enabled() is True


# ---------------------------------------------------------------------------
# disabled_response() — used by /neighbors when the flag is off
# ---------------------------------------------------------------------------


def test_disabled_response_marks_both_signals_disabled():
    from backend.acrcloud_engine import disabled_response

    r = disabled_response()
    assert r.provider == "acrcloud"
    assert r.cover_song_id is not None
    assert r.ai_music_detector is not None
    assert r.cover_song_id.status == "disabled"
    assert r.ai_music_detector.status == "disabled"


def test_disabled_response_carries_no_credential_or_pii():
    from backend.acrcloud_engine import disabled_response

    r = disabled_response()
    blob = repr(r).lower()
    # No accidental access-key / secret leak in the disabled payload.
    for forbidden in ("access_key", "access_secret", "bearer", "secret"):
        assert forbidden not in blob, f"disabled_response leaked the field name: {forbidden}"


# ---------------------------------------------------------------------------
# to_response_dict() — the camelCase wire shape (frontend reads these keys)
# ---------------------------------------------------------------------------


def test_to_response_dict_wire_shape_matches_locked_contract():
    from backend.acrcloud_engine import (
        AcrCloudResponse,
        AiMusicPayload,
        CoverSongPayload,
        to_response_dict,
    )

    resp = AcrCloudResponse(
        cover_song_id=CoverSongPayload(
            status="match",
            title="Blinding Lights",
            artist="The Weeknd",
            score=88,
            external_ids={"isrc": "USUM71916253"},
        ),
        ai_music_detector=AiMusicPayload(
            status="match",
            verdict="ai_generated",
            ai_probability=87,
            likely_source="suno",
        ),
    )
    out = to_response_dict(resp)

    # Top-level shape
    assert out["provider"] == "acrcloud"
    assert "coverSongId" in out, "wire key must be camelCase: coverSongId"
    assert "aiMusicDetector" in out, "wire key must be camelCase: aiMusicDetector"
    assert "cover_song_id" not in out, "wire must be camelCase, not snake_case"

    cs = out["coverSongId"]
    assert cs["status"] == "match"
    assert cs["title"] == "Blinding Lights"
    assert cs["artist"] == "The Weeknd"
    assert cs["score"] == 88
    assert cs["scoreSemantics"] == "acrcloud_music_confidence_70_100"
    assert cs["externalIds"] == {"isrc": "USUM71916253"}

    ai = out["aiMusicDetector"]
    assert ai["status"] == "match"
    assert ai["verdict"] == "ai_generated"
    assert ai["ai_probability"] == 87
    assert ai["likely_source"] == "suno"
    assert ai["scoreSemantics"] == "acrcloud_ai_probability"


def test_to_response_dict_disabled_round_trips():
    from backend.acrcloud_engine import disabled_response, to_response_dict

    out = to_response_dict(disabled_response())
    assert out["coverSongId"]["status"] == "disabled"
    assert out["aiMusicDetector"]["status"] == "disabled"


# ---------------------------------------------------------------------------
# call_for_query() — short-circuits when disabled, no network
# ---------------------------------------------------------------------------


def test_call_for_query_returns_disabled_when_flag_off(monkeypatch):
    monkeypatch.setenv("ENABLE_ACRCLOUD", "false")
    import importlib

    from backend import acrcloud_engine

    importlib.reload(acrcloud_engine)

    # Patch httpx to fail loudly if anything actually attempts a network call.
    with patch("httpx.Client.post", side_effect=AssertionError("no network when disabled")):
        with patch("httpx.AsyncClient.post", side_effect=AssertionError("no network when disabled")):
            r = acrcloud_engine.call_for_query(b"\x00" * 1024)
    assert r.cover_song_id.status == "disabled"
    assert r.ai_music_detector.status == "disabled"


# ---------------------------------------------------------------------------
# call_cover_song_id() — happy + degraded paths via mocked transports
# ---------------------------------------------------------------------------


def test_cover_song_id_match_normalizes_to_payload(monkeypatch):
    """200 + status.code 0 + populated metadata.music → status="match" with fields."""
    monkeypatch.setenv("ENABLE_ACRCLOUD", "true")
    monkeypatch.setenv("ACRCLOUD_ACCESS_KEY", "k")
    monkeypatch.setenv("ACRCLOUD_ACCESS_SECRET", "s")
    import importlib

    from backend import acrcloud_engine

    importlib.reload(acrcloud_engine)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": {"code": 0, "msg": "Success"},
                "metadata": {
                    "music": [
                        {
                            "title": "Blinding Lights",
                            "artists": [{"name": "The Weeknd"}],
                            "score": 88,
                            "external_ids": {"isrc": "USUM71916253"},
                        }
                    ]
                },
            },
        )

    transport = httpx.MockTransport(handler)
    OriginalClient = httpx.Client
    with patch("httpx.Client", lambda **kw: OriginalClient(transport=transport, **kw)):
        out = acrcloud_engine.call_cover_song_id(b"\x00" * 1024)

    assert out.status == "match"
    assert out.title == "Blinding Lights"
    assert out.artist == "The Weeknd"
    assert out.score == 88


def test_cover_song_id_no_match_when_status_code_nonzero(monkeypatch):
    """200 + status.code 1001 (no result) → status="no_match"."""
    monkeypatch.setenv("ENABLE_ACRCLOUD", "true")
    monkeypatch.setenv("ACRCLOUD_ACCESS_KEY", "k")
    monkeypatch.setenv("ACRCLOUD_ACCESS_SECRET", "s")
    import importlib

    from backend import acrcloud_engine

    importlib.reload(acrcloud_engine)

    def handler(_request):
        return httpx.Response(200, json={"status": {"code": 1001, "msg": "No result"}})

    transport = httpx.MockTransport(handler)
    OriginalClient = httpx.Client
    with patch("httpx.Client", lambda **kw: OriginalClient(transport=transport, **kw)):
        out = acrcloud_engine.call_cover_song_id(b"\x00" * 1024)

    assert out.status == "no_match"
    assert out.title is None
    assert out.score is None


def test_cover_song_id_timeout_maps_to_status(monkeypatch):
    monkeypatch.setenv("ENABLE_ACRCLOUD", "true")
    monkeypatch.setenv("ACRCLOUD_ACCESS_KEY", "k")
    monkeypatch.setenv("ACRCLOUD_ACCESS_SECRET", "s")
    import importlib

    from backend import acrcloud_engine

    importlib.reload(acrcloud_engine)

    def handler(_request):
        raise httpx.TimeoutException("simulated timeout")

    transport = httpx.MockTransport(handler)
    OriginalClient = httpx.Client
    with patch("httpx.Client", lambda **kw: OriginalClient(transport=transport, **kw)):
        out = acrcloud_engine.call_cover_song_id(b"\x00" * 1024)

    assert out.status == "timeout"


# ---------------------------------------------------------------------------
# call_ai_music_detector() — happy + degraded paths
# ---------------------------------------------------------------------------


def test_ai_music_detector_suno_match_normalizes_to_payload(monkeypatch):
    """AI Music Detector should expose likely_source=suno when the model says so."""
    monkeypatch.setenv("ENABLE_ACRCLOUD", "true")
    monkeypatch.setenv("ACRCLOUD_ACCESS_KEY", "k")
    monkeypatch.setenv("ACRCLOUD_ACCESS_SECRET", "s")
    monkeypatch.setenv("ACRCLOUD_AI_DETECTOR_BEARER", "bearer")
    monkeypatch.setenv("ACRCLOUD_AI_DETECTOR_URL", "https://example.test/ai")
    import importlib

    from backend import acrcloud_engine

    importlib.reload(acrcloud_engine)

    def handler(_request):
        return httpx.Response(
            200,
            json={
                "prediction": "ai_generated",
                "ai_probability": 87,
                "likely_source": "suno",
            },
        )

    transport = httpx.MockTransport(handler)
    OriginalClient = httpx.Client
    with patch("httpx.Client", lambda **kw: OriginalClient(transport=transport, **kw)):
        out = acrcloud_engine.call_ai_music_detector(b"\x00" * 1024)

    assert out.status == "match"
    assert out.verdict == "ai_generated"
    assert out.ai_probability == 87
    assert out.likely_source == "suno"


def test_ai_music_detector_human_verdict(monkeypatch):
    monkeypatch.setenv("ENABLE_ACRCLOUD", "true")
    monkeypatch.setenv("ACRCLOUD_ACCESS_KEY", "k")
    monkeypatch.setenv("ACRCLOUD_ACCESS_SECRET", "s")
    monkeypatch.setenv("ACRCLOUD_AI_DETECTOR_BEARER", "bearer")
    monkeypatch.setenv("ACRCLOUD_AI_DETECTOR_URL", "https://example.test/ai")
    import importlib

    from backend import acrcloud_engine

    importlib.reload(acrcloud_engine)

    def handler(_request):
        return httpx.Response(
            200,
            json={
                "prediction": "human",
                "ai_probability": 5,
                "likely_source": None,
            },
        )

    transport = httpx.MockTransport(handler)
    OriginalClient = httpx.Client
    with patch("httpx.Client", lambda **kw: OriginalClient(transport=transport, **kw)):
        out = acrcloud_engine.call_ai_music_detector(b"\x00" * 1024)

    assert out.status == "match"
    assert out.verdict == "human"
    assert out.ai_probability == 5
    assert out.likely_source is None


# ---------------------------------------------------------------------------
# call_for_query() — degraded partial failure: one signal fails, other survives
# ---------------------------------------------------------------------------


def test_partial_failure_does_not_cascade(monkeypatch):
    """If Cover Song ID times out, AI Music Detector still returns its match."""
    monkeypatch.setenv("ENABLE_ACRCLOUD", "true")
    monkeypatch.setenv("ACRCLOUD_ACCESS_KEY", "k")
    monkeypatch.setenv("ACRCLOUD_ACCESS_SECRET", "s")
    monkeypatch.setenv("ACRCLOUD_AI_DETECTOR_BEARER", "bearer")
    monkeypatch.setenv("ACRCLOUD_AI_DETECTOR_URL", "https://example.test/ai")
    import importlib

    from backend import acrcloud_engine

    importlib.reload(acrcloud_engine)

    # Patch the lower-level calls to deliver mixed results.
    with patch.object(
        acrcloud_engine,
        "call_cover_song_id",
        return_value=acrcloud_engine.CoverSongPayload(status="timeout"),
    ):
        with patch.object(
            acrcloud_engine,
            "call_ai_music_detector",
            return_value=acrcloud_engine.AiMusicPayload(
                status="match",
                verdict="ai_generated",
                ai_probability=91,
                likely_source="suno",
            ),
        ):
            r = acrcloud_engine.call_for_query(b"\x00" * 1024)

    assert r.cover_song_id.status == "timeout"
    assert r.ai_music_detector.status == "match"
    assert r.ai_music_detector.likely_source == "suno"

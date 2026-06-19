"""Endpoint-level tests for POST /narrative.

These tests mock `backend.rag_narrative.generate_narrative` directly so they
don't depend on Codex's module being fully implemented yet. The contract is
fixed (CODEX_RAG_NARRATIVE_IMPLEMENTATION.md). Codex's own module-level
fixtures live in `test_rag_narrative.py` — these endpoint tests cover the
HTTP layer: token verification, gating, and error mapping.

Coverage:
  - 503 narrative-disabled  — OPENAI_API_KEY missing
  - 503 narrative-disabled  — CONTEXT_TOKEN_HMAC_KEY missing
  - 422 unsupported-mode    — bad mode string
  - 401 invalid-token       — tampered signature
  - 400 malformed-token     — bad shape
  - 412 token-expired       — past expiresAt
  - 412 stale-token         — modelSha or catalogSha mismatch
  - 404 not-in-context      — trackId not in token
  - 200 happy path          — valid token + mocked generate_narrative
  - 200 LowConfidence       — gate short-circuits
  - 200 NarrativeUnavailable — citation validation rejects
"""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


_TEST_HMAC_SECRET = "test-hmac-secret-not-for-production"
_TEST_OPENAI_KEY = "sk-test-not-real"
_TEST_MODEL_SHA = "model-sha-test"
_TEST_CATALOG_SHA = "catalog-sha-test"


def _install_rag_narrative_stub():
    """Install a stand-in `backend.rag_narrative` module if Codex's hasn't
    landed yet. The stub exposes the Pydantic models + a `generate_narrative`
    callable so the endpoint can import and instantiate them; tests patch
    `generate_narrative` to control behavior. If Codex's real module is
    present, do nothing — the real module takes precedence.

    NOTE: when Codex's `rag_narrative.py` lands in the tree, this stub is
    inert (the real module is imported first by Python's import system).
    """
    try:
        from backend import rag_narrative  # noqa: F401
        return  # real module already importable
    except ImportError:
        pass

    from typing import Literal
    from pydantic import BaseModel

    NarrativeMode = Literal["whySimilar", "creatorAdvice"]
    CriterionId = Literal["tempo", "key", "harmonic", "timbre"]

    class CriterionContext(BaseModel):
        id: CriterionId
        queryValue: float | str | dict
        matchValue: float | str | dict
        agreement: float
        label: str

    class NarrativeContext(BaseModel):
        queryFingerprint: str
        trackId: str
        title: str
        artist: str | None = None
        queryWindow: tuple[float, float]
        matchWindow: tuple[float, float]
        rawCosine: float
        criteria: list[CriterionContext]
        acrcloudCoverSongId: dict | None = None

    class StructuredCitation(BaseModel):
        trackId: str
        side: Literal["query", "match"]
        timestampRange: tuple[float, float]
        criterionIds: list[CriterionId]
        citedValues: dict[str, float | str]

    class NarrativeResponse(BaseModel):
        kind: Literal["narrative"] = "narrative"
        mode: NarrativeMode
        prose: str
        citations: list[StructuredCitation]

    class LowConfidence(BaseModel):
        kind: Literal["low_confidence"] = "low_confidence"
        reason: str

    class NarrativeUnavailable(BaseModel):
        kind: Literal["unavailable"] = "unavailable"
        reason: str

    def generate_narrative(context, mode, *, model_sha, catalog_sha, model_id="gpt-4o-mini", openai_client=None):
        # Stubs are patched in each test; this default is never called.
        raise NotImplementedError("test stub — patch generate_narrative in each test")

    SYSTEM_PROMPTS = {
        "whySimilar": "stub prompt — whySimilar",
        "creatorAdvice": "stub prompt — creatorAdvice",
    }

    def cache_key(context, mode, *, model_sha, catalog_sha, model_id):
        return f"stub-cache-key:{mode}:{model_sha}:{catalog_sha}:{model_id}:{context.trackId}"

    stub = types.ModuleType("backend.rag_narrative")
    stub.CriterionContext = CriterionContext
    stub.NarrativeContext = NarrativeContext
    stub.StructuredCitation = StructuredCitation
    stub.NarrativeResponse = NarrativeResponse
    stub.LowConfidence = LowConfidence
    stub.NarrativeUnavailable = NarrativeUnavailable
    stub.generate_narrative = generate_narrative
    stub.SYSTEM_PROMPTS = SYSTEM_PROMPTS
    stub.cache_key = cache_key
    sys.modules["backend.rag_narrative"] = stub


_install_rag_narrative_stub()


# Now that the module exists in sys.modules, we can import its types for use
# in tests (whether the real module or the stub).
from backend import rag_narrative  # noqa: E402


@pytest.fixture
def configured_env(monkeypatch):
    """Both keys set, model + catalog SHA stubbed to test values.

    Stubs `_load_corpus` and `muq_engine.load` so the FastAPI lifespan can
    fire without overwriting the model_sha/catalog_sha values we set here
    and without spending 30 s loading MuQ-MuLan. The endpoint logic doesn't
    care whether a real corpus is loaded — only that the module globals are
    set.
    """
    monkeypatch.setenv("CONTEXT_TOKEN_HMAC_KEY", _TEST_HMAC_SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", _TEST_OPENAI_KEY)
    from backend import api, muq_engine
    monkeypatch.setattr(api, "_load_corpus", lambda: None, raising=True)
    monkeypatch.setattr(muq_engine, "load", lambda: None, raising=True)
    monkeypatch.setattr(api, "_model_sha", _TEST_MODEL_SHA, raising=True)
    monkeypatch.setattr(api, "_catalog_sha", _TEST_CATALOG_SHA, raising=True)
    yield


def _issue_test_token(neighbors: dict | None = None, *, ttl_seconds: int = 1800, now: int | None = None):
    from backend import context_token
    return context_token.issue(
        query_fingerprint="a" * 64,
        model_sha=_TEST_MODEL_SHA,
        catalog_sha=_TEST_CATALOG_SHA,
        neighbors=neighbors or _sample_neighbors_fragment(),
        acrcloud_cover_song_id=None,
        ttl_seconds=ttl_seconds,
        now=now,
    )


def _sample_neighbors_fragment() -> dict:
    return {
        "tier1:itunes:380907765": {
            "trackId": "tier1:itunes:380907765",
            "title": "Take On Me",
            "artist": "a-ha",
            "queryWindow": [20.0, 30.0],
            "matchWindow": [10.0, 20.0],
            "rawCosine": 0.881,
            "criteria": [
                {
                    "id": "tempo",
                    "queryValue": 100.0,
                    "matchValue": 100.5,
                    "agreement": 1.0,
                    "label": "same tempo",
                }
            ],
        }
    }


def _client():
    from backend.api import app
    return TestClient(app)


# ----- gating tests --------------------------------------------------------


def test_missing_openai_key_returns_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CONTEXT_TOKEN_HMAC_KEY", _TEST_HMAC_SECRET)
    with _client() as c:
        r = c.post(
            "/narrative",
            json={"contextToken": "anything", "trackId": "x", "mode": "whySimilar"},
        )
    assert r.status_code == 503
    assert r.json() == {"error": "narrative-disabled"}


def test_missing_hmac_key_returns_503(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", _TEST_OPENAI_KEY)
    monkeypatch.delenv("CONTEXT_TOKEN_HMAC_KEY", raising=False)
    with _client() as c:
        r = c.post(
            "/narrative",
            json={"contextToken": "anything", "trackId": "x", "mode": "whySimilar"},
        )
    assert r.status_code == 503
    assert r.json() == {"error": "narrative-disabled"}


def test_unsupported_mode_returns_422(configured_env):
    token = _issue_test_token()
    with _client() as c:
        r = c.post(
            "/narrative",
            json={"contextToken": token, "trackId": "tier1:itunes:380907765", "mode": "explain"},
        )
    assert r.status_code == 422
    assert r.json() == {"error": "unsupported-mode"}


# ----- token verification tests --------------------------------------------


def test_invalid_signature_returns_401(configured_env):
    token = _issue_test_token()
    body, sig = token.split(".", 1)
    bad_char = "0" if sig[-1] != "0" else "1"
    bad_token = f"{body}.{sig[:-1]}{bad_char}"
    with _client() as c:
        r = c.post(
            "/narrative",
            json={"contextToken": bad_token, "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
        )
    assert r.status_code == 401
    assert r.json() == {"error": "invalid-token"}


def test_malformed_token_returns_400(configured_env):
    with _client() as c:
        r = c.post(
            "/narrative",
            json={"contextToken": "no-dot-here", "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
        )
    assert r.status_code == 400
    assert r.json() == {"error": "malformed-token"}


def test_expired_token_returns_412(configured_env):
    past = int(time.time()) - 7200
    token = _issue_test_token(ttl_seconds=60, now=past)
    with _client() as c:
        r = c.post(
            "/narrative",
            json={"contextToken": token, "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
        )
    assert r.status_code == 412
    assert r.json() == {"error": "token-expired"}


def test_stale_model_sha_returns_412(configured_env, monkeypatch):
    token = _issue_test_token()
    from backend import api
    monkeypatch.setattr(api, "_model_sha", "different-model-sha", raising=True)
    with _client() as c:
        r = c.post(
            "/narrative",
            json={"contextToken": token, "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
        )
    assert r.status_code == 412
    assert r.json() == {"error": "stale-token"}


def test_stale_catalog_sha_returns_412(configured_env, monkeypatch):
    token = _issue_test_token()
    from backend import api
    monkeypatch.setattr(api, "_catalog_sha", "different-catalog-sha", raising=True)
    with _client() as c:
        r = c.post(
            "/narrative",
            json={"contextToken": token, "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
        )
    assert r.status_code == 412
    assert r.json() == {"error": "stale-token"}


def test_trackid_not_in_token_returns_404(configured_env):
    token = _issue_test_token()
    with _client() as c:
        r = c.post(
            "/narrative",
            json={"contextToken": token, "trackId": "tier1:itunes:000000000", "mode": "whySimilar"},
        )
    assert r.status_code == 404
    assert r.json() == {"error": "not-in-context"}


# ----- happy path + result variants ----------------------------------------


def test_valid_token_returns_narrative(configured_env):
    token = _issue_test_token()
    expected = rag_narrative.NarrativeResponse(
        mode="whySimilar",
        prose="The tracks share the same tempo and key.",
        citations=[
            rag_narrative.StructuredCitation(
                trackId="tier1:itunes:380907765",
                side="query",
                timestampRange=(20.0, 30.0),
                criterionIds=["tempo"],
                citedValues={"tempo.queryValue": 100.0},
            )
        ],
    )
    with patch("backend.rag_narrative.generate_narrative", return_value=expected) as mock_gen:
        with _client() as c:
            r = c.post(
                "/narrative",
                json={"contextToken": token, "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
            )

    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "narrative"
    assert body["mode"] == "whySimilar"
    assert body["prose"].startswith("The tracks share")
    assert body["citations"][0]["trackId"] == "tier1:itunes:380907765"

    # generate_narrative called with the rebuilt context and right mode.
    mock_gen.assert_called_once()
    call_args = mock_gen.call_args
    ctx_arg = call_args.args[0]
    assert ctx_arg.trackId == "tier1:itunes:380907765"
    assert ctx_arg.queryFingerprint == "a" * 64
    assert call_args.args[1] == "whySimilar"
    assert call_args.kwargs["model_sha"] == _TEST_MODEL_SHA
    assert call_args.kwargs["catalog_sha"] == _TEST_CATALOG_SHA


def test_low_confidence_returns_typed_payload(configured_env):
    token = _issue_test_token()
    result = rag_narrative.LowConfidence(reason="weak-evidence")
    with patch("backend.rag_narrative.generate_narrative", return_value=result):
        with _client() as c:
            r = c.post(
                "/narrative",
                json={"contextToken": token, "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
            )
    assert r.status_code == 200
    assert r.json() == {"kind": "low_confidence", "reason": "weak-evidence"}


def test_narrative_unavailable_returns_typed_payload(configured_env):
    token = _issue_test_token()
    result = rag_narrative.NarrativeUnavailable(reason="malformed-llm-output")
    with patch("backend.rag_narrative.generate_narrative", return_value=result):
        with _client() as c:
            r = c.post(
                "/narrative",
                json={"contextToken": token, "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
            )
    assert r.status_code == 200
    assert r.json() == {"kind": "unavailable", "reason": "malformed-llm-output"}


# ----- /narrative/stats endpoint -------------------------------------------


def test_stats_endpoint_returns_telemetry_snapshot(configured_env):
    """A real /narrative call should move counters that /narrative/stats reflects."""
    from backend import narrative_telemetry
    narrative_telemetry.reset()

    token = _issue_test_token()
    expected = rag_narrative.NarrativeResponse(
        mode="whySimilar",
        prose="Tracks share tempo and key.",
        citations=[
            rag_narrative.StructuredCitation(
                trackId="tier1:itunes:380907765",
                side="query",
                timestampRange=(20.0, 30.0),
                criterionIds=["tempo"],
                citedValues={"tempo.queryValue": 100.0},
            )
        ],
    )
    with patch("backend.rag_narrative.generate_narrative", return_value=expected):
        with _client() as c:
            c.post(
                "/narrative",
                json={"contextToken": token, "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
            )
            stats_resp = c.get("/narrative/stats")

    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_calls"] == 1
    assert stats["by_mode"]["whySimilar"] == 1
    assert stats["by_kind"]["narrative"] == 1
    assert stats["openai_calls"] == 1
    assert stats["latency_ms"]["sample_n"] == 1
    assert stats["cost_cents_estimate"] >= 0  # at least the prompt-chars contribution


def test_stats_endpoint_works_without_calls(configured_env):
    """Empty stats endpoint should still return a sane snapshot, not error."""
    from backend import narrative_telemetry
    narrative_telemetry.reset()
    with _client() as c:
        r = c.get("/narrative/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_calls"] == 0
    assert body["latency_ms"]["sample_n"] == 0
    assert body["latency_ms"]["p50"] is None


def test_stats_endpoint_tracks_low_confidence_and_errors(configured_env):
    """Mix of success / gate / token error should land in correct buckets."""
    from backend import narrative_telemetry
    narrative_telemetry.reset()

    token = _issue_test_token()
    low_conf = rag_narrative.LowConfidence(reason="weak-evidence")

    with patch("backend.rag_narrative.generate_narrative", return_value=low_conf):
        with _client() as c:
            # Call 1: gate short-circuits via low_confidence
            c.post(
                "/narrative",
                json={"contextToken": token, "trackId": "tier1:itunes:380907765", "mode": "whySimilar"},
            )
            # Call 2: unknown trackId → 404 not-in-context
            c.post(
                "/narrative",
                json={"contextToken": token, "trackId": "tier1:itunes:000000000", "mode": "whySimilar"},
            )
            stats = c.get("/narrative/stats").json()

    assert stats["total_calls"] == 2
    assert stats["by_kind"]["low_confidence"] == 1
    assert stats["gate_short_circuits"] == 1
    assert stats["by_error"]["not-in-context"] == 1

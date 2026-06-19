"""End-to-end wire-shape integration test for /neighbors → contextToken → /narrative.

The per-endpoint tests verify each side in isolation. This test exercises
the contract between them: the fragment shape api.py writes into the token
during /neighbors must be loadable back into a Pydantic NarrativeContext
during /narrative. A drift between the two sides — a renamed field, a
silently dropped key, a type mismatch — would silently break /narrative
even though both per-endpoint test suites pass.

Approach: simulate the exact code paths /neighbors uses (the
`_criteria_to_token_fragment` reshaper + `context_token.neighbor_context_fragment`
+ `context_token.issue`) → verify → build `NarrativeContext` exactly as
`/narrative` does. If any field is misnamed, the Pydantic validation at
the rebuild step will fail.
"""

from __future__ import annotations

import os

import pytest

# Ensure rag_narrative is importable (Codex's module ships with Commit A).
from backend import context_token, rag_narrative
from backend.api import _criteria_to_token_fragment


_TEST_SECRET = "e2e-test-secret"


@pytest.fixture(autouse=True)
def hmac_env(monkeypatch):
    monkeypatch.setenv("CONTEXT_TOKEN_HMAC_KEY", _TEST_SECRET)


def _sample_neighbors_block() -> dict:
    """Shape of the per-neighbor `criteria` block that /neighbors writes
    into its response (and from which the api.py reshaper builds the
    token fragment)."""
    return {
        "tempo": {
            "queryValue": 100.0,
            "matchValue": 100.5,
            "agreement": 1.0,
            "label": "same tempo",
        },
        "key": {
            "queryValue": "C major",
            "matchValue": "C major",
            "agreement": 1.0,
            "label": "same key",
        },
        "harmonic": {
            "agreement": 0.82,
            "label": "similar chord palette",
        },
        "timbre": {
            "agreement": 0.78,
            "label": "similar production feel",
        },
    }


def test_neighbors_to_narrative_wire_shape_roundtrip():
    """The shape /neighbors writes must be loadable back into NarrativeContext."""
    criteria_block = _sample_neighbors_block()

    # Stage 1: /neighbors flattens the block via _criteria_to_token_fragment.
    flat_criteria = _criteria_to_token_fragment(criteria_block)
    assert isinstance(flat_criteria, list)
    assert {c["id"] for c in flat_criteria} == {"tempo", "key", "harmonic", "timbre"}

    # Stage 2: /neighbors builds the per-neighbor fragment via the helper.
    fragment = context_token.neighbor_context_fragment(
        track_id="tier1:itunes:380907765",
        title="Take On Me",
        artist="a-ha",
        query_window=(20.0, 30.0),
        match_window=(10.0, 20.0),
        raw_cosine=0.881,
        criteria=flat_criteria,
    )

    # Stage 3: token issuance + verification (api.py does this on each side).
    token = context_token.issue(
        query_fingerprint="a" * 64,
        model_sha="model-sha-e2e",
        catalog_sha="catalog-sha-e2e",
        neighbors={fragment["trackId"]: fragment},
    )
    verified = context_token.verify(
        token,
        expected_model_sha="model-sha-e2e",
        expected_catalog_sha="catalog-sha-e2e",
    )
    verified_fragment = verified.neighbors["tier1:itunes:380907765"]

    # Stage 4: rebuild NarrativeContext exactly as /narrative does. This is
    # where any wire-shape drift will surface — Pydantic validation will
    # raise if a field name was renamed or the criteria shape changed.
    context = rag_narrative.NarrativeContext(
        queryFingerprint=verified.queryFingerprint,
        trackId=verified_fragment["trackId"],
        title=verified_fragment.get("title", ""),
        artist=verified_fragment.get("artist"),
        queryWindow=tuple(verified_fragment["queryWindow"]),
        matchWindow=tuple(verified_fragment["matchWindow"]),
        rawCosine=float(verified_fragment["rawCosine"]),
        criteria=[
            rag_narrative.CriterionContext(**c)
            for c in (verified_fragment.get("criteria") or [])
        ],
        acrcloudCoverSongId=verified.acrcloudCoverSongId,
    )

    # The rebuilt context must preserve every load-bearing field.
    assert context.queryFingerprint == "a" * 64
    assert context.trackId == "tier1:itunes:380907765"
    assert context.title == "Take On Me"
    assert context.artist == "a-ha"
    assert context.queryWindow == (20.0, 30.0)
    assert context.matchWindow == (10.0, 20.0)
    assert context.rawCosine == pytest.approx(0.881)
    assert {c.id for c in context.criteria} == {"tempo", "key", "harmonic", "timbre"}
    # Tempo / key carry queryValue + matchValue through the wire.
    tempo = next(c for c in context.criteria if c.id == "tempo")
    assert tempo.queryValue == pytest.approx(100.0)
    assert tempo.matchValue == pytest.approx(100.5)
    key = next(c for c in context.criteria if c.id == "key")
    assert key.queryValue == "C major"
    assert key.matchValue == "C major"
    # Harmonic / timbre use the elided-vector marker.
    harmonic = next(c for c in context.criteria if c.id == "harmonic")
    assert harmonic.queryValue == {"vector": "elided"}
    assert harmonic.matchValue == {"vector": "elided"}


def test_e2e_generate_narrative_runs_on_rebuilt_context():
    """The rebuilt context must pass through generate_narrative()'s gate
    + (mocked) LLM call cleanly."""
    from unittest.mock import patch

    fragment = context_token.neighbor_context_fragment(
        track_id="tier1:itunes:380907765",
        title="Take On Me",
        artist="a-ha",
        query_window=(20.0, 30.0),
        match_window=(10.0, 20.0),
        raw_cosine=0.881,
        criteria=_criteria_to_token_fragment(_sample_neighbors_block()),
    )

    token = context_token.issue(
        query_fingerprint="a" * 64,
        model_sha="model-sha-e2e",
        catalog_sha="catalog-sha-e2e",
        neighbors={fragment["trackId"]: fragment},
    )
    verified = context_token.verify(
        token,
        expected_model_sha="model-sha-e2e",
        expected_catalog_sha="catalog-sha-e2e",
    )
    rebuilt_fragment = verified.neighbors["tier1:itunes:380907765"]
    context = rag_narrative.NarrativeContext(
        queryFingerprint=verified.queryFingerprint,
        trackId=rebuilt_fragment["trackId"],
        title=rebuilt_fragment["title"],
        artist=rebuilt_fragment.get("artist"),
        queryWindow=tuple(rebuilt_fragment["queryWindow"]),
        matchWindow=tuple(rebuilt_fragment["matchWindow"]),
        rawCosine=float(rebuilt_fragment["rawCosine"]),
        criteria=[
            rag_narrative.CriterionContext(**c)
            for c in (rebuilt_fragment.get("criteria") or [])
        ],
        acrcloudCoverSongId=verified.acrcloudCoverSongId,
    )

    payload = {
        "kind": "narrative",
        "mode": "whySimilar",
        "prose": "Tracks share tempo and key.",
        "citations": [
            {
                "trackId": "tier1:itunes:380907765",
                "side": "query",
                "timestampRange": [20.0, 30.0],
                "criterionIds": ["tempo", "key"],
                "citedValues": {
                    "tempo.queryValue": 100.0,
                    "tempo.matchValue": 100.5,
                    "key.matchValue": "C major",
                    "rawCosine": 0.881,
                },
            }
        ],
    }

    with patch("backend.rag_narrative._call_openai_json", return_value=payload):
        result = rag_narrative.generate_narrative(
            context,
            "whySimilar",
            model_sha="model-sha-e2e",
            catalog_sha="catalog-sha-e2e",
        )

    assert isinstance(result, rag_narrative.NarrativeResponse)
    assert result.kind == "narrative"
    assert len(result.citations) == 1

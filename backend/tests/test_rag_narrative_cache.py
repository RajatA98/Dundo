"""Tests for the in-process narrative cache wired onto cache_key()."""

from __future__ import annotations

from unittest.mock import patch

from backend import rag_narrative
from backend.rag_narrative import CriterionContext, NarrativeContext, NarrativeResponse


def _context(raw_cosine: float = 0.9) -> NarrativeContext:
    return NarrativeContext(
        queryFingerprint="b" * 64,
        trackId="tier2:jamendo:999",
        title="Cache Test Track",
        artist="Cache Test Artist",
        queryWindow=(0.0, 10.0),
        matchWindow=(0.0, 10.0),
        rawCosine=raw_cosine,
        acrcloudCoverSongId=None,
        criteria=[
            CriterionContext(
                id="tempo",
                queryValue=120.0,
                matchValue=120.0,
                agreement=1.0,
                label="same tempo",
            )
        ],
    )


def _valid_payload() -> dict:
    # `_context()` supplies a tempo criterion, and a context with MIR criteria MUST
    # cite them (see `_citations_are_grounded`), so a genuine success cites tempo.
    # The citation is grounded against the tempo criterion only — no rawCosine cite —
    # so it validates for every `_context()` regardless of its rawCosine value.
    return {
        "kind": "narrative",
        "mode": "whySimilar",
        "prose": "This resonates because of shared tempo and feel.",
        "citations": [
            {
                "trackId": "tier2:jamendo:999",
                "side": "query",
                "timestampRange": [0.0, 10.0],
                "criterionIds": ["tempo"],
                "citedValues": [
                    {"name": "tempo.queryValue", "value": 120.0},
                    {"name": "tempo.matchValue", "value": 120.0},
                ],
            }
        ],
        "factCitations": [],
    }


def test_repeat_call_with_same_context_hits_cache_not_openai() -> None:
    ctx = _context()
    with patch(
        "backend.rag_narrative._call_openai_json", return_value=_valid_payload()
    ) as call:
        first = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha"
        )
        second = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha"
        )

    assert isinstance(first, NarrativeResponse)
    assert isinstance(second, NarrativeResponse)
    assert first.prose == second.prose
    call.assert_called_once()


def test_different_context_is_not_a_cache_hit() -> None:
    with patch(
        "backend.rag_narrative._call_openai_json", return_value=_valid_payload()
    ) as call:
        rag_narrative.generate_narrative(
            _context(raw_cosine=0.9), "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha"
        )
        rag_narrative.generate_narrative(
            _context(raw_cosine=0.5), "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha"
        )

    assert call.call_count == 2


def test_openai_error_is_not_cached_so_a_retry_can_succeed() -> None:
    ctx = _context()
    with patch("backend.rag_narrative._call_openai_json", return_value=None) as call:
        first = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha"
        )
    assert first.reason == "openai-error"
    assert call.call_count == 1

    # Same context, next call succeeds — the error was NOT frozen in the cache.
    with patch(
        "backend.rag_narrative._call_openai_json", return_value=_valid_payload()
    ) as call2:
        second = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha"
        )
    assert isinstance(second, NarrativeResponse)
    call2.assert_called_once()


def test_hallucination_rejection_is_not_cached() -> None:
    ctx = _context()
    hallucinated = {
        "kind": "narrative",
        "mode": "whySimilar",
        "prose": "cites a criterion that does not exist in context",
        "citations": [
            {
                "trackId": "tier2:jamendo:999",
                "side": "query",
                "timestampRange": [0.0, 10.0],
                "criterionIds": ["key"],  # context only supplies tempo
                "citedValues": [{"name": "key.queryValue", "value": "Z minor"}],
            }
        ],
        "factCitations": [],
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=hallucinated):
        first = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha"
        )
    assert first.kind == "unavailable"

    # A better draw on retry must be able to succeed.
    with patch(
        "backend.rag_narrative._call_openai_json", return_value=_valid_payload()
    ) as call2:
        second = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha"
        )
    assert isinstance(second, NarrativeResponse)
    call2.assert_called_once()

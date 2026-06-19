from __future__ import annotations

from unittest.mock import patch

from backend import rag_narrative
from backend.rag_narrative import (
    CriterionContext,
    LowConfidence,
    NarrativeContext,
    NarrativeResponse,
    NarrativeUnavailable,
)


def _context(*, criteria: list[CriterionContext] | None = None) -> NarrativeContext:
    return NarrativeContext(
        queryFingerprint="a" * 64,
        trackId="tier1:itunes:380907765",
        title="Take On Me",
        artist="a-ha",
        queryWindow=(20.0, 30.0),
        matchWindow=(10.0, 20.0),
        rawCosine=0.8812,
        criteria=criteria
        if criteria is not None
        else [
            CriterionContext(
                id="tempo",
                queryValue=100.0,
                matchValue=100.5,
                agreement=1.0,
                label="same tempo",
            ),
            CriterionContext(
                id="key",
                queryValue="C major",
                matchValue="C major",
                agreement=1.0,
                label="same key",
            ),
            CriterionContext(
                id="harmonic",
                queryValue={"shape": [12]},
                matchValue={"shape": [12]},
                agreement=0.82,
                label="similar chord palette",
            ),
        ],
        acrcloudCoverSongId=None,
    )


def _valid_payload(mode: str = "whySimilar") -> dict:
    return {
        "kind": "narrative",
        "mode": mode,
        "prose": (
            "The match is grounded in the same tempo, shared key, and a close harmonic "
            "palette around the cited windows. The cosine score is high enough to support "
            "the comparison, but this is still an acoustic explanation rather than a legal claim."
        ),
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


def test_valid_narrative_passes() -> None:
    ctx = _context()
    with patch("backend.rag_narrative._call_openai_json", return_value=_valid_payload()) as call:
        result = rag_narrative.generate_narrative(
            ctx,
            "whySimilar",
            model_sha="model-sha",
            catalog_sha="catalog-sha",
        )

    assert isinstance(result, NarrativeResponse)
    assert result.kind == "narrative"
    assert result.mode == "whySimilar"
    assert result.prose
    assert len(result.citations) >= 1
    call.assert_called_once()


def test_malformed_llm_json_returns_unavailable() -> None:
    ctx = _context()
    malformed = {"prose": 123, "citations": "not-a-list"}
    with patch("backend.rag_narrative._call_openai_json", return_value=malformed):
        result = rag_narrative.generate_narrative(
            ctx,
            "whySimilar",
            model_sha="model-sha",
            catalog_sha="catalog-sha",
        )

    assert isinstance(result, NarrativeUnavailable)
    assert result.reason == "malformed-llm-output"


def test_openai_none_returns_unavailable() -> None:
    ctx = _context()
    with patch("backend.rag_narrative._call_openai_json", return_value=None):
        result = rag_narrative.generate_narrative(
            ctx,
            "whySimilar",
            model_sha="model-sha",
            catalog_sha="catalog-sha",
        )

    assert isinstance(result, NarrativeUnavailable)
    assert result.reason == "openai-error"


def test_hallucinated_criterion_returns_unavailable() -> None:
    ctx = _context(
        criteria=[
            CriterionContext(
                id="key",
                queryValue="C major",
                matchValue="C major",
                agreement=1.0,
                label="same key",
            ),
            CriterionContext(
                id="harmonic",
                queryValue={"shape": [12]},
                matchValue={"shape": [12]},
                agreement=0.82,
                label="similar chord palette",
            ),
        ]
    )
    payload = _valid_payload()
    with patch("backend.rag_narrative._call_openai_json", return_value=payload):
        result = rag_narrative.generate_narrative(
            ctx,
            "whySimilar",
            model_sha="model-sha",
            catalog_sha="catalog-sha",
        )

    assert isinstance(result, NarrativeUnavailable)
    assert result.reason == "citation-hallucinated"


def test_wrong_trackid_returns_unavailable() -> None:
    ctx = _context()
    payload = _valid_payload()
    payload["citations"][0]["trackId"] = "tier1:itunes:999999999"
    with patch("backend.rag_narrative._call_openai_json", return_value=payload):
        result = rag_narrative.generate_narrative(
            ctx,
            "whySimilar",
            model_sha="model-sha",
            catalog_sha="catalog-sha",
        )

    assert isinstance(result, NarrativeUnavailable)
    assert result.reason == "citation-hallucinated"


def test_low_context_short_circuits_llm() -> None:
    ctx = _context(criteria=[])
    with patch("backend.rag_narrative._call_openai_json") as call:
        result = rag_narrative.generate_narrative(
            ctx,
            "whySimilar",
            model_sha="model-sha",
            catalog_sha="catalog-sha",
        )

    assert isinstance(result, LowConfidence)
    assert result.reason == "missing-criteria"
    call.assert_not_called()


def test_cache_key_stable_under_key_reordering() -> None:
    ctx_a = _context()
    ctx_b = _context(criteria=list(reversed(ctx_a.criteria)))

    key_a = rag_narrative.cache_key(
        ctx_a,
        "whySimilar",
        model_id="gpt-4o-mini",
        model_sha="model-sha",
        catalog_sha="catalog-sha",
    )
    key_b = rag_narrative.cache_key(
        ctx_b,
        "whySimilar",
        model_id="gpt-4o-mini",
        model_sha="model-sha",
        catalog_sha="catalog-sha",
    )

    assert key_a == key_b


def test_cache_key_changes_when_prompt_template_changes() -> None:
    ctx = _context()
    before = rag_narrative.cache_key(
        ctx,
        "whySimilar",
        model_id="gpt-4o-mini",
        model_sha="model-sha",
        catalog_sha="catalog-sha",
    )

    with patch.dict(
        rag_narrative.SYSTEM_PROMPTS,
        {"whySimilar": rag_narrative.SYSTEM_PROMPTS["whySimilar"] + "\nchanged"},
    ):
        after = rag_narrative.cache_key(
            ctx,
            "whySimilar",
            model_id="gpt-4o-mini",
            model_sha="model-sha",
            catalog_sha="catalog-sha",
        )

    assert before != after

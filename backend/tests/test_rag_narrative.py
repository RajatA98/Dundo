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
        trackId="tier2:jamendo:380907765",
        title="Small Hours",
        artist="Maya Lev",
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
                "trackId": "tier2:jamendo:380907765",
                "side": "query",
                "timestampRange": [20.0, 30.0],
                "criterionIds": ["tempo", "key"],
                "citedValues": [
                    {"name": "tempo.queryValue", "value": 100.0},
                    {"name": "tempo.matchValue", "value": 100.5},
                    {"name": "key.matchValue", "value": "C major"},
                    {"name": "rawCosine", "value": 0.881},
                ],
            }
        ],
    }


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _Response('{"kind":"narrative","mode":"whySimilar","prose":"ok","citations":[]}')


class _Client:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _Completions()})()


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


def test_openai_call_uses_strict_json_schema_and_larger_token_budget() -> None:
    client = _Client()

    payload = rag_narrative._call_openai_json(
        client,
        system_prompt="system",
        user_prompt="user",
        max_tokens=rag_narrative.MAX_COMPLETION_TOKENS,
        model_id="gpt-4o-mini",
    )

    kwargs = client.chat.completions.kwargs
    assert payload["kind"] == "narrative"
    assert rag_narrative.MAX_COMPLETION_TOKENS >= 1000
    assert kwargs["response_format"]["type"] == "json_schema"
    schema_payload = kwargs["response_format"]["json_schema"]
    assert schema_payload["strict"] is True
    assert schema_payload["schema"]["additionalProperties"] is False
    assert set(schema_payload["schema"]["required"]) == {"kind", "mode", "prose", "citations"}
    # citedValues is now a list of fixed-shape {name,value} objects (not an open
    # dict) so the schema is OpenAI-strict-valid. See test_strict_schema.py.
    cited_values_schema = schema_payload["schema"]["$defs"]["StructuredCitation"]["properties"]["citedValues"]
    assert cited_values_schema["type"] == "array"


def test_prompt_is_artist_framed_discovery_voice() -> None:
    ctx = _context()
    prompt = rag_narrative._build_user_prompt(ctx, "whySimilar")
    system = rag_narrative.SYSTEM_PROMPTS["whySimilar"]

    assert "why this artist resonates with what you made" in system
    assert "Maya Lev" in prompt
    assert "matchedArtist" in prompt
    assert "copyright" not in system.casefold()
    assert "legal" not in system.casefold()
    assert "risk" not in system.casefold()


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

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
    assert set(schema_payload["schema"]["required"]) == {"kind", "mode", "prose", "citations", "factCitations", "promptSnippet"}
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
    ctx.rawCosine = 0.5  # genuinely weak: no criteria, no evidence, low cosine -> gated
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


def test_evidence_descriptors_revive_gate_without_criteria() -> None:
    # At a WEAK cosine: no criteria + no evidence -> gated; shared descriptors revive it.
    bare = _context(criteria=[])
    bare.rawCosine = 0.5
    assert rag_narrative._context_gate_reason(bare) == "missing-criteria"
    grounded = _context(criteria=[])
    grounded.rawCosine = 0.5
    grounded.evidenceShared = [{"kind": "genre", "label": "rock", "confidence": 0.6}]
    assert rag_narrative._context_gate_reason(grounded) is None


def test_evidence_only_narrative_passes_with_empty_citations() -> None:
    # narrative grounded purely on shared descriptors (no MIR) — empty citations are valid.
    ctx = _context(criteria=[])
    ctx.evidenceShared = [{"kind": "genre", "label": "rock", "confidence": 0.6}]
    payload = {
        "kind": "narrative",
        "mode": "whySimilar",
        "prose": "You both live in atmospheric rock — that shared sound is why this resonates.",
        "citations": [],
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=payload) as call:
        result = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha",
        )
    assert isinstance(result, NarrativeResponse)
    call.assert_called_once()


def test_strong_acoustic_match_narrates_without_criteria_or_evidence() -> None:
    # No MIR criteria, no shared descriptors, but a strong acoustic match (rawCosine>=0.70):
    # every displayed top-3 match deserves an explanation, grounded on the resemblance in prose.
    ctx = _context(criteria=[])  # helper rawCosine=0.8812
    payload = {
        "kind": "narrative",
        "mode": "whySimilar",
        "prose": "Your tracks share a similar acoustic character — the resemblance is strongest "
        "in the matched section.",
        "citations": [],
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=payload) as call:
        result = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="model-sha", catalog_sha="catalog-sha",
        )
    assert isinstance(result, NarrativeResponse)
    call.assert_called_once()


def test_creator_advice_evidence_only_passes_with_empty_citations() -> None:
    # creatorAdvice grounded purely on shared descriptors (no MIR) — like whySimilar,
    # empty citations are valid and the gate must not reject it.
    ctx = _context(criteria=[])
    ctx.evidenceShared = [{"kind": "genre", "label": "pop", "confidence": 0.6}]
    payload = {
        "kind": "narrative",
        "mode": "creatorAdvice",
        "prose": "Lean further into your atmospheric pop edge, vary the rhythm to set "
        "yourself apart, and foreground a signature texture the shared sound doesn't have.",
        "citations": [],
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=payload) as call:
        result = rag_narrative.generate_narrative(
            ctx, "creatorAdvice", model_sha="model-sha", catalog_sha="catalog-sha",
        )
    assert isinstance(result, NarrativeResponse)
    assert result.mode == "creatorAdvice"
    call.assert_called_once()


def test_creator_advice_acoustic_only_passes_with_empty_citations() -> None:
    # No MIR criteria, no shared descriptors, strong acoustic match: creatorAdvice
    # must still narrate (grounded on the resemblance) rather than gate/hallucinate.
    ctx = _context(criteria=[])  # helper rawCosine=0.8812
    payload = {
        "kind": "narrative",
        "mode": "creatorAdvice",
        "prose": "Push the contrast where your tracks resemble each other most — shift the "
        "arrangement in the matched section, add a distinctive motif, and tighten your dynamics.",
        "citations": [],
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=payload) as call:
        result = rag_narrative.generate_narrative(
            ctx, "creatorAdvice", model_sha="model-sha", catalog_sha="catalog-sha",
        )
    assert isinstance(result, NarrativeResponse)
    assert result.mode == "creatorAdvice"
    call.assert_called_once()


def _ak() -> dict:
    return {
        "location": "Utrecht, Netherlands",
        "locationAliases": ["utrecht, netherlands", "utrecht", "netherlands", "nld", "dutch"],
        "genres": ["dream pop", "ambient"],
        "moods": ["dreamy"],
        "instruments": ["guitar"],
    }


def test_grounded_artist_facts_pass() -> None:
    # facts that ARE in artistKnowledge → narrative passes.
    ctx = _context(criteria=[])
    ctx.artistKnowledge = _ak()
    payload = {
        "kind": "narrative",
        "mode": "whySimilar",
        "prose": "Maya, a Utrecht-based artist whose catalog leans dream pop, shares your hazy atmosphere.",
        "citations": [],
        "factCitations": [
            {"type": "location", "value": "Utrecht"},
            {"type": "tag", "value": "dream pop"},
        ],
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=payload) as call:
        result = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="m", catalog_sha="c",
        )
    assert isinstance(result, NarrativeResponse)
    call.assert_called_once()


def test_hallucinated_location_fact_rejected() -> None:
    # LLM claims a place NOT in artistKnowledge → reject (the integrity gate for facts).
    ctx = _context(criteria=[])
    ctx.artistKnowledge = _ak()
    payload = {
        "kind": "narrative",
        "mode": "whySimilar",
        "prose": "This Tokyo-based artist shares your sound.",
        "citations": [],
        "factCitations": [{"type": "location", "value": "Tokyo"}],
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=payload):
        result = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="m", catalog_sha="c",
        )
    assert isinstance(result, NarrativeUnavailable)
    assert result.reason == "fact-hallucinated"


def test_artist_fact_claimed_with_no_knowledge_rejected() -> None:
    # artistKnowledge empty but the LLM still asserts a fact → reject (can't invent).
    ctx = _context(criteria=[])  # artistKnowledge defaults to {}
    payload = {
        "kind": "narrative",
        "mode": "whySimilar",
        "prose": "This French artist resonates with you.",
        "citations": [],
        "factCitations": [{"type": "location", "value": "France"}],
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=payload):
        result = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="m", catalog_sha="c",
        )
    assert isinstance(result, NarrativeUnavailable)
    assert result.reason == "fact-hallucinated"


def test_empty_fact_citations_always_ok() -> None:
    # No artist-fact claim (the common acoustic/evidence-only case) → unaffected.
    ctx = _context(criteria=[])
    ctx.artistKnowledge = _ak()
    payload = {
        "kind": "narrative",
        "mode": "whySimilar",
        "prose": "Your tracks share a similar acoustic character in the matched section.",
        "citations": [],
        "factCitations": [],
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=payload) as call:
        result = rag_narrative.generate_narrative(
            ctx, "whySimilar", model_sha="m", catalog_sha="c",
        )
    assert isinstance(result, NarrativeResponse)
    call.assert_called_once()


def test_creator_coach_returns_prompt_snippet() -> None:
    # The Suno coach (creatorAdvice) grounds on queryDescriptors and returns a
    # structured, copyable promptSnippet (style + lyricsTags + workflowTip).
    ctx = _context(criteria=[])
    ctx.queryDescriptors = {"tempoBpm": 92, "key": "G", "mode": "minor", "genres": ["dream pop"], "moods": ["dreamy"]}
    payload = {
        "kind": "narrative",
        "mode": "creatorAdvice",
        "prose": "Your track reads as dreamy dream-pop at ~92 BPM. To resonate more, open the "
        "verse and lift the chorus; to stand out, swap in one signature timbre. Suno guides — "
        "it won't guarantee, so expect a couple of re-rolls.",
        "citations": [],
        "factCitations": [],
        "promptSnippet": {
            "style": "dream-pop, hazy, analog synth pads, breathy vocal, 92 BPM",
            "lyricsTags": ["[Verse]", "[Build-Up]", "[Chorus]", "[Energy: High]"],
            "workflowTip": "Use Replace Section on the chorus, then export Stems to re-record the lead.",
        },
    }
    with patch("backend.rag_narrative._call_openai_json", return_value=payload) as call:
        result = rag_narrative.generate_narrative(ctx, "creatorAdvice", model_sha="m", catalog_sha="c")
    assert isinstance(result, NarrativeResponse)
    assert result.promptSnippet.style.startswith("dream-pop")
    assert "[Chorus]" in result.promptSnippet.lyricsTags
    assert result.promptSnippet.workflowTip
    call.assert_called_once()


def test_suno_kb_injected_into_creator_advice_only() -> None:
    coach = rag_narrative._system_prompt("creatorAdvice")
    why = rag_narrative._system_prompt("whySimilar")
    assert "SUNO COACHING KNOWLEDGE BASE" in coach
    assert "SUNO COACHING KNOWLEDGE BASE" not in why


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

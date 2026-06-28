"""Metadata-grounded narrative generation for Dundo match explanations.

This module is intentionally independent from FastAPI. `api.py` supplies a
trusted `NarrativeContext`; this module gates context quality, builds a
bounded prompt, calls OpenAI through one adapter, validates structured
citations, and returns typed Pydantic results for the frontend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, ValidationError

NarrativeMode = Literal["whySimilar", "creatorAdvice"]
CriterionId = Literal["tempo", "key", "harmonic", "timbre"]

RESPONSE_SCHEMA_VERSION = "v3"  # v3: creatorAdvice Suno coach — queryDescriptors + promptSnippet
CRITERIA_ALGORITHM_VERSION = "adr-0004-v1"
MAX_PROMPT_CHARS = 14000  # raised for the cached Suno-coach KB injected into creatorAdvice
MAX_COMPLETION_TOKENS = 1000

logger = logging.getLogger(__name__)


def _load_suno_kb() -> str:
    """Load the Suno coaching knowledge base once (cached). Injected into the
    creatorAdvice system prompt as grounding. Empty string if the file is absent."""
    global _SUNO_KB
    if _SUNO_KB is not None:
        return _SUNO_KB
    try:
        _SUNO_KB = (Path(__file__).parent / "knowledge" / "suno_coach.md").read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        logger.warning("suno_coach KB not loaded: %r", exc)
        _SUNO_KB = ""
    return _SUNO_KB


_SUNO_KB: str | None = None


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
    artist: str | None
    queryWindow: tuple[float, float]
    matchWindow: tuple[float, float]
    rawCosine: float
    criteria: list[CriterionContext]
    acrcloudCoverSongId: dict | None
    # Evidence Layer: gated shared descriptors [{kind,label,confidence}] — grounds the
    # narrative on genre/mood/instrument overlap even when MIR criteria are absent.
    evidenceShared: list[dict] = Field(default_factory=list)
    # Narrative v2 — catalog facts about the MATCHED artist (NOT the overlap):
    #   {location, locationAliases:[...], genres:[...], moods:[...], instruments:[...],
    #    similarArtists:[...]}
    # Lets the narrative add brief real context about who the artist is. Any
    # place/scene/era/history claim the LLM makes must be backed by a factCitation
    # validated against this object — otherwise the narrative is rejected. Empty {}
    # means "no catalog facts": the narrative stays acoustic/evidence-only and may
    # make NO artist-context claims.
    artistKnowledge: dict = Field(default_factory=dict)
    # creatorAdvice (Suno coach) — the UPLOAD's own detected descriptors:
    #   {tempoBpm, key, mode, genres:[...], moods:[...]}. INFERRED from audio + neighbor
    #   tags, not ground truth — the coach must hedge ("your track reads as…"). Empty {}
    #   for whySimilar / when unavailable.
    queryDescriptors: dict = Field(default_factory=dict)


class CitedValue(BaseModel):
    # A single grounded value the narrative cites. Modeled as a fixed-shape
    # {name, value} object (not an open dict) so the response schema is
    # OpenAI-strict-compatible — strict mode forbids open `additionalProperties`.
    # `name` is e.g. "tempo.queryValue", "key.matchValue", or "rawCosine".
    name: str
    value: float | str


class StructuredCitation(BaseModel):
    trackId: str
    side: Literal["query", "match"]
    # list[float] (not tuple) so the schema is `array of number` — strict mode
    # rejects the prefixItems/minItems/maxItems that a tuple type generates.
    timestampRange: list[float]
    criterionIds: list[CriterionId]
    citedValues: list[CitedValue]


FactType = Literal["location", "tag", "similarArtist"]


class FactCitation(BaseModel):
    # A typed artist-fact the narrative asserts. `value` must be present in the
    # supplied artistKnowledge (location alias / tag / similar artist) or the
    # narrative is rejected as `fact-hallucinated`. Mirrors the numeric-citation
    # integrity model, extended from numbers to facts.
    type: FactType
    value: str


class PromptSnippet(BaseModel):
    # creatorAdvice (Suno coach) — a copyable, ready-to-paste Suno artifact. Empty for
    # whySimilar. `style` = the Style-field line; `lyricsTags` = bracketed structure
    # metatags for the Lyrics box; `workflowTip` = a Suno workflow move (Extend /
    # Replace Section / Stems / Exclude Styles), not just a prompt string.
    style: str = ""
    lyricsTags: list[str] = Field(default_factory=list)
    workflowTip: str = ""


class NarrativeResponse(BaseModel):
    kind: Literal["narrative"] = "narrative"
    mode: NarrativeMode
    prose: str
    citations: list[StructuredCitation]
    # Optional in Pydantic (older mocks omit it) but the strict OpenAI schema marks
    # it required, so live calls always return it (possibly empty).
    factCitations: list[FactCitation] = Field(default_factory=list)
    # creatorAdvice only — the structured copyable Suno snippet (empty for whySimilar).
    promptSnippet: PromptSnippet = Field(default_factory=PromptSnippet)


class LowConfidence(BaseModel):
    kind: Literal["low_confidence"] = "low_confidence"
    reason: str


class NarrativeUnavailable(BaseModel):
    kind: Literal["unavailable"] = "unavailable"
    reason: str


# Runtime type alias — use Union[] (not the `X | Y` operator) so this module
# imports on Python 3.9+; `from __future__ import annotations` does not cover
# runtime expressions like this. (Fixes run_rag_eval TypeError on 3.9.)
NarrativeResult = Union[NarrativeResponse, LowConfidence, NarrativeUnavailable]


# Narrative v2 — shared rule appended to both modes. Grounds artist-context claims
# in the supplied catalog facts and forces a typed citation for each, so place/scene/
# era claims can be validated and invented ones rejected.
_ARTIST_KNOWLEDGE_RULE = (
    "You also receive `artistKnowledge`: catalog facts about the matched artist — their "
    "location and their own genre/mood/instrument tags (and sometimes similar artists). "
    "You MAY add ONE brief, real touch of context from it (e.g. where they are based, or "
    "what their catalog leans toward), used sparingly — never a tag dump. You MUST NOT "
    "state any location, scene, era, or history that is not present in `artistKnowledge`; "
    "when it is empty, make no such claim. For EVERY artist fact you use, add a "
    "`factCitations` entry with its `type` (location | tag | similarArtist) and the exact "
    "`value` drawn from `artistKnowledge`."
)

SYSTEM_PROMPTS: dict[NarrativeMode, str] = {
    "whySimilar": (
        "You are Dundo, a warm music-discovery assistant explaining why this "
        "artist resonates with what you made. You receive structured metadata "
        "about the uploaded audio and a matched catalog artist; you do not hear "
        "the audio. Ground the explanation in the supplied evidence: the tempo, "
        "key, harmonic, and timbre criteria when present, AND the shared "
        "genre/instrument/mood descriptors in `sharedDescriptors`. Cite only "
        "tracks, criteria, and values present in the supplied context, and "
        "reference only descriptors listed in `sharedDescriptors` — never invent "
        "a genre, mood, or instrument. When neither criteria nor shared descriptors are "
        "present, ground the explanation in the overall acoustic resemblance and the matched "
        "section — describe the shared sonic character honestly, without naming a genre. "
        + _ARTIST_KNOWLEDGE_RULE
        + " Output a single JSON object matching the "
        "schema. No additional text, no markdown."
    ),
    "creatorAdvice": (
        "You are Dundo's Suno coach — a warm music + Suno expert helping a creator improve "
        "the AI-generated track they uploaded. You do not hear the audio; you receive the "
        "track's DETECTED descriptors in `queryDescriptors` (tempo, key/mode, inferred "
        "genre/mood tags). These are INFERRED from the audio and its acoustic neighbors, "
        "NOT ground truth — so hedge: say 'your track reads as…' or 'Dundo detected…', "
        "never 'your song is definitely…'. Use the SUNO COACHING KNOWLEDGE BASE below as "
        "your expertise, and tailor to the creator's likely intent. "
        "Give exactly TWO moves in `prose`: (1) a 'make it resonate more' idea "
        "(emotional / hook / dynamics), and (2) a 'make it stand out' idea "
        "(distinctive / less-AI). Keep it warm, plain, and tied to the detected "
        "descriptors. Then fill `promptSnippet` with ONE consolidated, copyable Suno "
        "artifact: `style` (a Style-field line of comma-separated descriptors), "
        "`lyricsTags` (a few bracketed structure metatags like [Verse], [Build-Up], "
        "[Chorus]), and `workflowTip` (a FULL, specific one-sentence instruction naming a "
        "Suno surface and what to do — e.g. 'Use Replace Section on the chorus to re-roll "
        "just that part, then export Stems to swap the lead for a real take' — never a "
        "single word). "
        "Set `citations` to [] and `factCitations` to [] — you cite no MIR criteria and "
        "assert no facts about any matched artist. End the prose with a brief honest "
        "reminder that Suno prompts guide the output, they don't guarantee it. "
        "Output a single JSON object matching the schema. No additional text, no markdown."
    ),
}


def _system_prompt(mode: NarrativeMode) -> str:
    """The mode's system prompt, with the cached Suno KB appended for creatorAdvice
    only. Used for both the live call and the cache-key hash so they stay consistent."""
    base = SYSTEM_PROMPTS[mode]
    if mode == "creatorAdvice":
        kb = _load_suno_kb()
        if kb:
            return f"{base}\n\n--- SUNO COACHING KNOWLEDGE BASE ---\n{kb}"
    return base

USER_PROMPT_TEMPLATE = """Mode: {mode}

Return JSON with exactly this shape:
{{
  "kind": "narrative",
  "mode": "{mode}",
  "prose": "80-140 words for whySimilar, or 60-120 words for creatorAdvice",
  "citations": [
    {{
      "trackId": "{track_id}",
      "side": "query|match",
      "timestampRange": [start_seconds, end_seconds],
      "criterionIds": ["tempo|key|harmonic|timbre"],
      "citedValues": [
        {{"name": "<criterionId>.queryValue", "value": "exact supplied value when cited"}},
        {{"name": "<criterionId>.matchValue", "value": "exact supplied value when cited"}},
        {{"name": "rawCosine", "value": 0.0}}
      ]
    }}
  ],
  "factCitations": [
    {{"type": "location|tag|similarArtist", "value": "exact value drawn from artistKnowledge"}}
  ],
  "promptSnippet": {{"style": "Suno Style-field line (creatorAdvice only, else empty)", "lyricsTags": ["[Verse]", "[Chorus]"], "workflowTip": "a Suno workflow move"}}
}}

`citations` carry numeric/criterion evidence; `factCitations` carry artist facts (one per artist fact you state in prose). Use [] for either when you cite nothing of that kind. IMPORTANT: if the context's `criteria` array is empty, `citations` MUST be [] — never invent a tempo/key/timestamp citation. Use the supplied context only. For whySimilar, write one grounded paragraph about why the matched artist resonates with what the user made, and leave `promptSnippet` empty (style "", lyricsTags [], workflowTip ""). For creatorAdvice, follow the Suno-coach instructions: two moves in `prose` (resonate + stand out) grounded in `queryDescriptors`, and fill `promptSnippet` with the copyable Suno artifact.

Context:
{context_json}
"""


def cache_key(
    context: NarrativeContext,
    mode: NarrativeMode,
    *,
    model_sha: str,
    catalog_sha: str,
    model_id: str,
) -> str:
    """Return a stable cache key for the prompt-relevant narrative context."""
    payload = {
        "model_id": model_id,
        "model_sha": model_sha,
        "catalog_sha": catalog_sha,
        "prompt_template_hash": _prompt_template_hash(mode),
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "criteria_algorithm_version": CRITERIA_ALGORITHM_VERSION,
        "query_fingerprint": context.queryFingerprint,
        "track_id": context.trackId,
        "mode": mode,
        "criteria_rounded": [_criterion_for_cache(c) for c in sorted(context.criteria, key=lambda c: c.id)],
        # evidenceShared is prompt-relevant context, so it must affect the cache key.
        "evidence_shared": sorted(
            (str(d.get("kind")), str(d.get("label"))) for d in context.evidenceShared
        ),
        # artistKnowledge is prompt-relevant context, so it must affect the cache key.
        "artist_knowledge": _round_numbers(context.artistKnowledge),
        # queryDescriptors (creatorAdvice Suno coach) is prompt-relevant too.
        "query_descriptors": _round_numbers(context.queryDescriptors),
        "raw_cosine": round(float(context.rawCosine), 3),
    }
    return _sha256_json(payload)


def generate_narrative(
    context: NarrativeContext,
    mode: NarrativeMode,
    *,
    model_sha: str,
    catalog_sha: str,
    model_id: str = "gpt-4o-mini",
    openai_client=None,
) -> NarrativeResult:
    start = time.perf_counter()
    key = cache_key(context, mode, model_sha=model_sha, catalog_sha=catalog_sha, model_id=model_id)

    def finish(result: NarrativeResult, *, gate_result: str, success: bool) -> NarrativeResult:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "rag_narrative cache_key=%s mode=%s gate_result=%s latency_ms=%.1f success=%s",
            key,
            mode,
            gate_result,
            latency_ms,
            success,
        )
        return result

    gate_reason = _context_gate_reason(context)
    if gate_reason is not None:
        return finish(LowConfidence(reason=gate_reason), gate_result=gate_reason, success=False)

    system_prompt = _system_prompt(mode)
    user_prompt = _build_user_prompt(context, mode)
    if len(system_prompt) + len(user_prompt) > MAX_PROMPT_CHARS:
        return finish(
            LowConfidence(reason="context-cap-exceeded"),
            gate_result="context-cap-exceeded",
            success=False,
        )

    payload = _call_openai_json(
        openai_client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=MAX_COMPLETION_TOKENS,
        model_id=model_id,
    )
    if payload is None:
        return finish(
            NarrativeUnavailable(reason="openai-error"),
            gate_result="called",
            success=False,
        )

    try:
        narrative = NarrativeResponse.model_validate(payload)
    except ValidationError:
        return finish(
            NarrativeUnavailable(reason="malformed-llm-output"),
            gate_result="called",
            success=False,
        )

    if narrative.mode != mode:
        return finish(
            NarrativeUnavailable(reason="schema-mismatch"),
            gate_result="called",
            success=False,
        )

    if not _citations_are_grounded(narrative.citations, context):
        return finish(
            NarrativeUnavailable(reason="citation-hallucinated"),
            gate_result="called",
            success=False,
        )

    if not _fact_citations_are_grounded(narrative.factCitations, context):
        return finish(
            NarrativeUnavailable(reason="fact-hallucinated"),
            gate_result="called",
            success=False,
        )

    return finish(narrative, gate_result="called", success=True)


def _call_openai_json(
    client,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    model_id: str,
) -> dict | None:
    """Call OpenAI once and return parsed JSON, or None on SDK/parse failure."""
    try:
        if client is None:
            from openai import OpenAI

            client = OpenAI()

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=_response_format_json_schema(),
            max_tokens=max_tokens,
            temperature=0,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        logger.exception("rag_narrative OpenAI JSON call failed")
        return None


def _context_gate_reason(context: NarrativeContext) -> str | None:
    # Proceed when there's SOME grounding: MIR criteria, shared descriptors, OR a strong
    # acoustic match. A displayed top-3 match is >= the discovery threshold (0.70), so it
    # always carries the embedding resemblance + matched section to ground on — every shown
    # match deserves an explanation. Only genuinely weak context (low cosine, nothing else)
    # short-circuits.
    if not context.criteria and not context.evidenceShared and float(context.rawCosine) < 0.70:
        return "missing-criteria"
    if not context.title or not context.title.strip():
        return "missing-metadata"
    if not _window_is_valid(context.queryWindow) or not _window_is_valid(context.matchWindow):
        return "missing-metadata"
    # weak-evidence only applies when criteria EXIST but are uniformly weak.
    if (
        context.criteria
        and not any(float(c.agreement) >= 0.55 for c in context.criteria)
        and float(context.rawCosine) < 0.75
        and not context.evidenceShared
    ):
        return "weak-evidence"
    return None


def _window_is_valid(window: tuple[float, float]) -> bool:
    start, end = float(window[0]), float(window[1])
    return start >= 0 and end > start


def _build_user_prompt(context: NarrativeContext, mode: NarrativeMode) -> str:
    context_payload = {
        "queryFingerprint": context.queryFingerprint,
        "trackId": context.trackId,
        "title": context.title,
        "artist": context.artist,
        "matchedArtist": context.artist,
        "matchedTrackTitle": context.title,
        "queryWindow": list(context.queryWindow),
        "matchWindow": list(context.matchWindow),
        "rawCosine": round(float(context.rawCosine), 3),
        "criteria": [_criterion_for_prompt(c) for c in sorted(context.criteria, key=lambda c: c.id)],
        "sharedDescriptors": context.evidenceShared,
        "artistKnowledge": context.artistKnowledge,
        "queryDescriptors": context.queryDescriptors,
        "acrcloudCoverSongId": context.acrcloudCoverSongId,
    }
    return USER_PROMPT_TEMPLATE.format(
        mode=mode,
        track_id=context.trackId,
        context_json=json.dumps(context_payload, sort_keys=True, separators=(",", ":")),
    )


def _criterion_for_prompt(criterion: CriterionContext) -> dict[str, Any]:
    return {
        "id": criterion.id,
        "queryValue": criterion.queryValue,
        "matchValue": criterion.matchValue,
        "agreement": round(float(criterion.agreement), 3),
        "label": criterion.label,
    }


def _criterion_for_cache(criterion: CriterionContext) -> dict[str, Any]:
    return _round_numbers(_criterion_for_prompt(criterion))


def _round_numbers(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    if isinstance(value, list):
        return [_round_numbers(v) for v in value]
    if isinstance(value, tuple):
        return [_round_numbers(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _round_numbers(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    return value


def _citations_are_grounded(citations: list[StructuredCitation], context: NarrativeContext) -> bool:
    criteria = {c.id: c for c in context.criteria}
    if not citations:
        # A narrative with MIR criteria MUST cite them. But criteria-absent narratives —
        # grounded in prose on the shared descriptors and/or the overall acoustic resemblance
        # — legitimately carry no structured citations.
        return not context.criteria

    for citation in citations:
        if citation.trackId != context.trackId:
            return False
        if not all(criterion_id in criteria for criterion_id in citation.criterionIds):
            return False
        if not _timestamp_is_grounded(citation, context):
            return False
        for cited in citation.citedValues:
            key, cited_value = cited.name, cited.value
            if key == "rawCosine":
                if not _numeric_close(cited_value, context.rawCosine, tolerance=0.01):
                    return False
                continue
            if "." not in key:
                return False
            criterion_id, side = key.split(".", 1)
            if criterion_id not in criteria or side not in {"queryValue", "matchValue"}:
                return False
            criterion = criteria[criterion_id]
            expected = getattr(criterion, side)
            if criterion_id == "tempo":
                if not _numeric_close(cited_value, expected, tolerance=2.0):
                    return False
            elif criterion_id == "key":
                if str(cited_value) != str(expected):
                    return False
            elif criterion_id in {"harmonic", "timbre"}:
                if not isinstance(expected, dict):
                    return False
            else:
                return False
    return True


def _fact_citations_are_grounded(
    fact_citations: list[FactCitation], context: NarrativeContext
) -> bool:
    """Every asserted artist fact must be backed by the supplied artistKnowledge.

    The integrity model the numeric citations use, extended to facts: a location
    claim must match a canonical alias; a tag claim must be one of the artist's
    real catalog tags; a similarArtist claim must be in the supplied list. Any
    unsupported fact (or a fact at all when artistKnowledge is empty) → reject.
    Empty factCitations is always fine (the narrative made no artist-fact claim).
    """
    if not fact_citations:
        return True
    ak = context.artistKnowledge or {}
    loc_aliases = {str(a).strip().lower() for a in (ak.get("locationAliases") or []) if str(a).strip()}
    tags = {
        str(t).strip().lower()
        for key in ("genres", "moods", "instruments")
        for t in (ak.get(key) or [])
        if str(t).strip()
    }
    similar = {str(s).strip().lower() for s in (ak.get("similarArtists") or []) if str(s).strip()}
    for fc in fact_citations:
        value = str(fc.value).strip().lower()
        if not value:
            return False
        if fc.type == "location":
            # Exact normalized alias membership (Codex review). The prompt requires the
            # cited value to be an exact value drawn from artistKnowledge, so we do NOT
            # do fuzzy substring matching — "Utrecht-based" in prose still cites "Utrecht".
            if value not in loc_aliases:
                return False
        elif fc.type == "tag":
            if value not in tags:
                return False
        elif fc.type == "similarArtist":
            if value not in similar:
                return False
        else:
            return False
    return True


def _timestamp_is_grounded(citation: StructuredCitation, context: NarrativeContext) -> bool:
    start, end = citation.timestampRange
    if end <= start:
        return False
    window = context.queryWindow if citation.side == "query" else context.matchWindow
    return start >= window[0] - 0.5 and end <= window[1] + 0.5


def _numeric_close(actual: Any, expected: Any, *, tolerance: float) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def _prompt_template_hash(mode: NarrativeMode) -> str:
    return hashlib.sha256((_system_prompt(mode) + "\n" + USER_PROMPT_TEMPLATE).encode("utf-8")).hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _response_format_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dundo_narrative_response",
            "strict": True,
            "schema": _strict_json_schema(NarrativeResponse.model_json_schema()),
        },
    }


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make Pydantic's schema acceptable for OpenAI strict structured outputs."""
    def visit(node: Any) -> Any:
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node

        out = {key: visit(value) for key, value in node.items()}
        if "properties" in out:
            properties = out.get("properties") or {}
            out["additionalProperties"] = False
            out["required"] = sorted(properties)
        return out

    return visit(schema)

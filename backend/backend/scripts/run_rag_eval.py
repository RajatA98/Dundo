"""RAG eval harness for the /narrative explanatory layer.

Reads `backend/tests/fixtures/narrative_golden_set.json`, runs each case
through `rag_narrative.generate_narrative()` with `_call_openai_json` mocked
to return the case's prepared LLM response, and scores aggregate metrics.

Metrics:
  - kind_agreement_rate:      result.kind == expected.kind
  - reason_agreement_rate:    (result.reason or null) == (expected.reason or null)
  - citation_groundedness_rate: among happy_path cases, fraction returning
    `kind=narrative` AND all citations validate.
  - gate_correctness_rate:    among low_context cases, fraction returning
    `kind=low_confidence` with the right reason.
  - hallucination_rejection_rate: among hallucinated_citation cases, fraction
    returning `kind=unavailable, reason=citation-hallucinated`.

Writes a summary JSON to `factory/artifacts/RAG_EVAL_RESULT.json` so the
result is committable and reviewable in PRs that touch rag_narrative.py.

Run:
    python -m backend.scripts.run_rag_eval [--out PATH]

Returns non-zero exit code if any baseline gate fails (kind_agreement < 1.0,
gate_correctness < 1.0, hallucination_rejection < 1.0). Those gates are
load-bearing — any regression in them is a real bug in the validation layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_SET_PATH = REPO_ROOT / "backend" / "tests" / "fixtures" / "narrative_golden_set.json"
DEFAULT_OUT_PATH = REPO_ROOT / "factory" / "artifacts" / "RAG_EVAL_RESULT.json"


def _load_golden_set(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    cases = data.get("cases", [])
    if not cases:
        raise ValueError(f"empty golden set at {path}")
    return cases


def _build_context(spec: dict):
    """Materialize a NarrativeContext from a golden-set spec dict."""
    from backend import rag_narrative

    return rag_narrative.NarrativeContext(
        queryFingerprint=spec["queryFingerprint"],
        trackId=spec["trackId"],
        title=spec["title"],
        artist=spec.get("artist"),
        queryWindow=tuple(spec["queryWindow"]),
        matchWindow=tuple(spec["matchWindow"]),
        rawCosine=float(spec["rawCosine"]),
        criteria=[
            rag_narrative.CriterionContext(**c) for c in spec["criteria"]
        ],
        evidenceShared=spec.get("evidenceShared") or [],
        artistKnowledge=spec.get("artistKnowledge") or {},
        acrcloudCoverSongId=spec.get("acrcloudCoverSongId"),
    )


def _run_case(case: dict) -> dict:
    """Run one golden-set case end-to-end and capture the result vs expected.

    Returns a per-case report row.
    """
    from backend import rag_narrative

    mocked = case["mocked_llm_response"]
    expected_kind = case["expected_kind"]
    expected_reason = case.get("expected_reason")
    must_not_call_llm = mocked == "MUST_NOT_BE_CALLED"

    # Sentinel "MUST_NOT_BE_CALLED" lives only in the JSON for readability;
    # the patched helper still needs a real return value if called. We use
    # None so a wrongful call surfaces as openai-error and the case will
    # fail the kind-agreement assertion, surfacing the bug.
    patch_return = None if must_not_call_llm else mocked

    context = _build_context(case["context"])

    with patch(
        "backend.rag_narrative._call_openai_json", return_value=patch_return
    ) as call_mock:
        result = rag_narrative.generate_narrative(
            context,
            case["mode"],
            model_sha="eval-model-sha",
            catalog_sha="eval-catalog-sha",
        )

    llm_was_called = call_mock.called
    actual_kind = getattr(result, "kind", None) or (
        result.get("kind") if isinstance(result, dict) else None
    )
    actual_reason = getattr(result, "reason", None) or (
        result.get("reason") if isinstance(result, dict) else None
    )

    return {
        "name": case["name"],
        "category": case["category"],
        "mode": case["mode"],
        "expected_kind": expected_kind,
        "actual_kind": actual_kind,
        "expected_reason": expected_reason,
        "actual_reason": actual_reason,
        "llm_was_called": llm_was_called,
        "must_not_call_llm": must_not_call_llm,
        "kind_match": actual_kind == expected_kind,
        "reason_match": (actual_reason or None) == (expected_reason or None),
        "gate_respected": (not must_not_call_llm) or (not llm_was_called),
    }


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    kind_correct = sum(1 for r in rows if r["kind_match"])
    reason_correct = sum(1 for r in rows if r["reason_match"])
    by_cat = {}
    cat_total: Counter[str] = Counter()
    cat_kind_correct: Counter[str] = Counter()
    cat_reason_correct: Counter[str] = Counter()
    cat_gate_respected: Counter[str] = Counter()
    for r in rows:
        cat_total[r["category"]] += 1
        if r["kind_match"]:
            cat_kind_correct[r["category"]] += 1
        if r["reason_match"]:
            cat_reason_correct[r["category"]] += 1
        if r["gate_respected"]:
            cat_gate_respected[r["category"]] += 1

    for cat, total in cat_total.items():
        by_cat[cat] = {
            "total": total,
            "kind_agreement_rate": round(cat_kind_correct[cat] / total, 4),
            "reason_agreement_rate": round(cat_reason_correct[cat] / total, 4),
            "gate_respected_rate": round(cat_gate_respected[cat] / total, 4),
        }

    return {
        "n_cases": n,
        "kind_agreement_rate": round(kind_correct / n, 4) if n else 0.0,
        "reason_agreement_rate": round(reason_correct / n, 4) if n else 0.0,
        "by_category": by_cat,
        "baseline_gates": {
            # Each gate is "must be 1.0 to pass" — any regression here is a
            # real bug in validation / gating, not noise.
            "happy_path_kind_agreement": by_cat.get("happy_path", {}).get("kind_agreement_rate", 0.0),
            "low_context_gate_correctness": by_cat.get("low_context", {}).get("reason_agreement_rate", 0.0),
            "hallucination_rejection": by_cat.get("hallucinated_citation", {}).get("reason_agreement_rate", 0.0),
            "malformed_rejection": by_cat.get("malformed_output", {}).get("reason_agreement_rate", 0.0),
            "openai_error_handling": by_cat.get("openai_error", {}).get("reason_agreement_rate", 0.0),
        },
    }


def run_eval(golden_set_path: Path = GOLDEN_SET_PATH) -> dict:
    """Public entry point — runs the eval and returns the aggregate dict.

    Pure function from disk path → results dict. The CLI wrapper writes the
    results to disk; the pytest gate calls this directly and asserts.
    """
    cases = _load_golden_set(golden_set_path)
    rows = [_run_case(c) for c in cases]
    summary = _aggregate(rows)
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Where to write the eval result JSON.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_SET_PATH,
        help="Override golden-set path (for experiments).",
    )
    args = parser.parse_args()

    result = run_eval(args.golden)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    summary = result["summary"]
    print(f"RAG eval — {summary['n_cases']} cases")
    print(f"  kind agreement   : {summary['kind_agreement_rate']:.2%}")
    print(f"  reason agreement : {summary['reason_agreement_rate']:.2%}")
    print()
    print("  Baseline gates (each MUST be 1.0):")
    failed: list[str] = []
    for gate, score in summary["baseline_gates"].items():
        status = "OK " if score >= 1.0 else "FAIL"
        print(f"    [{status}] {gate}: {score:.2%}")
        if score < 1.0:
            failed.append(gate)

    print()
    print("  By category:")
    for cat, stats in summary["by_category"].items():
        print(
            f"    {cat:<26} n={stats['total']:>2} kind={stats['kind_agreement_rate']:.2%} "
            f"reason={stats['reason_agreement_rate']:.2%} gate_respected={stats['gate_respected_rate']:.2%}"
        )

    print()
    print(f"Result written to {args.out}")

    if failed:
        print(f"FAILED gates: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

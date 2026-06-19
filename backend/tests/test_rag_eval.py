"""Pytest gate around the RAG eval harness.

Runs `backend.scripts.run_rag_eval.run_eval()` in-process and asserts the
five baseline gates from ADR-0005's verification section:

  - happy_path_kind_agreement      → must be 1.0
  - low_context_gate_correctness   → must be 1.0
  - hallucination_rejection        → must be 1.0
  - malformed_rejection            → must be 1.0
  - openai_error_handling          → must be 1.0

A regression in any gate is a real bug in validation / gating / schema
enforcement — never noise. The eval is fully offline (OpenAI helper is
patched on every case), so this runs in CI alongside the unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.scripts.run_rag_eval import GOLDEN_SET_PATH, run_eval


def test_golden_set_loads_and_has_cases():
    data = json.loads(GOLDEN_SET_PATH.read_text())
    assert data["version"] == "1"
    assert isinstance(data["cases"], list)
    assert len(data["cases"]) >= 10, "golden set should cover at least 10 cases"


def test_rag_eval_runs_clean():
    result = run_eval()
    assert "summary" in result and "rows" in result
    assert result["summary"]["n_cases"] == len(result["rows"])


@pytest.mark.parametrize(
    "gate",
    [
        "happy_path_kind_agreement",
        "low_context_gate_correctness",
        "hallucination_rejection",
        "malformed_rejection",
        "openai_error_handling",
    ],
)
def test_baseline_gate_holds_at_perfect(gate):
    """Each named gate MUST be 1.0. Anything less is a validation regression."""
    result = run_eval()
    score = result["summary"]["baseline_gates"].get(gate)
    assert score is not None, f"gate {gate!r} missing from summary"
    assert score >= 1.0, f"baseline gate {gate!r} regressed to {score:.4f}"


def test_overall_kind_agreement_perfect():
    result = run_eval()
    assert result["summary"]["kind_agreement_rate"] == 1.0


def test_low_context_cases_never_call_llm():
    """Cookbook self-evaluation gate: low-context cases must short-circuit
    before any OpenAI spend."""
    result = run_eval()
    for row in result["rows"]:
        if row["category"] == "low_context":
            assert not row["llm_was_called"], (
                f"low_context case {row['name']!r} called the LLM — gate bypassed"
            )


def test_eval_result_writes_to_expected_path(tmp_path):
    """The CLI writes results next to the other factory artifacts; verify
    that path is honored by run_eval's downstream consumers."""
    from backend.scripts import run_rag_eval

    # Just confirm the constant points where ADR-0005 expects it to.
    expected = Path(run_rag_eval.DEFAULT_OUT_PATH)
    assert expected.parts[-2:] == ("artifacts", "RAG_EVAL_RESULT.json"), (
        f"unexpected eval output location: {expected}"
    )

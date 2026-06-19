"""Contract test: the Python `compute_report` must match the JS one bit-for-bit.

`parity_cases` (in conftest) shells out to the real JS `computeReport` in
`quality-scorer/src/lib/scoring.js` and returns the expected outputs. This
test asserts the Python port produces identical reports field-by-field.

If the JS source changes, the parity check picks it up automatically — the
test is the canonical contract guard between the two implementations.
"""

from __future__ import annotations

from backend.scoring import compute_report

# Per-signal severity is a float arithmetic chain; floats can drift by ulps.
SEVERITY_TOLERANCE = 1e-9


def test_top_level_fields_match(parity_cases: list[dict]) -> None:
    for case in parity_cases:
        got = compute_report(case["raw"])
        exp = case["expected"]
        name = case["name"]
        assert got["score"] == exp["score"], f"{name}: score got={got['score']} exp={exp['score']}"
        assert got["verdict"] == exp["verdict"], f"{name}: verdict"
        assert got["primaryFail"] == exp["primaryFail"], f"{name}: primaryFail"
        assert sorted(got["failModes"]) == sorted(exp["failModes"]), f"{name}: failModes"
        assert got["reason"] == exp["reason"], (
            f"{name}: reason\n  got: {got['reason']!r}\n  exp: {exp['reason']!r}"
        )


def test_per_signal_match(parity_cases: list[dict]) -> None:
    for case in parity_cases:
        got = compute_report(case["raw"])
        exp = case["expected"]
        name = case["name"]
        assert len(got["signals"]) == len(exp["signals"]) == 7, name
        for gs, es in zip(got["signals"], exp["signals"]):
            assert gs["id"] == es["id"], f"{name}: signal order"
            assert gs["status"] == es["status"], f"{name}/{gs['id']}: status"
            assert abs(gs["severity"] - es["severity"]) < SEVERITY_TOLERANCE, (
                f"{name}/{gs['id']}: severity got={gs['severity']} exp={es['severity']}"
            )
            assert gs["display"] == es["display"], (
                f"{name}/{gs['id']}: display got={gs['display']!r} exp={es['display']!r}"
            )

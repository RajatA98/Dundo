"""Tests for backend.narrative_telemetry — in-process counters + structured
logger + Sentry tag helper.

The module is observability, not control flow — these tests assert counter
behavior so a regression in the /narrative endpoint surfaces here, not as a
silent stats endpoint that lies.
"""

from __future__ import annotations

import logging

import pytest

from backend import narrative_telemetry


@pytest.fixture(autouse=True)
def reset_state():
    """Wipe counters before + after every test for isolation."""
    narrative_telemetry.reset()
    yield
    narrative_telemetry.reset()


def test_initial_snapshot_is_empty():
    snap = narrative_telemetry.snapshot()
    assert snap["total_calls"] == 0
    assert snap["by_mode"] == {}
    assert snap["by_kind"] == {}
    assert snap["by_error"] == {}
    assert snap["openai_calls"] == 0
    assert snap["gate_short_circuits"] == 0
    assert snap["cost_cents_estimate"] == 0.0
    assert snap["latency_ms"]["sample_n"] == 0
    assert snap["latency_ms"]["p50"] is None
    assert snap["latency_ms"]["p95"] is None
    assert snap["latency_ms"]["p99"] is None


def test_record_call_increments_counters():
    narrative_telemetry.record_call(
        mode="whySimilar",
        latency_ms=120.0,
        result_kind="narrative",
        openai_called=True,
        prompt_chars=400,
        completion_chars=80,
    )
    snap = narrative_telemetry.snapshot()
    assert snap["total_calls"] == 1
    assert snap["by_mode"]["whySimilar"] == 1
    assert snap["by_kind"]["narrative"] == 1
    assert snap["openai_calls"] == 1
    assert snap["gate_short_circuits"] == 0
    assert snap["cost_cents_estimate"] > 0  # non-zero from prompt + completion chars


def test_record_low_confidence_increments_gate_counter():
    narrative_telemetry.record_call(
        mode="whySimilar",
        latency_ms=2.0,
        result_kind="low_confidence",
        gate_short_circuit=True,
    )
    snap = narrative_telemetry.snapshot()
    assert snap["total_calls"] == 1
    assert snap["by_kind"]["low_confidence"] == 1
    assert snap["gate_short_circuits"] == 1
    assert snap["openai_calls"] == 0  # gate fired before any LLM call


def test_record_error_code_increments_error_counter():
    narrative_telemetry.record_call(
        mode="whySimilar",
        latency_ms=1.0,
        error_code="token-expired",
    )
    snap = narrative_telemetry.snapshot()
    assert snap["total_calls"] == 1
    assert snap["by_error"]["token-expired"] == 1
    assert snap["by_kind"] == {}


def test_unknown_error_code_buckets_to_other_sentinel():
    """Unknown codes go under "_other" so operators see them without the
    counter growing arbitrary keys per typo / new code."""
    narrative_telemetry.record_call(
        mode="whySimilar",
        latency_ms=1.0,
        error_code="some-typo-code-that-doesnt-exist",
    )
    snap = narrative_telemetry.snapshot()
    assert "some-typo-code-that-doesnt-exist" not in snap["by_error"]
    assert snap["by_error"]["_other"] == 1


def test_unknown_result_kind_buckets_to_other_sentinel():
    """Same protection for result_kind."""
    narrative_telemetry.record_call(
        mode="whySimilar",
        latency_ms=1.0,
        result_kind="some-future-kind",
    )
    snap = narrative_telemetry.snapshot()
    assert "some-future-kind" not in snap["by_kind"]
    assert snap["by_kind"]["_other"] == 1


def test_latency_percentiles_with_two_samples():
    narrative_telemetry.record_call(mode="whySimilar", latency_ms=100.0, result_kind="narrative")
    narrative_telemetry.record_call(mode="whySimilar", latency_ms=200.0, result_kind="narrative")
    snap = narrative_telemetry.snapshot()
    assert snap["latency_ms"]["sample_n"] == 2
    # p50 of [100, 200] with linear interpolation at rank 0.5 = 150
    assert snap["latency_ms"]["p50"] == 150.0
    # p99 close to 200
    assert snap["latency_ms"]["p99"] == pytest.approx(200.0, abs=1.0)


def test_latency_window_is_capped():
    for i in range(narrative_telemetry.LATENCY_WINDOW_SIZE + 50):
        narrative_telemetry.record_call(
            mode="whySimilar",
            latency_ms=float(i),
            result_kind="narrative",
        )
    snap = narrative_telemetry.snapshot()
    # Counter keeps growing; window is bounded.
    assert snap["total_calls"] == narrative_telemetry.LATENCY_WINDOW_SIZE + 50
    assert snap["latency_ms"]["sample_n"] == narrative_telemetry.LATENCY_WINDOW_SIZE


def test_cost_estimate_scales_with_chars():
    narrative_telemetry.record_call(
        mode="whySimilar",
        latency_ms=100.0,
        result_kind="narrative",
        prompt_chars=1000,
        completion_chars=200,
    )
    snap_a = narrative_telemetry.snapshot()
    cost_a = snap_a["cost_cents_estimate"]

    narrative_telemetry.record_call(
        mode="whySimilar",
        latency_ms=100.0,
        result_kind="narrative",
        prompt_chars=1000,
        completion_chars=200,
    )
    snap_b = narrative_telemetry.snapshot()
    cost_b = snap_b["cost_cents_estimate"]

    # Second identical call doubles the cost.
    assert cost_b == pytest.approx(2 * cost_a, rel=1e-6)


def test_measure_call_records_on_normal_exit():
    with narrative_telemetry.measure_call("whySimilar") as ctx:
        ctx.set(result_kind="narrative", openai_called=True)
    snap = narrative_telemetry.snapshot()
    assert snap["total_calls"] == 1
    assert snap["by_kind"]["narrative"] == 1
    assert snap["openai_calls"] == 1
    assert snap["latency_ms"]["sample_n"] == 1


def test_measure_call_records_on_exception():
    """A raised exception inside the context manager still records the call
    as a narrative-error so partial-call traffic isn't dropped from counters."""
    with pytest.raises(RuntimeError):
        with narrative_telemetry.measure_call("whySimilar"):
            raise RuntimeError("simulated failure")
    snap = narrative_telemetry.snapshot()
    assert snap["total_calls"] == 1
    assert snap["by_error"]["narrative-error"] == 1


def test_structured_log_line_emitted(caplog):
    """The INFO log line contains stable key=value fields so HF Space logs
    are grep-able without a parser."""
    with caplog.at_level(logging.INFO, logger="dundo.narrative"):
        narrative_telemetry.record_call(
            mode="whySimilar",
            latency_ms=42.0,
            result_kind="narrative",
            openai_called=True,
            prompt_chars=400,
            completion_chars=100,
            trackId="tier1:itunes:380907765",
            cache_key="abcdef1234567890",
        )
    matching = [r for r in caplog.records if "narrative.call" in r.getMessage()]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "mode=whySimilar" in msg
    assert "kind=narrative" in msg
    assert "openai_called=True" in msg
    assert "trackId=tier1:itunes:380907765" in msg
    # Cache key is truncated to 16 chars for readability.
    assert "cache_key=abcdef1234567890" in msg


def test_reset_clears_state():
    narrative_telemetry.record_call(mode="whySimilar", latency_ms=10.0, result_kind="narrative")
    assert narrative_telemetry.snapshot()["total_calls"] == 1
    narrative_telemetry.reset()
    assert narrative_telemetry.snapshot()["total_calls"] == 0

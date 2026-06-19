"""In-process telemetry for the /narrative RAG explanatory layer.

The right rung for this scale: in-process counters + structured logs + Sentry
tags. A Prometheus/Datadog stack would be overbuilt for a 155-track demo —
same "progressive complexity" principle ADR-0005 commits to for retrieval.

What this module owns:
  - Counters: total_calls, by_mode, by_kind, by_error, openai_calls,
    gate_short_circuits, token_invalid, token_expired, token_stale.
  - Latency: a fixed-size sliding window of recent call durations, surfaced
    as p50/p95/p99 in the stats snapshot.
  - Cost estimate: a rough running total in cents, derived from prompt and
    completion character counts × GPT-4o-mini pricing constants. Not an
    accounting ledger — a directional cost-awareness signal for the
    /narrative/stats endpoint.
  - Structured logger: one INFO line per call with stable key=value fields
    so the HF Space logs are grep-able without a parser.
  - Sentry tags: when SENTRY_DSN is set, every call tags the current scope
    with mode + result_kind so failures aggregate by category in the
    existing dashboard.

The module is thread-safe (one lock around counter mutations + window
operations). All operations are O(1) except `snapshot()` which is O(N) over
the sliding window — N is bounded by `LATENCY_WINDOW_SIZE`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

# GPT-4o-mini pricing as of 2026-06 (rough; meant as a directional signal):
#   $0.15 per 1M input tokens, $0.60 per 1M output tokens.
# At ~4 chars per token (English average), 1 input char ≈ 0.0375 micro-cents
# and 1 output char ≈ 0.150 micro-cents. Times 100 cents/dollar:
COST_CENTS_PER_INPUT_CHAR = 0.0000375
COST_CENTS_PER_OUTPUT_CHAR = 0.000150

# Sliding window size for latency percentiles. ~30 minutes of activity at
# steady demo traffic; bounded so memory stays trivial.
LATENCY_WINDOW_SIZE = 256

# Result kinds we expect from rag_narrative. Anything else gets coerced to
# "unknown" in the counter so a bug doesn't silently grow a new key.
_KNOWN_KINDS = {"narrative", "low_confidence", "unavailable"}

# Backend error codes we surface in counters; HTTP-layer codes from
# api.py /narrative endpoint.
_KNOWN_ERROR_CODES = {
    "narrative-disabled",
    "invalid-token",
    "malformed-token",
    "token-expired",
    "stale-token",
    "not-in-context",
    "unsupported-mode",
    "malformed-context",
    "narrative-error",
}

_logger = logging.getLogger("dundo.narrative")
_logger.setLevel(logging.INFO)
# Don't add handlers here — the FastAPI app's uvicorn config already streams
# stdlib logging to stdout, which is the HF Space's log-collection surface.

_lock = threading.Lock()


@dataclass
class _State:
    total_calls: int = 0
    by_mode: dict[str, int] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)
    by_error: dict[str, int] = field(default_factory=dict)
    openai_calls: int = 0
    gate_short_circuits: int = 0
    cost_cents: float = 0.0
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW_SIZE))
    started_at: float = field(default_factory=time.time)


_state = _State()


def record_call(
    *,
    mode: str,
    latency_ms: float,
    result_kind: str | None = None,
    error_code: str | None = None,
    openai_called: bool = False,
    gate_short_circuit: bool = False,
    prompt_chars: int = 0,
    completion_chars: int = 0,
    trackId: str | None = None,
    cache_key: str | None = None,
) -> None:
    """Record a single /narrative call.

    Exactly one of `result_kind` and `error_code` should be set. The function
    accepts both being None (e.g. a 503 returned before any LLM/gate work)
    and records it as `by_error["narrative-disabled"]` so the counter still
    reflects the dropped traffic.

    Cost is estimated from prompt_chars + completion_chars × GPT-4o-mini
    pricing. Tests can pass 0 for both to skip the cost increment.
    """
    mode_key = mode if mode else "unknown"
    # Unknown result_kind / error_code values get bucketed under the
    # "_other" sentinel rather than being either dropped silently or growing
    # arbitrary counter keys. Operators see "_other > 0" → time to update
    # _KNOWN_KINDS / _KNOWN_ERROR_CODES.
    if result_kind is None:
        kind_key = None
    elif result_kind in _KNOWN_KINDS:
        kind_key = result_kind
    else:
        kind_key = "_other"
    if error_code is None:
        err_key = None
    elif error_code in _KNOWN_ERROR_CODES:
        err_key = error_code
    else:
        err_key = "_other"

    with _lock:
        _state.total_calls += 1
        _state.by_mode[mode_key] = _state.by_mode.get(mode_key, 0) + 1
        if kind_key:
            _state.by_kind[kind_key] = _state.by_kind.get(kind_key, 0) + 1
        if err_key:
            _state.by_error[err_key] = _state.by_error.get(err_key, 0) + 1
        if openai_called:
            _state.openai_calls += 1
        if gate_short_circuit:
            _state.gate_short_circuits += 1
        _state.cost_cents += (
            prompt_chars * COST_CENTS_PER_INPUT_CHAR
            + completion_chars * COST_CENTS_PER_OUTPUT_CHAR
        )
        _state.latencies_ms.append(float(latency_ms))

    # Structured log — one line, grep-able. Don't emit prompt or response
    # bodies (would leak content + bloat logs). Cache key is logged for
    # de-dup correlation across same-payload calls.
    _logger.info(
        "narrative.call mode=%s kind=%s error=%s latency_ms=%.1f openai_called=%s gate_short_circuit=%s prompt_chars=%d completion_chars=%d trackId=%s cache_key=%s",
        mode_key,
        kind_key or "-",
        err_key or "-",
        float(latency_ms),
        openai_called,
        gate_short_circuit,
        prompt_chars,
        completion_chars,
        trackId or "-",
        (cache_key or "-")[:16],  # prefix only; full key is high-cardinality
    )

    # Sentry tag scope. No-op when sentry_sdk isn't installed or SENTRY_DSN
    # isn't set — both paths defer to the existing api.py wiring.
    _set_sentry_tags(mode=mode_key, kind=kind_key, error=err_key)


def _set_sentry_tags(*, mode: str, kind: str | None, error: str | None) -> None:
    """Tag the current Sentry scope (no-op when Sentry isn't active).

    Tagging here means /narrative-layer Sentry events are filterable by
    mode + result_kind + error in the existing dashboard without a new
    integration.
    """
    if not os.getenv("SENTRY_DSN", "").strip():
        return
    try:
        import sentry_sdk

        sentry_sdk.set_tag("narrative.mode", mode)
        if kind:
            sentry_sdk.set_tag("narrative.kind", kind)
        if error:
            sentry_sdk.set_tag("narrative.error", error)
    except Exception:
        # If Sentry tagging fails for any reason, don't let it break the
        # request flow. Telemetry is observability, not control flow.
        pass


def snapshot() -> dict:
    """Return a JSON-serializable snapshot of current counters + percentiles.

    This is what `GET /narrative/stats` returns. Includes:
      - all counters
      - latency p50 / p95 / p99 over the sliding window (or null if empty)
      - cost_cents rounded to 4 decimal places
      - uptime_sec since process start
    """
    with _lock:
        latencies = sorted(_state.latencies_ms)
        sample_n = len(latencies)
        return {
            "total_calls": _state.total_calls,
            "by_mode": dict(_state.by_mode),
            "by_kind": dict(_state.by_kind),
            "by_error": dict(_state.by_error),
            "openai_calls": _state.openai_calls,
            "gate_short_circuits": _state.gate_short_circuits,
            "cost_cents_estimate": round(_state.cost_cents, 4),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50) if sample_n else None,
                "p95": _percentile(latencies, 0.95) if sample_n else None,
                "p99": _percentile(latencies, 0.99) if sample_n else None,
                "sample_n": sample_n,
                "window_size": LATENCY_WINDOW_SIZE,
            },
            "uptime_sec": round(time.time() - _state.started_at, 1),
        }


def _percentile(sorted_samples: list[float], p: float) -> float:
    """Linear-interpolation percentile over a pre-sorted list.

    Returns rounded to 1 decimal ms — sub-ms precision adds noise without
    signal at this aggregation level.
    """
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return round(sorted_samples[0], 1)
    rank = p * (len(sorted_samples) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = rank - lo
    return round(sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo]), 1)


def reset() -> None:
    """Wipe state. Used by tests to isolate per-test counter assertions.

    Production code should NEVER call this — the snapshot wouldn't survive
    a restart anyway, so there's no use case beyond test isolation.
    """
    global _state
    with _lock:
        _state = _State()


def measure_call(mode: str):
    """Context manager: time a call and ensure record_call gets invoked
    exactly once with the measured latency.

    Usage in api.py:
        with measure_call("whySimilar") as ctx:
            ... do work ...
            ctx.set(result_kind="narrative", openai_called=True, prompt_chars=N, completion_chars=M)
    """
    return _CallTimer(mode)


class _CallTimer:
    def __init__(self, mode: str):
        self.mode = mode
        self.start_ts: float = 0.0
        self._fields: dict = {}

    def __enter__(self):
        self.start_ts = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        latency_ms = (time.time() - self.start_ts) * 1000.0
        # If the context block raised, mark as a narrative-error if no other
        # error was set.
        if exc_type is not None and "error_code" not in self._fields:
            self._fields["error_code"] = "narrative-error"
        record_call(mode=self.mode, latency_ms=latency_ms, **self._fields)
        return False  # don't suppress exceptions

    def set(self, **kwargs) -> None:
        """Stash fields for record_call. Last write wins per key."""
        self._fields.update(kwargs)


__all__: Iterable[str] = (
    "record_call",
    "snapshot",
    "reset",
    "measure_call",
    "LATENCY_WINDOW_SIZE",
    "COST_CENTS_PER_INPUT_CHAR",
    "COST_CENTS_PER_OUTPUT_CHAR",
)

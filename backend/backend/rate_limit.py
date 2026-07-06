"""In-process, per-client sliding-window rate limiter.

Deliberately simple: a module-level dict guarded by a lock, no Redis or
external store. This matches the actual deployment topology — a single
FastAPI process on one HF Space, already serializing all audio inference
onto one dedicated thread (see api.py's `_infer_executor`) — rather than
adding distributed-rate-limiting infrastructure for a single-instance demo.

Client identity comes from the first X-Forwarded-For hop (the Space sits
behind HF's reverse proxy and uvicorn runs without --proxy-headers, so
request.client.host is the proxy, not the visitor). XFF is spoofable, but
this is a cost cap, not auth: a spoofer sharding buckets still can't push
any single bucket past the ceiling, the same exposure class as rotating
source IPs.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)


def client_key(forwarded_for: str | None, fallback: str) -> str:
    """First X-Forwarded-For hop, or `fallback` when the header is absent/blank."""
    if forwarded_for and forwarded_for.strip():
        return forwarded_for.split(",")[0].strip()
    return fallback


def allow(key: str, *, max_requests: int, window_s: float, now: float | None = None) -> bool:
    """Return True (and record this call) if `key` has made fewer than
    `max_requests` calls in the trailing `window_s` seconds. Return False
    (without recording) if `key` is already at the limit — a rejected call
    doesn't consume a slot.

    `now` is an injectable clock for deterministic tests; production callers
    omit it and get `time.monotonic()`.
    """
    ts = now if now is not None else time.monotonic()
    cutoff = ts - window_s
    with _lock:
        hits = _hits[key]
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= max_requests:
            return False
        hits.append(ts)
        return True


def reset() -> None:
    """Clear all rate-limit state. Test-only hook."""
    with _lock:
        _hits.clear()

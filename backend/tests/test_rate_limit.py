from __future__ import annotations

import pytest

from backend import rate_limit


@pytest.fixture(autouse=True)
def _reset():
    rate_limit.reset()
    yield


def test_allows_up_to_max_requests_then_blocks():
    for _ in range(3):
        assert rate_limit.allow("ip1", max_requests=3, window_s=60, now=0.0) is True
    assert rate_limit.allow("ip1", max_requests=3, window_s=60, now=0.0) is False


def test_window_expiry_frees_up_a_slot():
    assert rate_limit.allow("ip1", max_requests=1, window_s=10, now=0.0) is True
    assert rate_limit.allow("ip1", max_requests=1, window_s=10, now=5.0) is False
    assert rate_limit.allow("ip1", max_requests=1, window_s=10, now=11.0) is True


def test_keys_are_independent():
    assert rate_limit.allow("ip1", max_requests=1, window_s=60, now=0.0) is True
    assert rate_limit.allow("ip2", max_requests=1, window_s=60, now=0.0) is True


def test_rejected_call_does_not_consume_a_slot():
    assert rate_limit.allow("ip1", max_requests=1, window_s=60, now=0.0) is True
    assert rate_limit.allow("ip1", max_requests=1, window_s=60, now=1.0) is False
    # a later, still-within-window call is still rejected — the rejection
    # above didn't quietly extend or reset anything
    assert rate_limit.allow("ip1", max_requests=1, window_s=60, now=2.0) is False


def test_client_key_prefers_first_forwarded_hop():
    assert rate_limit.client_key("203.0.113.7, 10.0.0.1", "10.0.0.99") == "203.0.113.7"
    assert rate_limit.client_key("203.0.113.7", "10.0.0.99") == "203.0.113.7"


def test_client_key_falls_back_when_no_forwarded_header():
    assert rate_limit.client_key(None, "10.0.0.99") == "10.0.0.99"
    assert rate_limit.client_key("", "10.0.0.99") == "10.0.0.99"
    assert rate_limit.client_key("   ", "10.0.0.99") == "10.0.0.99"

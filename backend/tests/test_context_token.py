"""Tests for backend.context_token — HMAC sign + verify roundtrip.

Stateless token replaces the in-memory cache Codex round-2 pushed back on.
These tests exercise every failure code in verify() so the api.py endpoint
mapping can rely on TokenError.code being precise.
"""

from __future__ import annotations

import os
import time

import pytest

from backend import context_token
from backend.context_token import TokenError


_TEST_SECRET = "test-secret-key-not-for-production"


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setenv("CONTEXT_TOKEN_HMAC_KEY", _TEST_SECRET)
    yield _TEST_SECRET


def _sample_neighbors() -> dict:
    return {
        "tier1:itunes:380907765": {
            "trackId": "tier1:itunes:380907765",
            "title": "Take On Me",
            "artist": "a-ha",
            "queryWindow": [20.0, 30.0],
            "matchWindow": [10.0, 20.0],
            "rawCosine": 0.881,
            "criteria": [],
        }
    }


def test_roundtrip_valid_token_decodes_payload(secret):
    token = context_token.issue(
        query_fingerprint="a" * 64,
        model_sha="model-sha-1",
        catalog_sha="catalog-sha-1",
        neighbors=_sample_neighbors(),
        acrcloud_cover_song_id={"status": "no_result"},
    )
    verified = context_token.verify(
        token,
        expected_model_sha="model-sha-1",
        expected_catalog_sha="catalog-sha-1",
    )
    assert verified.queryFingerprint == "a" * 64
    assert verified.modelSha == "model-sha-1"
    assert verified.catalogSha == "catalog-sha-1"
    assert verified.acrcloudCoverSongId == {"status": "no_result"}
    assert "tier1:itunes:380907765" in verified.neighbors


def test_tampered_signature_raises_invalid_signature(secret):
    token = context_token.issue(
        query_fingerprint="a" * 64,
        model_sha="model-sha-1",
        catalog_sha="catalog-sha-1",
        neighbors=_sample_neighbors(),
    )
    body, sig = token.split(".", 1)
    # Flip one hex char of the signature.
    bad_char = "0" if sig[-1] != "0" else "1"
    bad_token = f"{body}.{sig[:-1]}{bad_char}"
    with pytest.raises(TokenError) as exc:
        context_token.verify(
            bad_token,
            expected_model_sha="model-sha-1",
            expected_catalog_sha="catalog-sha-1",
        )
    assert exc.value.code == "invalid-signature"


def test_tampered_payload_raises_invalid_signature(secret):
    """Flipping the body breaks the HMAC; verify must reject."""
    token = context_token.issue(
        query_fingerprint="a" * 64,
        model_sha="model-sha-1",
        catalog_sha="catalog-sha-1",
        neighbors=_sample_neighbors(),
    )
    body, sig = token.split(".", 1)
    # Append a stray character to body — base64 decode may still succeed but
    # HMAC must mismatch.
    tampered = f"{body}A.{sig}"
    with pytest.raises(TokenError) as exc:
        context_token.verify(
            tampered,
            expected_model_sha="model-sha-1",
            expected_catalog_sha="catalog-sha-1",
        )
    assert exc.value.code == "invalid-signature"


def test_expired_token_raises_token_expired(secret):
    past = int(time.time()) - 7200  # 2h ago
    token = context_token.issue(
        query_fingerprint="a" * 64,
        model_sha="model-sha-1",
        catalog_sha="catalog-sha-1",
        neighbors=_sample_neighbors(),
        ttl_seconds=60,
        now=past,
    )
    with pytest.raises(TokenError) as exc:
        context_token.verify(
            token,
            expected_model_sha="model-sha-1",
            expected_catalog_sha="catalog-sha-1",
        )
    assert exc.value.code == "token-expired"


def test_stale_model_raises_stale_model(secret):
    token = context_token.issue(
        query_fingerprint="a" * 64,
        model_sha="old-model-sha",
        catalog_sha="catalog-sha-1",
        neighbors=_sample_neighbors(),
    )
    with pytest.raises(TokenError) as exc:
        context_token.verify(
            token,
            expected_model_sha="new-model-sha",
            expected_catalog_sha="catalog-sha-1",
        )
    assert exc.value.code == "stale-model"


def test_stale_catalog_raises_stale_catalog(secret):
    token = context_token.issue(
        query_fingerprint="a" * 64,
        model_sha="model-sha-1",
        catalog_sha="old-catalog-sha",
        neighbors=_sample_neighbors(),
    )
    with pytest.raises(TokenError) as exc:
        context_token.verify(
            token,
            expected_model_sha="model-sha-1",
            expected_catalog_sha="new-catalog-sha",
        )
    assert exc.value.code == "stale-catalog"


def test_malformed_token_string_raises_malformed(secret):
    for bad in ["", "no-dot-here", "too.many.dots", "good-body.not-hex-sig!@#"]:
        with pytest.raises(TokenError) as exc:
            context_token.verify(
                bad,
                expected_model_sha="model-sha-1",
                expected_catalog_sha="catalog-sha-1",
            )
        assert exc.value.code in ("malformed", "invalid-signature")


def test_missing_hmac_key_raises_hmac_key_missing(monkeypatch):
    monkeypatch.delenv("CONTEXT_TOKEN_HMAC_KEY", raising=False)
    with pytest.raises(TokenError) as exc:
        context_token.issue(
            query_fingerprint="a" * 64,
            model_sha="model-sha-1",
            catalog_sha="catalog-sha-1",
            neighbors=_sample_neighbors(),
        )
    assert exc.value.code == "hmac-key-missing"


def test_is_configured_reflects_env(monkeypatch):
    monkeypatch.delenv("CONTEXT_TOKEN_HMAC_KEY", raising=False)
    assert context_token.is_configured() is False
    monkeypatch.setenv("CONTEXT_TOKEN_HMAC_KEY", _TEST_SECRET)
    assert context_token.is_configured() is True


def test_neighbor_context_fragment_shape():
    frag = context_token.neighbor_context_fragment(
        track_id="tier1:itunes:380907765",
        title="Take On Me",
        artist="a-ha",
        query_window=(20.0, 30.0),
        match_window=(10.0, 20.0),
        raw_cosine=0.881,
        criteria=[{"id": "tempo", "queryValue": 100.0, "matchValue": 100.5}],
    )
    assert frag["trackId"] == "tier1:itunes:380907765"
    assert frag["queryWindow"] == [20.0, 30.0]
    assert frag["rawCosine"] == 0.881
    assert frag["criteria"][0]["id"] == "tempo"

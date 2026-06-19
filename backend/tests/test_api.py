"""FastAPI `/analyze` smoke via TestClient.

All tests marked `slow` because the lifespan startup loads CLAP (~2 GB). Once
the model is cached locally, the suite runs in a few seconds.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def clean_wav_bytes() -> bytes:
    sr = 22050
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False, dtype=np.float32)
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="WAV")
    return buf.getvalue()


@pytest.mark.slow
def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "model" in body
    assert "version" in body


@pytest.mark.slow
def test_analyze_returns_track_shape(client, clean_wav_bytes) -> None:
    r = client.post(
        "/analyze",
        files={"file": ("test.wav", clean_wav_bytes, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    track = r.json()
    assert track["id"] == "upload"
    assert track["source"] == "upload"
    assert track["title"] == "test"
    assert isinstance(track["genre"], str)
    assert len(track["genres"]) == 3
    assert all("label" in g and "score" in g for g in track["genres"])
    assert len(track["waveform"]) == 180
    assert all(0.0 <= v <= 1.0 for v in track["waveform"])
    assert len(track["signals"]) == 7
    assert track["verdict"] in ("keep", "drop")


@pytest.mark.slow
def test_analyze_rejects_bad_mime(client) -> None:
    r = client.post(
        "/analyze",
        files={"file": ("not_audio.txt", b"this is not audio", "text/plain")},
    )
    assert r.status_code == 415
    assert r.json() == {"error": "unsupported_media"}


@pytest.mark.slow
def test_analyze_rejects_undecodable(client) -> None:
    r = client.post(
        "/analyze",
        files={"file": ("broken.wav", b"not a real wav", "audio/wav")},
    )
    assert r.status_code == 422
    assert r.json() == {"error": "decode_failed"}


def test_decode_uses_temp_file_fallback_for_aac_lc() -> None:
    """Regression test: when librosa.load on a BytesIO fails (e.g. AAC-LC `.m4a`
    that libsndfile can't decode), `_decode_and_pipeline` must fall back to
    writing the bytes to a temp file with the upload's extension and retrying
    `librosa.load` on the path. The path-based load lets audioread dispatch
    to ffmpeg, which handles AAC-LC.

    Before the fix this scenario returned 422 `decode_failed` and blocked every
    iTunes Tier-1 preview from ever reaching the encoder.
    """
    from unittest.mock import patch
    import numpy as np
    from backend import api as api_module

    # Stub librosa.load: fail when called on a BytesIO, succeed when called on a path.
    sr_target = 22050
    sample_audio = (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, sr_target * 3, dtype=np.float32))).astype(np.float32)
    seen_paths: list[str] = []

    def fake_load(src, sr=None, mono=False):  # noqa: ARG001
        if isinstance(src, io.BytesIO):
            raise RuntimeError("libsndfile cannot decode AAC-LC")
        # Path-based call — succeeds.
        seen_paths.append(str(src))
        return sample_audio, sr_target

    # Stub sf.info so the duration probe doesn't blow up.
    class _StubInfo:
        duration = 3.0

    with patch.object(api_module, "librosa") as mock_librosa, \
         patch.object(api_module, "sf") as mock_sf:
        mock_librosa.load = fake_load
        mock_librosa.to_mono = lambda y: y if y.ndim == 1 else y.mean(axis=0)
        mock_sf.info = lambda _b: _StubInfo()
        # sf.write needs to work for the acrcloud_buf step — pass through to real.
        import soundfile as real_sf
        mock_sf.write = real_sf.write
        # analyze_array depends on librosa internals we don't want to stub.
        # Easiest: short-circuit by stubbing the whole pipeline path that follows decode.
        # The contract we're testing is that the temp-file fallback IS TAKEN, which is
        # detectable by seen_paths having an entry with the correct suffix.
        with patch.object(api_module, "analyze_array") as mock_analyze, \
             patch.object(api_module, "clap_windowed") as mock_clap, \
             patch.object(api_module, "muq_engine") as mock_muq, \
             patch.object(api_module, "compute_report") as mock_report:
            mock_analyze.return_value = {"raw": {}, "waveform_180": [0.0] * 180}
            mock_clap.encode_windowed = lambda *args, **kwargs: (np.zeros(512, dtype=np.float32), np.zeros((1, 512), dtype=np.float32))
            mock_muq.top_genres = lambda emb: [("Test", 1.0)]
            mock_report.return_value = {"score": 100, "verdict": "keep", "signals": []}
            result = api_module._decode_and_pipeline(b"\x00\x01\x02\x03fake-aac-bytes", ext=".m4a")

    assert not isinstance(result, type(api_module._err(0, ""))), (
        f"Expected fallback to succeed, got error response: {result}"
    )
    assert len(seen_paths) == 1, f"Expected the fallback path to be taken once, got: {seen_paths}"
    assert seen_paths[0].endswith(".m4a"), (
        f"Temp file suffix should match the upload extension (.m4a), got: {seen_paths[0]}"
    )


@pytest.mark.slow
def test_analyze_rejects_empty(client) -> None:
    r = client.post(
        "/analyze",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code == 422
    assert r.json() == {"error": "empty_file"}

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from backend.scripts import mir_backfill
from backend.corpus_dataset import validate_manifest_file_hashes


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_rehash_manifest_after_corpus_edit_round_trips_validation(tmp_path):
    corpus = tmp_path / "corpus.json"
    manifest = tmp_path / "manifest.json"
    _write_json(corpus, [{"track_id": "tier2:jamendo:1", "title": "Old"}])
    mir_backfill.rehash_manifest(manifest)

    payload = json.loads(corpus.read_text())
    payload[0]["mir_features"] = {"tempo_bpm": 100.0}
    _write_json(corpus, payload)

    with pytest.raises(ValueError, match="checksum"):
        validate_manifest_file_hashes(tmp_path)

    mir_backfill.rehash_manifest(manifest)
    validate_manifest_file_hashes(tmp_path)


def test_merge_shard_results_into_corpus_and_rehashes_manifest(tmp_path):
    corpus = tmp_path / "corpus.json"
    manifest = tmp_path / "manifest.json"
    shard_dir = tmp_path / "mir"
    shard_dir.mkdir()
    _write_json(corpus, [
        {"track_id": "tier2:jamendo:1", "title": "A"},
        {"track_id": "tier2:jamendo:2", "title": "B"},
    ])
    mir_backfill.rehash_manifest(manifest)
    (shard_dir / "shard_0.jsonl").write_text(
        json.dumps({"track_id": "tier2:jamendo:1", "mir_features": {"tempo_bpm": 101.2}}) + "\n"
        + json.dumps({"track_id": "tier2:jamendo:2", "mir_features": None}) + "\n"
    )

    summary = mir_backfill.merge_shard_results(corpus, shard_dir, manifest)

    rows = json.loads(corpus.read_text())
    assert rows[0]["mir_features"] == {"tempo_bpm": 101.2}
    assert rows[1]["mir_features"] is None
    assert summary == {"tracks": 2, "updated": 2, "withMir": 1, "nullMir": 1}
    validate_manifest_file_hashes(tmp_path)


def test_worker_skips_completed_rows_and_uses_download_fallback(tmp_path, monkeypatch):
    shard_file = tmp_path / "shard_0.tracks.json"
    out_dir = tmp_path / "mir"
    out_dir.mkdir()
    _write_json(shard_file, [
        {
            "track_id": "tier2:jamendo:1",
            "external_ids": {"jamendoAudioUrl": "https://audio.example/1.mp3"},
        },
        {
            "track_id": "tier2:jamendo:2",
            "external_ids": {"jamendoAudioUrl": "https://audio.example/2.mp3"},
        },
    ])
    (out_dir / "shard_0.jsonl").write_text(
        json.dumps({"track_id": "tier2:jamendo:1", "mir_features": {"tempo_bpm": 90.0}}) + "\n"
    )

    monkeypatch.setattr(mir_backfill, "load_audio_for_track", lambda *args, **kwargs: ([0.1, 0.2, 0.3], 22050))

    class FakeFeatures:
        def to_dict(self):
            return {"tempo_bpm": 120.0, "key": "C", "mode": "major", "key_confidence": 1.0}

    monkeypatch.setattr(mir_backfill, "compute_mir", lambda wav, sr: FakeFeatures())

    written = mir_backfill.run_worker(shard_file, out_dir)

    assert written == 1
    lines = [json.loads(line) for line in (out_dir / "shard_0.jsonl").read_text().splitlines()]
    assert [row["track_id"] for row in lines] == ["tier2:jamendo:1", "tier2:jamendo:2"]
    assert lines[1]["mir_features"]["tempo_bpm"] == 120.0


def test_supervisor_records_crashed_inflight_track_and_relaunches(tmp_path, monkeypatch):
    shard_file = tmp_path / "shard_0.tracks.json"
    out_dir = tmp_path / "mir"
    out_dir.mkdir()
    _write_json(shard_file, [{"track_id": "tier2:jamendo:1"}, {"track_id": "tier2:jamendo:2"}])
    (out_dir / "shard_0.current.json").write_text(json.dumps({"track_id": "tier2:jamendo:1"}))

    calls = {"n": 0}

    def fake_run_worker(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise mir_backfill.ShardCrashed("simulated segfault", returncode=139)
        mir_backfill.append_jsonl(
            out_dir / "shard_0.jsonl",
            {"track_id": "tier2:jamendo:2", "mir_features": {"tempo_bpm": 111.0}},
        )
        return 1

    monkeypatch.setattr(mir_backfill, "_run_worker_subprocess", fake_run_worker)

    summary = mir_backfill.supervise_shards([shard_file], out_dir, concurrency=1)

    rows = [json.loads(line) for line in (out_dir / "shard_0.jsonl").read_text().splitlines()]
    assert rows == [
        {"track_id": "tier2:jamendo:1", "mir_features": None},
        {"track_id": "tier2:jamendo:2", "mir_features": {"tempo_bpm": 111.0}},
    ]
    assert summary["crashed"] == 1
    assert summary["completed"] == 1


def test_driver_end_to_end_on_tiny_local_audio_fixture(tmp_path):
    corpus = tmp_path / "corpus.json"
    manifest = tmp_path / "manifest.json"
    shard_dir = tmp_path / "shards"
    out_dir = tmp_path / "mir"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(audio_dir / "1.wav", 0.2 * np.sin(2 * np.pi * 440 * t), sr)
    _write_json(corpus, [
        {
            "track_id": "tier2:jamendo:1",
            "title": "Synthetic",
            "external_ids": {"jamendoTrackId": "1"},
        }
    ])
    mir_backfill.rehash_manifest(manifest)

    shard_files = mir_backfill.prepare_shards(corpus, shard_dir, shards=1)
    summary = mir_backfill.supervise_shards(
        shard_files,
        out_dir,
        concurrency=1,
        audio_dir=audio_dir,
        timeout=1.0,
    )
    merge_summary = mir_backfill.merge_shard_results(corpus, out_dir, manifest)

    assert summary == {"completed": 1, "crashed": 0}
    assert merge_summary == {"tracks": 1, "updated": 1, "withMir": 1, "nullMir": 0}
    row = json.loads(corpus.read_text())[0]
    assert row["mir_features"]["tempo_bpm"] is not None
    assert len(row["mir_features"]["chroma_mean"]) == 12
    validate_manifest_file_hashes(tmp_path)

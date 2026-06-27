import json

from backend.scripts.build_catalog_tags import build, rehash_manifest_with_sidecar


def _write_tsv(path, rows):
    with open(path, "w") as f:
        f.write("TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n")
        for num, tags in rows:
            cols = [f"track_{num:07d}", "artist_1", "album_1", "p", "100.0", *tags]
            f.write("\t".join(cols) + "\n")


def _setup(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_tsv(cache / "autotagging_genre.tsv", [
        (214, ["genre---punkrock", "genre---rock"]),
        (382, ["genre---classical"]),
    ])
    _write_tsv(cache / "autotagging_moodtheme.tsv", [(214, ["mood/theme---energetic"])])
    _write_tsv(cache / "autotagging_instrument.tsv", [(382, ["instrument---piano"])])
    corpus = [
        {"track_id": "tier2:jamendo:214", "external_ids": {"jamendoTrackId": "214"}},
        {"track_id": "tier2:jamendo:382", "external_ids": {"jamendoTrackId": "382"}},
        {"track_id": "tier2:jamendo:999", "external_ids": {"jamendoTrackId": "999"}},  # untagged
    ]
    cpath = tmp_path / "corpus.json"
    cpath.write_text(json.dumps(corpus))
    out = tmp_path / "catalog_tags.json"
    build("rev", out, cache, str(cpath))
    return json.loads(out.read_text())


def test_join_coarse_and_no_backfill(tmp_path):
    data = _setup(tmp_path)
    t = data["tracks"]
    # join + coarse-genre mapping (punkrock + rock both -> rock)
    assert t["tier2:jamendo:214"]["genre"] == ["punkrock", "rock"]
    assert t["tier2:jamendo:214"]["coarseGenre"] == ["rock"]
    assert t["tier2:jamendo:214"]["mood"] == ["energetic"]
    # NO MuQ/backfill for a missing tag-type
    assert "instrument" not in t["tier2:jamendo:214"]
    assert "mood" not in t["tier2:jamendo:382"]
    assert t["tier2:jamendo:382"]["instrument"] == ["piano"]
    # untagged track is omitted entirely
    assert "tier2:jamendo:999" not in t


def test_support_counts_in_meta(tmp_path):
    data = _setup(tmp_path)
    sup = data["_meta"]["support"]
    assert sup["coarseGenre"]["rock"] == 1
    assert sup["coarseGenre"]["classical"] == 1
    assert sup["instrument"]["piano"] == 1
    assert sup["mood"]["energetic"] == 1


def test_rehash_manifest_adds_sidecar(tmp_path):
    sidecar = tmp_path / "catalog_tags.json"
    sidecar.write_text('{"x":1}')
    base = {"sha256": {"files": {"corpus.json": "abc"}, "combined": "old"}}
    out = rehash_manifest_with_sidecar({"sha256": {"files": {"corpus.json": "abc"}, "combined": "old"}}, sidecar)
    assert "catalog_tags.json" in out["sha256"]["files"]
    assert out["sha256"]["files"]["corpus.json"] == "abc"  # existing hash preserved
    assert out["sha256"]["combined"] != base["sha256"]["combined"]  # recomputed

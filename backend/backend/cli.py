"""Soundcheck CLI entry point.

Subcommands (built incrementally):
  score <path>   — analyze one file and print Track-shape JSON (step 4 ✓).
  ingest …       — build the corpus (step 7, added later).
  eval …         — measure the scorer against a golden set (step 11, later).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, clap_engine, config
from .librosa_engine import analyze_array
from .scoring import compute_report


def _analyze_with_genre(path: Path) -> tuple[dict, list[tuple[str, float]]]:
    """Single decode pass shared by librosa + CLAP. Returns (analysis_dict, genres)."""
    import librosa
    import soundfile as sf

    try:
        duration_full = float(sf.info(str(path)).duration)
    except Exception:
        duration_full = None

    y, sr = librosa.load(str(path), sr=config.ANALYSIS_SR, mono=False)
    analysis = analyze_array(y, sr, duration_override=duration_full)

    mono = librosa.to_mono(y) if y.ndim > 1 else y
    cap_n = int(config.CLIP_CAP_S * sr)
    if mono.shape[-1] > cap_n:
        mono = mono[:cap_n]

    clap_engine.load()
    emb = clap_engine.encode_audio(mono, sr)
    genres = clap_engine.top_genres(emb)

    return analysis, genres


def _track_from_analysis(
    path: Path,
    analysis: dict,
    source: str,
    *,
    genres: list[tuple[str, float]] | None = None,
) -> dict:
    """Wrap the librosa + CLAP + scoring output in the Track shape the UI consumes."""
    report = compute_report(analysis["raw"])
    g = genres or []
    return {
        "id": f"{source}-{path.stem}" if source != "upload" else "upload",
        "title": path.stem,
        "genre": g[0][0] if g else None,
        "genres": [{"label": lbl, "score": float(s)} for lbl, s in g],
        "durationSec": analysis["durationSec"],
        "source": source,
        "waveform": analysis["waveform"],
        "problems": analysis["problems"],
        **report,
    }


def cmd_score(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"score: file not found: {path}", file=sys.stderr)
        return 1
    analysis, genres = _analyze_with_genre(path)
    track = _track_from_analysis(path, analysis, source="local", genres=genres)
    json.dump(track, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def _process_source(src: dict):
    """Analyze one source descriptor → (Track-shape dict, embedding). None on failure."""
    import librosa
    import numpy as np
    import soundfile as sf

    path = Path(src["local_path"])
    if not path.exists():
        print(f"[ingest] missing audio: {path}", file=sys.stderr)
        return None

    try:
        duration_full = float(sf.info(str(path)).duration)
    except Exception:
        duration_full = src.get("duration")

    try:
        y, sr = librosa.load(str(path), sr=config.ANALYSIS_SR, mono=False)
    except Exception as e:
        print(f"[ingest] decode failed for {path.name}: {e}", file=sys.stderr)
        return None

    analysis = analyze_array(y, sr, duration_override=duration_full)

    mono = librosa.to_mono(y) if y.ndim > 1 else y
    cap_n = int(config.CLIP_CAP_S * sr)
    if mono.shape[-1] > cap_n:
        mono = mono[:cap_n]

    emb = clap_engine.encode_audio(mono, sr)
    genres = clap_engine.top_genres(emb)
    report = compute_report(analysis["raw"])

    title = path.stem
    if src.get("source") == "sonics":
        bits = []
        if src.get("source_genre"):
            bits.append(str(src["source_genre"]).title())
        if src.get("mood"):
            bits.append(str(src["mood"]).title())
        if bits:
            title = " · ".join(bits)

    track = {
        "id": src["id"],
        "title": title,
        "genre": genres[0][0] if genres else None,
        "genres": [{"label": lbl, "score": float(s)} for lbl, s in genres],
        "durationSec": analysis["durationSec"],
        "source": "corpus",
        "waveform": analysis["waveform"],
        "problems": analysis["problems"],
        **report,
    }
    if "attribution" in src:
        track["attribution"] = src["attribution"]
    if src.get("source_kind"):
        track["sourceKind"] = src["source_kind"]
    if src.get("source_genre"):
        track["sourceGenre"] = src["source_genre"]
    if src.get("corruption_kind"):
        track["corruptionKind"] = src["corruption_kind"]

    return track, emb


def cmd_ingest(args: argparse.Namespace) -> int:
    """SONICS + own + corruptions → corpus.json + embeddings.npy + corpus_stats.json + examples.json."""
    import random

    import numpy as np
    from tqdm import tqdm

    from . import corpus_io, corruptions, sonics

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(args.corpus_dir).resolve()
    corpus_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = corpus_io.read_existing_track_ids(out_dir)
    if existing_ids:
        print(
            f"[ingest] resuming: {len(existing_ids)} tracks already in {out_dir / 'corpus.json'}"
        )

    # 1. SONICS subsample (audio cached by huggingface_hub under HF_HOME).
    sonics_dir = (corpus_dir / "audio" / "sonics").resolve()
    sonics_picks: list[dict] = []
    if args.sonics_n > 0:
        # Parse --sonics-parts into a tuple of ints.
        if args.sonics_parts.strip().lower() == "all":
            parts = tuple(range(1, 11))
        else:
            parts = tuple(
                int(p) for p in args.sonics_parts.split(",") if p.strip()
            )
        print(f"[ingest] SONICS: requesting {args.sonics_n} picks, downloading parts {parts}")
        sonics_picks = sonics.download_subsample(
            args.sonics_n, sonics_dir, seed=args.seed, parts=parts
        )
        print(f"[ingest] SONICS: {len(sonics_picks)} tracks")

    # 2. Your own tracks (optional)
    own_picks: list[dict] = []
    if args.own_dir:
        own_dir = Path(args.own_dir).resolve()
        if own_dir.exists():
            AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
            for p in sorted(own_dir.iterdir()):
                if p.suffix.lower() in AUDIO_EXTS:
                    own_picks.append({
                        "id": f"own-{p.stem}",
                        "source": "own",
                        "local_path": str(p),
                    })
            print(f"[ingest] own: {len(own_picks)} tracks at {own_dir}")
        else:
            print(f"[ingest] own_dir not found ({own_dir}) — SONICS-only run")

    # 3. Load CLAP once for all analysis
    print("[ingest] loading CLAP …")
    clap_engine.load()

    new_tracks: list[dict] = []
    new_embeddings: list = []
    source_pool = sonics_picks + own_picks

    for src in tqdm(source_pool, desc="Analyze SONICS+own", unit="track"):
        if src["id"] in existing_ids:
            continue
        result = _process_source(src)
        if result is None:
            continue
        track, emb = result
        new_tracks.append(track)
        new_embeddings.append(emb)

    # 4. Pick clean tracks, synthesize 5 corruption variants of each, analyze them too.
    if args.corruption_subset > 0:
        clean_pairs = [
            (src, t)
            for src, t in zip(source_pool, new_tracks)
            if t.get("verdict") == "keep"
        ]
        if clean_pairs:
            rng = random.Random(args.seed + 1)
            n_corrupt = min(args.corruption_subset, len(clean_pairs))
            seeds = rng.sample(clean_pairs, n_corrupt)
            corruption_dir = (corpus_dir / "audio" / "corruptions").resolve()
            corruption_dir.mkdir(parents=True, exist_ok=True)

            for src, _ in tqdm(
                seeds, desc="Synthesize+analyze corruptions", unit="seed"
            ):
                src_path = Path(src["local_path"])
                for kind in corruptions.KINDS:
                    corrupted_id = f"{src['id']}__{kind}"
                    if corrupted_id in existing_ids:
                        continue
                    out_path = corruption_dir / f"{src_path.stem}__{kind}.wav"
                    cseed = corruptions.seed_from_id(corrupted_id)
                    try:
                        corruptions.corrupt_file(src_path, out_path, kind, seed=cseed)
                    except Exception as e:
                        print(
                            f"[ingest] {kind} corruption of {src_path.stem} failed: {e}",
                            file=sys.stderr,
                        )
                        continue
                    cs = {
                        "id": corrupted_id,
                        "source": "corruption",
                        "local_path": str(out_path),
                        "corruption_kind": kind,
                        "source_id": src["id"],
                    }
                    result = _process_source(cs)
                    if result is None:
                        continue
                    track, emb = result
                    new_tracks.append(track)
                    new_embeddings.append(emb)

    # 5. Merge with any existing payload for resumability
    if existing_ids and (out_dir / "corpus.json").exists():
        existing_payload = json.loads((out_dir / "corpus.json").read_text())
        all_tracks = existing_payload.get("tracks", []) + new_tracks
        existing_emb_path = out_dir / "embeddings.npy"
        existing_emb = (
            np.load(existing_emb_path)
            if existing_emb_path.exists()
            else np.zeros((0, config.CLAP_EMBED_DIM), dtype=np.float32)
        )
        if new_embeddings:
            embeddings = np.vstack([existing_emb, np.stack(new_embeddings)])
        else:
            embeddings = existing_emb
    else:
        all_tracks = new_tracks
        embeddings = (
            np.stack(new_embeddings)
            if new_embeddings
            else np.zeros((0, config.CLAP_EMBED_DIM), dtype=np.float32)
        )

    # 6. Write static assets
    stats = corpus_io.write_corpus(out_dir, all_tracks, embeddings)

    print()
    print(f"Ingest complete → {out_dir}")
    print(f"  Total tracks: {stats['total']}")
    print(f"  Kept: {stats['kept']} ({stats['keepPct']}%)")
    print(f"  Dropped: {stats['dropped']}")
    if stats["byMode"]:
        print(f"  By failure mode: {stats['byMode']}")
    print(f"  Embeddings: {embeddings.shape}")
    return 0


# --- step 11: golden set + eval --------------------------------------------

_MODE_LABELS = {
    "silence": "Silence",
    "clipping": "Clipping",
    "noise": "Noise",
    "truncation": "Truncation",
    "channel": "Dead channel",
    "duration": "Duration",
}

_CORRUPTION_TO_MODE = {
    "silence": "silence",
    "clip": "clipping",
    "truncate": "truncation",
    "noise": "noise",
    "dead_channel": "channel",
}


def _eval_scaffold(args: argparse.Namespace) -> int:
    """Emit a golden.json template pre-populated with the engine's predictions.

    Stratified pick (~30% corruptions covering all 5 critical modes + ~70%
    SONICS/own). Default labels = predictions, so labelling is "agree or
    correct" rather than "from scratch".
    """
    import random
    from datetime import datetime, timezone

    out_path = Path(args.golden).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus).resolve()
    if not corpus_path.exists():
        print(f"eval scaffold: corpus.json not found at {corpus_path}", file=sys.stderr)
        return 1

    corpus = json.loads(corpus_path.read_text())
    tracks = corpus.get("tracks", [])
    if not tracks:
        print("eval scaffold: corpus.json has no tracks", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    n_target = min(args.n, len(tracks))

    corruption_tracks = [t for t in tracks if t.get("corruptionKind")]
    other_tracks = [t for t in tracks if not t.get("corruptionKind")]
    n_corrupt_target = min(round(n_target * 0.3), len(corruption_tracks))

    # Stratify corruptions by kind so every failure mode appears in the golden set.
    by_kind: dict[str, list[dict]] = {}
    for t in corruption_tracks:
        by_kind.setdefault(t["corruptionKind"], []).append(t)
    for ks in by_kind.values():
        rng.shuffle(ks)

    corruption_picks: list[dict] = []
    if by_kind:
        per_kind = max(1, n_corrupt_target // max(1, len(by_kind)))
        for kind, ts in by_kind.items():
            corruption_picks.extend(ts[:per_kind])
        corruption_picks = corruption_picks[:n_corrupt_target]

    n_other_target = n_target - len(corruption_picks)
    other_picks = (
        rng.sample(other_tracks, min(n_other_target, len(other_tracks)))
        if other_tracks
        else []
    )

    picked = corruption_picks + other_picks
    rng.shuffle(picked)

    rows = []
    for t in picked:
        pred_verdict = t["verdict"]
        pred_primary = t.get("primaryFail")
        # Corruption tracks have known ground truth from the recipe; pre-fill
        # those labels accordingly. Others default to the prediction.
        if t.get("corruptionKind"):
            label_verdict = "drop"
            label_primary = _CORRUPTION_TO_MODE.get(t["corruptionKind"])
        else:
            label_verdict = pred_verdict
            label_primary = pred_primary
        rows.append({
            "id": t["id"],
            "title": t.get("title", t["id"]),
            "predicted_verdict": pred_verdict,
            "predicted_primary_fail": pred_primary,
            "label_verdict": label_verdict,
            "label_primary_fail": label_primary,
        })

    payload = {
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(rows),
        "instructions": (
            "For each row, leave `label_*` as-is to AGREE with the prediction, "
            "or change to CORRECT it. Then run `python -m backend.cli eval`."
        ),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {len(rows)}-row golden-set template → {out_path}")
    print("Edit `label_verdict` / `label_primary_fail` per row, then:")
    print("  python -m backend.cli eval")
    return 0


def _eval_run(args: argparse.Namespace) -> int:
    """Read labeled golden.json + corpus.json; write eval.json (page-shaped)."""
    from collections import Counter
    from datetime import datetime, timezone

    golden_path = Path(args.golden).resolve()
    if not golden_path.exists():
        print(f"eval: golden.json not found at {golden_path}", file=sys.stderr)
        print("      run `python -m backend.cli eval scaffold` first", file=sys.stderr)
        return 1

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(golden_path.read_text()).get("rows", [])
    if not rows:
        print("eval: golden.json has no rows", file=sys.stderr)
        return 1

    # Confusion matrix on verdict (predicted × labeled).
    kg = sum(1 for r in rows if r["predicted_verdict"] == "keep" and r["label_verdict"] == "keep")
    kb = sum(1 for r in rows if r["predicted_verdict"] == "keep" and r["label_verdict"] == "drop")
    dg = sum(1 for r in rows if r["predicted_verdict"] == "drop" and r["label_verdict"] == "keep")
    db = sum(1 for r in rows if r["predicted_verdict"] == "drop" and r["label_verdict"] == "drop")
    labeled_good = kg + dg
    labeled_bad = kb + db
    total = labeled_good + labeled_bad

    keep_total = kg + kb
    keep_precision = kg / keep_total if keep_total > 0 else 1.0

    # Per-failure-mode precision / recall (only modes with ≥1 label).
    by_mode: list[dict] = []
    for mid, mlabel in _MODE_LABELS.items():
        tp = sum(1 for r in rows if r["predicted_primary_fail"] == mid and r["label_primary_fail"] == mid)
        fp = sum(1 for r in rows if r["predicted_primary_fail"] == mid and r["label_primary_fail"] != mid)
        fn = sum(1 for r in rows if r["predicted_primary_fail"] != mid and r["label_primary_fail"] == mid)
        n_labeled = sum(1 for r in rows if r["label_primary_fail"] == mid)
        if n_labeled == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        by_mode.append({
            "label": mlabel,
            "n": n_labeled,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        })

    payload = {
        "goldenSet": total,
        "labeledGood": labeled_good,
        "labeledBad": labeled_bad,
        "confusion": {"keptGood": kg, "keptBad": kb, "droppedGood": dg, "droppedBad": db},
        "byMode": by_mode,
        "version": __version__,
        "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Eval complete → {out_path}")
    print(f"  Golden set: {total} tracks  ({labeled_good} good · {labeled_bad} broken)")
    print(f"  Confusion: kept {kg} good + {kb} broken · dropped {dg} good + {db} broken")
    print(f"  Keep-precision: {keep_precision:.3f}   (target: ≥ 0.90)")
    if keep_precision < 0.90 and kb > 0:
        slipped = Counter(
            r.get("label_primary_fail")
            for r in rows
            if r["predicted_verdict"] == "keep" and r["label_verdict"] == "drop"
        )
        print(f"  ⚠  {kb} broken tracks slipped into KEEP. Most common labeled modes:")
        for m, c in slipped.most_common(3):
            print(f"      {m}: {c}")
        print("      Consider tightening the relevant thresholds in backend/backend/signals.py.")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    return _eval_scaffold(args) if args.action == "scaffold" else _eval_run(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.cli",
        description=f"Soundcheck CLI (v{__version__})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score", help="Score one audio file → Track-shape JSON")
    p_score.add_argument("path", help="path to an audio file")
    p_score.set_defaults(func=cmd_score)

    p_ing = sub.add_parser(
        "ingest",
        help="Build the static corpus (SONICS + own + corruptions → corpus.json + embeddings.npy)",
    )
    p_ing.add_argument("--sonics-n", type=int, default=300, help="how many SONICS tracks to sample")
    p_ing.add_argument(
        "--sonics-parts",
        default="1",
        help=(
            "Which SONICS zip parts to download (1–10). Comma-separated, e.g. "
            "'1,2,3' or 'all'. Each part is ~3.5 GB; one part yields ~10%% of "
            "the requested --sonics-n picks. Use 'all' for full coverage (~35 GB)."
        ),
    )
    p_ing.add_argument(
        "--own-dir",
        default="../corpus/audio/own",
        help="dir with your own tracks (mp3/wav/etc.); empty/missing → SONICS-only",
    )
    p_ing.add_argument(
        "--corruption-subset",
        type=int,
        default=60,
        help="how many clean tracks to corrupt (5 variants each)",
    )
    p_ing.add_argument(
        "--out",
        default="../quality-scorer/public/corpus",
        help="output dir for corpus.json + embeddings.npy",
    )
    p_ing.add_argument(
        "--corpus-dir",
        default="../corpus",
        help="working dir for downloaded audio + synthesized corruptions",
    )
    p_ing.add_argument("--seed", type=int, default=42)
    p_ing.set_defaults(func=cmd_ingest)

    p_eval = sub.add_parser(
        "eval",
        help="Golden-set evaluation: `eval scaffold` then `eval` (run).",
    )
    p_eval.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=["run", "scaffold"],
        help="`scaffold` to emit a golden.json template, default `run` to compute metrics",
    )
    p_eval.add_argument("--n", type=int, default=120, help="(scaffold) target golden-set size")
    p_eval.add_argument(
        "--corpus",
        default="../quality-scorer/public/corpus/corpus.json",
    )
    p_eval.add_argument(
        "--golden",
        default="../corpus/data/golden.json",
        help="path to golden.json (read for run, written for scaffold)",
    )
    p_eval.add_argument(
        "--out",
        default="../quality-scorer/public/corpus/eval.json",
        help="(run) eval.json output",
    )
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

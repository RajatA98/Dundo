"""FastAPI service — `POST /analyze`, `POST /neighbors`, `GET /health`.

The lifespan startup loads CLAP **and** the corpus (`corpus.json` +
`embeddings.npy` + `segment_embeddings.npz`) once into memory so similarity
queries are microsecond-fast.

Endpoints:
  - `/analyze`   — single-track scoring (Soundcheck): the technical-quality gate.
  - `/neighbors` — similarity audit (Twin Check): given an uploaded track, return
    the top-k most similar tracks already in the catalog with mean-pooled and
    max-segment similarity metrics.

Errors are returned as `{"error": "<code>"}` to match the frontend's `api.js`:
  - `unsupported_media` (415) — wrong mime / extension
  - `empty_file`        (422) — zero-byte upload
  - `file_too_large`    (413) — > MAX_UPLOAD_BYTES (~50 MB)
  - `decode_failed`     (422) — librosa couldn't decode
  - `empty_audio`       (422) — decoded but no samples
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ADR-0002: clap_engine is no longer the primary encoder; muq_engine took its
# place via clap_windowed's swap. We still import clap_engine here only because
# legacy code paths may reference it; the encoder load + genre tagging both go
# through muq_engine.
from . import __version__, artist_response, context_token, corpus_dataset, muq_engine, narrative_telemetry, clap_windowed, config, evidence_tags, mir_features, similarity
from .artist import ArtistNeighborsResponse, Criterion, EvidenceTags
from .librosa_engine import analyze_array
from .scoring import compute_report

# ADR-0007: SIMILARITY_BACKEND flag lets us swap NumPy → FAISS Flat without
# changing any return shape. Default stays NumPy. The bench at
# `python -m backend.scripts.bench_similarity` documents the crossover
# (~10k tracks) where FAISS starts paying for itself.
_SIMILARITY_BACKEND = os.getenv("SIMILARITY_BACKEND", "numpy").strip().lower()
if _SIMILARITY_BACKEND == "faiss":
    from . import similarity_faiss
    if not similarity_faiss.is_available():
        print("[api] SIMILARITY_BACKEND=faiss but faiss is not installed; falling back to numpy.")
        _SIMILARITY_BACKEND = "numpy"
        _top_k_neighbors = similarity.top_k_neighbors
    else:
        print("[api] SIMILARITY_BACKEND=faiss — using similarity_faiss.top_k_neighbors_faiss")
        _top_k_neighbors = similarity_faiss.top_k_neighbors_faiss
else:
    _top_k_neighbors = similarity.top_k_neighbors

# Optional Sentry error tracking. No-op when SENTRY_DSN is unset.
_sentry_dsn = os.getenv("SENTRY_DSN", "").strip()
if _sentry_dsn:
    import sentry_sdk
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
        release=__version__,
    )

# CPU torch isn't reliably thread-safe; serialize CLAP encodes.
_clap_lock = threading.Lock()

# All MuQ/torch inference runs on ONE dedicated thread (max_workers=1).
#
# Why a single dedicated thread and not Starlette's run_in_threadpool:
#   - Heavy sync inference must run OFF the event loop, or one upload blocks the
#     loop and every overlapping request (and /health) stalls.
#   - But torch's OpenMP intra-op pool is bound to the thread that first runs a
#     forward pass. Starlette's threadpool hands each call a DIFFERENT worker
#     thread, so the multi-threaded OMP pool deadlocks on the thread-affinity
#     mismatch — the server wedges on the very first real upload. (Capping
#     OMP/torch to 1 thread hid this but made each inference ~40s instead of ~7s.)
#   - A single-worker executor fixes both: every forward pass (warm-up included,
#     see _warm_model_and_corpus) runs on the SAME thread, so the OMP pool is
#     created once and reused — multi-threaded (all cores, ~7s) AND deadlock-free.
#     max_workers=1 also serialises the pipeline, so nothing oversubscribes.
# numba stays single-thread via the Dockerfile ENV (NUMBA_NUM_THREADS=1).
_infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")


async def _run_pipeline(raw: bytes, ext: str) -> dict | JSONResponse:
    """Run _decode_and_pipeline on the dedicated inference thread (off the loop)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_infer_executor, _decode_and_pipeline, raw, ext)

# In-memory corpus for /neighbors. Loaded once at startup.
_corpus_tracks: list[dict] = []
_corpus_embeddings: np.ndarray | None = None
_corpus_by_id: dict[str, dict] = {}
_flat_catalog: similarity.FlatCatalog | None = None
_catalog_cosine_distribution: np.ndarray | None = None  # sorted upper-tri off-diag pairwise cosines
_model_sha: str = ""
_catalog_sha: str = ""  # sha256 of manifest.json bytes; used in contextToken claims
_threshold_default: float = config.SIMILARITY_THRESHOLD_DEFAULT
_artists_by_id: dict[str, dict] | None = None
_track_to_artist: dict[str, str] | None = None
# Evidence Layer: {track_id: {coarseGenre/instrument/mood: [...]}} — support-filtered MTG tags.
_catalog_tags: dict[str, dict] = {}
EVIDENCE_SUPPORT_MIN = 50
_warm_ready = False
_warm_started = False
_warm_error: str | None = None
_warm_lock = threading.Lock()


def _default_corpus_dir() -> Path:
    """Search for the corpus next to the repo's quality-scorer/."""
    here = Path(__file__).resolve()
    # backend/backend/api.py → repo_root/quality-scorer/public/corpus
    return here.parents[2] / "quality-scorer" / "public" / "corpus"


def _load_catalog_tags(corpus_dir: Path) -> dict[str, dict]:
    """Load the Evidence Layer sidecar, keeping only labels above the support floor.

    Returns {track_id: {coarseGenre/instrument/mood: [labels]}}. Empty when absent (older
    catalogs) — the evidence block then simply never renders.
    """
    tpath = corpus_dir / "catalog_tags.json"
    if not tpath.exists():
        return {}
    raw = json.loads(tpath.read_text())
    support = (raw.get("_meta") or {}).get("support") or {}
    allowed = {f: {l for l, c in counts.items() if c >= EVIDENCE_SUPPORT_MIN} for f, counts in support.items()}
    out: dict[str, dict] = {}
    for tid, entry in (raw.get("tracks") or {}).items():
        kept = {}
        for field in ("coarseGenre", "instrument", "mood"):
            vals = [v for v in (entry.get(field) or []) if not allowed.get(field) or v in allowed[field]]
            if vals:
                kept[field] = vals
        if kept:
            out[tid] = kept
    return out


def _load_corpus() -> None:
    """Populate corpus globals from disk if all corpus artifacts are present."""
    global _corpus_tracks, _corpus_embeddings, _corpus_by_id, _flat_catalog
    global _catalog_cosine_distribution
    global _model_sha, _catalog_sha, _threshold_default
    global _artists_by_id, _track_to_artist, _catalog_tags
    corpus_dir = corpus_dataset.resolve_corpus_dir(Path(os.getenv("CORPUS_DIR", str(_default_corpus_dir()))))
    cpath = corpus_dir / "corpus.json"
    epath = corpus_dir / "embeddings.npy"
    spath = corpus_dir / "segment_embeddings.npz"
    mpath = corpus_dir / "manifest.json"
    apath = corpus_dir / "artists.json"
    missing = [p.name for p in (cpath, epath, spath, mpath) if not p.exists()]
    if missing:
        print(
            f"[api] corpus not found at {corpus_dir} "
            f"(missing: {', '.join(missing)}) "
            f"— /neighbors will return no_corpus"
        )
        _corpus_tracks = []
        _corpus_embeddings = None
        _corpus_by_id = {}
        _flat_catalog = None
        _model_sha = ""
        _catalog_sha = ""
        _threshold_default = config.SIMILARITY_THRESHOLD_DEFAULT
        _artists_by_id = None
        _track_to_artist = None
        _catalog_tags = {}
        return
    try:
        data = json.loads(cpath.read_text())
        _corpus_tracks = data if isinstance(data, list) else data.get("tracks", [])
        _corpus_embeddings = np.load(epath).astype(np.float32)
        with np.load(spath) as npz:
            segment_embeddings = {k: npz[k].astype(np.float32) for k in npz.files}
        manifest_bytes = mpath.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        _model_sha = str(manifest.get("model_sha") or "unpinned")
        if _model_sha == "unpinned":
            print("[api] WARNING manifest missing model_sha; using 'unpinned'")
        # catalog_sha = sha256 of manifest.json bytes. Captures every
        # meaningful catalog regeneration (model swap, threshold change,
        # track count change) in a single stable hash. Embedded in every
        # contextToken so /narrative can detect stale tokens after redeploy.
        _catalog_sha = hashlib.sha256(manifest_bytes).hexdigest()
        _threshold_default = similarity.threshold_from_manifest(manifest)
        _flat_catalog = similarity.build_flat_catalog(_corpus_tracks, _corpus_embeddings, segment_embeddings)
        del segment_embeddings
        _catalog_cosine_distribution = similarity.compute_catalog_distribution(_flat_catalog)
        _corpus_by_id = {str(row["track_id"]): row for row in _corpus_tracks if row.get("track_id")}
        if apath.exists():
            artists_list = json.loads(apath.read_text())
            if not isinstance(artists_list, list):
                raise ValueError("artists.json must contain a list")
            _artists_by_id = artist_response.index_artists(artists_list)
            _track_to_artist = artist_response.build_track_to_artist(artists_list)
            print(f"[api] artists loaded: {len(_artists_by_id)} artists")
        else:
            _artists_by_id = None
            _track_to_artist = None
            print("[api] artists.json not found — /neighbors will use legacy track-shaped response")
        _catalog_tags = _load_catalog_tags(corpus_dir)
        if _catalog_tags:
            print(f"[api] evidence tags loaded: {len(_catalog_tags)} tracks")
        if _corpus_embeddings.shape[0] != len(_corpus_tracks):
            print(
                f"[api] WARNING corpus length {len(_corpus_tracks)} ≠ embeddings rows "
                f"{_corpus_embeddings.shape[0]} — /neighbors may be inconsistent"
            )
        print(
            f"[api] corpus loaded: {len(_corpus_tracks)} tracks · "
            f"embeddings {_corpus_embeddings.shape} · segments {_flat_catalog.segs_flat.shape[0]}"
        )
    except Exception as e:
        print(f"[api] corpus load failed: {e!r}")
        _corpus_tracks = []
        _corpus_embeddings = None
        _corpus_by_id = {}
        _flat_catalog = None
        _catalog_cosine_distribution = None
        _model_sha = ""
        _catalog_sha = ""
        _threshold_default = config.SIMILARITY_THRESHOLD_DEFAULT
        _artists_by_id = None
        _track_to_artist = None
        _catalog_tags = {}


@asynccontextmanager
async def lifespan(_app):
    _start_warmup_thread()
    yield


app = FastAPI(title="Dundo", version=__version__, lifespan=lifespan)

_CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_CORS_ORIGIN],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# --- shared validation + decode + analyze --------------------------------------

def _err(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code})


def _start_warmup_thread() -> None:
    """Start model + corpus loading without blocking ASGI startup."""
    global _warm_started, _warm_error
    with _warm_lock:
        if _warm_ready or _warm_started:
            return
        _warm_started = True
        _warm_error = None
        threading.Thread(target=_warm_model_and_corpus, name="dundo-warmup", daemon=True).start()


def _warm_inference_once() -> None:
    """Run one MuQ encode + one numba MIR compute on 2s of silence.

    Executed on _infer_executor so torch's OpenMP pool and numba's JIT bind to
    the dedicated inference thread — the same thread every real request uses.
    """
    sr = config.AUDIO_ENCODER_SAMPLE_RATE
    silence = np.zeros(sr * 2, dtype=np.float32)
    muq_engine.encode_audio(silence, sr)
    try:
        mir_features.compute(silence, sr)
    except Exception as _exc:  # MIR warm-up is best-effort
        print(f"[api] MIR warm-up skipped: {_exc!r}")


def _warm_model_and_corpus() -> None:
    global _warm_ready, _warm_started, _warm_error
    try:
        muq_engine.load()
        _load_corpus()
        # Warm the inference path ON THE DEDICATED EXECUTOR THREAD. Two reasons:
        # (1) the first MuQ forward pass + numba MIR JIT are ~3x slower cold, which
        # could push the first upload past the gateway timeout; (2) crucially, this
        # binds torch's OpenMP pool (and numba's JIT) to the SAME thread every real
        # request will run on, so the multi-threaded OMP pool never deadlocks on a
        # thread-affinity mismatch. Run it through _infer_executor and wait.
        try:
            _infer_executor.submit(_warm_inference_once).result()
        except Exception as _exc:
            print(f"[api] inference warm-up skipped: {_exc!r}")
        _warm_ready = True
    except Exception as exc:
        _warm_error = repr(exc)
        print(f"[api] warmup failed: {exc!r}")
    finally:
        _warm_started = False


def _warming_err() -> JSONResponse | None:
    if _warm_ready:
        return None
    return _err(503, "warming-up")


def _validate_upload(file: UploadFile, raw: bytes) -> JSONResponse | None:
    ext = Path(file.filename or "").suffix.lower()
    mime = (file.content_type or "").lower()
    if ext not in config.ALLOWED_EXTENSIONS and not mime.startswith(config.ALLOWED_MIME_PREFIX):
        return _err(415, "unsupported_media")
    if not raw:
        return _err(422, "empty_file")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        return _err(413, "file_too_large")
    return None


def _decode_and_pipeline(raw: bytes, ext: str = "") -> dict | JSONResponse:
    """Decode bytes; run librosa + CLAP; return all artifacts (analysis, embedding, genres, report).

    Returns a dict or a JSONResponse on error.

    `ext` is the upload's file extension (e.g. ".m4a"). It's used only as the
    suffix on the temp-file fallback path: when librosa.load on a BytesIO
    fails for an AAC-LC `.m4a` upload (libsndfile can't decode AAC, and
    audioread's ffmpeg fallback requires a path not a BytesIO), we write the
    bytes to a temp file with the right suffix and retry. The suffix matters
    because ffmpeg's format dispatch is partially extension-driven.
    """
    try:
        duration_full = float(sf.info(io.BytesIO(raw)).duration)
    except Exception:
        duration_full = None
    try:
        y, sr = librosa.load(io.BytesIO(raw), sr=config.ANALYSIS_SR, mono=False)
    except Exception:
        # AAC-LC `.m4a` / other libsndfile-unsupported formats hit this path.
        # Write to a temp file (with the upload's extension as the suffix) and
        # retry — audioread will then dispatch to ffmpeg with a real path.
        import tempfile
        suffix = ext if ext and ext.startswith(".") else ""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(raw)
                tmp.flush()
                y, sr = librosa.load(tmp.name, sr=config.ANALYSIS_SR, mono=False)
        except Exception:
            return _err(422, "decode_failed")
    if (y if y.ndim == 1 else y).shape[-1] == 0:
        return _err(422, "empty_audio")

    analysis = analyze_array(y, sr, duration_override=duration_full)
    mono = librosa.to_mono(y) if y.ndim > 1 else y
    cap_n = int(config.CLIP_CAP_S * sr)
    if mono.shape[-1] > cap_n:
        mono = mono[:cap_n]
    with _clap_lock:
        emb, segment_embeddings = clap_windowed.encode_windowed(mono, sr, max_seconds=None)
    genres = muq_engine.top_genres(emb)
    report = compute_report(analysis["raw"])

    # ADR-0004: compute the four locked MIR criteria on the query audio.
    # The same mono+capped buffer that drives MuQ-MuLan is the right input —
    # we want the criteria computed against the same time region the embedding
    # was computed over so the per-criterion comparisons are self-consistent.
    try:
        query_mir = mir_features.compute(mono, sr)
    except Exception as exc:
        print(f"[api] mir_features.compute failed: {exc!r}")
        query_mir = None

    return {
        "analysis": analysis,
        "report": report,
        "genres": genres,
        "emb": emb,
        "segment_embeddings": segment_embeddings,
        "mir": query_mir,
    }


def _build_criteria_block(query_mir, match_mir_dict) -> dict | None:
    """Per ADR-0004: compose the four-criterion comparison block for one neighbor.

    Args:
        query_mir: MirFeatures dataclass (or None) computed on the upload.
        match_mir_dict: dict from corpus.json `mir_features` field, or None.

    Returns:
        Dict with `tempo`/`key`/`harmonic`/`timbre` entries, or None when
        either side is missing MIR data (which is the case for un-backfilled
        catalog tracks during the rollout window).
    """
    if query_mir is None or not match_mir_dict:
        return None
    try:
        match_mir = mir_features.MirFeatures.from_dict(match_mir_dict)
    except (KeyError, TypeError, ValueError):
        return None

    tempo_cmp = similarity.compare_tempos(query_mir.tempo_bpm, match_mir.tempo_bpm)
    key_cmp = similarity.compare_keys(query_mir.key, query_mir.mode, match_mir.key, match_mir.mode)
    chroma_cmp = similarity.compare_chroma_vectors(query_mir.chroma_mean, match_mir.chroma_mean)
    timbre_cmp = similarity.compare_timbre_vectors(query_mir.timbre_mean, match_mir.timbre_mean)

    return {
        "tempo": {
            "queryValue": round(float(query_mir.tempo_bpm), 1),
            "matchValue": round(float(match_mir.tempo_bpm), 1),
            "agreement": float(tempo_cmp["agreement"]),
            "label": str(tempo_cmp["label"]),
        },
        "key": {
            "queryValue": f"{query_mir.key} {query_mir.mode}",
            "matchValue": f"{match_mir.key} {match_mir.mode}",
            "agreement": float(key_cmp["agreement"]),
            "label": str(key_cmp["label"]),
        },
        "harmonic": {
            "agreement": float(chroma_cmp["agreement"]),
            "label": str(chroma_cmp["label"]),
        },
        "timbre": {
            "agreement": float(timbre_cmp["agreement"]),
            "label": str(timbre_cmp["label"]),
        },
    }


def _build_track(file: UploadFile, pipeline: dict, *, source: str, id_: str) -> dict:
    return {
        "id": id_,
        "title": Path(file.filename or id_).stem or id_,
        "genre": pipeline["genres"][0][0] if pipeline["genres"] else None,
        "genres": [{"label": lbl, "score": float(s)} for lbl, s in pipeline["genres"]],
        "durationSec": pipeline["analysis"]["durationSec"],
        "source": source,
        "waveform": pipeline["analysis"]["waveform"],
        "problems": pipeline["analysis"]["problems"],
        **pipeline["report"],
    }


def _energy_band(loudness_db: float | None) -> str | None:
    """Map mean RMS loudness (dBFS) to an honest qualitative intensity band.
    A band, never a fake 0-1 score — 'measured, not claimed'."""
    if loudness_db is None:
        return None
    if loudness_db >= -16.0:
        return "High"
    if loudness_db >= -24.0:
        return "Medium"
    return "Low"


def _build_query_summary(pipeline: dict, tag_profile: dict[str, list[dict]]) -> dict:
    """The 'Your song's stats' snapshot for the upload — honest, from analysis we
    already run: tempo, key+mode (+confidence), duration, an energy band, and the
    upload's real k-NN-propagated genre/mood/instrument tags. Perceptual things
    are bands/tags, never fabricated decimals."""
    mir = pipeline.get("mir")
    analysis = pipeline.get("analysis") or {}
    summary: dict = {
        "durationSec": analysis.get("durationSec"),
        "energyBand": _energy_band(analysis.get("loudnessDb")),
        "tags": tag_profile or {},
    }
    if mir is not None:
        tempo = float(getattr(mir, "tempo_bpm", 0.0) or 0.0)
        summary["tempoBpm"] = round(tempo) if tempo > 0 else None
        summary["key"] = getattr(mir, "key", None)
        summary["mode"] = getattr(mir, "mode", None)
        summary["keyConfidence"] = round(float(getattr(mir, "key_confidence", 0.0) or 0.0), 2)
    return summary


_CRITERION_LABELS = {
    "tempo": "Tempo",
    "key": "Key",
    "harmonic": "Harmonic",
    "timbre": "Timbre",
}


def _criteria_block_to_artist_criteria(criteria_block: dict | None) -> list[Criterion]:
    if not criteria_block:
        return []
    criteria: list[Criterion] = []
    for cid in ("tempo", "key", "harmonic", "timbre"):
        entry = criteria_block.get(cid)
        if not entry:
            continue
        criteria.append(
            Criterion(
                label=_CRITERION_LABELS[cid],
                detail=str(entry.get("label") or ""),
                agreement=float(entry.get("agreement", 0.0)),
            )
        )
    return criteria


def _track_preview_url(track: dict) -> str | None:
    external_ids = track.get("external_ids") or {}
    return (
        external_ids.get("jamendoAudioUrl")
        or external_ids.get("previewUrl")
        or track.get("previewUrl")
        or track.get("audioUrl")
    )


# ISO3 → (country name, demonym) for the common CC/indie countries, so location
# facts validate whether the LLM writes "AUS", "Australia", or "Australian".
# Unknown codes degrade gracefully (alias set is just city + code).
_ISO3_COUNTRY = {
    "USA": ("United States", "American"), "GBR": ("United Kingdom", "British"),
    "DEU": ("Germany", "German"), "FRA": ("France", "French"), "ITA": ("Italy", "Italian"),
    "ESP": ("Spain", "Spanish"), "NLD": ("Netherlands", "Dutch"), "BEL": ("Belgium", "Belgian"),
    "SWE": ("Sweden", "Swedish"), "NOR": ("Norway", "Norwegian"), "FIN": ("Finland", "Finnish"),
    "DNK": ("Denmark", "Danish"), "POL": ("Poland", "Polish"), "CZE": ("Czechia", "Czech"),
    "AUT": ("Austria", "Austrian"), "CHE": ("Switzerland", "Swiss"), "PRT": ("Portugal", "Portuguese"),
    "IRL": ("Ireland", "Irish"), "AUS": ("Australia", "Australian"), "NZL": ("New Zealand", "New Zealander"),
    "CAN": ("Canada", "Canadian"), "BRA": ("Brazil", "Brazilian"), "ARG": ("Argentina", "Argentine"),
    "MEX": ("Mexico", "Mexican"), "JPN": ("Japan", "Japanese"), "RUS": ("Russia", "Russian"),
    "UKR": ("Ukraine", "Ukrainian"), "GRC": ("Greece", "Greek"), "ROU": ("Romania", "Romanian"),
    "HUN": ("Hungary", "Hungarian"), "R4S": ("Serbia", "Serbian"), "HRV": ("Croatia", "Croatian"),
    "TUR": ("Turkey", "Turkish"), "ISR": ("Israel", "Israeli"), "IND": ("India", "Indian"),
    "CHL": ("Chile", "Chilean"), "COL": ("Colombia", "Colombian"), "ZAF": ("South Africa", "South African"),
    "SVN": ("Slovenia", "Slovenian"), "SVK": ("Slovakia", "Slovak"), "EST": ("Estonia", "Estonian"),
    "NCL": ("New Caledonia", "New Caledonian"),
}


def _artist_knowledge_for(track_id: str, match) -> dict:
    """Build the matched artist's catalog-fact object (Narrative v2).

    Real facts we already hold: the artist's location (with canonical aliases so a
    location fact validates regardless of spelling) and their own MTG genre/mood/
    instrument tags (support-filtered, capped — used sparingly, never a tag dump).
    Empty dict when we know nothing → the narrative stays acoustic/evidence-only.
    """
    tags = _catalog_tags.get(track_id) or {}
    genres = list(tags.get("coarseGenre") or [])[:4]
    moods = list(tags.get("mood") or [])[:4]
    instruments = list(tags.get("instrument") or [])[:4]

    location_raw = (getattr(match, "location", None) or "").strip()
    display_location = location_raw
    aliases: set[str] = set()
    if location_raw:
        parts = [p.strip() for p in location_raw.split(",") if p.strip()]
        aliases.add(location_raw.lower())
        for p in parts:
            aliases.add(p.lower())
        # last part is often an ISO3 country code → expand to name + demonym
        if parts:
            code = parts[-1].upper()
            named = _ISO3_COUNTRY.get(code)
            if named:
                name, demonym = named
                aliases.update({name.lower(), demonym.lower()})
                city = parts[0] if len(parts) > 1 else None
                display_location = f"{city}, {name}" if city else name

    ak: dict = {}
    if display_location:
        # The display value itself is a valid alias, so exact-membership validation
        # accepts the LLM citing the location string it was shown (Codex review).
        aliases.add(display_location.lower())
        ak["location"] = display_location
        ak["locationAliases"] = sorted(a for a in aliases if a)
    if genres:
        ak["genres"] = genres
    if moods:
        ak["moods"] = moods
    if instruments:
        ak["instruments"] = instruments
    # similarArtists left empty for v1 (build-time graph is the Codex-approved fast-follow).
    return ak


def _compact_query_descriptors(query_summary: dict) -> dict:
    """Compact subset of querySummary for the Suno coach (signed into the token):
    tempo, key/mode, and the upload's top inferred genre/mood tags."""
    tags = query_summary.get("tags") or {}
    out: dict = {}
    for k in ("tempoBpm", "key", "mode"):
        if query_summary.get(k):
            out[k] = query_summary[k]
    genres = [t["label"] for t in (tags.get("genre") or [])[:3] if t.get("label")]
    moods = [t["label"] for t in (tags.get("mood") or [])[:3] if t.get("label")]
    if genres:
        out["genres"] = genres
    if moods:
        out["moods"] = moods
    return out


def _issue_context_token_for_neighbors(
    query_fingerprint: str, neighbors: list[dict], query_descriptors: dict | None = None
) -> str | None:
    if not neighbors or not context_token.is_configured():
        return None
    neighbor_fragments: dict[str, dict] = {}
    for nb in neighbors:
        track = nb.get("track") or {}
        ts = nb.get("matchTimestamp") or {}
        neighbor_fragments[str(nb["trackId"])] = context_token.neighbor_context_fragment(
            track_id=str(nb["trackId"]),
            title=str(track.get("title") or nb["trackId"]),
            artist=nb.get("artistName") or track.get("artist"),
            query_window=(
                float(ts.get("queryStartSec", 0.0)),
                float(ts.get("queryEndSec", 0.0)),
            ),
            match_window=(
                float(ts.get("catalogStartSec", 0.0)),
                float(ts.get("catalogEndSec", 0.0)),
            ),
            raw_cosine=float(nb.get("rawCosine", 0.0)),
            criteria=_criteria_to_token_fragment(nb.get("criteria")),
            evidence_shared=nb.get("evidenceShared"),
            artist_knowledge=nb.get("artistKnowledge"),
        )
    return context_token.issue(
        query_fingerprint=query_fingerprint,
        model_sha=_model_sha or "unpinned",
        catalog_sha=_catalog_sha or "no-catalog",
        neighbors=neighbor_fragments,
        query_descriptors=query_descriptors,
    )


def _legacy_neighbors_response(
    query_track: dict,
    neighbors: list[dict],
    query_fingerprint: str,
    pipeline: dict,
) -> dict:
    specificity = float(similarity.query_specificity(pipeline["emb"].astype(np.float32), _flat_catalog))
    return {
        "query": query_track,
        "neighbors": neighbors,
        "topMeanPooledSimilarity": float(neighbors[0]["meanPooledSimilarity"]) if neighbors else 0.0,
        "topMaxSegmentSimilarity": float(neighbors[0]["maxSegmentSimilarity"]) if neighbors else 0.0,
        "topPercentileRank": float(neighbors[0]["percentileRank"]) if neighbors else 0.0,
        "topSimilarityLabel": neighbors[0]["similarityLabel"] if neighbors else "weak",
        "querySpecificity": specificity,
        "modelSha": _model_sha,
        "thresholdDefault": _threshold_default,
        "queryFingerprint": query_fingerprint,
        "contextToken": _issue_context_token_for_neighbors(query_fingerprint, neighbors),
    }


# --- endpoints -----------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model": muq_engine.model_id(),
        "modelSha": _model_sha,
        "version": __version__,
        "corpus": len(_corpus_tracks),
        "segments": int(_flat_catalog.segs_flat.shape[0]) if _flat_catalog else 0,
        "ready": _warm_ready,
        "warming": _warm_started and not _warm_ready,
        "warmupError": _warm_error,
    }


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    if (err := _warming_err()) is not None:
        return err
    raw = await file.read()
    if (err := _validate_upload(file, raw)) is not None:
        return err
    ext = Path(file.filename or "").suffix.lower()
    pipeline = await _run_pipeline(raw, ext)
    if isinstance(pipeline, JSONResponse):
        return pipeline
    return _build_track(file, pipeline, source="upload", id_="upload")


@app.post("/neighbors")
async def neighbors_endpoint(file: UploadFile = File(...), k: int = 5):
    """Similarity audit: top-k nearest tracks in the catalog."""
    if (err := _warming_err()) is not None:
        return err
    raw = await file.read()
    if (err := _validate_upload(file, raw)) is not None:
        return err
    # queryFingerprint: SHA-256 of the upload bytes. Embedded in contextToken
    # so /narrative can verify the same query is still in play. Stable across
    # re-uploads of the same file; cheap to compute.
    query_fingerprint = hashlib.sha256(raw).hexdigest()
    ext = Path(file.filename or "").suffix.lower()
    pipeline = await _run_pipeline(raw, ext)
    if isinstance(pipeline, JSONResponse):
        return pipeline
    query_track = _build_track(file, pipeline, source="upload", id_="upload")

    if _flat_catalog is None:
        return ArtistNeighborsResponse(matches=[], contextToken=None).model_dump()

    artist_mode = _artists_by_id is not None and _track_to_artist is not None

    neighbors = _top_k_neighbors(
        pipeline["emb"].astype(np.float32),
        pipeline["segment_embeddings"].astype(np.float32),
        _flat_catalog,
        k=max(int(k), 40) if artist_mode else int(k),
    )

    # ADR-0001: calibrate raw cosines against the catalog distribution so the
    # UI can render meaningful labels instead of "99.8% / 99.7% / 99.7%".
    distribution = _catalog_cosine_distribution if _catalog_cosine_distribution is not None else np.empty((0,), dtype=np.float32)
    for nb in neighbors:
        nb["track"] = _corpus_by_id.get(nb["trackId"], {})
        raw = float(nb["meanPooledSimilarity"])
        seg = float(nb["maxSegmentSimilarity"])
        pct = similarity.cosine_to_percentile(raw, distribution)
        nb["rawCosine"] = raw
        nb["percentileRank"] = float(pct)
        nb["similarityLabel"] = similarity.similarity_label(pct)
        nb["segmentSupport"] = seg
        # Calibrated 0-1 score for the UI bar width — uses percentile rank.
        nb["calibratedScore"] = float(pct)
        # Timestamp of the strongest segment match — what part of the query
        # lined up with what part of the catalog track. Window indices come
        # straight out of similarity.top_k_neighbors; we convert to seconds
        # using the locked 10 s window protocol.
        q_win = int(nb.pop("matchQueryWindow", 0))
        c_win = int(nb.pop("matchCatalogWindow", 0))
        win_s = float(config.CLAP_WINDOW_SECONDS)
        nb["matchTimestamp"] = {
            "queryStartSec": q_win * win_s,
            "queryEndSec": (q_win + 1) * win_s,
            "catalogStartSec": c_win * win_s,
            "catalogEndSec": (c_win + 1) * win_s,
            "windowSeconds": win_s,
        }

        # ADR-0004: attach the four-criterion comparison block when both the
        # query and the catalog track have MIR features available. Missing
        # MIR data on either side → null criteria; the frontend handles that
        # gracefully (criteria table just hides).
        nb["criteria"] = _build_criteria_block(pipeline.get("mir"), nb["track"].get("mir_features"))

    if not artist_mode:
        return _legacy_neighbors_response(query_track, neighbors, query_fingerprint, pipeline)

    ranked_tracks = [(str(nb["trackId"]), float(nb["rawCosine"])) for nb in neighbors]
    match_pairs = artist_response.build_artist_match_pairs(
        ranked_tracks,
        _artists_by_id,
        _track_to_artist,
        threshold=_threshold_default,
        k=3,
    )

    neighbors_by_id = {str(nb["trackId"]): nb for nb in neighbors}
    # Evidence Layer pool: every neighbor as (track_id, artist_id, sim) for k-NN tag propagation.
    # The artist id comes from _track_to_artist so the per-card candidate-exclusion is consistent.
    ev_neighbors = (
        [
            evidence_tags.Neighbor(
                track_id=str(nb["trackId"]),
                artist=_track_to_artist.get(str(nb["trackId"])),
                sim=float(nb["rawCosine"]),
            )
            for nb in neighbors
        ]
        if _catalog_tags
        else []
    )
    matches = []
    winning_neighbors: list[dict] = []
    for match, winning_track_id in match_pairs:
        nb = neighbors_by_id.get(winning_track_id)
        if not nb:
            continue
        track = nb.get("track") or {}
        match.similarity = float(nb["rawCosine"])
        match.representativeTrackId = winning_track_id
        match.criteria = _criteria_block_to_artist_criteria(nb.get("criteria"))
        if preview_url := _track_preview_url(track):
            match.previewUrl = preview_url
        if ev_neighbors:
            # candidate's artist excluded inside assemble_evidence_tags (no circular evidence).
            block = evidence_tags.assemble_evidence_tags(
                ev_neighbors,
                candidate_track_id=winning_track_id,
                candidate_artist=match.artistId,
                catalog_tags=_catalog_tags,
            )
            if block is not None:
                match.evidenceTags = EvidenceTags.model_validate(block)
                # carry the gated shared descriptors into the context token so /narrative
                # can ground on them even when MIR criteria are absent.
                nb["evidenceShared"] = block.get("shared") or []
        # Narrative v2: the matched artist's own catalog facts (location + tags), for the
        # context token. Distinct from evidenceShared (the overlap); validated in /narrative.
        nb["artistKnowledge"] = _artist_knowledge_for(winning_track_id, match)
        # The human display name ("Marc Teichert"), so the narrative names the artist
        # rather than the raw MTG slug ("artist_355362") carried on the track.
        nb["artistName"] = match.name
        matches.append(match)
        winning_neighbors.append(nb)

    # "Your song's stats": honest snapshot of the upload (tempo/key/duration/energy +
    # its real k-NN-propagated genre/mood/instrument tags, voted over all neighbors).
    tag_profile = (
        evidence_tags.assemble_query_profile(ev_neighbors, _catalog_tags) if ev_neighbors else {}
    )
    query_summary = _build_query_summary(pipeline, tag_profile)
    # The Suno coach grounds on the upload's detected descriptors — signed top-level in
    # the token (identical for all matches), per Codex's review of CODEX_SUNO_COACH.md.
    query_descriptors = _compact_query_descriptors(query_summary)
    ctx_token = _issue_context_token_for_neighbors(
        query_fingerprint, winning_neighbors, query_descriptors
    )
    return ArtistNeighborsResponse(
        matches=matches, contextToken=ctx_token, querySummary=query_summary
    ).model_dump()


def _criteria_to_token_fragment(criteria_block: dict | None) -> list[dict] | None:
    """Reshape /neighbors' criteria block into the list-of-CriterionContext
    form Codex's rag_narrative module expects.

    The /neighbors response groups criteria by id under a top-level dict;
    NarrativeContext takes a flat list of {id, queryValue, matchValue,
    agreement, label}. Convert here so the token payload matches the
    NarrativeContext shape directly.
    """
    if not criteria_block:
        return None
    out: list[dict] = []
    for cid in ("tempo", "key", "harmonic", "timbre"):
        entry = criteria_block.get(cid)
        if not entry:
            continue
        # harmonic + timbre come back from /neighbors without queryValue /
        # matchValue (only agreement + label) because we don't ship the raw
        # vectors. Substitute a shape marker so Codex's citation validator
        # has something to check the keys against without exposing internals.
        q_val = entry.get("queryValue")
        m_val = entry.get("matchValue")
        if cid in ("harmonic", "timbre") and q_val is None and m_val is None:
            q_val = {"vector": "elided"}
            m_val = {"vector": "elided"}
        out.append({
            "id": cid,
            "queryValue": q_val,
            "matchValue": m_val,
            "agreement": float(entry.get("agreement", 0.0)),
            "label": str(entry.get("label", "")),
        })
    return out or None


# --- /narrative -------------------------------------------------------------
#
# Stateless RAG explanatory layer over /neighbors. Client sends the
# contextToken received from /neighbors plus the trackId + mode it wants
# narrated; backend verifies the token (signature, expiry, model/catalog
# version), rebuilds NarrativeContext from the embedded claims, and delegates
# to Codex's rag_narrative module.
#
# Failure shape: typed `{"error": "<code>"}` JSON, status code by class:
#   503 narrative-disabled  — OPENAI_API_KEY or CONTEXT_TOKEN_HMAC_KEY absent
#   401 invalid-token       — signature mismatch (tampered or wrong secret)
#   412 token-expired       — past expiresAt
#   412 stale-token         — modelSha/catalogSha changed since issuance
#   400 malformed-token     — bad shape; not <body>.<sig>
#   404 not-in-context      — trackId wasn't part of the issued token
#   422 unsupported-mode    — mode wasn't "whySimilar" or "creatorAdvice"


class NarrativeRequest(BaseModel):
    contextToken: str = Field(..., min_length=1)
    trackId: str = Field(..., min_length=1)
    mode: str = Field(..., min_length=1)


_TOKEN_ERROR_TO_HTTP = {
    "malformed": (400, "malformed-token"),
    "invalid-signature": (401, "invalid-token"),
    "token-expired": (412, "token-expired"),
    "stale-model": (412, "stale-token"),
    "stale-catalog": (412, "stale-token"),
    "hmac-key-missing": (503, "narrative-disabled"),
}


@app.post("/narrative")
async def narrative_endpoint(req: NarrativeRequest):
    """RAG explanatory layer — see ADR-0005 for the full spec."""
    if (err := _warming_err()) is not None:
        return err
    with narrative_telemetry.measure_call(req.mode) as tel:
        # Gate 1: OpenAI key present. Without it we can't call GPT-4o-mini.
        if not os.getenv("OPENAI_API_KEY", "").strip():
            tel.set(error_code="narrative-disabled")
            return _err(503, "narrative-disabled")
        # Gate 2: HMAC key present. Without it we can't trust the token.
        if not context_token.is_configured():
            tel.set(error_code="narrative-disabled")
            return _err(503, "narrative-disabled")
        # Gate 3: mode is one of the supported values.
        if req.mode not in ("whySimilar", "creatorAdvice"):
            tel.set(error_code="unsupported-mode")
            return _err(422, "unsupported-mode")

        # Verify the token. TokenError.code maps directly to a typed HTTP response.
        try:
            verified = context_token.verify(
                req.contextToken,
                expected_model_sha=_model_sha or "unpinned",
                expected_catalog_sha=_catalog_sha or "no-catalog",
            )
        except context_token.TokenError as exc:
            status, code = _TOKEN_ERROR_TO_HTTP.get(exc.code, (400, "malformed-token"))
            tel.set(error_code=code)
            return _err(status, code)

        # Look up the requested trackId inside the verified token claims.
        fragment = verified.neighbors.get(req.trackId)
        if not fragment:
            tel.set(error_code="not-in-context", trackId=req.trackId)
            return _err(404, "not-in-context")

        # Lazy-import Codex's module. Keeping this inside the handler means the
        # FastAPI app boots and /neighbors keeps working even if rag_narrative
        # hasn't shipped yet. If it's missing at request time, surface as 503
        # narrative-disabled so the frontend's no-key fallback path handles it.
        try:
            from . import rag_narrative
        except ImportError:
            tel.set(error_code="narrative-disabled")
            return _err(503, "narrative-disabled")

        # Build NarrativeContext from the verified fragment. This is the Pydantic
        # model Codex defined; instantiating it here also validates the shape.
        try:
            context = rag_narrative.NarrativeContext(
                queryFingerprint=verified.queryFingerprint,
                trackId=fragment["trackId"],
                title=fragment.get("title", ""),
                artist=fragment.get("artist"),
                queryWindow=tuple(fragment["queryWindow"]),
                matchWindow=tuple(fragment["matchWindow"]),
                rawCosine=float(fragment["rawCosine"]),
                criteria=[
                    rag_narrative.CriterionContext(**c)
                    for c in (fragment.get("criteria") or [])
                ],
                evidenceShared=fragment.get("evidenceShared") or [],
                artistKnowledge=fragment.get("artistKnowledge") or {},
                queryDescriptors=verified.queryDescriptors or {},
                acrcloudCoverSongId=verified.acrcloudCoverSongId,
            )
        except Exception:
            # If the token fragment fails to materialize into a NarrativeContext,
            # surface as malformed rather than blowing up internally.
            tel.set(error_code="malformed-context", trackId=req.trackId)
            return _err(422, "malformed-context")

        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
        try:
            result = rag_narrative.generate_narrative(
                context,
                req.mode,
                model_sha=_model_sha or "unpinned",
                catalog_sha=_catalog_sha or "no-catalog",
                model_id=model_id,
            )
        except Exception as exc:
            print(f"[api] /narrative generate_narrative raised: {exc!r}")
            tel.set(error_code="narrative-error", trackId=req.trackId)
            return _err(500, "narrative-error")

        # Record the result kind. result.kind is the discriminator on all
        # three Pydantic variants (NarrativeResponse / LowConfidence /
        # NarrativeUnavailable). Approximate cost via prose char count;
        # we don't have token counts without re-tokenizing, but char-count
        # is the right directional signal for the stats endpoint.
        result_kind = getattr(result, "kind", None)
        completion_chars = 0
        if result_kind == "narrative":
            completion_chars = len(getattr(result, "prose", "") or "")
        # Rough prompt size estimate — system + user prompt char count.
        # narrative_telemetry treats this as char-not-token because tokenizer
        # access isn't worth the overhead for an in-process counter.
        prompt_chars_estimate = len(fragment.get("title", "")) + 600  # base + metadata
        tel.set(
            result_kind=result_kind,
            openai_called=(result_kind == "narrative" or result_kind == "unavailable"),
            gate_short_circuit=(result_kind == "low_confidence"),
            prompt_chars=prompt_chars_estimate,
            completion_chars=completion_chars,
            trackId=req.trackId,
        )

        # Pydantic v2 .model_dump() — uniform shape regardless of which result
        # variant came back. The `kind` discriminator lets the frontend route
        # rendering.
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result


@app.get("/narrative/stats")
def narrative_stats_endpoint() -> dict:
    """Return the in-process counters snapshot for the /narrative layer.

    Senior-reviewer-friendly visibility into what's actually happening in
    production — call counts, latency percentiles, mode distribution,
    error distribution, rough cost estimate. Counters reset on restart;
    this is not a long-term metrics store, it's a "right now" snapshot.

    Cost estimate is char-based × GPT-4o-mini pricing — directional, not
    accounting-grade. The honest framing from ADR-0005 holds.
    """
    return narrative_telemetry.snapshot()


def run() -> None:
    """Convenience launcher: `python -m backend.api` or `uvicorn backend.api:app`."""
    import uvicorn

    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    run()

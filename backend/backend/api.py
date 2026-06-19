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

import hashlib
import io
import json
import os
import threading
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
from . import __version__, acrcloud_engine, context_token, muq_engine, narrative_telemetry, clap_windowed, config, mir_features, similarity
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

# In-memory corpus for /neighbors. Loaded once at startup.
_corpus_tracks: list[dict] = []
_corpus_embeddings: np.ndarray | None = None
_corpus_by_id: dict[str, dict] = {}
_flat_catalog: similarity.FlatCatalog | None = None
_catalog_cosine_distribution: np.ndarray | None = None  # sorted upper-tri off-diag pairwise cosines
_model_sha: str = ""
_catalog_sha: str = ""  # sha256 of manifest.json bytes; used in contextToken claims
_threshold_default: float = config.SIMILARITY_THRESHOLD_DEFAULT


def _default_corpus_dir() -> Path:
    """Search for the corpus next to the repo's quality-scorer/."""
    here = Path(__file__).resolve()
    # backend/backend/api.py → repo_root/quality-scorer/public/corpus
    return here.parents[2] / "quality-scorer" / "public" / "corpus"


def _load_corpus() -> None:
    """Populate corpus globals from disk if all corpus artifacts are present."""
    global _corpus_tracks, _corpus_embeddings, _corpus_by_id, _flat_catalog
    global _catalog_cosine_distribution
    global _model_sha, _catalog_sha, _threshold_default
    corpus_dir = Path(os.getenv("CORPUS_DIR", str(_default_corpus_dir())))
    cpath = corpus_dir / "corpus.json"
    epath = corpus_dir / "embeddings.npy"
    spath = corpus_dir / "segment_embeddings.npz"
    mpath = corpus_dir / "manifest.json"
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
        _catalog_cosine_distribution = similarity.compute_catalog_distribution(_flat_catalog)
        _corpus_by_id = {str(row["track_id"]): row for row in _corpus_tracks if row.get("track_id")}
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


@asynccontextmanager
async def lifespan(_app):
    muq_engine.load()
    _load_corpus()
    yield


app = FastAPI(title="PiedPiper", version=__version__, lifespan=lifespan)

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
    acrcloud_n = int(15 * sr)
    acrcloud_slice = mono[:acrcloud_n]
    acrcloud_buf = io.BytesIO()
    sf.write(acrcloud_buf, acrcloud_slice, sr, format="WAV", subtype="PCM_16")
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
        "acrcloud_audio": acrcloud_buf.getvalue(),
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
        "acrcloudEnabled": acrcloud_engine.is_enabled(),
    }


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    raw = await file.read()
    if (err := _validate_upload(file, raw)) is not None:
        return err
    ext = Path(file.filename or "").suffix.lower()
    pipeline = _decode_and_pipeline(raw, ext=ext)
    if isinstance(pipeline, JSONResponse):
        return pipeline
    return _build_track(file, pipeline, source="upload", id_="upload")


@app.post("/neighbors")
async def neighbors_endpoint(file: UploadFile = File(...), k: int = 5):
    """Similarity audit: top-k nearest tracks in the catalog."""
    raw = await file.read()
    if (err := _validate_upload(file, raw)) is not None:
        return err
    # queryFingerprint: SHA-256 of the upload bytes. Embedded in contextToken
    # so /narrative can verify the same query is still in play. Stable across
    # re-uploads of the same file; cheap to compute.
    query_fingerprint = hashlib.sha256(raw).hexdigest()
    ext = Path(file.filename or "").suffix.lower()
    pipeline = _decode_and_pipeline(raw, ext=ext)
    if isinstance(pipeline, JSONResponse):
        return pipeline
    query_track = _build_track(file, pipeline, source="upload", id_="upload")

    if _flat_catalog is None:
        return {
            "query": query_track,
            "neighbors": [],
            "verdict": "no_corpus",
            "topMeanPooledSimilarity": 0.0,
            "topMaxSegmentSimilarity": 0.0,
            "modelSha": _model_sha,
            "thresholdDefault": _threshold_default,
            "acrcloud": acrcloud_engine.to_response_dict(acrcloud_engine.disabled_response()),
            "queryFingerprint": query_fingerprint,
            "contextToken": None,
        }

    neighbors = _top_k_neighbors(
        pipeline["emb"].astype(np.float32),
        pipeline["segment_embeddings"].astype(np.float32),
        _flat_catalog,
        k=k,
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

    specificity = float(similarity.query_specificity(pipeline["emb"].astype(np.float32), _flat_catalog))
    acr = acrcloud_engine.call_for_query(pipeline["acrcloud_audio"])
    acr_response = acrcloud_engine.to_response_dict(acr)

    # Codex round-2 Q3: stateless signed token replaces the in-memory cache.
    # /narrative will verify this token and rebuild context server-side from
    # the embedded claims. Token is None when HMAC key isn't configured —
    # /narrative also 503s in that case so the gating is consistent.
    ctx_token = None
    if context_token.is_configured():
        neighbor_fragments: dict[str, dict] = {}
        for nb in neighbors:
            track = nb.get("track") or {}
            ts = nb.get("matchTimestamp") or {}
            neighbor_fragments[str(nb["trackId"])] = context_token.neighbor_context_fragment(
                track_id=str(nb["trackId"]),
                title=str(track.get("title") or nb["trackId"]),
                artist=track.get("artist"),
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
            )
        ctx_token = context_token.issue(
            query_fingerprint=query_fingerprint,
            model_sha=_model_sha or "unpinned",
            catalog_sha=_catalog_sha or "no-catalog",
            neighbors=neighbor_fragments,
            acrcloud_cover_song_id=acr_response.get("coverSongId"),
        )

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
        "acrcloud": acr_response,
        "queryFingerprint": query_fingerprint,
        "contextToken": ctx_token,
    }


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

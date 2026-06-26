"""Robust sharded MIR backfill for Phase-2 corpus artifacts.

The supervisor launches each shard as its own Python subprocess. That keeps
librosa/native decoder crashes isolated to one shard and lets the parent mark
the in-flight track as an honest MIR gap before resuming.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np

from backend.mir_features import compute as compute_mir
from backend.scripts.enrich_mir_features import APPLE_UA, _audio_url_for
from backend.scripts.phase2_catalog import artifact_file_hashes, write_json

THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
COMMON_AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac")


class ShardCrashed(RuntimeError):
    def __init__(self, message: str, *, returncode: int):
        super().__init__(message)
        self.returncode = returncode


def load_corpus_tracks(corpus_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(corpus_path.read_text())
    tracks = raw if isinstance(raw, list) else raw.get("tracks", [])
    if not isinstance(tracks, list):
        raise ValueError(f"{corpus_path} must contain a track list or a tracks object")
    return tracks


def prepare_shards(
    corpus_path: Path,
    shard_dir: Path,
    shards: int,
    *,
    force: bool = False,
    limit: int | None = None,
) -> list[Path]:
    tracks = load_corpus_tracks(corpus_path)
    todo = [track for track in tracks if force or "mir_features" not in track]
    if limit is not None:
        todo = todo[: max(0, int(limit))]
    shard_count = max(1, int(shards))
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for shard_idx in range(shard_count):
        shard_tracks = todo[shard_idx::shard_count]
        path = shard_dir / f"shard_{shard_idx}.tracks.json"
        write_json(path, shard_tracks)
        paths.append(path)
    return paths


def run_worker(
    shard_file: Path,
    out_dir: Path,
    *,
    audio_dir: Path | None = None,
    timeout: float = 30.0,
    sleep: float = 0.0,
    allow_download: bool = True,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_name = _shard_name(shard_file)
    out_path = out_dir / f"{shard_name}.jsonl"
    current_path = out_dir / f"{shard_name}.current.json"
    tracks = json.loads(shard_file.read_text())
    completed = read_completed_track_ids(out_path)
    audio_index = build_audio_index(audio_dir) if audio_dir else {}

    written = 0
    for track in tracks:
        track_id = str(track.get("track_id") or "")
        if not track_id or track_id in completed:
            continue
        write_json(current_path, {"track_id": track_id})
        features: dict[str, Any] | None = None
        try:
            wav, sr = load_audio_for_track(
                track,
                audio_dir=audio_dir,
                timeout=timeout,
                audio_index=audio_index,
                allow_download=allow_download,
            )
            if np.asarray(wav).size == 0:
                raise ValueError("decoded audio is empty")
            features = compute_mir(wav, sr).to_dict()
        except Exception as exc:
            print(f"[mir_backfill:{shard_name}] {track_id}: {exc!r}", file=sys.stderr, flush=True)
        append_jsonl(out_path, {"track_id": track_id, "mir_features": features})
        completed.add(track_id)
        written += 1
        if sleep:
            time.sleep(sleep)

    current_path.unlink(missing_ok=True)
    return written


def supervise_shards(
    shard_files: Iterable[Path],
    out_dir: Path,
    *,
    concurrency: int = 1,
    audio_dir: Path | None = None,
    timeout: float = 30.0,
    sleep: float = 0.0,
    python: str | None = None,
    allow_download: bool = True,
) -> dict[str, int]:
    pending = list(shard_files)
    running: dict[subprocess.Popen, Path] = {}
    summary = {"completed": 0, "crashed": 0}
    concurrency = max(1, int(concurrency))

    while pending or running:
        while pending and len(running) < concurrency:
            shard_file = pending.pop(0)
            try:
                result = _run_worker_subprocess(
                    shard_file,
                    out_dir,
                    audio_dir=audio_dir,
                    timeout=timeout,
                    sleep=sleep,
                    python=python,
                    allow_download=allow_download,
                    wait=False,
                )
            except ShardCrashed:
                _record_crashed_inflight(shard_file, out_dir)
                summary["crashed"] += 1
                pending.append(shard_file)
            else:
                if isinstance(result, subprocess.Popen):
                    running[result] = shard_file
                else:
                    summary["completed"] += 1

        if not running:
            continue

        time.sleep(0.2)
        for proc, shard_file in list(running.items()):
            returncode = proc.poll()
            if returncode is None:
                continue
            running.pop(proc)
            if returncode == 0:
                summary["completed"] += 1
            else:
                summary["crashed"] += 1
                _record_crashed_inflight(shard_file, out_dir)
                pending.append(shard_file)

    return summary


def _run_worker_subprocess(
    shard_file: Path,
    out_dir: Path,
    *,
    audio_dir: Path | None = None,
    timeout: float = 30.0,
    sleep: float = 0.0,
    python: str | None = None,
    allow_download: bool = True,
    wait: bool = True,
) -> int | subprocess.Popen:
    cmd = [
        python or sys.executable,
        "-m",
        "backend.scripts.mir_backfill",
        "worker",
        "--shard-file",
        str(shard_file),
        "--out-dir",
        str(out_dir),
        "--timeout",
        str(timeout),
        "--sleep",
        str(sleep),
    ]
    if audio_dir:
        cmd.extend(["--audio-dir", str(audio_dir)])
    if not allow_download:
        cmd.append("--no-download")
    env = os.environ.copy()
    env.update(THREAD_LIMIT_ENV)
    if not wait:
        return subprocess.Popen(cmd, env=env)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise ShardCrashed(f"{shard_file} exited {proc.returncode}", returncode=proc.returncode)
    return proc.returncode


def _record_crashed_inflight(shard_file: Path, out_dir: Path) -> None:
    shard_name = _shard_name(shard_file)
    out_path = out_dir / f"{shard_name}.jsonl"
    current_path = out_dir / f"{shard_name}.current.json"
    if not current_path.exists():
        return
    payload = json.loads(current_path.read_text())
    track_id = str(payload.get("track_id") or "")
    if not track_id or track_id in read_completed_track_ids(out_path):
        return
    append_jsonl(out_path, {"track_id": track_id, "mir_features": None})


def merge_shard_results(corpus_path: Path, shard_dir: Path, manifest_path: Path | None = None) -> dict[str, int]:
    raw = json.loads(corpus_path.read_text())
    tracks = raw if isinstance(raw, list) else raw.get("tracks", [])
    if not isinstance(tracks, list):
        raise ValueError(f"{corpus_path} must contain a track list or a tracks object")

    results: dict[str, Any] = {}
    for path in sorted(shard_dir.glob("shard_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            track_id = str(row.get("track_id") or "")
            if track_id:
                results[track_id] = row.get("mir_features")

    updated = 0
    with_mir = 0
    null_mir = 0
    for track in tracks:
        track_id = str(track.get("track_id") or "")
        if track_id not in results:
            continue
        track["mir_features"] = results[track_id]
        updated += 1
        if results[track_id] is None:
            null_mir += 1
        else:
            with_mir += 1

    write_json(corpus_path, raw)
    rehash_manifest(manifest_path or corpus_path.parent / "manifest.json")
    return {"tracks": len(tracks), "updated": updated, "withMir": with_mir, "nullMir": null_mir}


def rehash_manifest(manifest_path: Path) -> dict[str, Any]:
    corpus_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    sha = manifest.setdefault("sha256", {})
    existing_files = set((sha.get("files") or {}).keys())
    if not existing_files:
        existing_files = {"corpus.json"}
    file_hashes = artifact_file_hashes(corpus_dir, existing_files)
    if (corpus_dir / "corpus.json").exists():
        file_hashes["corpus.json"] = artifact_file_hashes(corpus_dir, ["corpus.json"])["corpus.json"]

    combined = __import__("hashlib").sha256()
    for name in sorted(file_hashes):
        combined.update(name.encode("utf-8"))
        combined.update(file_hashes[name].encode("ascii"))
    sha["files"] = file_hashes
    sha["combined"] = combined.hexdigest()
    write_json(manifest_path, manifest)
    return manifest


def load_audio_for_track(
    track: dict[str, Any],
    *,
    audio_dir: Path | None = None,
    timeout: float = 30.0,
    audio_index: dict[str, Path] | None = None,
    allow_download: bool = True,
) -> tuple[Any, int]:
    local_path = resolve_local_audio_path(track, audio_dir, audio_index or {}) if audio_dir else None
    if local_path:
        return librosa.load(str(local_path), sr=22050, mono=True)
    if not allow_download:
        raise ValueError("no local audio file")
    url = _audio_url_for(track)
    if not url:
        raise ValueError("no local audio file or source URL")
    return _decode_audio_bytes(_download(url, timeout))


def resolve_local_audio_path(
    track: dict[str, Any],
    audio_dir: Path | None,
    audio_index: dict[str, Path],
) -> Path | None:
    if not audio_dir:
        return None
    for token in _track_tokens(track):
        if token in audio_index:
            return audio_index[token]
        for ext in COMMON_AUDIO_EXTS:
            direct = audio_dir / f"{token}{ext}"
            if direct.exists():
                return direct
    return None


def build_audio_index(audio_dir: Path | None) -> dict[str, Path]:
    if not audio_dir or not audio_dir.exists():
        return {}
    index: dict[str, Path] = {}
    for path in audio_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in COMMON_AUDIO_EXTS:
            continue
        stem = path.stem
        index.setdefault(stem, path)
        # Jamendo files are named "<id>.low.mp3" → path.stem is "<id>.low";
        # also index the leading dot-segment so the bare numeric id resolves.
        base = stem.split(".")[0]
        if base and base != stem:
            index.setdefault(base, path)
        for token in stem.replace("-", "_").split("_"):
            if token:
                index.setdefault(token, path)
    return index


def _track_tokens(track: dict[str, Any]) -> list[str]:
    ext = track.get("external_ids") or {}
    candidates = [
        ext.get("jamendoTrackId"),
        ext.get("trackId"),
        track.get("sourceTrackId"),
        str(track.get("track_id") or "").split(":")[-1],
    ]
    return [str(value).strip() for value in candidates if str(value or "").strip()]


def _download(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": APPLE_UA})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _decode_audio_bytes(audio_bytes: bytes) -> tuple[Any, int]:
    try:
        wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
        if getattr(wav, "size", 0) > 0:
            return wav, sr
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        return librosa.load(tmp.name, sr=22050, mono=True)


def read_completed_track_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        track_id = str(row.get("track_id") or "")
        if track_id:
            completed.add(track_id)
    return completed


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _shard_name(shard_file: Path) -> str:
    name = shard_file.name
    return name.removesuffix(".tracks.json").removesuffix(".json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="split corpus tracks into shard files")
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--shard-dir", type=Path, required=True)
    prepare.add_argument("--shards", type=int, required=True)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--force", action="store_true")

    run = sub.add_parser("run", help="prepare shards and supervise workers")
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--shard-dir", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--shards", type=int, required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--audio-dir", type=Path)
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("--sleep", type=float, default=0.0)
    run.add_argument("--force", action="store_true")
    run.add_argument("--no-download", action="store_true")

    worker = sub.add_parser("worker", help="run one shard; intended for supervisor use")
    worker.add_argument("--shard-file", type=Path, required=True)
    worker.add_argument("--out-dir", type=Path, required=True)
    worker.add_argument("--audio-dir", type=Path)
    worker.add_argument("--timeout", type=float, default=30.0)
    worker.add_argument("--sleep", type=float, default=0.0)
    worker.add_argument("--no-download", action="store_true")

    merge = sub.add_parser("merge", help="merge shard JSONL into corpus and rehash manifest")
    merge.add_argument("--corpus", type=Path, required=True)
    merge.add_argument("--shard-dir", type=Path, required=True)
    merge.add_argument("--manifest", type=Path)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "prepare":
        paths = prepare_shards(args.corpus, args.shard_dir, args.shards, force=args.force, limit=args.limit)
        print(json.dumps({"shards": len(paths)}))
        return 0
    if args.command == "run":
        paths = prepare_shards(args.corpus, args.shard_dir, args.shards, force=args.force, limit=args.limit)
        summary = supervise_shards(
            paths,
            args.out_dir,
            concurrency=args.concurrency,
            audio_dir=args.audio_dir,
            timeout=args.timeout,
            sleep=args.sleep,
            allow_download=not args.no_download,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "worker":
        written = run_worker(
            args.shard_file,
            args.out_dir,
            audio_dir=args.audio_dir,
            timeout=args.timeout,
            sleep=args.sleep,
            allow_download=not args.no_download,
        )
        print(json.dumps({"written": written}))
        return 0
    if args.command == "merge":
        summary = merge_shard_results(args.corpus, args.shard_dir, args.manifest)
        print(json.dumps(summary, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())

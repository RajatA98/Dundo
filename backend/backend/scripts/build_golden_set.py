"""Validate the user's golden_set_input.yaml and write the structured golden_set.json.

Phase 6 — pre-eval step. The user fills `backend/eval_input/golden_set_input.yaml`
after generating Suno tracks (see the *.example.yaml template alongside this file).
This script validates the input, resolves audio paths to absolute, verifies each
file exists, and writes `backend/eval_input/golden_set.json` for `run_eval.py`
to consume.

Usage:
    python -m backend.scripts.build_golden_set
    # or with explicit paths:
    python -m backend.scripts.build_golden_set \\
        --input backend/eval_input/golden_set_input.yaml \\
        --out backend/eval_input/golden_set.json

Outputs the validated `golden_set.json` shape:

    {
      "positives": [
        {
          "id": "pos_seed01_v1",
          "seed_track_id": "tier1:itunes:1499378034",
          "seed_title": "Blinding Lights",
          "query_audio_path": "/abs/path/to/seed01_v1.mp3",
          "prompt_used": "..."
        },
        ...
      ],
      "negatives": [
        {
          "id": "neg_001",
          "query_audio_path": "/abs/path/to/neg_001.mp3",
          "prompt_used": "..."
        },
        ...
      ]
    }

Re-running is safe and deterministic given the same input file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "backend" / "eval_input" / "golden_set_input.yaml"
DEFAULT_OUT = REPO_ROOT / "backend" / "eval_input" / "golden_set.json"


def main() -> None:
    args = _parse_args()
    raw = _load_input(args.input)
    validated = validate(raw, base_dir=REPO_ROOT)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(validated, indent=2))
    print(
        f"[build_golden_set] wrote {args.out} — "
        f"{len(validated['positives'])} positives + "
        f"{len(validated['negatives'])} negatives"
    )


def validate(raw: dict, *, base_dir: Path) -> dict:
    """Validate the YAML input shape and produce the structured golden_set.json dict.

    Args:
        raw: parsed YAML — top-level dict with `positives` (list of seed entries) +
             `negatives` (list of negative entries).
        base_dir: repo root, used to resolve relative audio paths.

    Returns:
        Validated dict in the shape documented in the module docstring.

    Raises:
        ValueError if structural validation fails:
          - missing `positives` or `negatives` keys
          - a positive entry missing `seed_track_id` or `suno_audio_paths`
          - an audio file referenced but not on disk
          - a positive has 0 audio paths (need ≥ 1 per seed)
    """
    if not isinstance(raw, dict):
        raise ValueError("golden set input must be a mapping")
    positives_raw = raw.get("positives")
    negatives_raw = raw.get("negatives")
    if not isinstance(positives_raw, list):
        raise ValueError("positives must be a list")
    if not isinstance(negatives_raw, list):
        raise ValueError("negatives must be a list")

    positives: list[dict] = []
    for seed in positives_raw:
        if not isinstance(seed, dict):
            raise ValueError("positive entries must be mappings")
        seed_track_id = str(seed.get("seed_track_id") or "").strip()
        if not seed_track_id:
            raise ValueError("positive entry missing seed_track_id")
        audio_paths = seed.get("suno_audio_paths")
        if not isinstance(audio_paths, list) or not audio_paths:
            raise ValueError(f"{seed_track_id}: suno_audio_paths must contain at least one path")

        seed_short = _slug(seed_track_id.rsplit(":", 1)[-1])
        for idx, audio_path in enumerate(audio_paths, start=1):
            resolved = _resolve_audio(audio_path, base_dir=base_dir)
            positives.append(
                {
                    "id": f"pos_{seed_short}_v{idx}",
                    "seed_track_id": seed_track_id,
                    "seed_title": seed.get("seed_title") or "",
                    "query_audio_path": str(resolved),
                    "prompt_used": seed.get("prompt_used") or "",
                }
            )

    negatives: list[dict] = []
    for row in negatives_raw:
        if not isinstance(row, dict):
            raise ValueError("negative entries must be mappings")
        neg_id = str(row.get("id") or "").strip()
        if not neg_id:
            raise ValueError("negative entry missing id")
        audio_path = row.get("audio_path")
        if not audio_path:
            raise ValueError(f"{neg_id}: missing audio_path")
        resolved = _resolve_audio(audio_path, base_dir=base_dir)
        negatives.append(
            {
                "id": neg_id,
                "query_audio_path": str(resolved),
                "prompt_used": row.get("prompt_used") or "",
            }
        )

    return {"positives": positives, "negatives": negatives}


def _load_input(path: Path) -> dict:
    """Parse the YAML input. Raises FileNotFoundError if path doesn't exist."""
    return yaml.safe_load(path.read_text()) or {}


def _resolve_audio(path_value: object, *, base_dir: Path) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("audio path must be a non-empty string")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise ValueError(f"audio file not found: {path}")
    return path


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "seed"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


if __name__ == "__main__":
    main()

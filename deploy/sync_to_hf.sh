#!/usr/bin/env bash
# Sync the PiedPiper repo into the staging shape the HF Space Dockerfile expects.
#
# Why this exists: the Hugging Face Space repo has its own git remote and expects
# a flat layout — Dockerfile + README.md + app.py + requirements.txt + backend/ +
# corpus/ + eval_audio/, all at the repo root. The PiedPiper monorepo nests these
# files under `deploy/hf_space/`, `backend/backend/`, and `quality-scorer/public/`,
# so we sync them into a staging directory which the user then pushes to the Space.
#
# Usage:
#   bash deploy/sync_to_hf.sh [STAGING_DIR]
#
# Default STAGING_DIR is ../piedpiper-hf-space (sibling of the repo).
#
# After running:
#   cd ../piedpiper-hf-space
#   git init && git remote add origin https://huggingface.co/spaces/<user>/piedpiper
#   git add . && git commit -m "Initial Space build"
#   git push -u origin main

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING_DIR="${1:-${REPO_ROOT}/../piedpiper-hf-space}"

echo "[sync] repo:    $REPO_ROOT"
echo "[sync] staging: $STAGING_DIR"

# Refuse to clobber a non-empty staging directory unless it's already a Space repo.
if [ -e "$STAGING_DIR" ] && [ ! -d "$STAGING_DIR/.git" ] && [ -n "$(ls -A "$STAGING_DIR" 2>/dev/null || true)" ]; then
  echo "[sync] ERROR: $STAGING_DIR exists and is not a git repo. Refusing to overwrite." >&2
  echo "[sync]        Either delete it, pick a different path, or 'git init' it first." >&2
  exit 1
fi

mkdir -p "$STAGING_DIR"

# Top-level files for the Space.
cp "$REPO_ROOT/deploy/hf_space/Dockerfile"        "$STAGING_DIR/Dockerfile"
cp "$REPO_ROOT/deploy/hf_space/README.md"         "$STAGING_DIR/README.md"
cp "$REPO_ROOT/deploy/hf_space/app.py"            "$STAGING_DIR/app.py"
cp "$REPO_ROOT/deploy/hf_space/requirements.txt"  "$STAGING_DIR/requirements.txt"

# Backend Python package — flatten one level. The PiedPiper repo has
# `backend/backend/api.py`; the Space expects `/app/backend/api.py` after
# `COPY backend /app/backend`, so the inner package directory is what we ship.
rm -rf "$STAGING_DIR/backend"
mkdir -p "$STAGING_DIR/backend"
# Use rsync if available for the --exclude support; fall back to cp + find.
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$REPO_ROOT/backend/backend/" "$STAGING_DIR/backend/"
else
  cp -R "$REPO_ROOT/backend/backend/." "$STAGING_DIR/backend/"
  find "$STAGING_DIR/backend" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
  find "$STAGING_DIR/backend" -type f -name '*.pyc' -delete 2>/dev/null || true
fi

# Corpus + eval audio — bake them into the image so /neighbors works on cold start.
rm -rf "$STAGING_DIR/corpus" "$STAGING_DIR/eval_audio"
cp -R "$REPO_ROOT/quality-scorer/public/corpus"       "$STAGING_DIR/corpus"
if [ -d "$REPO_ROOT/quality-scorer/public/eval_audio" ]; then
  cp -R "$REPO_ROOT/quality-scorer/public/eval_audio" "$STAGING_DIR/eval_audio"
else
  mkdir -p "$STAGING_DIR/eval_audio"
  echo "# Named eval-example audio drops here; safe to ship empty." > "$STAGING_DIR/eval_audio/README.md"
fi

# A .gitattributes for HF git-lfs — embeddings.npy and segment_embeddings.npz
# can be tens of MB. HF requires LFS for files over 10 MB on free tier.
cat > "$STAGING_DIR/.gitattributes" <<'EOF'
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.mp3 filter=lfs diff=lfs merge=lfs -text
*.wav filter=lfs diff=lfs merge=lfs -text
EOF

# A .gitignore to keep the staging dir clean if the user re-runs.
cat > "$STAGING_DIR/.gitignore" <<'EOF'
__pycache__/
*.pyc
.DS_Store
.hf_cache/
EOF

echo "[sync] done. Layout:"
( cd "$STAGING_DIR" && find . -maxdepth 2 -not -path '*/.git*' | sort | sed 's/^/  /' )
echo ""
echo "[sync] Next steps:"
echo "  1. cd $STAGING_DIR"
echo "  2. git init && git lfs install (if not already)"
echo "  3. git remote add origin https://huggingface.co/spaces/<your-user>/piedpiper"
echo "  4. git add . && git commit -m 'Initial PiedPiper Space build'"
echo "  5. git push -u origin main"
echo ""
echo "  Then in the Space settings, add the ACRCloud secrets + CORS_ORIGIN."

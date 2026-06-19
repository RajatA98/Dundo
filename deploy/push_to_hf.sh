#!/usr/bin/env bash
# One-shot: clone the HF Space repo with auth, overlay the latest backend
# code via sync_to_hf.sh, commit + push to trigger a rebuild.
#
# Auth: reads ~/.cache/huggingface/token (the same file huggingface-cli
# writes on login). If the token is missing, prints the URL to create one.
#
# Usage:
#   bash deploy/push_to_hf.sh
#
# Idempotent: deletes the staging dir before re-cloning so re-runs are clean.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING_DIR="$(dirname "$REPO_ROOT")/piedpiper-hf-fresh"
SPACE_NAME="rajata98/piedpiper"

if [ ! -r "$HOME/.cache/huggingface/token" ]; then
  echo "[push] ERROR: ~/.cache/huggingface/token not readable." >&2
  echo "[push] Get a token from https://huggingface.co/settings/tokens" >&2
  echo "[push] then save it: echo YOUR_TOKEN > ~/.cache/huggingface/token" >&2
  exit 1
fi

# git-lfs is required to push the corpus binaries (embeddings.npy +
# segment_embeddings.npz). Without it the HF pre-receive hook rejects the
# push with a "binary files must use xet/LFS" error.
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[push] ERROR: git-lfs is not installed." >&2
  echo "[push] Install it: brew install git-lfs" >&2
  exit 1
fi

TOKEN="$(tr -d ' \n' < "$HOME/.cache/huggingface/token")"
if [ -z "$TOKEN" ]; then
  echo "[push] ERROR: token file is empty." >&2
  exit 1
fi

echo "[push] Cleaning staging dir: $STAGING_DIR"
rm -rf "$STAGING_DIR"

echo "[push] Cloning HF Space repo..."
# Pipe through sed so the token never lands in the captured output.
git clone "https://USER:${TOKEN}@huggingface.co/spaces/${SPACE_NAME}" "$STAGING_DIR" 2>&1 \
  | sed "s|${TOKEN}|***|g"

# Install LFS hooks in the fresh clone BEFORE sync overwrites .gitattributes.
# The clean filter consults .gitattributes at `git add` time, so as long as
# both LFS hooks and the .gitattributes patterns are in place when add runs,
# binary files convert to LFS pointers automatically.
cd "$STAGING_DIR"
git lfs install --local

echo "[push] Syncing latest backend code..."
bash "$REPO_ROOT/deploy/sync_to_hf.sh" "$STAGING_DIR" 2>&1 | sed "s|${TOKEN}|***|g"

cd "$STAGING_DIR"

echo "[push] Staging changes..."
git add .

if git diff --cached --quiet; then
  echo "[push] No changes to commit. The HF Space is already up to date."
  exit 0
fi

git -c user.email="rajat1998@gmail.com" -c user.name="Rajat Arora" \
  commit -m "Deploy RAG narrative layer + telemetry + RAG eval harness"

echo "[push] Pushing to HF Space..."
git push origin main 2>&1 | sed "s|${TOKEN}|***|g"

echo ""
echo "[push] Done. Watch build progress at:"
echo "       https://huggingface.co/spaces/${SPACE_NAME}/logs"
echo ""
echo "       When build finishes, smoke test:"
echo "         curl -sS https://rajata98-piedpiper.hf.space/narrative/stats"
echo "       Expected: {\"total_calls\": 0, ...}"

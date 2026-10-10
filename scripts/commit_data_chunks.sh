#!/usr/bin/env bash
# Commit one month of daily Parquet chunks at a time. Requires Git LFS and an existing Git remote.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Run this from inside the cloned project repository." >&2
  exit 2
}
cd "$ROOT"
DATA_ROOT="data/processed/BTCUSDT_1s"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "No downloaded data at $DATA_ROOT" >&2
  exit 2
fi

git lfs install
shopt -s nullglob
MONTH_DIRS=("$DATA_ROOT"/[0-9][0-9][0-9][0-9]-[0-9][0-9])
if [[ ${#MONTH_DIRS[@]} -eq 0 ]]; then
  echo "No month chunks found under $DATA_ROOT" >&2
  exit 0
fi

for month_dir in "${MONTH_DIRS[@]}"; do
  month="$(basename "$month_dir")"
  git add -- "$month_dir"
  if git diff --cached --quiet; then
    echo "No new files for $month"
    continue
  fi
  git commit -m "Add BTCUSDT 1-second data chunk $month"
  if [[ "${PUSH:-0}" == "1" ]]; then
    git push origin HEAD
  fi
done

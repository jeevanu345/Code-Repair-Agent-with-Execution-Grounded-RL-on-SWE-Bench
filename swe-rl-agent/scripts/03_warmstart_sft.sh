#!/usr/bin/env bash
set -euo pipefail
if [[ "${SWE_RL_ALLOW_HEAVY_TRAINING:-}" != "YES" ]]; then
  echo "Refusing model download/training. Set SWE_RL_ALLOW_HEAVY_TRAINING=YES on a prepared GPU host." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
OUT="${1:-./outputs/sft}"
python -m swe_rl.cli train sft --output-dir "${OUT}" --use-humanevalpack

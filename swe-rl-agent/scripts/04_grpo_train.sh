#!/usr/bin/env bash
set -euo pipefail
if [[ "${SWE_RL_ALLOW_HEAVY_TRAINING:-}" != "YES" ]]; then
  echo "Refusing model download/training. Set SWE_RL_ALLOW_HEAVY_TRAINING=YES on a prepared GPU host." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
OUT="${1:-./outputs/grpo}"
MAX_INSTANCES="${MAX_INSTANCES:-100}"
python -m swe_rl.cli train grpo --max-instances "${MAX_INSTANCES}" --output-dir "${OUT}"

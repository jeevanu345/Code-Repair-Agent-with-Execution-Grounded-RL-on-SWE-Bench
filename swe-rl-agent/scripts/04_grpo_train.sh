#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-./outputs/grpo}"
MAX_INSTANCES="${MAX_INSTANCES:-100}"
python -m swe_rl.cli train grpo --max-instances "${MAX_INSTANCES}" --output-dir "${OUT}"

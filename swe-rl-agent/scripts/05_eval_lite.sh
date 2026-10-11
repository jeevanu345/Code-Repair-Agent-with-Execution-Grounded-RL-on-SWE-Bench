#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-base}"
SUBSET="${2:-lite}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MAX_INSTANCES="${MAX_INSTANCES:-300}"
MODEL_ARGS=()
if [[ "${CHECKPOINT}" != "base" ]]; then
  MODEL_ARGS=(--model "${CHECKPOINT}")
fi

mkdir -p outputs/predictions outputs/reports

python -m swe_rl.cli eval lite \
    --subset "${SUBSET}" \
    "${MODEL_ARGS[@]}" \
    --max-instances "${MAX_INSTANCES}" \
    --output-predictions "outputs/predictions/${RUN_ID}.json" \
    --report-dir "outputs/reports/${RUN_ID}" \
    --run-id "${RUN_ID}" \
    --base-model "${CHECKPOINT}"

echo "report: outputs/reports/${RUN_ID}/leaderboard.md"

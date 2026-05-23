#!/usr/bin/env bash
# Package predictions for SWE-bench leaderboard submission.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
PRED="outputs/predictions/${RUN_ID}.json"
REPORT_DIR="outputs/reports/${RUN_ID}"

if [[ ! -f "${PRED}" ]]; then
  echo "missing predictions: ${PRED}" >&2
  exit 1
fi

OUT_DIR="outputs/submission/${RUN_ID}"
mkdir -p "${OUT_DIR}"
cp "${PRED}" "${OUT_DIR}/predictions.json"
[[ -f "${REPORT_DIR}/leaderboard.md" ]] && cp "${REPORT_DIR}/leaderboard.md" "${OUT_DIR}/leaderboard.md"
[[ -f "${REPORT_DIR}/summary.json" ]] && cp "${REPORT_DIR}/summary.json" "${OUT_DIR}/summary.json"

cat > "${OUT_DIR}/model_card.md" <<EOF
# swe-rl-agent — ${RUN_ID}

- Base model: ${MODEL_NAME:-Qwen/Qwen2.5-Coder-14B-Instruct}
- Training: SFT warm-start -> DPO -> GRPO with execution-grounded reward
- Reward: binary (FAIL_TO_PASS pass AND PASS_TO_PASS still pass) inside ephemeral Docker
- Sandbox: 4GB RAM / 2 CPU / 600s wallclock / network disabled post-install
- See leaderboard.md for headline number and per-repo breakdown.
EOF

tar czf "${OUT_DIR}.tar.gz" -C "$(dirname "${OUT_DIR}")" "$(basename "${OUT_DIR}")"
echo "submission package: ${OUT_DIR}.tar.gz"

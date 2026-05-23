#!/usr/bin/env bash
# End-to-end smoke: pick one Lite instance, apply the gold patch in the sandbox,
# run F2P + P2P, expect reward = 1.0.
set -euo pipefail
cd "$(dirname "$0")/.."

INSTANCE_ID="${1:-}"

if [[ -z "${INSTANCE_ID}" ]]; then
  python -m swe_rl.cli smoke rollout --gold
else
  python -m swe_rl.cli smoke rollout --instance-id "${INSTANCE_ID}" --gold
fi

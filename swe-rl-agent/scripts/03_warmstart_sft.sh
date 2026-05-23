#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-./outputs/sft}"
python -m swe_rl.cli train sft --output-dir "${OUT}" --use-humanevalpack

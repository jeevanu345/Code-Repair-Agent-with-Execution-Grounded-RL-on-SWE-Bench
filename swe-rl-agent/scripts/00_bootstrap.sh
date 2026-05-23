#!/usr/bin/env bash
# Bootstrap a workstation for swe-rl-agent.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  uv pip install -e ".[dev]"
else
  python -m pip install --upgrade pip
  python -m pip install -e ".[dev]"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
fi

docker build -f docker/sandbox.Dockerfile -t swe-rl/sandbox:latest .

mkdir -p logs outputs/trajectories outputs/predictions outputs/reports outputs/checkpoints

echo "bootstrap ok"

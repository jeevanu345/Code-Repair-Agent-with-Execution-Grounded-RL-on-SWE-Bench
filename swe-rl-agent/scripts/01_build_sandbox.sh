#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -f docker/sandbox.Dockerfile -t swe-rl/sandbox:latest .
docker images swe-rl/sandbox

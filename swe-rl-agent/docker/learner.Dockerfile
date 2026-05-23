FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential pkg-config curl \
        ca-certificates ripgrep jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/swe-rl
COPY pyproject.toml /opt/swe-rl/pyproject.toml
COPY src /opt/swe-rl/src

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ".[dev]"

ENV PYTHONPATH=/opt/swe-rl/src
CMD ["bash"]

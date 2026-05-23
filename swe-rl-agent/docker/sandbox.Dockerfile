# Sandbox image: per-instance ephemeral container for SWE-bench rollouts.
# - Ubuntu 22.04 + Python 3.11
# - common build/test tooling
# - non-root user `agent`
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        wget \
        git \
        gnupg \
        build-essential \
        pkg-config \
        make \
        cmake \
        libssl-dev \
        libffi-dev \
        zlib1g-dev \
        libbz2-dev \
        libsqlite3-dev \
        libreadline-dev \
        liblzma-dev \
        libncurses-dev \
        tk-dev \
        xz-utils \
        unzip \
        jq \
        ripgrep \
        universal-ctags \
        sudo \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        pytest pytest-xdist pytest-timeout \
        tox \
        virtualenv \
        unidiff

RUN useradd -ms /bin/bash agent && \
    mkdir -p /workspace /home/agent/.cache && \
    chown -R agent:agent /workspace /home/agent

USER agent
WORKDIR /workspace

CMD ["/bin/bash", "-l"]

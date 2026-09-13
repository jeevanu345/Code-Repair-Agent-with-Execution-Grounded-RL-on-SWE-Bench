# Sandbox image: per-instance ephemeral container for SWE-bench rollouts.
# - Ubuntu 22.04 + Miniconda
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
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create Conda environments for Python 3.5 through 3.11
RUN conda create -y -n py35 python=3.5 && \
    conda create -y -n py36 python=3.6 && \
    conda create -y -n py37 python=3.7 && \
    conda create -y -n py38 python=3.8 && \
    conda create -y -n py39 python=3.9 && \
    conda create -y -n py310 python=3.10 && \
    conda create -y -n py311 python=3.11 && \
    conda clean -afy

# Pre-install SWE-bench test requirements into all environments
RUN for py in py35 py36 py37 py38 py39 py310 py311; do \
        $CONDA_DIR/envs/$py/bin/python -m pip install --upgrade pip setuptools wheel && \
        $CONDA_DIR/envs/$py/bin/python -m pip install pytest pytest-xdist pytest-timeout pytest-json-report tox virtualenv unidiff; \
    done

RUN useradd -ms /bin/bash agent && \
    mkdir -p /workspace /home/agent/.cache && \
    chown -R agent:agent /workspace /home/agent /opt/conda

USER agent
WORKDIR /workspace

CMD ["/bin/bash", "-l"]

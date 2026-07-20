# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8

# Place GPU specifications here or pass them with "--build-arg"
ARG TORCH_CUDA_ARCH_LIST="7.0"
ARG CUDAARCHS="70"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    CUDAARCHS="${CUDAARCHS}" \
    FORCE_CUDA=1 \
    MAX_JOBS=${MAX_JOBS} \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        build-essential \
        gcc \
        g++ \
        ninja-build \
        cmake \
        git \
        ca-certificates \
        pkg-config \
        libegl1-mesa-dev \
        libgl1-mesa-dev \
        libgles2-mesa-dev \
        libglvnd-dev \
        libx11-dev \
        libxext-dev \
        libxi-dev \
        libxrandr-dev \
        libxrender-dev \
        libxfixes-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv

RUN python --version \
    && python -c "import sys; assert sys.version_info[:2] == (3, 10)"

WORKDIR /build-input

COPY requirements.txt .
COPY aux/trimesh.path.packing.patched.py .

COPY third_party/nvdiffrast/ \
     /opt/extensions/nvdiffrast/

COPY third_party/diffoctreerast/ \
     /opt/extensions/diffoctreerast/

COPY third_party/mip-splatting/ \
     /opt/extensions/mip-splatting/

COPY install/install-docker-dependencies.sh \
     /usr/local/bin/install-docker-dependencies

RUN chmod +x /usr/local/bin/install-docker-dependencies

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/torch_extensions \
    /usr/local/bin/install-docker-dependencies

WORKDIR /workspace

COPY . /workspace

CMD ["python", "/workspace/src/server.py"]


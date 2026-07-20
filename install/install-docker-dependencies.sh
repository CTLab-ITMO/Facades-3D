#!/usr/bin/env bash

set -Eeuo pipefail

EXTENSIONS_ROOT="/opt/extensions"
WHEEL_DIR="/tmp/extension-wheels"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
export CUDAARCHS="${CUDAARCHS:-70}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-4}"

required_paths=(
    "${EXTENSIONS_ROOT}/nvdiffrast"
    "${EXTENSIONS_ROOT}/diffoctreerast"
    "${EXTENSIONS_ROOT}/mip-splatting/submodules/diff-gaussian-rasterization"
)

for path in "${required_paths[@]}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Missing extension source: ${path}" >&2
        echo "Initialize Git submodules before building:" >&2
        echo "  git submodule update --init --recursive" >&2
        exit 1
    fi
done

echo "Python:"
python --version

echo "CUDA compiler:"
nvcc --version

python -m pip install --upgrade \
    pip \
    setuptools \
    wheel

python -m pip install \
    ninja \
    packaging \
    psutil

echo "Installing PyTorch..."

python -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

echo "Installing application requirements..."

python -m pip install \
    -r /build-input/requirements.txt

echo "Patching trimesh..."

SITE_PACKAGES="$(
    python -c 'import site; print(site.getsitepackages()[0])'
)"

TRIMESH_TARGET="${SITE_PACKAGES}/trimesh/path/packing.py"

test -f "${TRIMESH_TARGET}"

install \
    -m 0644 \
    /build-input/trimesh.path.packing.patched.py \
    "${TRIMESH_TARGET}"

echo "Installing xformers..."

python -m pip install \
    xformers==0.0.29.post3 \
    --index-url https://download.pytorch.org/whl/cu126

echo "Installing Kaolin..."

python -m pip install \
    kaolin==0.18.0 \
    --find-links \
    https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu126.html

rm -rf "${WHEEL_DIR}"
mkdir -p "${WHEEL_DIR}"

build_and_install() {
    local name="$1"
    local source="$2"
    local output="${WHEEL_DIR}/${name}"

    echo
    echo "Building ${name} from ${source}"

    rm -rf "${output}"
    mkdir -p "${output}"

    python -m pip wheel \
        --verbose \
        --no-deps \
        --no-build-isolation \
        --wheel-dir "${output}" \
        "${source}"

    mapfile -t wheels < <(
        find "${output}" \
            -maxdepth 1 \
            -type f \
            -name '*.whl' \
            -print
    )

    if [[ "${#wheels[@]}" -ne 1 ]]; then
        echo "Expected one wheel for ${name}, found ${#wheels[@]}" >&2
        exit 1
    fi

    python -m pip install \
        --no-deps \
        "${wheels[0]}"
}

build_and_install \
    nvdiffrast \
    "${EXTENSIONS_ROOT}/nvdiffrast"

build_and_install \
    diffoctreerast \
    "${EXTENSIONS_ROOT}/diffoctreerast"

build_and_install \
    diff-gaussian-rasterization \
    "${EXTENSIONS_ROOT}/mip-splatting/submodules/diff-gaussian-rasterization"

python -m pip check

python - <<'PY'
import torch
import torchvision
import torchaudio
import trimesh
import xformers
import nvdiffrast
import nvdiffrast.torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("trimesh:", trimesh.__version__)
print("xformers:", xformers.__version__)
print("nvdiffrast:", nvdiffrast.__file__)
PY

rm -rf "${WHEEL_DIR}"


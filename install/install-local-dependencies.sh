#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${1:-.}"
PROJECT_ROOT="$(cd -- "${PROJECT_ROOT}" && pwd)"

REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
TRIMESH_PATCH="${PROJECT_ROOT}/aux/trimesh.path.packing.patched.py"
EXTENSIONS_ROOT="${PROJECT_ROOT}/third_party"

NVDIFFRAST_SOURCE="${EXTENSIONS_ROOT}/nvdiffrast"
DIFFOCTREERAST_SOURCE="${EXTENSIONS_ROOT}/diffoctreerast"
DIFF_GAUSSIAN_SOURCE="${EXTENSIONS_ROOT}/mip-splatting/submodules/diff-gaussian-rasterization"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-4}"

WHEEL_DIR="$(mktemp -d -t extension-wheels.XXXXXXXXXX)"
trap 'rm -rf -- "${WHEEL_DIR}"' EXIT

fail() {
    echo "Error: $*" >&2
    exit 1
}

command -v python >/dev/null 2>&1 || fail "python is not available in PATH"
command -v nvcc >/dev/null 2>&1 || fail "nvcc is not available in PATH"
command -v gcc >/dev/null 2>&1 || fail "gcc is not available in PATH"
command -v g++ >/dev/null 2>&1 || fail "g++ is not available in PATH"

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"Error: Python 3.10 is required, but the active interpreter is "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

print("Python executable:", sys.executable)
print("Virtual environment:", sys.prefix)
print("Python version:", sys.version.split()[0])
PY

required_files=(
    "${REQUIREMENTS_FILE}"
    "${TRIMESH_PATCH}"
)

required_directories=(
    "${NVDIFFRAST_SOURCE}"
    "${DIFFOCTREERAST_SOURCE}"
    "${DIFF_GAUSSIAN_SOURCE}"
)

for path in "${required_files[@]}"; do
    [[ -f "${path}" ]] || fail "missing required file: ${path}"
done

for path in "${required_directories[@]}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Missing extension source: ${path}" >&2
        echo "Initialize Git submodules from the project root:" >&2
        echo "  git submodule update --init --recursive" >&2
        exit 1
    fi
done

echo "Project root: ${PROJECT_ROOT}"
echo "CUDA_HOME: ${CUDA_HOME}"
[ -v TORCH_CUDA_ARCH_LIST ] && echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
[ -v CUDAARCHS ] && echo "CUDAARCHS: ${CUDAARCHS}"
echo "MAX_JOBS: ${MAX_JOBS}"

echo
echo "CUDA compiler:"
nvcc --version

echo
echo "Upgrading Python build tooling..."
python -m pip install --upgrade \
    pip \
    setuptools \
    wheel

python -m pip install \
    ninja \
    packaging \
    psutil

echo
echo "Installing PyTorch..."
python -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

echo
echo "Installing application requirements..."
python -m pip install \
    -r "${REQUIREMENTS_FILE}"

echo
echo "Patching trimesh..."
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
TRIMESH_TARGET="${SITE_PACKAGES}/trimesh/path/packing.py"

[[ -f "${TRIMESH_TARGET}" ]] || fail "trimesh target was not found: ${TRIMESH_TARGET}"

install \
    -m 0644 \
    "${TRIMESH_PATCH}" \
    "${TRIMESH_TARGET}"

echo
echo "Installing xformers..."
python -m pip install \
    xformers==0.0.29.post3 \
    --index-url https://download.pytorch.org/whl/cu126

echo
echo "Installing Kaolin..."
python -m pip install \
    kaolin==0.18.0 \
    --find-links \
    https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu126.html

build_and_install() {
    local name="$1"
    local source="$2"
    local output="${WHEEL_DIR}/${name}"
    local -a wheels=()

    echo
    echo "Building ${name} from ${source}"

    rm -rf -- "${output}"
    mkdir -p -- "${output}"

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
        fail "expected one wheel for ${name}, found ${#wheels[@]}"
    fi

    python -m pip install \
        --no-deps \
        "${wheels[0]}"
}

build_and_install \
    nvdiffrast \
    "${NVDIFFRAST_SOURCE}"

build_and_install \
    diffoctreerast \
    "${DIFFOCTREERAST_SOURCE}"

build_and_install \
    diff-gaussian-rasterization \
    "${DIFF_GAUSSIAN_SOURCE}"

echo
echo "Checking installed dependencies..."
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

echo
echo "Dependency installation completed successfully."


#!/usr/bin/env bash
# ==============================================================================
# scripts/run_on_existing_g4.sh
# Turnkey launcher for an existing G4 VM (g4-standard-192 4-GPU or g4-standard-384 8-GPU)
# ==============================================================================
# Usage:
#   ./scripts/run_on_existing_g4.sh \
#     --model-source gs://<YOUR_BUCKET>/DeepSeek-V4.1-Flash \
#     --port 7080
#
#   Or download directly from Hugging Face:
#   ./scripts/run_on_existing_g4.sh \
#     --model-source deepseek-ai/DeepSeek-V4.1-Flash \
#     --hf-revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SOURCE="${MODEL_SOURCE:-}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/mnt/localssd/models/DeepSeek-V4.1-Flash}"
HF_REVISION="${HF_REVISION:-2cba9e42aa026125f3ed06c6d98c1db82f7ca027}"
PORT="${PORT:-7080}"
BASE_IMAGE="${BASE_IMAGE:-us-docker.pkg.dev/agent-platform-mg-public/containers/sglang-airlock:ds41-e56358a-rev1}"
CONTAINER_NAME="${CONTAINER_NAME:-deepseek-v41-flash-g4}"
DETACH="${DETACH:-1}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --model-source <URI_OR_HF_ID>   GCS path (gs://bucket/path), local path (/path/to/model),
                                  or Hugging Face ID (default: deepseek-ai/DeepSeek-V4.1-Flash)
  --local-model-dir <DIR>         Local directory on the G4 VM to cache weights
                                  (default: /mnt/localssd/models/DeepSeek-V4.1-Flash, falls back to /tmp/models)
  --hf-revision <COMMIT>          Hugging Face commit hash (default: ${HF_REVISION})
  --port <PORT>                   Serving HTTP port (default: ${PORT})
  --image <IMAGE_URI>             Base or pre-built container image (default: ${BASE_IMAGE})
  --foreground                    Run container in foreground instead of detached background daemon
  -h, --help                      Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-source) MODEL_SOURCE="$2"; shift 2 ;;
    --local-model-dir) LOCAL_MODEL_DIR="$2"; shift 2 ;;
    --hf-revision) HF_REVISION="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --image) BASE_IMAGE="$2"; shift 2 ;;
    --foreground) DETACH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${MODEL_SOURCE}" ]]; then
  MODEL_SOURCE="deepseek-ai/DeepSeek-V4.1-Flash"
fi

# 1. Verify NVIDIA GPUs on host
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi not found. Please run this script on a G4 GPU instance with NVIDIA drivers installed." >&2
  exit 1
fi

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "[setup] Detected ${NUM_GPUS} NVIDIA GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if (( NUM_GPUS < 4 )); then
  echo "[error] At least 4 GPUs (g4-standard-192 or g4-standard-384) are required for TP4+EP4." >&2
  exit 1
fi

# Choose writable local model cache directory
if ! mkdir -p "${LOCAL_MODEL_DIR}" 2>/dev/null; then
  LOCAL_MODEL_DIR="${HOME}/models/DeepSeek-V4.1-Flash"
  mkdir -p "${LOCAL_MODEL_DIR}"
fi

# 2. Stage model weights if not already present locally
if [[ -d "${MODEL_SOURCE}" && -f "${MODEL_SOURCE}/config.json" ]]; then
  LOCAL_MODEL_DIR="$(cd "${MODEL_SOURCE}" && pwd)"
  echo "[setup] Using existing local model directory: ${LOCAL_MODEL_DIR}"
elif [[ -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
  echo "[setup] Found cached model weights at ${LOCAL_MODEL_DIR}"
elif [[ "${MODEL_SOURCE}" == gs://* ]]; then
  echo "[setup] Downloading model weights from ${MODEL_SOURCE} to ${LOCAL_MODEL_DIR}..."
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp -r "${MODEL_SOURCE%/}/*" "${LOCAL_MODEL_DIR}/"
  else
    gsutil -m cp -r "${MODEL_SOURCE%/}/*" "${LOCAL_MODEL_DIR}/"
  fi
else
  echo "[setup] Downloading ${MODEL_SOURCE} (revision ${HF_REVISION}) from Hugging Face to ${LOCAL_MODEL_DIR}..."
  python3 -c "
import subprocess, sys
try:
    import huggingface_hub
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub[cli,hf_transfer]'])
from huggingface_hub import snapshot_download
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
snapshot_download(
    repo_id='${MODEL_SOURCE}',
    revision='${HF_REVISION}',
    local_dir='${LOCAL_MODEL_DIR}',
)
"
fi

# 3. Stop any previous container with the same name
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# 4. Launch container with repo overrides & patches mounted
DOCKER_FLAGS=(
  --gpus all
  --shm-size 40g
  --ulimit memlock=-1
  --ulimit core=0
  --network host
  --ipc host
  -e "PORT=${PORT}"
  -e "HF_TOKEN=${HF_TOKEN:-}"
  -v "${LOCAL_MODEL_DIR}:/models/deepseek-ai/DeepSeek-V4.1-Flash:ro"
  -v "${REPO_ROOT}/sglang_overrides:/sgl-workspace/sglang/python/sglang"
  -v "${REPO_ROOT}/patches:/opt/dsv41_g4_patches:ro"
  -v "${REPO_ROOT}/scripts/vertex_g4_entrypoint.sh:/usr/local/bin/vertex_g4_entrypoint.sh:ro"
  --entrypoint /usr/local/bin/vertex_g4_entrypoint.sh
)

if [[ "${DETACH}" == "1" ]]; then
  echo "[setup] Starting container ${CONTAINER_NAME} in background on port ${PORT}..."
  docker run -d --name "${CONTAINER_NAME}" "${DOCKER_FLAGS[@]}" "${BASE_IMAGE}" \
    --model-path=/models/deepseek-ai/DeepSeek-V4.1-Flash --port="${PORT}"
  echo ""
  echo "[setup] Container launched! Follow startup logs with:"
  echo "  docker logs -f ${CONTAINER_NAME}"
  echo ""
  echo "[setup] Once ready, run the automated verification & benchmark suite with:"
  echo "  python3 ${REPO_ROOT}/benchmarks/verify_and_bench.py --url http://127.0.0.1:${PORT} --quick-smoke"
else
  echo "[setup] Starting container ${CONTAINER_NAME} in foreground on port ${PORT}..."
  exec docker run --rm --name "${CONTAINER_NAME}" "${DOCKER_FLAGS[@]}" "${BASE_IMAGE}" \
    --model-path=/models/deepseek-ai/DeepSeek-V4.1-Flash --port="${PORT}"
fi

#!/usr/bin/env bash
# ==============================================================================
# DeepSeek-V4.1-Flash — G4 (NVIDIA RTX PRO 6000 SM120) Serving Entrypoint
# ==============================================================================
# Features:
#   - 1M context window (--context-length=1048576) with 128 MiB bounded FP4 indexer
#   - DSpark speculative decoding (--speculative-algorithm=DSPARK, 5 draft tokens)
#   - Tool call & reasoning parsers (--tool-call-parser=deepseekv41,
#     --reasoning-parser=deepseek-v41, SGLANG_TOOL_STRICT_LEVEL=2)
#   - Automatic topology dispatch:
#       * 4 GPUs (g4-standard-192): 1x TP4+EP4 replica on port 7080 (64 max seqs, 1.6M KV tokens)
#       * 8 GPUs (g4-standard-384): 2x TP4+EP4 replicas (GPUs 0-3 :30000 + GPUs 4-7 :30001)
#         fronted by sglang_router --policy cache_aware on port 7080 (128 max seqs, 3.2M KV tokens)
# ==============================================================================
set -euo pipefail

ulimit -c 0

PORT="${PORT:-7080}"
MODEL_PATH="${MODEL_PATH:-${AIP_STORAGE_URI:-/models/deepseek-ai/DeepSeek-V4.1-Flash}}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/models/deepseek-ai/DeepSeek-V4.1-Flash}"
PATCH_DIR="${PATCH_DIR:-/opt/dsv41_g4_patches}"
SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"

for arg in "$@"; do
  case "$arg" in
    --model=*|--model-path=*)
      MODEL_PATH="${arg#*=}"
      ;;
    --port=*)
      PORT="${arg#*=}"
      ;;
  esac
done

echo "[entrypoint] Starting DeepSeek-V4.1-Flash G4 serving recipe..."

# 1. Stage weights from GCS if MODEL_PATH is a gs:// URI
if [[ "${MODEL_PATH}" == gs://* ]]; then
  mkdir -p "${LOCAL_MODEL_DIR}"
  if [[ ! -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
    echo "[entrypoint] Staging checkpoint from ${MODEL_PATH} to ${LOCAL_MODEL_DIR}..."
    if command -v gcloud >/dev/null 2>&1; then
      gcloud storage cp -r "${MODEL_PATH%/}/*" "${LOCAL_MODEL_DIR}/"
    elif command -v gsutil >/dev/null 2>&1; then
      gsutil -m cp -r "${MODEL_PATH%/}/*" "${LOCAL_MODEL_DIR}/"
    else
      echo "[entrypoint] ERROR: Neither gcloud nor gsutil found to stage ${MODEL_PATH}" >&2
      exit 1
    fi
  fi
  SERVE_MODEL_PATH="${LOCAL_MODEL_DIR}"
else
  SERVE_MODEL_PATH="${MODEL_PATH}"
fi

# 2. Verify / apply SM120 kernel patches idempotently
if [[ -d "${PATCH_DIR}" && -d "${SGLANG_ROOT}" ]]; then
  echo "[entrypoint] Verifying SM120 kernel optimization patches..."
  pushd "${SGLANG_ROOT}" >/dev/null
  PATCHES=(
    "22-mhc-mix-stats-tf32-gated.diff"
    "26-mhc-mix-stats-gather-pretrans-swizzle.diff"
    "36-sparse-prefill-qtile.diff"
    "36b-sparse-prefill-v2.diff"
    "37-mhc-post-pre-fuse.diff"
    "38-sparse-decode-splitk.diff"
    "41-prefill-shard-mhc.diff"
    "44-prefill-attn-rowshard.diff"
    "45-sparse-prefill-h64.diff"
    "46-fp4-gemm-direct.diff"
    "48-indexer-mask-incr.diff"
  )
  for p in "${PATCHES[@]}"; do
    pf="${PATCH_DIR}/${p}"
    if [[ -f "${pf}" ]]; then
      if patch -p1 -R --dry-run < "${pf}" >/dev/null 2>&1; then
        echo "  [ok] ${p} (already applied)"
      else
        patch -p1 --forward < "${pf}" || true
        echo "  [applied] ${p}"
      fi
    fi
  done
  find "${SGLANG_ROOT}/python/sglang" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  popd >/dev/null
fi

# 3. Export SM120 Kernel & Runtime Environment Variables
export SGLANG_TOOL_STRICT_LEVEL=2
export SGLANG_SM120_FLASHMLA_BACKEND=triton
export NCCL_P2P_LEVEL=SYS
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private
export SGLANG_CUSTOM_AR_ALLOW_PCIE_P2P=1
export SGLANG_FP8_PREFILL_AR=1
export SGLANG_FP8_PREFILL_AR2=1
export SGLANG_DENSE_NVFP4=1
export SGLANG_DENSE_NVFP4_LAYERS=.
export SGLANG_WO_A_FP8_REQUANT=1
export SGLANG_DSV41_FUSE_MERGE_SINK=1
export SPARSE_MLA_DECODE_DSV4P1_G4=1
export DECODE_HEAD_TRIM_DSV4P1_G4=1
export SGLANG_DSV41_MHC_MIX_TF32=1
export SGLANG_DSV41_MHC_GATHER=1
export SGLANG_DSV41_INDEXER_CHUNK_MB="${SGLANG_DSV41_INDEXER_CHUNK_MB:-128}"
export SGLANG_DSV41_SPARSE_PREFILL_QTILE=2
export SGLANG_DSV41_MHC_FUSE_POST_PRE_TRITON=1
export SGLANG_DSV41_SPARSE_DECODE_SPLITK=1
export SGLANG_DSV41_PREFILL_SHARD_MHC=1
export SGLANG_DSV41_PREFILL_ATTN_ROWSHARD=1
export SGLANG_DSV41_SPARSE_PREFILL_H64=1
export SGLANG_DSV41_FP4_GEMM_DIRECT=1
export SGLANG_DSV41_INDEXER_MASK_INCR=1
export SGLANG_FLASHINFER_AUTOTUNE_CACHE=0

# 4. SGLang Server Arguments
COMMON_FLAGS=(
  --model-path="${SERVE_MODEL_PATH}"
  --trust-remote-code
  --tp-size=4
  --ep-size=4
  --context-length=1048576
  --kv-cache-dtype=fp8_e4m3
  --moe-runner-backend=flashinfer_mxfp4
  --fp8-gemm-backend=flashinfer_cutlass
  --enable-deepseek-v4-fp4-indexer
  --mem-fraction-static=0.856
  --swa-full-tokens-ratio=0.035
  --chunked-prefill-size=8192
  --max-running-requests=64
  --cuda-graph-max-bs-decode=64
  --min-free-slots-delay=1
  --speculative-algorithm=DSPARK
  --enable-hierarchical-cache
  --hicache-ratio=2
  --enable-metrics
  --enable-cache-report
  --reasoning-parser=deepseek-v41
  --tool-call-parser=deepseekv41
  --dist-timeout=1200
  --nnodes=1
  --node-rank=0
  --host=0.0.0.0
)

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)
echo "[entrypoint] Detected ${NUM_GPUS} GPUs."

wait_for_replica() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-2200}"
  local start_s=$SECONDS
  echo "[entrypoint] Waiting up to ${timeout_s}s for ${name} (${url})..."
  while (( SECONDS - start_s < timeout_s )); do
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 10 "${url}" || true)
    if [[ "${code}" == "200" ]]; then
      echo "[entrypoint] ${name} is READY (${url} -> 200)."
      return 0
    fi
    sleep 15
  done
  echo "[entrypoint] ERROR: Timed out waiting for ${name} (${url})" >&2
  return 1
}

if (( NUM_GPUS >= 8 )); then
  echo "[entrypoint] Launching 2 co-resident TP4+EP4 replicas (TP4xDP2, 128 total max-running-requests)..."
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server "${COMMON_FLAGS[@]}" \
    --port=30000 \
    --dist-init-addr=127.0.0.1:20010 \
    > /tmp/sglang.30000.log 2>&1 &

  wait_for_replica "http://127.0.0.1:30000/health_generate" "Replica A" 2200

  CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server "${COMMON_FLAGS[@]}" \
    --port=30001 \
    --dist-init-addr=127.0.0.1:20020 \
    > /tmp/sglang.30001.log 2>&1 &

  wait_for_replica "http://127.0.0.1:30001/health_generate" "Replica B" 2200

  echo "[entrypoint] Both TP4+EP4 replicas ready. Starting cache-aware router on port ${PORT}..."
  exec python3 -m sglang_router.launch_router \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --worker-urls "http://127.0.0.1:30000" "http://127.0.0.1:30001" \
    --policy cache_aware
else
  echo "[entrypoint] Launching single TP4+EP4 replica on GPUs 0-3, port ${PORT}..."
  exec python3 -m sglang.launch_server "${COMMON_FLAGS[@]}" \
    --port="${PORT}" \
    --dist-init-addr=127.0.0.1:20010
fi

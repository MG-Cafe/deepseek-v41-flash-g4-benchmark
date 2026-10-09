#!/usr/bin/env bash
# ==============================================================================
# scripts/deploy_vertex_g4.sh
# Turnkey deployment of DeepSeek-V4.1-Flash to Vertex AI Online Prediction on G4
# Supports 4-GPU (g4-standard-192) or 8-GPU (g4-standard-384), On-Demand or Flex-Start
# ==============================================================================
# Usage:
#   ./scripts/deploy_vertex_g4.sh \
#     --project <YOUR_PROJECT_ID> \
#     --region us-central1 \
#     --model-gcs gs://<YOUR_BUCKET>/DeepSeek-V4.1-Flash \
#     --image-repo us-central1-docker.pkg.dev/<YOUR_PROJECT_ID>/containers/dsv41-flash-g4:latest \
#     --machine-type g4-standard-192 \
#     --mode on-demand
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
ENDPOINT_ID="${ENDPOINT_ID:-}"
ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-deepseek-v41-flash-g4-endpoint}"
MODEL_GCS_URI="${MODEL_GCS_URI:-}"
IMAGE_URI="${IMAGE_URI:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
MACHINE_TYPE="${MACHINE_TYPE:-g4-standard-192}"
MODE="${MODE:-on-demand}" # on-demand | flex-start
MAX_RUNTIME_DURATION="${MAX_RUNTIME_DURATION:-604800s}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"
MAX_REPLICAS="${MAX_REPLICAS:-1}"
SKIP_BUILD=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 --project <PROJECT_ID> --model-gcs <GS_URI> --image-repo <IMAGE_URI> [options]

Options:
  --project <PROJECT_ID>          Google Cloud Project ID (required)
  --region <REGION>               Vertex AI region (default: ${REGION})
  --model-gcs <GS_URI>            GCS URI containing DeepSeek-V4.1-Flash weights (required)
  --image-repo <IMAGE_URI>        Target Artifact Registry URI to build/push image (required)
  --endpoint-id <ID>              Existing Vertex AI Endpoint ID (if omitted, creates a new endpoint)
  --service-account <SA_EMAIL>    Vertex AI Serving Service Account email
                                  (default: vertex-model-serving@<PROJECT_ID>.iam.gserviceaccount.com)
  --machine-type <TYPE>           g4-standard-192 (4 GPUs) or g4-standard-384 (8 GPUs)
                                  (default: ${MACHINE_TYPE})
  --mode <MODE>                   on-demand | flex-start (default: ${MODE})
  --replicas <N>                  Replica count (default: 1; set to 2 on g4-standard-192 for 128 seqs)
  --skip-build                    Skip docker build & push (use existing --image-repo)
  --dry-run                       Generate the Vertex JSON payload and print curl command without executing
  -h, --help                      Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --model-gcs) MODEL_GCS_URI="$2"; shift 2 ;;
    --image-repo) IMAGE_URI="$2"; shift 2 ;;
    --endpoint-id) ENDPOINT_ID="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --replicas) MIN_REPLICAS="$2"; MAX_REPLICAS="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${PROJECT_ID}" || -z "${MODEL_GCS_URI}" || -z "${IMAGE_URI}" ]]; then
  echo "[error] --project, --model-gcs, and --image-repo are required." >&2
  usage
  exit 1
fi

if [[ -z "${SERVICE_ACCOUNT}" ]]; then
  SERVICE_ACCOUNT="vertex-model-serving@${PROJECT_ID}.iam.gserviceaccount.com"
fi

if [[ "${MACHINE_TYPE}" == "g4-standard-384" ]]; then
  ACCEL_COUNT=8
else
  ACCEL_COUNT=4
fi

# 1. Build and push self-contained G4 image
if [[ "${SKIP_BUILD}" == "0" && "${DRY_RUN}" == "0" ]]; then
  echo "[vertex] Building self-contained G4 container image: ${IMAGE_URI}..."
  docker build -t "${IMAGE_URI}" "${REPO_ROOT}"
  echo "[vertex] Pushing ${IMAGE_URI} to Artifact Registry..."
  docker push "${IMAGE_URI}"
fi

# 2. Render Vertex AI upload & deploy payload
PAYLOAD_FILE="/tmp/vertex_deploy_${MACHINE_TYPE}_${MODE}.json"
FLEX_BLOCK=""
if [[ "${MODE}" == "flex-start" || "${MODE}" == "dws-flex" ]]; then
  FLEX_BLOCK=", \"flexStart\": { \"maxRuntimeDuration\": \"${MAX_RUNTIME_DURATION}\" }"
fi

python3 -c "
import json
with open('${REPO_ROOT}/configs/deploy_vertex_g4_384_tp4x2.json') as f:
    cfg = json.load(f)

cfg['deployedModel']['displayName'] = 'deepseek-v41-flash-${MACHINE_TYPE}'
cfg['deployedModel']['serviceAccount'] = '${SERVICE_ACCOUNT}'
cfg['deployedModel']['dedicatedResources']['machineSpec'] = {
    'machineType': '${MACHINE_TYPE}',
    'acceleratorType': 'NVIDIA_RTX_PRO_6000',
    'acceleratorCount': ${ACCEL_COUNT}
}
cfg['deployedModel']['dedicatedResources']['minReplicaCount'] = ${MIN_REPLICAS}
cfg['deployedModel']['dedicatedResources']['maxReplicaCount'] = ${MAX_REPLICAS}
if '${MODE}' in ('flex-start', 'dws-flex'):
    cfg['deployedModel']['dedicatedResources']['flexStart'] = {'maxRuntimeDuration': '${MAX_RUNTIME_DURATION}'}

cfg['model']['displayName'] = 'deepseek-v41-flash-${MACHINE_TYPE}'
cfg['model']['containerSpec']['imageUri'] = '${IMAGE_URI}'
cfg['model']['containerSpec']['args'] = [
    '--model=${MODEL_GCS_URI}',
    '--port=7080'
]

with open('${PAYLOAD_FILE}', 'w') as out:
    json.dump(cfg, out, indent=2)
print('[vertex] Generated deployment payload at ${PAYLOAD_FILE}')
"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${PAYLOAD_FILE}"
  exit 0
fi

# 3. Create Vertex AI Endpoint if not provided
if [[ -z "${ENDPOINT_ID}" ]]; then
  echo "[vertex] Creating Vertex AI Endpoint '${ENDPOINT_DISPLAY_NAME}' in ${REGION}..."
  ENDPOINT_ID=$(gcloud ai endpoints create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${ENDPOINT_DISPLAY_NAME}" \
    --format="value(name)" | awk -F'/' '{print $NF}')
  echo "[vertex] Created Endpoint ID: ${ENDPOINT_ID}"
fi

echo "[vertex] Uploading model to Vertex AI Model Registry in ${REGION}..."
UPLOAD_BODY="/tmp/vertex_upload_model.json"
python3 -c "
import json
with open('${PAYLOAD_FILE}') as f:
    cfg = json.load(f)
with open('${UPLOAD_BODY}', 'w') as out:
    json.dump({'model': cfg['model']}, out, indent=2)
"

UPLOAD_OP=$(curl -sS -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://${REGION}-aiplatform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/models:upload" \
  -d @"${UPLOAD_BODY}")

echo "[vertex] Upload initiated: ${UPLOAD_OP}"
echo "[vertex] Payload saved at ${PAYLOAD_FILE} to deploy to endpoint ${ENDPOINT_ID}."

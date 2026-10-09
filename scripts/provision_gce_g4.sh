#!/usr/bin/env bash
# ==============================================================================
# scripts/provision_gce_g4.sh
# Create a Google Compute Engine G4 instance (On-Demand or DWS Flex-Start)
# Supports both 4-GPU (g4-standard-192) and 8-GPU (g4-standard-384)
# ==============================================================================
# Examples:
#   1. Create 4-GPU G4 VM (g4-standard-192) via On-Demand:
#      ./scripts/provision_gce_g4.sh \
#        --project <YOUR_PROJECT_ID> \
#        --zone us-central1-a \
#        --machine-type g4-standard-192 \
#        --mode on-demand
#
#   2. Create 8-GPU G4 VM (g4-standard-384) via DWS Flex-Start (7-day max duration):
#      ./scripts/provision_gce_g4.sh \
#        --project <YOUR_PROJECT_ID> \
#        --zone us-central1-a \
#        --machine-type g4-standard-384 \
#        --mode dws-flex \
#        --max-run-duration 7d
# ==============================================================================
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-deepseek-v41-flash-g4}"
MACHINE_TYPE="${MACHINE_TYPE:-g4-standard-192}"
MODE="${MODE:-on-demand}" # on-demand | dws-flex | spot
MAX_RUN_DURATION="${MAX_RUN_DURATION:-7d}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-1000}"
NETWORK="${NETWORK:-default}"
SUBNET="${SUBNET:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 --project <PROJECT_ID> [options]

Options:
  --project <PROJECT_ID>          Google Cloud Project ID (required)
  --zone <ZONE>                   GCE zone offering G4 (e.g. us-central1-a, us-east1-b, us-east4-a)
                                  (default: ${ZONE})
  --name <INSTANCE_NAME>          VM instance name (default: ${INSTANCE_NAME})
  --machine-type <TYPE>           g4-standard-192 (4x RTX PRO 6000) or g4-standard-384 (8x RTX PRO 6000)
                                  (default: ${MACHINE_TYPE})
  --mode <MODE>                   Provisioning mode: on-demand | dws-flex | spot (default: ${MODE})
  --max-run-duration <DURATION>   Max run duration for dws-flex mode (default: ${MAX_RUN_DURATION})
  --boot-disk-size <GB>           Hyperdisk Balanced boot disk size in GB (default: ${BOOT_DISK_SIZE_GB})
  --network <NETWORK>             VPC network name (default: ${NETWORK})
  --subnet <SUBNET>               Optional VPC subnet name
  --service-account <SA_EMAIL>    Optional GCE service account email
  --dry-run                       Print the gcloud command without executing it
  -h, --help                      Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --name) INSTANCE_NAME="$2"; shift 2 ;;
    --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --max-run-duration) MAX_RUN_DURATION="$2"; shift 2 ;;
    --boot-disk-size) BOOT_DISK_SIZE_GB="$2"; shift 2 ;;
    --network) NETWORK="$2"; shift 2 ;;
    --subnet) SUBNET="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${PROJECT_ID}" ]]; then
  echo "[error] --project <YOUR_PROJECT_ID> is required." >&2
  usage
  exit 1
fi

GCLOUD_CMD=(
  gcloud beta compute instances create "${INSTANCE_NAME}"
  --project="${PROJECT_ID}"
  --zone="${ZONE}"
  --machine-type="${MACHINE_TYPE}"
  --maintenance-policy=TERMINATE
  --image-family=common-cu128-ubuntu-2204-nvidia-570
  --image-project=deeplearning-platform-release
  --boot-disk-type=hyperdisk-balanced
  --boot-disk-size="${BOOT_DISK_SIZE_GB}GB"
  --network="${NETWORK}"
  --scopes=https://www.googleapis.com/auth/cloud-platform
)

if [[ -n "${SUBNET}" ]]; then
  GCLOUD_CMD+=(--subnet="${SUBNET}")
fi

if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  GCLOUD_CMD+=(--service-account="${SERVICE_ACCOUNT}")
fi

case "${MODE}" in
  on-demand)
    GCLOUD_CMD+=(--provisioning-model=STANDARD)
    ;;
  dws-flex|flex-start|flex)
    GCLOUD_CMD+=(
      --provisioning-model=FLEX_START
      --max-run-duration="${MAX_RUN_DURATION}"
      --instance-termination-action=DELETE
      --reservation-affinity=none
    )
    ;;
  spot)
    GCLOUD_CMD+=(
      --provisioning-model=SPOT
      --instance-termination-action=DELETE
    )
    ;;
  *)
    echo "[error] Invalid --mode '${MODE}'. Choose from: on-demand | dws-flex | spot" >&2
    exit 1
    ;;
esac

echo "[gce] Provisioning ${MACHINE_TYPE} (${MODE}) in ${PROJECT_ID}/${ZONE}..."
echo "  Command: ${GCLOUD_CMD[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${GCLOUD_CMD[@]}"

cat <<EOF

[gce] Instance '${INSTANCE_NAME}' created in ${ZONE}!
Next steps to deploy DeepSeek-V4.1-Flash on the instance:
  1. SSH into the VM:
     gcloud compute ssh ${INSTANCE_NAME} --project=${PROJECT_ID} --zone=${ZONE}
  2. Clone this repository and run the turnkey setup script:
     ./scripts/run_on_existing_g4.sh --model-source gs://<YOUR_BUCKET>/DeepSeek-V4.1-Flash
EOF

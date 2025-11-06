#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
STAGE2_IMAGE_NAME='npu-dev:24.04'
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-image)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output-image requires a value <name:tag>" >&2
        exit 1
      fi
      STAGE2_IMAGE_NAME="$2"; shift 2 ;;
    --)
      shift; break ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
set -a
source "$PROJECT_DIR/merged.env"
set +a
docker build   -f "$PROJECT_DIR/merged.Dockerfile"   -t "$STAGE2_IMAGE_NAME"   --add-host=host.docker.internal:host-gateway   --build-arg BASE_IMAGE_1 \
  --build-arg WITH_ESSENTIAL_APPS \
  --build-arg WITH_SSH \
  --build-arg SSH_USER_NAME \
  --build-arg SSH_USER_PASSWORD \
  --build-arg SSH_USER_UID \
  --build-arg SSH_PUBKEY_FILE \
  --build-arg SSH_PRIVKEY_FILE \
  --build-arg SSH_CONTAINER_PORT \
  --build-arg APT_SOURCE_FILE \
  --build-arg KEEP_APT_SOURCE_FILE \
  --build-arg APT_USE_PROXY \
  --build-arg APT_KEEP_PROXY \
  --build-arg PEI_HTTP_PROXY_1 \
  --build-arg PEI_HTTPS_PROXY_1 \
  --build-arg ENABLE_GLOBAL_PROXY \
  --build-arg REMOVE_GLOBAL_PROXY_AFTER_BUILD \
  --build-arg PEI_STAGE_HOST_DIR_1 \
  --build-arg PEI_STAGE_DIR_1 \
  --build-arg ROOT_PASSWORD \
  --build-arg BASE_IMAGE_1 \
  --build-arg WITH_ESSENTIAL_APPS \
  --build-arg PEI_STAGE_HOST_DIR_2 \
  --build-arg PEI_STAGE_DIR_2 \
  --build-arg PEI_PREFIX_APPS \
  --build-arg PEI_PREFIX_DATA \
  --build-arg PEI_PREFIX_WORKSPACE \
  --build-arg PEI_PREFIX_VOLUME \
  --build-arg PEI_PREFIX_IMAGE \
  --build-arg PEI_PATH_HARD \
  --build-arg PEI_PATH_SOFT \
  --build-arg PEI_HTTP_PROXY_2 \
  --build-arg PEI_HTTPS_PROXY_2 \
  --build-arg ENABLE_GLOBAL_PROXY \
  --build-arg REMOVE_GLOBAL_PROXY_AFTER_BUILD \
  "$PROJECT_DIR"

echo "[merge] Done. Final image: $STAGE2_IMAGE_NAME"

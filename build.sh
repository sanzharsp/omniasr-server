#!/bin/bash
#
# Build script for Omnilingual-ASR server Docker image.
#
# This image contains no model weights. Weights are downloaded at container
# startup according to the MODEL_NAME (and optionally MODEL_CHECKPOINT_URL /
# MODEL_TOKENIZER_URL) environment variables.
#
# Environment Variables:
#
#   NAMESPACE     - Namespace/registry prefix for the image name (optional)
#                   If provided, images will be tagged as NAMESPACE/omniasr-server
#                   Example: abc/omniasr-server
#                   If not provided, defaults to omniasr-server
#
#   LATEST_TAG    - Set to "true" to also tag the image as "latest"
#                   (default: false)
#
#   PUSH          - Set to "true" to push the image to the registry after building
#                   (default: false)
#
# Example usage:
#
#   # Build
#   bash build.sh
#
#   # Build and tag as latest
#   LATEST_TAG=true bash build.sh
#
#   # Build and push to registry
#   PUSH=true bash build.sh
#
#   # Build with namespace and push
#   NAMESPACE=abc PUSH=true bash build.sh
#

BASE_TAG=cu126-pt280

# Build image name with optional namespace
if [ -n "$NAMESPACE" ]; then
    IMAGE_NAME="$NAMESPACE/omniasr-server"
else
    IMAGE_NAME="omniasr-server"
fi

# Build tags
TAGS="-t $IMAGE_NAME:$BASE_TAG"

# Handle latest tag
if [ "${LATEST_TAG:-false}" = "true" ]; then
    TAGS="$TAGS -t $IMAGE_NAME:latest"
fi

# Build command
BUILD_CMD="docker buildx build \
    --platform linux/amd64 \
    $TAGS"

# Optionally push
if [ "${PUSH:-false}" = "true" ]; then
    BUILD_CMD="$BUILD_CMD --push"
fi

# Execute build command
$BUILD_CMD .

echo "Docker image built successfully! You can run it with:"
echo "    docker run --gpus all -p 8080:8080 -e MODEL_NAME=omniASR_CTC_300M_v2 $IMAGE_NAME:$BASE_TAG"

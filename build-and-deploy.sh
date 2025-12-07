#!/bin/bash
# Build and deploy script for forex-pairs-bot on OpenShift/CRC

set -euo pipefail

# Configuration
IMAGE_NAME="forex-pairs-bot"
PROJECT_NAME="${1:-forex-bot}"

echo "========================================"
echo "Forex Pairs Trading Bot - Build & Deploy"
echo "========================================"

# Check if logged into OpenShift
if ! oc whoami &> /dev/null; then
    echo "[ERROR] Not logged into OpenShift. Run 'oc login' first."
    exit 1
fi

# Create project if it doesn't exist
if ! oc get project "$PROJECT_NAME" &> /dev/null; then
    echo "[INFO] Creating project: $PROJECT_NAME"
    oc new-project "$PROJECT_NAME"
else
    echo "[INFO] Using existing project: $PROJECT_NAME"
    oc project "$PROJECT_NAME"
fi

# Build the image using OpenShift's internal build
echo ""
echo "[BUILD] Creating BuildConfig and ImageStream..."

# Create ImageStream
oc create imagestream "$IMAGE_NAME" --dry-run=client -o yaml | oc apply -f -

# Create BuildConfig for Docker build
cat <<EOF | oc apply -f -
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: ${IMAGE_NAME}
  labels:
    app: ${IMAGE_NAME}
spec:
  source:
    type: Binary
  strategy:
    type: Docker
    dockerStrategy:
      dockerfilePath: Containerfile
  output:
    to:
      kind: ImageStreamTag
      name: ${IMAGE_NAME}:latest
EOF

echo ""
echo "[BUILD] Starting build from local source..."
oc start-build "$IMAGE_NAME" --from-dir=. --follow

echo ""
echo "[DEPLOY] Applying Kubernetes manifests..."

# Apply configmap
oc apply -f k8s/configmap.yaml

# Check if secret exists - remind user to create it if not
if ! oc get secret oanda-credentials &> /dev/null; then
    echo ""
    echo "[WARN] Secret 'oanda-credentials' not found!"
    echo "[WARN] Edit k8s/secret.yaml with your OANDA credentials and run:"
    echo "       oc apply -f k8s/secret.yaml"
    echo ""
    echo "[INFO] Skipping deployment until secret is created."
    exit 0
fi

# Update deployment to use internal image
cat k8s/deployment.yaml | sed "s|image: forex-pairs-bot:latest|image: image-registry.openshift-image-registry.svc:5000/${PROJECT_NAME}/${IMAGE_NAME}:latest|" | oc apply -f -

echo ""
echo "[DONE] Deployment complete!"
echo ""
echo "Useful commands:"
echo "  oc get pods                    # Check pod status"
echo "  oc logs -f deploy/${IMAGE_NAME}  # Follow bot logs"
echo "  oc get configmap forex-bot-config -o yaml  # View config"
echo ""

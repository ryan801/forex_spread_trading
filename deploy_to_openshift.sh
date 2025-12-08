#!/bin/bash
# Deploy forex-pairs-bot to OpenShift
# Run this after the GitHub Actions pipeline has built and pushed the image

set -euo pipefail

PROJECT_NAME="${1:-forex-bot}"

echo "========================================"
echo "Forex Pairs Bot - Deploy to OpenShift"
echo "========================================"

# Check if logged into OpenShift
if ! oc whoami &> /dev/null; then
    echo "[ERROR] Not logged into OpenShift. Run 'oc login' first."
    exit 1
fi

echo "[INFO] Logged in as: $(oc whoami)"
echo "[INFO] Target project: $PROJECT_NAME"
echo ""

# Create project if it doesn't exist
if ! oc get project "$PROJECT_NAME" &> /dev/null; then
    echo "[INFO] Creating project: $PROJECT_NAME"
    oc new-project "$PROJECT_NAME"
else
    echo "[INFO] Switching to project: $PROJECT_NAME"
    oc project "$PROJECT_NAME"
fi

echo ""

# Check for ghcr pull secret
if ! oc get secret ghcr-pull-secret &> /dev/null; then
    echo "[WARN] Pull secret 'ghcr-pull-secret' not found!"
    echo ""
    echo "Create it with:"
    echo "  oc create secret docker-registry ghcr-pull-secret \\"
    echo "    --docker-server=ghcr.io \\"
    echo "    --docker-username=YOUR_GITHUB_USERNAME \\"
    echo "    --docker-password=YOUR_GITHUB_PAT \\"
    echo "    --docker-email=YOUR_EMAIL"
    echo ""
    echo "  oc secrets link default ghcr-pull-secret --for=pull"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for OANDA credentials
if ! oc get secret oanda-credentials &> /dev/null; then
    echo "[WARN] Secret 'oanda-credentials' not found!"
    echo ""
    echo "Create it with:"
    echo "  oc apply -f k8s/secret.yaml"
    echo ""
    echo "(After editing k8s/secret.yaml with your OANDA API credentials)"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "[INFO] Applying ConfigMap..."
oc apply -f k8s/configmap.yaml

echo "[INFO] Applying Deployment..."
oc apply -f k8s/deployment.yaml

# Check if this is an update or fresh deploy
if oc get deploy forex-pairs-bot &> /dev/null; then
    echo "[INFO] Triggering rollout to pull latest image..."
    oc rollout restart deploy/forex-pairs-bot
    oc rollout status deploy/forex-pairs-bot --timeout=120s
fi

echo ""
echo "========================================"
echo "[DONE] Deployment complete!"
echo "========================================"
echo ""
echo "Useful commands:"
echo "  oc get pods                        # Check pod status"
echo "  oc logs -f deploy/forex-pairs-bot  # Follow bot logs"
echo "  oc rollout restart deploy/forex-pairs-bot  # Redeploy latest image"
echo ""
#!/bin/bash
PROJECT_ID="gen-lang-client-0896070179"
REGION="us-central1"
CLUSTER_NAME="helloworld-cluster"

# Enable APIs
gcloud services enable container.googleapis.com cloudbuild.googleapis.com

# Build image
gcloud builds submit --config cloudbuild.yaml --substitutions=SHORT_SHA=$(git rev-parse --short HEAD) .

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID

# Create Namespace
kubectl apply -f k8s/namespace.yaml

# Create Secret from .env
# Note: This requires the local .env to be present and correct
kubectl create secret generic rag-secrets --from-env-file=.env -n rag-service --dry-run=client -o yaml | kubectl apply -f -

# Apply manifests
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

echo "Deployment initiated. Check status with: kubectl get pods -n rag-service"

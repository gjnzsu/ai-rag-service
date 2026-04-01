# GKE Deployment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the AI RAG service to Google Kubernetes Engine (GKE) in the `gen-lang-client-0896070179` project.

**Architecture:** Containerized FastAPI application using Artifact Registry for storage, GKE for orchestration, and a PersistentVolumeClaim for ChromaDB storage. Environment variables will be managed via Kubernetes Secrets.

**Tech Stack:** Docker, Kubernetes (GKE), Google Cloud Build, Google Artifact Registry.

---

### Task 1: Dockerize the Application

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Create .dockerignore**

```text
__pycache__/
*.py[cod]
*$py.class
.env
.git/
.gitignore
chroma_db/
.pytest_cache/
```

- [ ] **Step 2: Create Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Build local image to verify**

Run: `docker build -t rag-service:latest .`
Expected: Build completes successfully.

- [ ] **Step 4: Commit Docker configuration**

```bash
git add Dockerfile .dockerignore
git commit -m "deploy: add Dockerfile and .dockerignore"
```

### Task 2: Infrastructure as Code (Kubernetes Manifests)

**Files:**
- Create: `k8s/namespace.yaml`
- Create: `k8s/pvc.yaml`
- Create: `k8s/deployment.yaml`
- Create: `k8s/service.yaml`

- [ ] **Step 1: Create Namespace manifest**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-service
```

- [ ] **Step 2: Create PersistentVolumeClaim for ChromaDB**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: chroma-pvc
  namespace: rag-service
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

- [ ] **Step 3: Create Deployment manifest**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
  namespace: rag-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag-service
        image: gcr.io/gen-lang-client-0896070179/rag-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: CHROMA_PERSIST_DIR
          value: "/data/chroma_db"
        envFrom:
        - secretRef:
            name: rag-secrets
        volumeMounts:
        - name: chroma-data
          mountPath: /data
      volumes:
      - name: chroma-data
        persistentVolumeClaim:
          claimName: chroma-pvc
```

- [ ] **Step 4: Create Service manifest**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-service
  namespace: rag-service
spec:
  type: LoadBalancer
  selector:
    app: rag-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

- [ ] **Step 5: Commit Kubernetes manifests**

```bash
git add k8s/
git commit -m "deploy: add kubernetes manifests"
```

### Task 3: Cloud Build Configuration

**Files:**
- Create: `cloudbuild.yaml`

- [ ] **Step 1: Create cloudbuild.yaml**

```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/rag-service:$SHORT_SHA', '-t', 'gcr.io/$PROJECT_ID/rag-service:latest', '.']

- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/rag-service:$SHORT_SHA']

- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/rag-service:latest']

images:
- 'gcr.io/$PROJECT_ID/rag-service:$SHORT_SHA'
- 'gcr.io/$PROJECT_ID/rag-service:latest'
```

- [ ] **Step 2: Commit Cloud Build config**

```bash
git add cloudbuild.yaml
git commit -m "deploy: add cloudbuild.yaml"
```

### Task 4: Deployment Execution Script

**Files:**
- Create: `deploy.sh`

- [ ] **Step 1: Create deployment script**

```bash
#!/bin/bash
PROJECT_ID="gen-lang-client-0896070179"
REGION="us-central1"
CLUSTER_NAME="rag-cluster"

# Enable APIs
gcloud services enable container.googleapis.com cloudbuild.googleapis.com

# Build image
gcloud builds submit --config cloudbuild.yaml .

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID

# Create Namespace
kubectl apply -f k8s/namespace.yaml

# Create Secret from .env
kubectl create secret generic rag-secrets --from-env-file=.env -n rag-service --dry-run=client -o yaml | kubectl apply -f -

# Apply manifests
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

echo "Deployment initiated. Check status with: kubectl get pods -n rag-service"
```

- [ ] **Step 2: Make script executable and commit**

```bash
chmod +x deploy.sh
git add deploy.sh
git commit -m "deploy: add deployment script"
```

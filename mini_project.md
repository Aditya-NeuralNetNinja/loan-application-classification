---
title: "Scalable ML Serving on AWS ECS Fargate — Mini-Project Learning Path"
type: feat
date: 2026-03-29
---

# Scalable ML Serving on AWS — Learning Path

7 mini-projects that progressively teach you AWS infrastructure and ML deployment, each building on the last. By the end, you'll have a fully automated, scalable, monitored ML inference service.

```
Mini-Project 1        Mini-Project 2        Mini-Project 3
Docker + Local     →  S3 + IAM           →  ECR + ECS Fargate
"It runs on my       "Artifacts live        "It runs in
 machine"             in the cloud"          the cloud"
      │                    │                      │
      ▼                    ▼                      ▼
Mini-Project 4        Mini-Project 5        Mini-Project 6
ALB + Networking   →  GitHub Actions CI/CD → Auto-Scaling +
"It has a URL"        "It deploys itself"    Monitoring
                                             "It heals itself"
                                                  │
                                                  ▼
                                            Mini-Project 7
                                            MLOps + Drift
                                            "It stays smart"
```

---

# Mini-Project 1: Fix Docker & Run Locally

> **Learn:** Docker fundamentals, multi-stage builds, container debugging
>
> **AWS Services:** None (local only)
>
> **Time Estimate:** Half a day
>
> **Prerequisite:** Docker Desktop installed

## Problem

The Dockerfile references `data/processed/models/best_gbt` which **does not exist** in the repo. `docker build` fails immediately. You need to fix this before anything else.

## What You'll Do

### 1.1 — Generate the missing model artifact

- [ ] Run notebook `4a_model_training.ipynb` to generate `data/processed/models/best_gbt/`
- [ ] Verify the directory exists and contains Spark model metadata files
- [ ] Confirm `data/processed/pipeline_model/` also exists (should already be committed)

### 1.2 — Build and run the Docker image locally

- [ ] Run `docker build -t hmda-classifier .` and confirm it succeeds
- [ ] Run in API mode: `docker run -p 8000:8000 -e APP_MODE=api hmda-classifier`
- [ ] Run in Streamlit mode: `docker run -p 8000:8000 hmda-classifier`
- [ ] Hit `http://localhost:8000/health` and confirm a 200 response
- [ ] Hit `http://localhost:8000/docs` to see the FastAPI Swagger UI
- [ ] Upload `app/assets/demo_inference_100_rows.csv` to `/inference/predict-csv` and confirm predictions come back

### 1.3 — Optimize the Docker image

- [ ] Check image size with `docker images hmda-classifier` (expect ~1.5-2 GB)
- [ ] Add a `.dockerignore` review — confirm `data/raw/`, notebooks, `.git/` are excluded
- [ ] Try a multi-stage build: one stage for dependencies, one for the app (aim for < 1.5 GB)

### 1.4 — Improve the health check

The current `/health` endpoint only checks if a JSON file exists. It doesn't verify Spark or the model.

- [ ] Add a `/health/ready` endpoint in `app/api.py` that:
  - Calls `_spark_session()` to verify JVM is alive
  - Calls `_load_models_bundle()` to verify model loads
  - Returns `{"status": "ready", "spark": true, "model": true}`
- [ ] Update the Dockerfile `HEALTHCHECK` to use `/health/ready`

## Key Concepts You'll Learn

| Concept | What It Means |
|---|---|
| `COPY` vs `ADD` | How Docker bakes files into images |
| Multi-stage builds | Keeping images small by separating build from runtime |
| `HEALTHCHECK` | How Docker knows your app is actually working |
| Port mapping (`-p`) | Connecting your machine to the container's network |
| Environment variables (`-e`) | Configuring containers without changing code |

## Checkpoint

You're done when: `docker run` starts the app, `/health/ready` returns `200`, and you can score a CSV through the API.

---

# Mini-Project 2: S3 for Model Artifacts

> **Learn:** S3 buckets, versioning, IAM policies, AWS CLI, boto3
>
> **AWS Services:** S3, IAM
>
> **Cost:** ~$0.03/month (under free tier)
>
> **Prerequisite:** AWS account, AWS CLI configured (`aws configure`)

## Problem

Model artifacts are baked into the Docker image. Every time you retrain, you need to rebuild the entire image. You want to decouple model updates from code deployments.

## What You'll Do

### 2.1 — Create an S3 bucket with versioning

- [ ] Create the bucket:
  ```bash
  aws s3api create-bucket --bucket hmda-ml-artifacts-<your-initials> --region us-east-1
  ```
- [ ] Enable versioning:
  ```bash
  aws s3api put-bucket-versioning --bucket hmda-ml-artifacts-<your-initials> \
    --versioning-configuration Status=Enabled
  ```
- [ ] Verify: `aws s3api get-bucket-versioning --bucket hmda-ml-artifacts-<your-initials>`

### 2.2 — Upload model artifacts with a versioned path structure

- [ ] Upload the pipeline model:
  ```bash
  aws s3 cp data/processed/pipeline_model/ \
    s3://hmda-ml-artifacts-<your-initials>/models/pipeline_model/v1.0.0/ --recursive
  ```
- [ ] Upload the GBT model:
  ```bash
  aws s3 cp data/processed/models/best_gbt/ \
    s3://hmda-ml-artifacts-<your-initials>/models/best_gbt/v1.0.0/ --recursive
  ```
- [ ] Upload metadata:
  ```bash
  aws s3 cp data/processed/feature_metadata.json \
    s3://hmda-ml-artifacts-<your-initials>/metadata/v1.0.0/feature_metadata.json
  aws s3 cp data/processed/optimal_threshold.json \
    s3://hmda-ml-artifacts-<your-initials>/metadata/v1.0.0/optimal_threshold.json
  ```
- [ ] Create a pointer file `production/current.json`:
  ```json
  {"pipeline_version": "v1.0.0", "gbt_version": "v1.0.0", "updated_at": "2026-03-29"}
  ```
  Upload it: `aws s3 cp current.json s3://hmda-ml-artifacts-<your-initials>/production/current.json`
- [ ] Verify everything: `aws s3 ls s3://hmda-ml-artifacts-<your-initials>/ --recursive`

### 2.3 — Create an IAM policy for least-privilege S3 access

- [ ] Create a policy `hmda-s3-read-policy` that allows ONLY `s3:GetObject` and `s3:ListBucket` on your bucket
- [ ] Create an IAM user `hmda-app-user` (for local testing only — we'll use roles for ECS later)
- [ ] Attach the policy and generate access keys
- [ ] Test: `aws s3 ls s3://hmda-ml-artifacts-<your-initials>/ --profile hmda-app-user`
- [ ] Test that it CANNOT write: `aws s3 cp test.txt s3://hmda-ml-artifacts-<your-initials>/ --profile hmda-app-user` (should fail)

### 2.4 — Modify the app to download models from S3 at startup

- [ ] Add `MODEL_BUCKET` and `MODEL_VERSION` to `app/config.py`:
  ```python
  MODEL_BUCKET = os.getenv("MODEL_BUCKET", "")
  MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
  ```
- [ ] Create a script `scripts/download_models.sh`:
  ```bash
  #!/bin/bash
  if [ -n "$MODEL_BUCKET" ]; then
    echo "Downloading models from s3://$MODEL_BUCKET..."
    aws s3 cp s3://$MODEL_BUCKET/models/pipeline_model/$MODEL_VERSION/ \
      /app/data/processed/pipeline_model/ --recursive
    aws s3 cp s3://$MODEL_BUCKET/models/best_gbt/$MODEL_VERSION/ \
      /app/data/processed/models/best_gbt/ --recursive
    echo "Models downloaded."
  else
    echo "MODEL_BUCKET not set, using local artifacts."
  fi
  ```
- [ ] Update `entrypoint.sh` to call `download_models.sh` before starting the app
- [ ] Update the Dockerfile: remove the `COPY` lines for model artifacts, install `awscli`
- [ ] Test locally:
  ```bash
  docker build -t hmda-classifier .
  docker run -p 8000:8000 \
    -e APP_MODE=api \
    -e MODEL_BUCKET=hmda-ml-artifacts-<your-initials> \
    -e AWS_ACCESS_KEY_ID=... \
    -e AWS_SECRET_ACCESS_KEY=... \
    hmda-classifier
  ```

### 2.5 — Explore S3 versioning

- [ ] Upload a "v1.1.0" version of the model (just copy v1.0.0 for now)
- [ ] Update `production/current.json` to point to `v1.1.0`
- [ ] Restart the container with `MODEL_VERSION=v1.1.0` — confirm it picks up the new version
- [ ] Revert `current.json` to `v1.0.0` — this is your first rollback!

## Key Concepts You'll Learn

| Concept | What It Means |
|---|---|
| S3 bucket versioning | Every file change is preserved, enables rollback |
| Path-based versioning | Organizing artifacts by version (`/v1.0.0/`, `/v1.1.0/`) |
| IAM policies | Who can do what on which resources (least privilege) |
| Pointer file pattern | A single file that says "this is the active version" |
| Decoupling artifacts from code | Model updates don't need Docker rebuilds |

## Checkpoint

You're done when: the Docker container starts with NO model files baked in, downloads them from S3, and serves predictions successfully.

---

# Mini-Project 3: Push to ECR & Run on ECS Fargate

> **Learn:** Container registries, ECS task definitions, Fargate serverless compute, task roles
>
> **AWS Services:** ECR, ECS Fargate, IAM (task roles)
>
> **Cost:** ~$4-5/day while running (stop tasks when not testing!)
>
> **Prerequisite:** Mini-Projects 1 & 2 complete

## Problem

The app runs locally in Docker. Now you need to run it in the cloud so anyone can access it.

## What You'll Do

### 3.1 — Create an ECR repository and push your image

- [ ] Create the repo:
  ```bash
  aws ecr create-repository \
    --repository-name hmda-loan-classifier \
    --image-scanning-configuration scanOnPush=true \
    --region us-east-1
  ```
- [ ] Authenticate Docker to ECR:
  ```bash
  aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
  ```
- [ ] Tag and push:
  ```bash
  docker tag hmda-classifier:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/hmda-loan-classifier:latest
  docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/hmda-loan-classifier:latest
  ```
- [ ] Add a lifecycle policy to auto-clean old images (keep last 10):
  ```bash
  aws ecr put-lifecycle-policy --repository-name hmda-loan-classifier \
    --lifecycle-policy-text file://ecr-lifecycle-policy.json
  ```
- [ ] Verify the image appears in ECR console and the vulnerability scan ran

### 3.2 — Create an ECS cluster

- [ ] Create cluster with Container Insights:
  ```bash
  aws ecs create-cluster \
    --cluster-name hmda-cluster \
    --settings name=containerInsights,value=enhanced
  ```
- [ ] Verify: `aws ecs describe-clusters --clusters hmda-cluster`

### 3.3 — Create IAM roles for ECS

You need TWO roles (this is a common confusion point):

| Role | Purpose | Who Assumes It |
|---|---|---|
| `ecsTaskExecutionRole` | Lets ECS **pull your image** from ECR and **write logs** to CloudWatch | ECS service (infrastructure) |
| `hmda-task-role` | Lets your **app code** read from S3 | Your container (application) |

- [ ] Create `ecsTaskExecutionRole` with `AmazonECSTaskExecutionRolePolicy` attached
- [ ] Create `hmda-task-role` with your `hmda-s3-read-policy` from Mini-Project 2
- [ ] Both roles need a trust policy allowing `ecs-tasks.amazonaws.com` to assume them

### 3.4 — Write and register a task definition

Create `task-definition.json`:

```json
{
  "family": "hmda-inference",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/hmda-task-role",
  "containerDefinitions": [{
    "name": "hmda-app",
    "image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/hmda-loan-classifier:latest",
    "essential": true,
    "portMappings": [{ "containerPort": 8000, "protocol": "tcp" }],
    "environment": [
      { "name": "APP_MODE", "value": "api" },
      { "name": "PORT", "value": "8000" },
      { "name": "SPARK_DRIVER_MEMORY", "value": "1g" },
      { "name": "MODEL_BUCKET", "value": "hmda-ml-artifacts-<your-initials>" },
      { "name": "MODEL_VERSION", "value": "v1.0.0" }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/hmda-inference",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs",
        "awslogs-create-group": "true"
      }
    },
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
      "interval": 30,
      "timeout": 10,
      "retries": 5,
      "startPeriod": 180
    }
  }]
}
```

- [ ] Register it: `aws ecs register-task-definition --cli-input-json file://task-definition.json`
- [ ] Note the `startPeriod: 180` — this gives Spark 3 minutes to start before health checks kick in

### 3.5 — Run a standalone task (not a service yet)

- [ ] Run a one-off task to test:
  ```bash
  aws ecs run-task \
    --cluster hmda-cluster \
    --task-definition hmda-inference \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={
      subnets=[<your-subnet-id>],
      securityGroups=[<your-sg-id>],
      assignPublicIp=ENABLED
    }"
  ```
- [ ] Watch it start in the ECS console (Tasks tab)
- [ ] Check CloudWatch logs at `/ecs/hmda-inference` — you should see Spark initialization logs
- [ ] Find the task's public IP in the console
- [ ] Hit `http://<PUBLIC_IP>:8000/health` — confirm it responds!
- [ ] Stop the task when done (to save costs)

### 3.6 — Debug common failures

Expect at least one of these to happen (this is the learning!):

| Symptom | Likely Cause | Fix |
|---|---|---|
| Task stops immediately | OOM — not enough memory for Spark + JVM | Increase `memory` in task def |
| Task starts, health check fails | Spark hasn't initialized in time | Increase `startPeriod` |
| "Unable to pull image" | ECR auth or role issue | Check `ecsTaskExecutionRole` has ECR permissions |
| "Unable to assume role" | Trust policy wrong | Verify `ecs-tasks.amazonaws.com` in trust policy |
| App starts but can't read S3 | Task role missing S3 permissions | Check `hmda-task-role` has your S3 policy |

## Key Concepts You'll Learn

| Concept | What It Means |
|---|---|
| ECR | Docker Hub but private, inside your AWS account |
| Task Definition | The "recipe" — what image, how much CPU/RAM, what env vars |
| Fargate | Serverless containers — no servers to manage, pay per second |
| Execution Role vs Task Role | Infrastructure permissions vs application permissions |
| awsvpc networking | Each task gets its own IP address in your VPC |
| `startPeriod` | Grace time before health checks start (critical for JVM apps) |

## Checkpoint

You're done when: a Fargate task starts, downloads models from S3, passes health checks, and you can hit the API from your browser via the task's public IP.

---

# Mini-Project 4: ALB + Networking

> **Learn:** Load balancers, target groups, VPC networking, security groups, DNS
>
> **AWS Services:** ALB, VPC, Route 53 (optional)
>
> **Cost:** ~$0.70/day for ALB + running tasks
>
> **Prerequisite:** Mini-Project 3 complete

## Problem

Your task has a public IP, but it changes every time the task restarts. You need a stable URL with health-check-based routing.

## What You'll Do

### 4.1 — Create an Application Load Balancer

- [ ] Create ALB in your VPC (must be in public subnets, at least 2 AZs):
  ```bash
  aws elbv2 create-load-balancer \
    --name hmda-alb \
    --subnets subnet-xxx subnet-yyy \
    --security-groups sg-zzz \
    --scheme internet-facing \
    --type application
  ```
- [ ] Note the ALB DNS name (e.g., `hmda-alb-123456.us-east-1.elb.amazonaws.com`) — this is your stable URL

### 4.2 — Create a target group

- [ ] Create target group (MUST use `ip` type for Fargate):
  ```bash
  aws elbv2 create-target-group \
    --name hmda-tg \
    --protocol HTTP \
    --port 8000 \
    --vpc-id vpc-xxx \
    --target-type ip \
    --health-check-path "/health" \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 10 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 5
  ```

### 4.3 — Create a listener to connect ALB to target group

- [ ] Create listener:
  ```bash
  aws elbv2 create-listener \
    --load-balancer-arn <ALB_ARN> \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=<TG_ARN>
  ```

### 4.4 — Create an ECS service (replaces the standalone task)

This is the key upgrade: a **service** maintains your desired task count and integrates with the ALB.

- [ ] Create the service:
  ```bash
  aws ecs create-service \
    --cluster hmda-cluster \
    --service-name hmda-inference-service \
    --task-definition hmda-inference \
    --desired-count 2 \
    --launch-type FARGATE \
    --platform-version "1.4.0" \
    --health-check-grace-period-seconds 180 \
    --deployment-configuration "maximumPercent=200,minimumHealthyPercent=100" \
    --network-configuration "awsvpcConfiguration={
      subnets=[subnet-xxx,subnet-yyy],
      securityGroups=[sg-zzz],
      assignPublicIp=ENABLED
    }" \
    --load-balancers "targetGroupArn=<TG_ARN>,containerName=hmda-app,containerPort=8000"
  ```
- [ ] `healthCheckGracePeriodSeconds: 180` is critical — without it, ECS kills tasks before Spark finishes starting

### 4.5 — Configure security groups properly

- [ ] ALB security group: allow inbound 80 (HTTP) from `0.0.0.0/0`
- [ ] ECS task security group: allow inbound 8000 **only from the ALB security group** (not the whole internet)
- [ ] This means: internet → ALB (port 80) → task (port 8000). Tasks are not directly accessible.

### 4.6 — Test the full flow

- [ ] Wait for ECS to show 2 running tasks and the target group to show them as "healthy"
- [ ] Hit `http://<ALB_DNS>/health` — should return 200
- [ ] Hit `http://<ALB_DNS>/docs` — FastAPI Swagger UI
- [ ] Upload a CSV via Swagger UI — confirm predictions work
- [ ] Kill one task manually in the console — watch ECS automatically launch a replacement
- [ ] Test rolling update: update the task definition, update the service — watch zero-downtime deployment

### 4.7 — (Optional) Add HTTPS with a custom domain

- [ ] Register a domain or use an existing one
- [ ] Create an ACM certificate for your domain
- [ ] Add an HTTPS (443) listener to the ALB with the certificate
- [ ] Create a Route 53 alias record pointing to the ALB

## Key Concepts You'll Learn

| Concept | What It Means |
|---|---|
| ALB (Application Load Balancer) | Routes traffic to healthy containers, provides a stable URL |
| Target Group | A set of containers the ALB can send traffic to |
| Health checks (ALB) | ALB continuously pings `/health` — unhealthy targets get no traffic |
| ECS Service vs Task | A service maintains N tasks, replaces failures, does rolling deploys |
| `healthCheckGracePeriodSeconds` | Tells ECS "don't check ALB health for X seconds after launch" |
| Security group chaining | ALB SG → Task SG: tasks only accept traffic from the load balancer |
| Rolling deployment | Replace tasks one at a time so there's always capacity (zero downtime) |

## Checkpoint

You're done when: `http://<ALB_DNS>/health` returns 200, you can score CSVs through the ALB URL, and killing a task causes ECS to auto-replace it.

---

# Mini-Project 5: CI/CD with GitHub Actions

> **Learn:** GitHub Actions, OIDC federation, automated Docker builds, deployment pipelines
>
> **AWS Services:** IAM (OIDC provider), ECR, ECS
>
> **Cost:** Free (GitHub Actions free tier: 2,000 min/month)
>
> **Prerequisite:** Mini-Project 4 complete

## Problem

Every change requires you to manually build, push, and deploy. You want: push to `main` → automatically deployed.

## What You'll Do

### 5.1 — Set up OIDC trust (no stored AWS keys!)

This is the modern way. GitHub proves its identity to AWS using a token, not stored secrets.

- [ ] Create an OIDC identity provider in AWS IAM:
  - Provider URL: `https://token.actions.githubusercontent.com`
  - Audience: `sts.amazonaws.com`
- [ ] Create IAM role `github-actions-ecs-deploy` with trust policy:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": { "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com" },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
        "StringLike": { "token.actions.githubusercontent.com:sub": "repo:<YOUR_ORG>/<YOUR_REPO>:*" }
      }
    }]
  }
  ```
- [ ] Attach policies: ECR push, ECS deploy, ECS task definition registration

### 5.2 — Create the GitHub Actions workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to ECS
on:
  push:
    branches: [main]

permissions:
  id-token: write
  contents: read

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: hmda-loan-classifier
  ECS_CLUSTER: hmda-cluster
  ECS_SERVICE: hmda-inference-service

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::<ACCOUNT_ID>:role/github-actions-ecs-deploy
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to ECR
        id: ecr-login
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push
        id: build
        env:
          ECR_REGISTRY: ${{ steps.ecr-login.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
            $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

      - name: Render task definition
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: task-definition.json
          container-name: hmda-app
          image: ${{ steps.build.outputs.image }}

      - name: Deploy to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: ${{ env.ECS_SERVICE }}
          cluster: ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
```

### 5.3 — Add a security scan step

- [ ] Add Trivy vulnerability scanning BEFORE deploy:
  ```yaml
  - name: Security scan
    uses: aquasecurity/trivy-action@master
    with:
      image-ref: ${{ steps.ecr-login.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }}
      severity: CRITICAL,HIGH
      exit-code: 1
  ```
- [ ] This blocks deployment if critical CVEs are found in your image

### 5.4 — Test the full pipeline

- [ ] Push a small change to `main` (e.g., update the `/health` response message)
- [ ] Watch the Actions tab on GitHub — see each step execute
- [ ] Confirm the new image appears in ECR with the commit SHA tag
- [ ] Confirm ECS performs a rolling update (old tasks drain, new tasks start)
- [ ] Hit the ALB URL — see your change live
- [ ] Check how long the full pipeline takes (target: < 15 min)

### 5.5 — Add a test job (bonus)

- [ ] Add a `test` job that runs before `deploy`:
  ```yaml
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.deploy.txt pytest
      - run: pytest tests/ -v
  ```
- [ ] Create a basic test file `tests/test_api.py` that imports the app and checks `/health`
- [ ] Add `needs: test` to the deploy job so it won't deploy if tests fail

## Key Concepts You'll Learn

| Concept | What It Means |
|---|---|
| GitHub Actions | Automated workflows triggered by git events (push, PR, etc.) |
| OIDC federation | GitHub proves identity to AWS using tokens — no stored secrets |
| Image tagging with commit SHA | Every image is traceable to exactly which code built it |
| `wait-for-service-stability` | Pipeline fails if the new deployment doesn't become healthy |
| Trivy scanning | Catches known vulnerabilities in your Docker image before deploy |
| Rolling deployment (observed) | Watch zero-downtime replacement happen automatically |

## Checkpoint

You're done when: pushing to `main` automatically builds, scans, pushes, and deploys your app — and you can see it live at your ALB URL within 15 minutes.

---

# Mini-Project 6: Auto-Scaling + Monitoring

> **Learn:** CloudWatch, auto-scaling policies, alarms, dashboards, structured logging
>
> **AWS Services:** CloudWatch, Application Auto Scaling, SNS
>
> **Cost:** ~$10-15/month for CloudWatch
>
> **Prerequisite:** Mini-Project 5 complete

## Problem

You have one or two tasks running, but you don't know if they're healthy, overloaded, or sitting idle. And if traffic spikes, nothing scales up.

## What You'll Do

### 6.1 — Enable CloudWatch Container Insights

- [ ] Enable enhanced monitoring on your cluster:
  ```bash
  aws ecs update-cluster-settings \
    --cluster hmda-cluster \
    --settings name=containerInsights,value=enhanced
  ```
- [ ] Wait 5 minutes, then check the CloudWatch console → Container Insights
- [ ] Explore the out-of-the-box metrics: CPU, Memory, Network I/O per task

### 6.2 — Add structured logging to the app

Replace `print()` statements with structured JSON logging:

- [ ] Add a JSON log formatter in `app/api.py`:
  ```python
  import logging, json, time

  class JSONFormatter(logging.Formatter):
      def format(self, record):
          return json.dumps({
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "level": record.levelname,
              "message": record.getMessage(),
              "module": record.module,
          })

  handler = logging.StreamHandler()
  handler.setFormatter(JSONFormatter())
  logger = logging.getLogger("hmda")
  logger.addHandler(handler)
  logger.setLevel(logging.INFO)
  ```
- [ ] Log inference requests: row count, latency, model version, denial rate
- [ ] Deploy and verify structured logs appear in CloudWatch Logs

### 6.3 — Add custom CloudWatch metrics

Emit application-level metrics using CloudWatch Embedded Metric Format:

- [ ] Add to `app/inference.py` after scoring completes:
  ```python
  def emit_metrics(rows_scored, latency_ms, denial_rate):
      metric_log = {
          "_aws": {
              "Timestamp": int(time.time() * 1000),
              "CloudWatchMetrics": [{
                  "Namespace": "HMDA/Inference",
                  "Dimensions": [["ModelVersion"]],
                  "Metrics": [
                      {"Name": "RowsScored", "Unit": "Count"},
                      {"Name": "InferenceLatencyMs", "Unit": "Milliseconds"},
                      {"Name": "DenialRate", "Unit": "None"},
                  ]
              }]
          },
          "ModelVersion": "v1.0.0",
          "RowsScored": rows_scored,
          "InferenceLatencyMs": latency_ms,
          "DenialRate": denial_rate,
      }
      print(json.dumps(metric_log))  # CloudWatch picks this up automatically
  ```
- [ ] Deploy, run some inference requests, verify metrics appear in CloudWatch under "HMDA/Inference" namespace

### 6.4 — Configure auto-scaling

- [ ] Register your service as a scalable target:
  ```bash
  aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --resource-id service/hmda-cluster/hmda-inference-service \
    --scalable-dimension ecs:service:DesiredCount \
    --min-capacity 1 \
    --max-capacity 8
  ```
- [ ] Create a target tracking policy (scale on CPU):
  ```bash
  aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --resource-id service/hmda-cluster/hmda-inference-service \
    --scalable-dimension ecs:service:DesiredCount \
    --policy-name hmda-cpu-scaling \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration '{
      "TargetValue": 60.0,
      "PredefinedMetricSpecification": {
        "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
      },
      "ScaleOutCooldown": 60,
      "ScaleInCooldown": 300
    }'
  ```

### 6.5 — Create CloudWatch alarms

- [ ] Memory alarm (critical for JVM):
  ```bash
  aws cloudwatch put-metric-alarm \
    --alarm-name hmda-memory-high \
    --namespace ECS/ContainerInsights \
    --metric-name MemoryUtilized \
    --dimensions Name=ClusterName,Value=hmda-cluster Name=ServiceName,Value=hmda-inference-service \
    --statistic Average --period 300 --threshold 85 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:us-east-1:<ACCOUNT_ID>:hmda-alerts
  ```
- [ ] Create an SNS topic `hmda-alerts` and subscribe your email
- [ ] Create alarms for: CPU > 80%, Memory > 85%, 5xx error rate > 1%

### 6.6 — Build a CloudWatch dashboard

- [ ] Create a dashboard with 4 panels:
  1. **Infrastructure**: CPU utilization, memory utilization, running task count
  2. **API Performance**: request count, latency (from custom metrics)
  3. **Model Health**: denial rate trend, rows scored per hour
  4. **Costs**: estimated Fargate spend

### 6.7 — Load test and watch it scale

- [ ] Use `curl` in a loop or a simple Python script to send many concurrent CSV uploads
- [ ] Watch the CloudWatch dashboard — see CPU spike
- [ ] Watch ECS — see new tasks launch automatically
- [ ] Stop the load — watch tasks scale back down after the cooldown period

## Key Concepts You'll Learn

| Concept | What It Means |
|---|---|
| Container Insights | Pre-built metrics for ECS (CPU, memory, network) without code changes |
| Embedded Metric Format | Emit custom CloudWatch metrics by printing special JSON to stdout |
| Target tracking scaling | "Keep CPU at 60%" — AWS figures out how many tasks to run |
| Cooldown periods | Prevents scaling flapping (adding/removing tasks too rapidly) |
| CloudWatch Alarms + SNS | "If X crosses threshold, send me an email/Slack message" |
| Structured logging | JSON logs that CloudWatch can search, filter, and visualize |

## Checkpoint

You're done when: your dashboard shows live metrics, you've received at least one alarm email, and you've watched auto-scaling add and remove tasks under load.

---

# Mini-Project 7: MLOps — Model Updates & Drift Detection

> **Learn:** Model versioning workflows, blue-green deployments, drift monitoring, rollback
>
> **AWS Services:** S3, ECS (rolling updates), CloudWatch (custom metrics)
>
> **Cost:** No additional cost beyond existing infrastructure
>
> **Prerequisite:** Mini-Project 6 complete

## Problem

The model is static. When you retrain on new HMDA data, there's no safe way to deploy the new model, verify it's working, and roll back if it isn't.

## What You'll Do

### 7.1 — Define a model update workflow

- [ ] Document the manual workflow:
  1. Retrain model (run notebooks) → produces new `best_gbt/` directory
  2. Upload to S3 as a new version (`v1.1.0`)
  3. Update `production/current.json` to point to `v1.1.0`
  4. Update ECS task definition with `MODEL_VERSION=v1.1.0`
  5. ECS performs rolling update → new tasks download new model from S3

- [ ] Create a script `scripts/deploy_model.sh` that automates steps 2-4:
  ```bash
  #!/bin/bash
  VERSION=$1
  BUCKET="hmda-ml-artifacts-<your-initials>"

  echo "Uploading model $VERSION..."
  aws s3 cp data/processed/pipeline_model/ s3://$BUCKET/models/pipeline_model/$VERSION/ --recursive
  aws s3 cp data/processed/models/best_gbt/ s3://$BUCKET/models/best_gbt/$VERSION/ --recursive

  echo "Updating pointer to $VERSION..."
  echo "{\"pipeline_version\": \"$VERSION\", \"gbt_version\": \"$VERSION\", \"updated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    | aws s3 cp - s3://$BUCKET/production/current.json

  echo "Triggering ECS rolling restart..."
  aws ecs update-service --cluster hmda-cluster --service hmda-inference-service --force-new-deployment

  echo "Done. New model $VERSION will be live after rolling update completes."
  ```
- [ ] Test: `./scripts/deploy_model.sh v1.1.0`

### 7.2 — Implement rollback

- [ ] Rollback script `scripts/rollback_model.sh`:
  ```bash
  #!/bin/bash
  VERSION=$1
  BUCKET="hmda-ml-artifacts-<your-initials>"

  echo "Rolling back to model $VERSION..."
  echo "{\"pipeline_version\": \"$VERSION\", \"gbt_version\": \"$VERSION\", \"updated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    | aws s3 cp - s3://$BUCKET/production/current.json

  aws ecs update-service --cluster hmda-cluster --service hmda-inference-service --force-new-deployment
  echo "Rollback to $VERSION initiated."
  ```
- [ ] Practice a rollback: deploy v1.1.0, then roll back to v1.0.0
- [ ] Verify the API serves predictions from the correct model version

### 7.3 — Add model version to API responses

- [ ] Update `/health/ready` to include the active model version:
  ```json
  {"status": "ready", "model_version": "v1.0.0", "spark": true}
  ```
- [ ] Update inference response to include `model_version` field
- [ ] This makes it easy to verify which model is serving after a deploy/rollback

### 7.4 — Build drift detection

Monitor whether incoming data is shifting away from what the model was trained on:

- [ ] Track **prediction distribution** — emit `DenialRate` and `AvgDenialProbability` as CloudWatch metrics (already done in Mini-Project 6)
- [ ] Create CloudWatch alarms:
  - Denial rate shifts ±10% from training baseline (~25.9%)
  - Average denial probability shifts ±0.05 from training baseline
- [ ] Create a simple drift check script `scripts/check_drift.py`:
  ```python
  """Query CloudWatch metrics and compare to training baseline."""
  import boto3
  from datetime import datetime, timedelta

  TRAINING_DENIAL_RATE = 0.259  # From feature_metadata.json
  DRIFT_THRESHOLD = 0.10  # 10% shift

  cw = boto3.client("cloudwatch")
  response = cw.get_metric_statistics(
      Namespace="HMDA/Inference",
      MetricName="DenialRate",
      StartTime=datetime.utcnow() - timedelta(hours=24),
      EndTime=datetime.utcnow(),
      Period=3600,
      Statistics=["Average"],
  )

  if response["Datapoints"]:
      current_rate = response["Datapoints"][-1]["Average"]
      drift = abs(current_rate - TRAINING_DENIAL_RATE)
      print(f"Current denial rate: {current_rate:.3f}")
      print(f"Training baseline:   {TRAINING_DENIAL_RATE:.3f}")
      print(f"Drift:               {drift:.3f} ({'ALERT' if drift > DRIFT_THRESHOLD else 'OK'})")
  ```

### 7.5 — Test the full MLOps cycle

Run through the complete lifecycle:

- [ ] **Deploy v1.0.0** → verify predictions via ALB
- [ ] **Run inference requests** → see metrics in CloudWatch dashboard
- [ ] **Deploy v1.1.0** → watch rolling update, verify new version in `/health/ready`
- [ ] **Simulate drift** → manually check with `check_drift.py`
- [ ] **Rollback to v1.0.0** → verify rollback completes, old model is serving
- [ ] **Check CloudWatch** → confirm all events are visible in logs and metrics

## Key Concepts You'll Learn

| Concept | What It Means |
|---|---|
| Model versioning | Multiple model versions coexist in S3, one is "active" |
| Pointer file pattern | A single JSON file controls which version is in production |
| Rolling restart | ECS replaces tasks one at a time — zero downtime model updates |
| Rollback | Revert the pointer and restart — back to the previous model in minutes |
| Data drift | The real world changes, but the model was trained on old data |
| Concept drift | The relationship between features and outcomes shifts |
| Prediction monitoring | Track output distribution to catch problems before users notice |

## Checkpoint

You're done when: you can deploy a new model version, verify it's serving, detect drift in CloudWatch, and roll back — all without downtime.

---

# Summary: What You've Built

```
Mini-Project  AWS Services Learned         Skill Unlocked
──────────────────────────────────────────────────────────────
   1          Docker                        Containerization
   2          S3, IAM                       Cloud storage & security
   3          ECR, ECS Fargate              Serverless containers
   4          ALB, VPC, Security Groups     Networking & load balancing
   5          IAM OIDC, GitHub Actions      CI/CD automation
   6          CloudWatch, Auto Scaling      Observability & resilience
   7          S3 versioning, CloudWatch     MLOps & model lifecycle
```

## Total AWS Cost While Learning

| Phase | Estimated Daily Cost | Notes |
|---|---|---|
| Mini-Project 1 | $0 | Local only |
| Mini-Project 2 | $0 | S3 free tier |
| Mini-Project 3 | ~$4-5 | **Stop tasks when not testing** |
| Mini-Project 4 | ~$5-6 | ALB + 2 Fargate tasks |
| Mini-Project 5 | ~$5-6 | Same infra, CI/CD is free |
| Mini-Project 6 | ~$5-6 | +$0.30/day for CloudWatch |
| Mini-Project 7 | ~$5-6 | Same infra |

**Tip:** Tear down ECS services (`desired-count 0`) and ALB when not actively testing. S3 and ECR cost nearly nothing to keep.

## What's Next After These 7 Projects

- **HTTPS + Custom domain** — ACM certificate + Route 53
- **Authentication** — API keys or AWS Cognito + ALB integration
- **Async batch inference** — SQS queue for large uploads, presigned S3 URLs for results
- **Infrastructure as Code** — Terraform or CDK to define all resources as code
- **Multi-environment** — Separate staging/production ECS clusters
- **SageMaker integration** — Managed retraining pipelines with automated model promotion
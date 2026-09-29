# Deploying the backend to Google Cloud Run (guide, written 2026-09-30; nothing here has been run)

This is for the owner to follow once a GCP project exists.

- **Image:** `docker/Dockerfile` (docs/container-measurements.md). It carries no tile data, models or secrets.
- **Library:** tiles are static files on Vercel (web build); the backend serves only uploads and Search Online (CDSE)
  scenes.
- **Depth inference:** stays on the HF ZeroGPU Space.

Placeholders: `PROJECT_ID`, `REGION` (e.g. `asia-south1`, Mumbai), `BILLING_ACCOUNT_ID`, `SERVICE=dw2-backend`,
`REPO=dw2`.

## 1. Project and billing

```sh
gcloud auth login
gcloud projects create PROJECT_ID --name="DepthWizard2"
gcloud billing accounts list                                   # note BILLING_ACCOUNT_ID
gcloud billing projects link PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
gcloud config set project PROJECT_ID
gcloud config set run/region REGION
```

## 2. Budget alert (do this BEFORE the first deploy)

A small monthly budget, with alerts at 50 % / 90 % / 100 %. Alerts email the billing admins; **they do not stop
spending**. At min-instances 0, idle costs nothing.

```sh
gcloud services enable billingbudgets.googleapis.com
gcloud billing budgets create --billing-account=BILLING_ACCOUNT_ID \
  --display-name="dw2 monthly" --budget-amount=10USD \
  --filter-projects=projects/PROJECT_ID \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
```

## 3. APIs

```sh
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  cloudbuild.googleapis.com earthengine.googleapis.com
```

## 4. Artifact Registry, keeping only the last 2 images

```sh
gcloud artifacts repositories create REPO --repository-format=docker --location=REGION \
  --description="dw2 backend images"
cat > /tmp/dw2-cleanup.json <<'JSON'
[
  {"name": "keep-last-2", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
  {"name": "delete-rest", "action": {"type": "Delete"}, "condition": {"tagState": "any"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies REPO --location=REGION \
  --policy=/tmp/dw2-cleanup.json --no-dry-run
gcloud auth configure-docker REGION-docker.pkg.dev
```

## 5. Build and push (linux/amd64: what Cloud Run runs; this Mac is arm64)

```sh
IMAGE=REGION-docker.pkg.dev/PROJECT_ID/REPO/dw2-backend:$(git rev-parse --short HEAD)
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t "$IMAGE" --push .
```

The measured image is 116.7 MB compressed (2026-09-30).

## 6. Secrets (Secret Manager; never baked into the image, never in `--set-env-vars`)

The backend reads the following from its environment:

| variable | what | how |
|---|---|---|
| `HF_TOKEN` | calls the ZeroGPU Space (depth for uploads / CDSE scenes) | secret |
| `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET` | Search Online (Copernicus Data Space) | secrets |
| `EARTHENGINE_PROJECT` | the GCP project Earth Engine bills / authorises against | plain env var |
| Earth Engine credentials | FABDEM fetch | see below |
| `CORS_ORIGINS` | the Vercel origin(s), e.g. `https://depthwizard-studio.vercel.app` | plain env var |
| `DW2_NO_DOTENV=1`, `DW2_*_DIR` | already set in the image | — |

```sh
printf %s "$HF_TOKEN_VALUE" | gcloud secrets create hf-token --data-file=-
printf %s "$CDSE_ID_VALUE" | gcloud secrets create cdse-client-id --data-file=-
printf %s "$CDSE_SECRET_VALUE" | gcloud secrets create cdse-client-secret --data-file=-
```

**Earth Engine on Cloud Run:**
- Run the service as a dedicated service account that is **registered for Earth Engine** (Earth Engine → register the
  Cloud project, then give the service account the `roles/earthengine.viewer` role, or writer if needed).
- `ee.Initialize(project=...)` then uses the runtime's default credentials, with **no key file**.
- `backend/dem/fabdem.py` calls `ee.Initialize(project=EARTHENGINE_PROJECT)`, which picks these up.
- Test one FABDEM fetch right after the first deploy. If it fails, the GeoTIFF / CDSE flows fall back to GLO-30 only.

```sh
gcloud iam service-accounts create dw2-backend --display-name="dw2 backend"
SA=dw2-backend@PROJECT_ID.iam.gserviceaccount.com
for s in hf-token cdse-client-id cdse-client-secret; do
  gcloud secrets add-iam-policy-binding $s --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
done
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:$SA" --role=roles/earthengine.viewer
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:$SA" --role=roles/serviceusage.serviceUsageConsumer
```

## 7. Deploy

```sh
gcloud run deploy dw2-backend --image "$IMAGE" --region REGION \
  --service-account "$SA" \
  --memory 512Mi --cpu 1 --cpu-boost \
  --concurrency 4 --timeout 300 \
  --min-instances 0 --max-instances 3 \
  --execution-environment gen2 --port 8080 \
  --allow-unauthenticated \
  --set-env-vars "EARTHENGINE_PROJECT=PROJECT_ID,CORS_ORIGINS=https://depthwizard-studio.vercel.app" \
  --set-secrets "HF_TOKEN=hf-token:latest,CDSE_CLIENT_ID=cdse-client-id:latest,CDSE_CLIENT_SECRET=cdse-client-secret:latest"
```

Why these values (measured 2026-09-30, docs/container-measurements.md Part 4):

- **`--memory 512Mi`:**
  - The worst peak is 287 MiB RSS (3 heavy flows at once, amd64); idle is 80 MiB.
  - Cloud Run's `/tmp` is in memory and the backend does not clean it (about 2.5–5 MB per upload / scene
    generation), so watch memory on long-lived instances.
  - Raise to `1Gi` if you raise the concurrency or ever add Method 6.
- **`--concurrency 4`:** each extra concurrent heavy flow adds about 40–60 MiB. 4 keeps the worst case under 512 MiB.
- **`--timeout 300`:** a CDSE scene + FABDEM + Space call takes up to about 35 s; this leaves margin for a slow Space
  cold start.
- **`--cpu-boost`:** extra CPU during start-up, which shortens cold starts.
- **`--max-instances 3`:** a cost cap for a demo.

**min-instances:** keep 0 by default (scale to zero, no idle cost).

For judging / demo days:
```sh
gcloud run services update dw2-backend --region REGION --min-instances 1
```
Afterwards, back to zero:
```sh
gcloud run services update dw2-backend --region REGION --min-instances 0
```

Then:
- update the frontend's `VITE_API_BASE` (Vercel project env) to the service URL;
- redeploy the frontend;
- retire Render.

## 8. Measure the real cold start after the first deploy, and decide

```sh
URL=$(gcloud run services describe dw2-backend --region REGION --format='value(status.url)')
for i in 1 2 3 4 5; do
  # force a fresh instance: a no-op revision (or wait > 15 min idle), then time the first request
  gcloud run services update dw2-backend --region REGION --update-labels=coldtest=$i --quiet >/dev/null
  curl -s -o /dev/null -w "cold $i: %{time_total}s\n" "$URL/health"
  curl -s -o /dev/null -w "warm $i: %{time_total}s\n" "$URL/health"
done
```

Also time one real Search Online scene and one GeoTIFF upload from the web app. Their first request imports rasterio /
Earth Engine lazily.

**Decision rule:**
- If the **median cold `/health` is ≤ 5 s**, keep scale-to-zero (min-instances 0), and use min-instances 1 only on
  judging days (the two commands in §7).
- Above 5 s, reconsider:
  - look at the image-pull time in the logs;
  - try `--cpu 2` during start-up;
  - or accept min-instances 1 at its monthly cost.
- The container itself was ready in about 0.5 s locally (indicative, arm64), so most of a cold start will be the
  platform's image pull and boot.

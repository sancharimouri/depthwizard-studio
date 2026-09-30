# Deploy the backend to Cloud Run: runbook (rewritten 2026-09-30 for `release-candidate-2026-09-30`)

From "just logged into GCP" to "backend live and verified". The steps are copy-paste; run them in order from the repo
root on the branch you are deploying.

- **Fill in every `<<LIKE_THIS>>` value first**, in section 0.
- **Render keeps running until step 11 passes.** It is the rollback (section 13).
- **Background:**
  - measurements: `docs/release-candidate-2026-09-30.md` (Parts 2–5) and `docs/container-measurements.md`;
  - image: `docker/Dockerfile`: non-root, logs to stdout, SIGTERM-clean, and the /tmp cap on.

## 0. Values to fill in (once, in your shell)

```sh
export PROJECT_ID=<<PROJECT_ID>>                 # the Cloud project Cloud Run runs in (see step 4: ideally your EE project)
export EE_PROJECT=<<EARTHENGINE_PROJECT>>        # the Earth Engine-registered project (= PROJECT_ID if you reuse it)
export BILLING_ACCOUNT_ID=<<BILLING_ACCOUNT_ID>> # from: gcloud billing accounts list
export REGION=asia-south1                        # Mumbai; change only if you have a reason
export SERVICE=dw2-backend
export REPO=dw2                                  # Artifact Registry repository name
export SA=dw2-backend@${PROJECT_ID}.iam.gserviceaccount.com
export TAG=rc-2026-09-30-$(git rev-parse --short HEAD)
export IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${TAG}
export ORIGINS="https://depthwizard-studio.vercel.app,http://localhost:5173,http://127.0.0.1:5173"  # <<add any other Vercel domain>>
```

Also have ready, from the repo `.env`, **never pasted into a file you commit**:
- `<<HF_TOKEN>>`: a **read** token;
- `<<CDSE_CLIENT_ID>>`;
- `<<CDSE_CLIENT_SECRET>>`.

## 1. Project and billing

```sh
gcloud auth login
# EITHER reuse the existing EE project (recommended; skip the create):
gcloud config set project "$PROJECT_ID"
# OR create a new one:
# gcloud projects create "$PROJECT_ID" --name="DepthWizard2" && gcloud config set project "$PROJECT_ID"
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT_ID"
gcloud config set run/region "$REGION"
```

## 2. Budget alert at $50 / $150 / $250 (before the first deploy)

One monthly budget of $250 with thresholds at 20%, 60% and 100%, which is $50, $150 and $250. Alerts email the
billing admins; **they do not stop spending**. `--max-instances 3` (step 9) is the hard cap.

```sh
gcloud services enable billingbudgets.googleapis.com
gcloud billing budgets create --billing-account="$BILLING_ACCOUNT_ID" \
  --display-name="dw2 monthly" --budget-amount=250USD \
  --filter-projects="projects/$PROJECT_ID" \
  --threshold-rule=percent=0.2 --threshold-rule=percent=0.6 --threshold-rule=percent=1.0
gcloud billing budgets list --billing-account="$BILLING_ACCOUNT_ID"   # check: one budget, three thresholds
```

## 3. Enable the APIs

```sh
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  iam.googleapis.com earthengine.googleapis.com serviceusage.googleapis.com --project "$PROJECT_ID"
# if EE lives in a different project:
[ "$EE_PROJECT" != "$PROJECT_ID" ] && gcloud services enable earthengine.googleapis.com --project "$EE_PROJECT"
```

## 4. Earth Engine registration and the service account

**What the backend does:**
- `backend/dem/fabdem.py` calls `ee.Initialize(project=EARTHENGINE_PROJECT)` with no key file.
- On Cloud Run, earthengine-api falls back to Application Default Credentials: the service's **runtime service
  account**. Verified in earthengine-api 1.7.43 `ee.data.get_persistent_credentials`.

**Registration:**
1. **If `$EE_PROJECT` is already registered** for Earth Engine (your current `EARTHENGINE_PROJECT` is), skip this.
2. **Otherwise** (a new project), register it for **noncommercial** use: open
   `https://code.earthengine.google.com/register` → *Use with a Cloud Project* → *Unpaid usage / Academia &
   Research* → select `<<EE_PROJECT>>` → confirm.

```sh
gcloud iam service-accounts create dw2-backend --display-name="DepthWizard2 backend (Cloud Run)" --project "$PROJECT_ID"
# Earth Engine read access + permission to bill EE usage to the EE project (FABDEM is read-only: viewer is enough)
gcloud projects add-iam-policy-binding "$EE_PROJECT" --member="serviceAccount:$SA" --role=roles/earthengine.viewer
gcloud projects add-iam-policy-binding "$EE_PROJECT" --member="serviceAccount:$SA" --role=roles/serviceusage.serviceUsageConsumer
```

## 5. Secret Manager

`printf %s` keeps a trailing newline out of the secret. Run these in a shell whose history you then clear, or type
the values at a `read -s` prompt.

```sh
read -s -p "HF_TOKEN: " V && printf %s "$V" | gcloud secrets create hf-token --replication-policy=automatic --data-file=- ; echo
read -s -p "CDSE_CLIENT_ID: " V && printf %s "$V" | gcloud secrets create cdse-client-id --replication-policy=automatic --data-file=- ; echo
read -s -p "CDSE_CLIENT_SECRET: " V && printf %s "$V" | gcloud secrets create cdse-client-secret --replication-policy=automatic --data-file=- ; echo
unset V
for s in hf-token cdse-client-id cdse-client-secret; do
  gcloud secrets add-iam-policy-binding "$s" --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
done
```

## 6. Artifact Registry, keeping the last 2 images

```sh
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" \
  --description="DepthWizard2 backend images"
cat > /tmp/dw2-cleanup.json <<'JSON'
[
  {"name": "keep-last-2", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "any"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies "$REPO" --location="$REGION" \
  --policy=/tmp/dw2-cleanup.json --no-dry-run
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

The Keep rule takes precedence over Delete: the 2 newest versions always stay, and older ones are removed by the
periodic cleanup, not instantly.

## 7. Build and push (amd64)

Build for `linux/amd64`, which is Cloud Run's architecture. On Apple Silicon use Colima with Rosetta
(`colima start --vm-type vz --vz-rosetta`).

```sh
git status --short | grep -v '^??'     # must print nothing: deploy a committed tree
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t "$IMAGE" --push .
gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}" --include-tags
```

The expected size is about 119 MB compressed (Part 2).

## 8. (Optional) local check of the exact image

```sh
docker run --rm -d --name dw2check --platform linux/amd64 -p 127.0.0.1:18080:8080 "$IMAGE"
sleep 3 && curl -s http://127.0.0.1:18080/health && docker stop dw2check
```

## 9. Deploy

- **Timeout:** the longest measured flow (a CDSE scene) took 29–105 s, dominated by upstream CDSE / Earth Engine /
  Space latency, and the Space call alone may take up to its 120 s timeout. **`--timeout 300`** gives margin.
- **Memory:** measured worst peak 287.6 MiB RSS with 3 concurrent flows, so 512Mi fits with ~225 MiB spare. The /tmp
  cap is 64 MiB.

```sh
gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" \
  --service-account "$SA" \
  --memory 512Mi --cpu 1 --concurrency 4 \
  --min-instances 0 --max-instances 3 \
  --cpu-boost --timeout 300 \
  --execution-environment gen2 --port 8080 \
  --allow-unauthenticated \
  --set-env-vars "^@^EARTHENGINE_PROJECT=${EE_PROJECT}@CORS_ORIGINS=${ORIGINS}" \
  --set-secrets "HF_TOKEN=hf-token:latest,CDSE_CLIENT_ID=cdse-client-id:latest,CDSE_CLIENT_SECRET=cdse-client-secret:latest"
export URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo "$URL"
```

- `^@^` switches the env-var delimiter to `@`, because `CORS_ORIGINS` contains commas.
- `--allow-unauthenticated`: the browser calls it directly; CORS restricts which sites can.
- **Do not** set `CORS_ORIGIN_REGEX` (no wildcards).

## 10. Smoke tests (curl)

```sh
curl -s "$URL/health"                                   # {"ok":true}
# CORS: the Vercel origin is allowed, another origin is not
curl -s -o /dev/null -D - -H "Origin: https://depthwizard-studio.vercel.app" "$URL/health" | grep -i access-control-allow-origin
curl -s -o /dev/null -D - -H "Origin: https://example.com" "$URL/health" | grep -ci access-control-allow-origin   # 0
# Facts + Scenario (bundled ThinkHazard + live sources): Darjeeling
curl -s "$URL/api/facts?bbox=88.21,27.0,88.31,27.09" | python3 -c "import json,sys;d=json.load(sys.stdin);print(len(d['facts']),{k:len(v) for k,v in d['scenario'].items()},d['status'])"
# Upload flow (PNG, no georeference): upload -> generate -> fetch an asset
ID=$(curl -s -F "file=@build/slim/inputs/flow_png.png;type=image/png" "$URL/api/input/upload" | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
curl -s -X POST "$URL/api/generate/input/$ID" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['meta']['job'], d['depth']['host'])"
# GeoTIFF + FABDEM (exercises Earth Engine through the service account): expect a "dem" block
ID=$(curl -s -F "file=@build/slim/inputs/flow_geotiff.tif;type=image/tiff" "$URL/api/input/upload" | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
curl -s -X POST "$URL/api/input/$ID/fabdem" | python3 -c "import json,sys;print(json.load(sys.stdin)['dem']['source'])"
# CDSE search (exercises the CDSE secrets)
curl -s -X POST "$URL/api/cdse/search" -H 'content-type: application/json' \
  -d '{"lat":30.21,"lon":74.95,"aoi_km":10,"date_from":"2025-11-01","date_to":"2025-12-31","max_cloud":10}' | python3 -c "import json,sys;print(len(json.load(sys.stdin)['scenes']),'scenes')"
gcloud run services logs read "$SERVICE" --region "$REGION" --limit 50   # stdout logs; look for errors
```

- The inputs are the bench's `build/slim/inputs/`. Regenerate them with
  `python scripts/container_bench.py start <image> --label x` if missing.
- **If FABDEM fails** with an EE auth error, re-check step 4: registration, the two roles, and `EARTHENGINE_PROJECT`.

## 11. Measure the real cold start

Keep `--min-instances 0` and let the service scale to zero; there is no traffic for ~15 minutes. Then:

```sh
for i in 1 2 3 4 5; do
  curl -s -o /dev/null -w "cold $i: %{time_total}s\n" "$URL/health"
  sleep 1200   # 20 min idle between samples so each one is cold
done
```

- Also see Cloud Logging's "Container startup latency" for the revision.
- **Decision rule** (from the slim-container session): keep scale-to-zero if the median cold `/health` is ≤ 5 s;
  otherwise use min-instances 1 on judging days (step 12).
- The local start-to-health median was 0.75 s; Cloud Run adds image pull and sandbox start.

**Cutover:** only after steps 10–11 pass.
1. In Vercel set `VITE_API_BASE` to `$URL`, for the Production environment. It is baked at build time, so it needs a
   rebuild:
   ```sh
   cd frontend && npx vercel env rm VITE_API_BASE production && printf %s "$URL" | npx vercel env add VITE_API_BASE production && npx vercel deploy --prod
   ```
2. Check the live site: open a library tile (static, no backend), a CDSE search, and an upload.
3. **Keep Render running** for at least a day.

## 12. min-instances 1 on judging days (and back)

```sh
gcloud run services update "$SERVICE" --region "$REGION" --min-instances 1   # before judging: no cold starts
gcloud run services update "$SERVICE" --region "$REGION" --min-instances 0   # after: back to scale-to-zero
```

One always-on 512Mi / 1 vCPU instance is billed continuously; about a few dollars a day. Check it against the budget
in step 2.

## 13. ROLLBACK

- **The frontend points back to Render** (the fastest; Render is still running):
  ```sh
  cd frontend && npx vercel env rm VITE_API_BASE production && printf %s "https://depthwizard2-api.onrender.com" | npx vercel env add VITE_API_BASE production && npx vercel deploy --prod
  ```
- **A bad Cloud Run revision:** route traffic back to the previous one.
  ```sh
  gcloud run revisions list --service "$SERVICE" --region "$REGION"
  gcloud run services update-traffic "$SERVICE" --region "$REGION" --to-revisions <<PREVIOUS_REVISION>>=100
  ```
- **Stop Cloud Run spending** without deleting anything:
  `gcloud run services update "$SERVICE" --region "$REGION" --max-instances 0`. This rejects requests, so the
  frontend must point at Render first.
- **Decommission Render** only after Cloud Run has served the live site cleanly for your chosen period. Owner
  decision.

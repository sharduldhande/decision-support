# Coronary Decision Support on AWS Lambda

> *This is a demo and should not be used for real world decision making.*

Upload an ICA or CCTA report (PDF) and get a PCI vs. CABG recommendation grounded in the **2021 ACC/AHA/SCAI Coronary Revascularization Guideline**, with the Class of Recommendation and Level of Evidence behind each point.

This branch is the serverless AWS version. The original FastAPI version that runs on a single VPS is on [`main`](https://github.com/sharduldhande/decision-support/tree/main).

**Live demo:** [aws.sharduldhande.com](https://aws.sharduldhande.com)

---

## What it does

1. **Extract**: Claude reads the PDF (scans included) into a validated Pydantic schema: segment-level stenosis, dominance, LV function, comorbidities, presentation. Lesion severity is computed in code from the guideline's thresholds (≥70%, or ≥50% for left main, is significant; 40–69% is intermediate).
2. **Scope**: STEMI, posterior MI, stable and non-coronary reports are declined with an explanation.
3. **Retrieve**: rule-based queries (PCI vs. CABG, lesion summary, comorbidities, vessel count, plus left main, proximal LAD, low EF, prior CABG, age, valve disease, diabetes and FFR/iFR when relevant) and up to four model-written queries for anything the rules miss search a Chroma index of the guideline text and its 135 graded recommendations. Hits are grouped by guideline section, and the top eight sections are sent whole with all their recommendations (long sections send the matching passages and their neighbours). STEMI-only sections are left out.
4. **Decide**: Claude picks PCI, CABG or medical therapy using only the retrieved guideline text, and returns confidence, the factors favouring each option, next steps and verbatim citations.

---

## Architecture

The analysis takes 15–45 seconds, so the API never waits on it. The browser starts a job, uploads the PDF straight to S3, and polls for the result.

```
 browser                        AWS
 ───────                        ───
 1. POST /uploads ────────────► API Gateway ─► upload Lambda
    (Turnstile token)                           ├─ verifies the token with Cloudflare
                                                └─ creates the job in DynamoDB (waiting)
    ◄──────────────────────────  job id + presigned S3 POST

 2. POST the PDF ─────────────► S3  uploads/<job id>.pdf
                                     │ ObjectCreated event
                                     ▼
                                worker Lambda (container image)
                                     ├─ claims the job (waiting → reading)
                                     ├─ extract → retrieve → decide, updating the status
                                     └─ stores the result and deletes the PDF

 3. GET /jobs/{id} ───────────► API Gateway ─► status Lambda ─► DynamoDB
    every 2 s until done
```

Design choices:

- **Only the upload is guarded by Turnstile.** The bucket is private and the presigned POST allows one `application/pdf` of at most 20 MB for 5 minutes. Job ids are random UUIDs. The worker has no URL; only S3 events can start it, and its conditional claim means one upload is analysed at most once (retries are off).
- **Two small zip Lambdas, one container Lambda.** The upload and status functions use only the standard library and boto3. The worker image carries LangChain, Chroma and the prebuilt index, which it copies to `/tmp` at start-up.
- **Secrets stay out of Terraform.** The Anthropic key and the Turnstile secret are SSM SecureString parameters created by hand. The functions get only the parameter names and read the values at start-up.
- **Limits at the edge.** The HTTP API throttles `POST /uploads` to 2 requests/s (burst 5) and writes JSON access logs. All log groups keep 14 days. Uploads expire after a day and job records after 24 hours (DynamoDB TTL).

---

## Tech stack

| Component | Technology |
|---|---|
| Front end | Static page on S3 + CloudFront (OAC, ACM certificate), Cloudflare Turnstile |
| API | API Gateway HTTP API, Python 3.13 Lambdas (zip) |
| Jobs | S3 presigned POST, S3 event notifications, DynamoDB (on-demand, TTL) |
| Worker | Lambda container image (arm64) from ECR |
| LLM | Claude Haiku 5.5 with structured output |
| Orchestration | LangChain, LangSmith tracing |
| Embeddings | Amazon Titan Text Embeddings V2 on Bedrock |
| Vector database | ChromaDB, built ahead of time and shipped inside the image |
| Secrets | SSM Parameter Store |
| Infrastructure | Terraform |

---

## Deploying

### Prerequisites

- An AWS account with Bedrock access to `amazon.titan-embed-text-v2:0` in us-east-2, and the AWS CLI signed in
- Terraform, and Docker with `buildx`
- An [Anthropic API](https://platform.claude.com) key
- A [Cloudflare Turnstile](https://developers.cloudflare.com/turnstile/) widget for your domain
- The guideline EPUB at `data/ACC_AHA_Guidelines.epub` (not included; it is copyrighted)

To use your own domain, change `site_domain` in `site.tf`, `site_origins` in `main.tf`, `TURNSTILE_HOSTNAMES` in `api.tf`, and the site key in `site/app.js`.

### 1. Build the guideline index

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python ingest.py
```

This splits the guideline into chunks, adds the transcribed recommendations from `data/guideline_recommendations.json`, embeds everything with Titan and writes `app/chroma_db`. Titan V2 allows 60 requests a minute, so a full run takes about 11 minutes. A rerun resumes where it stopped.

### 2. Store the secrets

```bash
aws ssm put-parameter --name /decision-support/anthropic-api-key --type SecureString --value '...'
aws ssm put-parameter --name /decision-support/turnstile-secret --type SecureString --value '...'
```

### 3. Push the worker image and apply

The worker function needs an image in ECR before it can be created, so the first deploy creates the repository on its own:

```bash
terraform init
terraform apply -target=aws_ecr_repository.app

REPO=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%%/*}"
docker buildx build --platform linux/arm64 --provenance=false -t "$REPO:latest" --push .

terraform apply
```

The apply waits for the ACM certificate. Add the validation record from the `certificate_validation_record` output to your DNS, then point your domain at the `cloudfront_domain` output with a CNAME. Set `API_URL` in `site/app.js` to the `api_url` output, then apply again to upload the page.

After code changes, rebuild and push the image, then run `terraform apply`; it picks up the new image digest.

### Running the pipeline locally

```bash
python invoke_local.py path/to/report.pdf
```

Runs extract → retrieve → decide on your machine without Lambda, S3 or DynamoDB, and prints the timing of each stage and the result. It needs `ANTHROPIC_API_KEY` in `.env` and AWS credentials for the Titan embeddings.

---

## Project structure

```
├── main.tf                  # providers and shared settings
├── worker.tf                # ECR repository, worker Lambda and its role
├── storage.tf               # uploads bucket, S3 trigger, DynamoDB jobs table
├── api.tf                   # upload and status Lambdas, HTTP API, throttling
├── site.tf                  # site bucket, CloudFront, ACM certificate
├── Dockerfile               # worker image: Lambda Python 3.13 base + app/
├── app/
│   ├── handler.py           # worker: S3 event → claim job → run pipeline → store result
│   ├── pipeline.py          # LangChain pipeline: extract → retrieve → decide
│   └── schemas.py           # Pydantic schemas for findings and decisions
├── lambdas/
│   ├── upload/handler.py    # POST /uploads: Turnstile, job, presigned POST
│   └── status/handler.py    # GET /jobs/{id}
├── site/                    # static page, script and sample report
├── ingest.py                # builds app/chroma_db with Titan embeddings
├── invoke_local.py          # runs the pipeline locally
└── data/
    └── guideline_recommendations.json  # transcribed recommendation tables (COR/LOE)
```

---

## Clinical background

**NSTEMI** (Non-ST-Elevation Myocardial Infarction) is a heart attack in which one or more coronary arteries are partially blocked. Many NSTEMI patients have multi-vessel coronary artery disease, which calls for a decision on how to revascularize:

- **PCI (Percutaneous Coronary Intervention):** a catheter-based procedure that opens blocked arteries with balloons and stents. Less invasive, shorter recovery.
- **CABG (Coronary Artery Bypass Grafting):** open-heart surgery that routes blood around blocked vessels with grafts. Generally preferred for complex multi-vessel disease or left main stenosis.

The 2021 ACC/AHA/SCAI guideline grades its recommendations for specific anatomical and clinical scenarios by Class of Recommendation and Level of Evidence.

import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

import pipeline  # noqa: E402  (needs the API key from .env loaded first)

log = logging.getLogger("decision-support")

TURNSTILE_SECRET = os.environ.get("TURNSTILE_SECRET", "")
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
TURNSTILE_ACTION = "analyze"
TURNSTILE_HOSTNAMES = set(os.environ.get("TURNSTILE_HOSTNAMES", "decisionsupport.sharduldhande.com").split(","))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://decisionsupport.sharduldhande.com").split(",")

MAX_PDF_BYTES = 10 * 1024 * 1024
RATE_LIMIT = 5
RATE_WINDOW_SECONDS = 3600
recent_requests = defaultdict(deque)


@asynccontextmanager
async def lifespan(app):
    if not TURNSTILE_SECRET:
        raise RuntimeError("TURNSTILE_SECRET is not set")
    pipeline.get_retriever()  # load PubMedBERT now, not on the first upload
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["POST"], allow_headers=["*"])


def check_rate_limit(ip):
    now = time.time()
    times = recent_requests[ip]
    while times and now - times[0] > RATE_WINDOW_SECONDS:
        times.popleft()
    if len(times) >= RATE_LIMIT:
        raise HTTPException(429, f"Limit of {RATE_LIMIT} analyses per hour reached. Please try again later.")
    times.append(now)


def verify_turnstile(token, ip):
    if not token or len(token) > 2048:
        raise HTTPException(403, "Verification failed. Please complete the check and try again.")
    try:
        result = httpx.post(
            TURNSTILE_VERIFY_URL,
            data={"secret": TURNSTILE_SECRET, "response": token, "remoteip": ip},
            timeout=10,
        ).json()
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, "Verification service unavailable. Please try again.")
    if not (result.get("success") and result.get("action") == TURNSTILE_ACTION
            and result.get("hostname") in TURNSTILE_HOSTNAMES):
        log.warning("Turnstile rejected: %s", {k: result.get(k) for k in ("error-codes", "action", "hostname")})
        raise HTTPException(403, "Verification failed. Please complete the check and try again.")


def ndjson(event):
    return json.dumps(event) + "\n"


def stream_analysis(pdf_bytes):
    try:
        for event in pipeline.run_steps(pdf_bytes):
            if "result" in event:
                r = event["result"]
                event = {"result": {
                    "findings": r.findings.model_dump(mode="json"),
                    "decision": r.decision.model_dump(mode="json") if r.decision else None,
                    "message": r.message,
                    "note": r.note,
                }}
            yield ndjson(event)
    except Exception:
        log.exception("Analysis failed")
        yield ndjson({"error": "Something went wrong while analysing the report. Please try again."})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: Request, file: UploadFile = File(...), token: str = Form("")):
    ip = request.client.host
    verify_turnstile(token, ip)
    check_rate_limit(ip)

    pdf_bytes = file.file.read(MAX_PDF_BYTES + 1)
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(413, "The PDF is larger than 10 MB.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(415, "Please upload a PDF file.")

    return StreamingResponse(stream_analysis(pdf_bytes), media_type="application/x-ndjson")

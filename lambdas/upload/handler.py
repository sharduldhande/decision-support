"""POST /uploads: check Turnstile, create a job, and return a presigned S3 upload.

The browser then posts the PDF straight to S3, and the upload starts the worker.
Only the standard library and boto3 (built into the Lambda runtime), so this ships as a zip.
"""
import json
import logging
import os
import time
import urllib.parse
import urllib.request
import uuid

import boto3
from botocore.config import Config

log = logging.getLogger()
log.setLevel(logging.INFO)

REGION = os.environ["AWS_REGION"]
UPLOAD_BUCKET = os.environ["UPLOAD_BUCKET"]

MAX_PDF_BYTES = 20 * 1024 * 1024  # base64 for Claude stays under its 32 MB request limit
UPLOAD_URL_SECONDS = 300
JOB_TTL_SECONDS = 24 * 60 * 60

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
TURNSTILE_ACTION = os.environ.get("TURNSTILE_ACTION", "analyze")
TURNSTILE_HOSTNAMES = set(os.environ["TURNSTILE_HOSTNAMES"].split(","))
TURNSTILE_SECRET = boto3.client("ssm").get_parameter(
    Name=os.environ["TURNSTILE_SECRET_PARAM"], WithDecryption=True
)["Parameter"]["Value"]

jobs = boto3.resource("dynamodb").Table(os.environ["JOBS_TABLE"])
# Sign against the regional endpoint so the upload URL works for a brand-new bucket.
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://s3.{REGION}.amazonaws.com",
    config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
)


class TurnstileUnavailable(Exception):
    pass


def response(status, body):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body),
    }


def verify_turnstile(token, ip):
    if not token or len(token) > 2048:
        return False
    data = urllib.parse.urlencode({"secret": TURNSTILE_SECRET, "response": token, "remoteip": ip}).encode()
    try:
        with urllib.request.urlopen(TURNSTILE_VERIFY_URL, data=data, timeout=10) as reply:
            result = json.load(reply)
    except (OSError, ValueError) as e:
        raise TurnstileUnavailable from e

    # Cloudflare's public test secrets pass any token and report no real action or
    # hostname. Only a test secret we configured can produce this flag.
    if result.get("metadata", {}).get("result_with_testing_key"):
        log.warning("Turnstile test secret in use: every token passes")
        return bool(result.get("success"))

    passed = (
        result.get("success")
        and result.get("action") == TURNSTILE_ACTION
        and result.get("hostname") in TURNSTILE_HOSTNAMES
    )
    if not passed:
        log.warning("Turnstile rejected: %s", {k: result.get(k) for k in ("error-codes", "action", "hostname")})
    return bool(passed)


def handler(event, context):
    headers = event.get("headers") or {}
    ip = event["requestContext"]["http"]["sourceIp"]
    try:
        if not verify_turnstile(headers.get("x-turnstile-token", ""), ip):
            return response(403, {"error": "Verification failed. Please complete the check and try again."})
    except TurnstileUnavailable:
        log.exception("Turnstile unreachable")
        return response(503, {"error": "Verification service unavailable. Please try again."})

    # A random id doubles as the only way to read the job later, so it must not be guessable.
    job_id = str(uuid.uuid4())
    now = int(time.time())
    jobs.put_item(Item={
        "id": job_id,
        "status": "waiting",
        "created_at": now,
        "updated_at": now,
        "expires_at": now + JOB_TTL_SECONDS,
    })

    # S3 enforces these conditions itself: one key, PDF content type, at most 20 MB, 5 minutes.
    upload = s3.generate_presigned_post(
        Bucket=UPLOAD_BUCKET,
        Key=f"uploads/{job_id}.pdf",
        Fields={"Content-Type": "application/pdf"},
        Conditions=[{"Content-Type": "application/pdf"}, ["content-length-range", 1, MAX_PDF_BYTES]],
        ExpiresIn=UPLOAD_URL_SECONDS,
    )
    log.info("Created job %s", job_id)
    return response(201, {"job_id": job_id, "upload": upload})

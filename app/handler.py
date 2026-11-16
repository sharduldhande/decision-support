"""The worker: runs one analysis for each PDF that lands in uploads/.

Started only by S3 events. It claims the job, reads the PDF and deletes it, runs the
pipeline while recording each stage in DynamoDB, then stores the result or the error.
"""
import json
import logging
import os
import re
import shutil
import time
import urllib.parse
from pathlib import Path

import boto3

# Lambda's code directory is read-only and Chroma needs to write lock and journal
# files, so on Lambda copy the baked-in index to /tmp once per environment.
if "LAMBDA_TASK_ROOT" in os.environ:
    tmp_db = Path("/tmp/chroma_db")
    if not tmp_db.exists():
        shutil.copytree(Path(__file__).parent / "chroma_db", tmp_db)
    os.environ["CHROMA_PATH"] = str(tmp_db)

# On Lambda the config holds only the parameter's name. Fetch and decrypt the key once
# per environment, before pipeline builds its Claude client. Locally, .env provides it.
if "ANTHROPIC_API_KEY_PARAM" in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = boto3.client("ssm").get_parameter(
        Name=os.environ["ANTHROPIC_API_KEY_PARAM"], WithDecryption=True
    )["Parameter"]["Value"]

import pipeline  # noqa: E402

log = logging.getLogger()
log.setLevel(logging.INFO)

UPLOAD_BUCKET = os.environ["UPLOAD_BUCKET"]
MAX_PDF_BYTES = 20 * 1024 * 1024
KEY_PATTERN = re.compile(r"^uploads/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.pdf$")
GENERIC_ERROR = "Something went wrong while analysing the report. Please try again."

jobs = boto3.resource("dynamodb").Table(os.environ["JOBS_TABLE"])
s3 = boto3.client("s3")

# Load the embedding model and Chroma index during init, not on the first job.
pipeline.get_retriever()


class JobError(Exception):
    """A problem with the upload itself, shown to the user as is."""


def update_job(job_id, status, **fields):
    fields = {"status": status, "updated_at": int(time.time()), **fields}
    jobs.update_item(
        Key={"id": job_id},
        UpdateExpression="SET " + ", ".join(f"#{name} = :{name}" for name in fields),
        ExpressionAttributeNames={f"#{name}": name for name in fields},
        ExpressionAttributeValues={f":{name}": value for name, value in fields.items()},
    )


def claim(job_id):
    """Move the job from waiting to reading. Only one run can win, so a duplicate S3
    event (they are delivered at least once) or an unknown job does nothing."""
    try:
        jobs.update_item(
            Key={"id": job_id},
            UpdateExpression="SET #status = :reading, #updated_at = :now",
            ConditionExpression="#status = :waiting",
            ExpressionAttributeNames={"#status": "status", "#updated_at": "updated_at"},
            ExpressionAttributeValues={":reading": "reading", ":waiting": "waiting", ":now": int(time.time())},
        )
        return True
    except jobs.meta.client.exceptions.ConditionalCheckFailedException:
        return False


def read_upload(key):
    """Read the PDF and delete it right away: the report is only needed in memory."""
    pdf = s3.get_object(Bucket=UPLOAD_BUCKET, Key=key)["Body"].read()
    s3.delete_object(Bucket=UPLOAD_BUCKET, Key=key)
    if len(pdf) > MAX_PDF_BYTES:
        raise JobError("The PDF is larger than 20 MB.")
    if not pdf.startswith(b"%PDF"):
        raise JobError("Please upload a PDF file.")
    return pdf


def to_dict(result):
    return {
        "findings": result.findings.model_dump(mode="json") if result.findings else None,
        "decision": result.decision.model_dump(mode="json") if result.decision else None,
        "message": result.message,
        "note": result.note,
    }


def process(job_id, key):
    pdf = read_upload(key)
    result = None
    for step in pipeline.run_steps(pdf):
        if step.get("stage") not in (None, "reading"):  # "reading" was set by claim()
            update_job(job_id, step["stage"])
        result = step.get("result", result)
    # Stored as a JSON string: DynamoDB rejects Python floats in nested maps.
    update_job(job_id, "done", result=json.dumps(to_dict(result)))


def handler(event, context):
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        match = KEY_PATTERN.match(key)
        if bucket != UPLOAD_BUCKET or not match:
            log.warning("Ignoring unexpected object %s/%s", bucket, key)
            continue

        job_id = match[1]
        if not claim(job_id):
            log.warning("Job %s is not waiting; skipping", job_id)
            continue

        started = time.time()
        try:
            process(job_id, key)
            log.info("Job %s done in %.1f s", job_id, time.time() - started)
        except JobError as e:
            log.info("Job %s rejected: %s", job_id, e)
            update_job(job_id, "failed", error=str(e))
        except Exception:
            log.exception("Job %s failed", job_id)
            update_job(job_id, "failed", error=GENERIC_ERROR)

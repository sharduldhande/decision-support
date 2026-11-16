"""GET /jobs/{id}: report a job's status, and its result once it is done.

Statuses: waiting -> reading -> searching -> writing -> done, or failed at any point.
"""
import json
import os
import uuid

import boto3

jobs = boto3.resource("dynamodb").Table(os.environ["JOBS_TABLE"])


def response(status, body):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body),
    }


def handler(event, context):
    raw_id = (event.get("pathParameters") or {}).get("id", "")
    try:
        job_id = str(uuid.UUID(raw_id))  # rejects anything that isn't a job id
    except ValueError:
        return response(404, {"error": "No such job."})

    # Strongly consistent, so a poll right after the worker finishes sees "done".
    item = jobs.get_item(Key={"id": job_id}, ConsistentRead=True).get("Item")
    if not item:
        return response(404, {"error": "No such job."})

    body = {"status": item["status"]}
    if "result" in item:
        body["result"] = json.loads(item["result"])
    if "error" in item:
        body["error"] = item["error"]
    return response(200, body)

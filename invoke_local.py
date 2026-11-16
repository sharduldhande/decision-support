"""Run the pipeline on this machine, without Lambda, S3 or DynamoDB.

Usage: python invoke_local.py path/to/report.pdf
Needs ANTHROPIC_API_KEY in .env, and AWS credentials for the Titan embeddings on Bedrock.
"""
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "app"))

started = time.time()
import pipeline  # noqa: E402

pipeline.get_retriever()
print(f"init took {time.time() - started:.1f} s", file=sys.stderr)

pdf = Path(sys.argv[1]).read_bytes()
started, result = time.time(), None
for step in pipeline.run_steps(pdf):
    if "stage" in step:
        print(f"{time.time() - started:5.1f} s  {step['stage']}", file=sys.stderr)
    result = step.get("result", result)
print(f"{time.time() - started:5.1f} s  done", file=sys.stderr)

print(json.dumps({
    "findings": result.findings.model_dump(mode="json") if result.findings else None,
    "decision": result.decision.model_dump(mode="json") if result.decision else None,
    "message": result.message,
    "note": result.note,
}, indent=2))

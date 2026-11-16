# AWS's Lambda base image: Amazon Linux + Python 3.13 + the runtime client that
# receives events and calls the handler.
FROM public.ecr.aws/lambda/python:3.13

# Dependencies first, so code-only changes reuse this cached layer.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# The app and its prebuilt Chroma index (app/chroma_db, built by ingest.py).
COPY app/ ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]

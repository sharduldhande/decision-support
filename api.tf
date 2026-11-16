# The public side: one HTTP API with two routes, each backed by a small zip Lambda.
#   POST /uploads    -> upload: checks Turnstile, creates a job, returns a presigned S3 upload
#   GET  /jobs/{id}  -> status: returns the job's status, and its result once done

# Turnstile secret in SSM, created by hand like the Anthropic key.
locals {
  turnstile_secret_param = "/decision-support/turnstile-secret"
}

# ---------- upload ----------

data "archive_file" "upload" {
  type        = "zip"
  source_dir  = "${path.module}/lambdas/upload"
  output_path = "${path.module}/build/upload.zip"
}

resource "aws_iam_role" "upload" {
  name               = "decision-support-upload"
  assume_role_policy = local.lambda_trust
}

resource "aws_iam_role_policy_attachment" "upload_logs" {
  role       = aws_iam_role.upload.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Create jobs, sign uploads into uploads/ (a presigned POST carries this role's
# permission), and read the Turnstile secret.
resource "aws_iam_role_policy" "upload" {
  name = "create-jobs"
  role = aws_iam_role.upload.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "dynamodb:PutItem"
        Resource = aws_dynamodb_table.jobs.arn
      },
      {
        Effect   = "Allow"
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.uploads.arn}/uploads/*"
      },
      {
        Effect   = "Allow"
        Action   = "ssm:GetParameter"
        Resource = "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.turnstile_secret_param}"
      },
    ]
  })
}

resource "aws_cloudwatch_log_group" "upload" {
  name              = "/aws/lambda/decision-support-upload"
  retention_in_days = 14
}

resource "aws_lambda_function" "upload" {
  function_name    = "decision-support-upload"
  role             = aws_iam_role.upload.arn
  runtime          = "python3.13"
  handler          = "handler.handler"
  architectures    = ["arm64"]
  filename         = data.archive_file.upload.output_path
  source_code_hash = data.archive_file.upload.output_base64sha256
  memory_size      = 256
  timeout          = 15

  environment {
    variables = {
      JOBS_TABLE             = aws_dynamodb_table.jobs.name
      UPLOAD_BUCKET          = aws_s3_bucket.uploads.bucket
      TURNSTILE_SECRET_PARAM = local.turnstile_secret_param
      TURNSTILE_HOSTNAMES    = "aws.sharduldhande.com"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.upload_logs,
    aws_iam_role_policy.upload,
    aws_cloudwatch_log_group.upload,
  ]
}

# ---------- status ----------

data "archive_file" "status" {
  type        = "zip"
  source_dir  = "${path.module}/lambdas/status"
  output_path = "${path.module}/build/status.zip"
}

resource "aws_iam_role" "status" {
  name               = "decision-support-status"
  assume_role_policy = local.lambda_trust
}

resource "aws_iam_role_policy_attachment" "status_logs" {
  role       = aws_iam_role.status.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "status" {
  name = "read-jobs"
  role = aws_iam_role.status.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "dynamodb:GetItem"
      Resource = aws_dynamodb_table.jobs.arn
    }]
  })
}

resource "aws_cloudwatch_log_group" "status" {
  name              = "/aws/lambda/decision-support-status"
  retention_in_days = 14
}

resource "aws_lambda_function" "status" {
  function_name    = "decision-support-status"
  role             = aws_iam_role.status.arn
  runtime          = "python3.13"
  handler          = "handler.handler"
  architectures    = ["arm64"]
  filename         = data.archive_file.status.output_path
  source_code_hash = data.archive_file.status.output_base64sha256
  memory_size      = 256
  timeout          = 5

  environment {
    variables = {
      JOBS_TABLE = aws_dynamodb_table.jobs.name
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.status_logs,
    aws_iam_role_policy.status,
    aws_cloudwatch_log_group.status,
  ]
}

# ---------- HTTP API ----------

resource "aws_apigatewayv2_api" "api" {
  name          = "decision-support"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = local.site_origins
    allow_methods = ["GET", "POST"]
    allow_headers = ["content-type", "x-turnstile-token"]
    max_age       = 3600
  }
}

resource "aws_apigatewayv2_integration" "upload" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.upload.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_integration" "status" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.status.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "upload" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /uploads"
  target    = "integrations/${aws_apigatewayv2_integration.upload.id}"
}

resource "aws_apigatewayv2_route" "status" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "GET /jobs/{id}"
  target    = "integrations/${aws_apigatewayv2_integration.status.id}"
}

# Let the API (and only this API) invoke each function.
resource "aws_lambda_permission" "api_upload" {
  statement_id  = "AllowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.upload.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*/uploads"
}

resource "aws_lambda_permission" "api_status" {
  statement_id  = "AllowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.status.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*/jobs/*"
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/apigateway/decision-support"
  retention_in_days = 14
}

# One stage, deployed automatically on every change. Rate limits apply to all
# callers together (requests per second, with a short burst allowance); requests
# over the limit get a 429 from the API and never reach a Lambda.
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_rate_limit  = 10
    throttling_burst_limit = 20
  }

  # Uploads start the expensive work, so they get the tightest limit.
  route_settings {
    route_key              = aws_apigatewayv2_route.upload.route_key
    throttling_rate_limit  = 2
    throttling_burst_limit = 5
  }

  # A page polls every 2 s, so this allows about 20 analyses in progress at once.
  route_settings {
    route_key              = aws_apigatewayv2_route.status.route_key
    throttling_rate_limit  = 10
    throttling_burst_limit = 20
  }

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api.arn
    format = jsonencode({
      requestId = "$context.requestId"
      time      = "$context.requestTime"
      ip        = "$context.identity.sourceIp"
      route     = "$context.routeKey"
      status    = "$context.status"
      latencyMs = "$context.responseLatency"
      error     = "$context.error.message"
    })
  }
}

output "api_url" {
  value = aws_apigatewayv2_api.api.api_endpoint
}

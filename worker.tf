# The worker: the analysis pipeline as a container image in ECR, run by Lambda.
# It has no URL. Only an upload landing in S3 can start it.

# The Anthropic key lives in SSM Parameter Store as a SecureString (encrypted with
# the account's default aws/ssm KMS key). Terraform deliberately does not manage the
# parameter: the provider reads the decrypted value back into terraform.tfstate.
# It is created once by hand with `aws ssm put-parameter`; Terraform only knows its name.
locals {
  anthropic_key_param = "/decision-support/anthropic-api-key"
}

# Private registry that holds the image.
resource "aws_ecr_repository" "app" {
  name                 = "decision-support"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# Keep only the 5 newest images.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

# The image pushed as :latest. Its digest changes on every push, which is
# what makes Terraform update the function.
data "aws_ecr_image" "app" {
  repository_name = aws_ecr_repository.app.name
  image_tag       = "latest"
}

resource "aws_iam_role" "app" {
  name = "decision-support-lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "app_logs" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Titan embeddings, and nothing else on Bedrock.
resource "aws_iam_role_policy" "app_bedrock" {
  name = "invoke-titan-embeddings"
  role = aws_iam_role.app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "bedrock:InvokeModel"
      Resource = "arn:aws:bedrock:us-east-2::foundation-model/amazon.titan-embed-text-v2:0"
    }]
  })
}

# Read that one parameter. No kms:Decrypt needed: the aws/ssm key's policy lets any
# principal in the account decrypt through SSM once IAM allows the GetParameter.
resource "aws_iam_role_policy" "app_ssm" {
  name = "read-anthropic-key"
  role = aws_iam_role.app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      Resource = "arn:aws:ssm:${local.region}:${local.account_id}:parameter${local.anthropic_key_param}"
    }]
  })
}

# Read and then delete the uploaded PDF, and update its job.
resource "aws_iam_role_policy" "app_jobs" {
  name = "process-uploads"
  role = aws_iam_role.app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.uploads.arn}/uploads/*"
      },
      {
        Effect   = "Allow"
        Action   = "dynamodb:UpdateItem"
        Resource = aws_dynamodb_table.jobs.arn
      },
    ]
  })
}

resource "aws_lambda_function" "app" {
  function_name = "decision-support"
  role          = aws_iam_role.app.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}@${data.aws_ecr_image.app.image_digest}"
  architectures = ["arm64"]
  memory_size   = 1024
  timeout       = 120

  environment {
    variables = {
      ANTHROPIC_API_KEY_PARAM = local.anthropic_key_param # a name, not the secret
      UPLOAD_BUCKET           = aws_s3_bucket.uploads.bucket
      JOBS_TABLE              = aws_dynamodb_table.jobs.name
      ANONYMIZED_TELEMETRY    = "False" # stop Chroma phoning home on every cold start
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.app_logs,
    aws_iam_role_policy.app_ssm,
    aws_iam_role_policy.app_jobs,
    aws_cloudwatch_log_group.app,
  ]
}

# Lambda would retry a failed S3-triggered run twice, which means up to three
# rounds of Claude calls for one bad PDF. The worker records failures instead.
resource "aws_lambda_function_event_invoke_config" "app" {
  function_name          = aws_lambda_function.app.function_name
  maximum_retry_attempts = 0
}

# Let S3 start the worker, but only for this bucket in this account.
resource "aws_lambda_permission" "s3_invokes_worker" {
  statement_id   = "AllowS3Uploads"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.app.function_name
  principal      = "s3.amazonaws.com"
  source_arn     = aws_s3_bucket.uploads.arn
  source_account = local.account_id
}

# Lambda created this log group on the first run, with no expiry. The import block
# adopts it (instead of failing on "already exists") so logs are kept two weeks.
import {
  to = aws_cloudwatch_log_group.app
  id = "/aws/lambda/decision-support"
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/lambda/decision-support"
  retention_in_days = 14
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

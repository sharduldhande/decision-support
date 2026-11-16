# Uploaded reports and job state. Both hold patient data briefly and expire after a day.

resource "aws_s3_bucket" "uploads" {
  bucket        = "decision-support-uploads-${local.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "uploads" {
  bucket                  = aws_s3_bucket.uploads.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# The worker deletes each PDF as soon as it reads it; this catches anything left behind.
resource "aws_s3_bucket_lifecycle_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  rule {
    id     = "expire-uploads"
    status = "Enabled"
    filter {
      prefix = "uploads/"
    }
    expiration {
      days = 1
    }
  }
}

# Browsers upload straight to S3 with a presigned POST, so the bucket needs CORS.
resource "aws_s3_bucket_cors_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  cors_rule {
    allowed_methods = ["POST"]
    allowed_origins = local.site_origins
    allowed_headers = ["*"]
    max_age_seconds = 3600
  }
}

# A new PDF under uploads/ starts the worker. This is the only way it runs.
resource "aws_s3_bucket_notification" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  lambda_function {
    lambda_function_arn = aws_lambda_function.app.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "uploads/"
    filter_suffix       = ".pdf"
  }
  depends_on = [aws_lambda_permission.s3_invokes_worker]
}

# One item per job: {id, status, result | error, created_at, updated_at, expires_at}.
resource "aws_dynamodb_table" "jobs" {
  name         = "decision-support-jobs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

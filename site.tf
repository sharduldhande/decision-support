# The static page at https://aws.sharduldhande.com: files in a private S3 bucket,
# served by CloudFront with an ACM certificate. DNS lives at GoDaddy, so the two
# records (certificate validation and the CNAME to CloudFront) are added there by hand.

locals {
  site_domain = "aws.sharduldhande.com"
  content_types = {
    html = "text/html; charset=utf-8"
    js   = "text/javascript; charset=utf-8"
    css  = "text/css; charset=utf-8"
    pdf  = "application/pdf"
    svg  = "image/svg+xml"
    png  = "image/png"
  }
  # Every file under site/, skipping dotfiles like .DS_Store.
  site_files = toset([for f in fileset("${path.module}/site", "**") : f if !startswith(basename(f), ".")])
}

resource "aws_s3_bucket" "site" {
  bucket        = "decision-support-site-${local.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket                  = aws_s3_bucket.site.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# `terraform apply` uploads the site. A changed file gets a new etag and is re-uploaded.
# HTML and JS are revalidated on every visit so a deploy shows up at once; the
# sample PDF rarely changes and is cached for a day.
resource "aws_s3_object" "site" {
  for_each      = local.site_files
  bucket        = aws_s3_bucket.site.id
  key           = each.value
  source        = "${path.module}/site/${each.value}"
  etag          = filemd5("${path.module}/site/${each.value}")
  content_type  = lookup(local.content_types, reverse(split(".", each.value))[0], "application/octet-stream")
  cache_control = endswith(each.value, ".pdf") ? "public, max-age=86400" : "no-cache"
}

# ---------- certificate ----------

resource "aws_acm_certificate" "site" {
  provider          = aws.us_east_1
  domain_name       = local.site_domain
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

# Waits (up to 75 minutes) until ACM sees the validation record at GoDaddy.
resource "aws_acm_certificate_validation" "site" {
  provider        = aws.us_east_1
  certificate_arn = aws_acm_certificate.site.arn
}

# ---------- CloudFront ----------

# CloudFront signs its requests to S3, and the bucket policy accepts only this distribution.
resource "aws_cloudfront_origin_access_control" "site" {
  name                              = "decision-support-site"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

data "aws_cloudfront_cache_policy" "optimized" {
  name = "Managed-CachingOptimized"
}

resource "aws_cloudfront_response_headers_policy" "site" {
  name = "decision-support-site"

  # A demo, not something for search engines.
  custom_headers_config {
    items {
      header   = "X-Robots-Tag"
      value    = "noindex, nofollow"
      override = true
    }
  }

  security_headers_config {
    strict_transport_security {
      access_control_max_age_sec = 31536000
      include_subdomains         = false
      override                   = true
    }
    content_type_options {
      override = true
    }
    frame_options {
      frame_option = "DENY"
      override     = true
    }
    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }
  }
}

resource "aws_cloudfront_distribution" "site" {
  enabled             = true
  comment             = "decision-support site"
  aliases             = [local.site_domain]
  default_root_object = "index.html"
  price_class         = "PriceClass_100" # North America and Europe edges only: cheapest
  http_version        = "http2and3"
  is_ipv6_enabled     = true

  origin {
    origin_id                = "site"
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  default_cache_behavior {
    target_origin_id           = "site"
    viewer_protocol_policy     = "redirect-to-https"
    allowed_methods            = ["GET", "HEAD"]
    cached_methods             = ["GET", "HEAD"]
    compress                   = true
    cache_policy_id            = data.aws_cloudfront_cache_policy.optimized.id
    response_headers_policy_id = aws_cloudfront_response_headers_policy.site.id
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.site.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
}

resource "aws_s3_bucket_policy" "site" {
  bucket = aws_s3_bucket.site.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "cloudfront.amazonaws.com" }
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.site.arn}/*"
      Condition = { StringEquals = { "AWS:SourceArn" = aws_cloudfront_distribution.site.arn } }
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.site]
}

output "site_url" {
  value = "https://${local.site_domain}"
}

# GoDaddy: CNAME "aws" -> this name.
output "cloudfront_domain" {
  value = aws_cloudfront_distribution.site.domain_name
}

# GoDaddy: the certificate validation CNAME (also shown by ACM while apply waits).
output "certificate_validation_record" {
  value = {
    for o in aws_acm_certificate.site.domain_validation_options : o.domain_name => {
      name  = o.resource_record_name
      type  = o.resource_record_type
      value = o.resource_record_value
    }
  }
}

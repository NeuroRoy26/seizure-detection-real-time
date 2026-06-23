# S3 Bucket for Raw and Preprocessed EEG Datasets
resource "aws_s3_bucket" "data_bucket" {
  bucket        = "${var.project_name}-data-${var.environment}"
  force_destroy = false

  tags = {
    Name        = "${var.project_name}-data-${var.environment}"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_bucket_encryption" {
  bucket = aws_s3_bucket.data_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "data_bucket_public_block" {
  bucket = aws_s3_bucket.data_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


# S3 Bucket for SageMaker Model Training Artifacts
resource "aws_s3_bucket" "model_bucket" {
  bucket        = "${var.project_name}-models-${var.environment}"
  force_destroy = true

  tags = {
    Name        = "${var.project_name}-models-${var.environment}"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_bucket_encryption" {
  bucket = aws_s3_bucket.model_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "model_bucket_public_block" {
  bucket = aws_s3_bucket.model_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

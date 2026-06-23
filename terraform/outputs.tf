output "data_bucket_name" {
  value       = aws_s3_bucket.data_bucket.id
  description = "The name of the S3 bucket used for datasets."
}

output "data_bucket_arn" {
  value       = aws_s3_bucket.data_bucket.arn
  description = "The ARN of the S3 bucket used for datasets."
}

output "model_bucket_name" {
  value       = aws_s3_bucket.model_bucket.id
  description = "The name of the S3 bucket used for model artifacts."
}

output "model_bucket_arn" {
  value       = aws_s3_bucket.model_bucket.arn
  description = "The ARN of the S3 bucket used for model artifacts."
}

output "sagemaker_execution_role_arn" {
  value       = aws_iam_role.sagemaker_execution_role.arn
  description = "The ARN of the IAM role for SageMaker execution."
}

output "developer_ci_policy_arn" {
  value       = aws_iam_policy.developer_ci_policy.arn
  description = "The ARN of the developer and CI/CD access policy."
}

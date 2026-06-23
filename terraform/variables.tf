variable "aws_region" {
  type        = string
  description = "The AWS region where resources will be created."
  default     = "eu-central-1"
}

variable "environment" {
  type        = string
  description = "The deployment environment (e.g., dev, staging, prod)."
  default     = "dev"
}

variable "project_name" {
  type        = string
  description = "The name of the project."
  default     = "seizure-detection"
}

variable "skip_credentials_validation" {
  type        = bool
  description = "Skip credentials validation (set to true for offline tests)"
  default     = false
}

variable "skip_requesting_account_id" {
  type        = bool
  description = "Skip requesting AWS account ID (set to true for offline tests)"
  default     = false
}

variable "skip_metadata_api_check" {
  type        = bool
  description = "Skip AWS Metadata API checks (set to true for offline tests)"
  default     = false
}

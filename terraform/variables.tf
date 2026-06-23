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

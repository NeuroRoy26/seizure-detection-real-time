terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "neuroroy-tfstate-bucket"
    key            = "seizure-detection/terraform.tfstate"
    region         = "eu-central-1"
    encrypt        = true
    dynamodb_table = "neuroroy-tfstate-lock"
  }
}

provider "aws" {
  region                      = var.aws_region
  skip_credentials_validation = var.skip_credentials_validation
  skip_requesting_account_id  = var.skip_requesting_account_id
  skip_metadata_api_check     = var.skip_metadata_api_check
}

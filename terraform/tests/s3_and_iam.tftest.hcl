# Unit tests for the Seizure Detection Terraform module

run "verify_s3_buckets_configuration" {
  command = plan

  # Check data bucket attributes
  assert {
    condition     = aws_s3_bucket.data_bucket.bucket == "seizure-detection-data-dev"
    error_message = "Data bucket name did not match expected 'seizure-detection-data-dev'"
  }

  assert {
    condition     = aws_s3_bucket.data_bucket.force_destroy == false
    error_message = "Data bucket force_destroy should be false to prevent data loss"
  }

  # Check model bucket attributes
  assert {
    condition     = aws_s3_bucket.model_bucket.bucket == "seizure-detection-models-dev"
    error_message = "Model bucket name did not match expected 'seizure-detection-models-dev'"
  }

  assert {
    condition     = aws_s3_bucket.model_bucket.force_destroy == true
    error_message = "Model bucket force_destroy should be true for easy teardown of transient outputs"
  }
}

run "verify_iam_roles_and_policies" {
  command = plan

  # Check SageMaker role assume relationship
  assert {
    condition     = jsondecode(aws_iam_role.sagemaker_execution_role.assume_role_policy).Statement[0].Principal.Service == "sagemaker.amazonaws.com"
    error_message = "SageMaker execution role must allow assume_role by 'sagemaker.amazonaws.com'"
  }

  # Check policy exists
  assert {
    condition     = aws_iam_policy.sagemaker_access_policy.name == "seizure-detection-sagemaker-access-dev"
    error_message = "SageMaker access policy name is incorrect"
  }

  # Check developer policy name
  assert {
    condition     = aws_iam_policy.developer_ci_policy.name == "seizure-detection-developer-ci-dev"
    error_message = "Developer CI policy name is incorrect"
  }
}

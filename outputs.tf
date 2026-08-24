output "s3_bucket_name" {
  description = "Name of the Jenkins IaC demonstration bucket."
  value       = aws_s3_bucket.iac_demo.bucket
}

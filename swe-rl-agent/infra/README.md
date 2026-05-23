# Infra

Terraform module for deploying swe-rl-agent on AWS.

```
terraform init
terraform plan -var db_password=<secret>
terraform apply -var db_password=<secret>
```

Components:
- VPC + 2 AZ (public/private subnets, NAT)
- RDS Postgres for trajectory + checkpoint metadata
- ElastiCache Redis for the rollout queue
- S3 bucket for trajectory + report artifacts
- ECR for `sandbox` and `learner` images
- ECS cluster with privileged tasks for Docker-in-Docker rollouts
- Secrets Manager for HF_TOKEN and WANDB_API_KEY

The learner runs on a separate GPU EC2 (e.g. `p5.48xlarge` for 14B). Bring it
up with `aws ec2 run-instances` referencing the learner ECR image.

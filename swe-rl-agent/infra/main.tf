terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
  backend "s3" {
    bucket = "swe-rl-tfstate"
    key    = "swe-rl-agent/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  default = "us-east-1"
}

variable "project" {
  default = "swe-rl-agent"
}

variable "vpc_cidr" {
  default = "10.20.0.0/16"
}

# --- VPC ---------------------------------------------------------------------

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.13"

  name = "${var.project}-vpc"
  cidr = var.vpc_cidr
  azs  = ["${var.region}a", "${var.region}b"]

  private_subnets = ["10.20.1.0/24", "10.20.2.0/24"]
  public_subnets  = ["10.20.101.0/24", "10.20.102.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true
}

# --- Storage -----------------------------------------------------------------

resource "aws_s3_bucket" "trajectories" {
  bucket = "${var.project}-trajectories-${data.aws_caller_identity.me.account_id}"
}

resource "aws_s3_bucket_versioning" "trajectories" {
  bucket = aws_s3_bucket.trajectories.id
  versioning_configuration { status = "Enabled" }
}

# --- Postgres ----------------------------------------------------------------

resource "aws_db_subnet_group" "default" {
  name       = "${var.project}-db"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "db" {
  name   = "${var.project}-db"
  vpc_id = module.vpc.vpc_id
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

resource "aws_db_instance" "postgres" {
  identifier             = "${var.project}-postgres"
  engine                 = "postgres"
  engine_version         = "16.3"
  instance_class         = "db.t4g.medium"
  allocated_storage      = 50
  storage_type           = "gp3"
  username               = "swe_rl"
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.default.name
  vpc_security_group_ids = [aws_security_group.db.id]
  skip_final_snapshot    = true
  publicly_accessible    = false
}

variable "db_password" {
  type      = string
  sensitive = true
}

# --- Redis (queue) -----------------------------------------------------------

resource "aws_elasticache_subnet_group" "default" {
  name       = "${var.project}-redis"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name   = "${var.project}-redis"
  vpc_id = module.vpc.vpc_id
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

resource "aws_elasticache_replication_group" "default" {
  replication_group_id       = "${var.project}-redis"
  description                = "swe-rl queue"
  engine                     = "redis"
  node_type                  = "cache.t4g.micro"
  num_cache_clusters         = 1
  parameter_group_name       = "default.redis7"
  subnet_group_name          = aws_elasticache_subnet_group.default.name
  security_group_ids         = [aws_security_group.redis.id]
  automatic_failover_enabled = false
  port                       = 6379
}

# --- ECS rollout cluster -----------------------------------------------------

resource "aws_ecs_cluster" "rollout" {
  name = "${var.project}-rollout"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecr_repository" "sandbox" {
  name = "${var.project}/sandbox"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_repository" "learner" {
  name = "${var.project}/learner"
  image_scanning_configuration { scan_on_push = true }
}

# --- IAM ---------------------------------------------------------------------

resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.project}-ecs-task-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- Secrets -----------------------------------------------------------------

resource "aws_secretsmanager_secret" "hf_token" {
  name = "${var.project}/hf-token"
}

resource "aws_secretsmanager_secret" "wandb_api_key" {
  name = "${var.project}/wandb-api-key"
}

# --- Outputs -----------------------------------------------------------------

data "aws_caller_identity" "me" {}

output "vpc_id" {
  value = module.vpc.vpc_id
}
output "ecs_cluster_arn" {
  value = aws_ecs_cluster.rollout.arn
}
output "postgres_endpoint" {
  value     = aws_db_instance.postgres.address
  sensitive = true
}
output "redis_endpoint" {
  value = aws_elasticache_replication_group.default.primary_endpoint_address
}
output "ecr_sandbox_url" {
  value = aws_ecr_repository.sandbox.repository_url
}
output "ecr_learner_url" {
  value = aws_ecr_repository.learner.repository_url
}
output "trajectories_bucket" {
  value = aws_s3_bucket.trajectories.id
}

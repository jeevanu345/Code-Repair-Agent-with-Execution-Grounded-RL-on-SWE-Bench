# ECS task definition for rollout workers (privileged for Docker-in-Docker).
# Runs in private subnets with NAT for HF/HuggingFace fetches; sandboxes inside
# the task have their network disabled post-install.

resource "aws_cloudwatch_log_group" "rollout" {
  name              = "/swe-rl/rollout"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "rollout_worker" {
  family                   = "${var.project}-rollout-worker"
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc"
  cpu                      = "4096"
  memory                   = "16384"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([{
    name      = "rollout"
    image     = "${aws_ecr_repository.learner.repository_url}:latest"
    essential = true
    privileged = true
    environment = [
      { name = "DATABASE_URL", value = "postgresql+psycopg://swe_rl:${var.db_password}@${aws_db_instance.postgres.address}:5432/swe_rl" },
      { name = "REDIS_URL", value = "redis://${aws_elasticache_replication_group.default.primary_endpoint_address}:6379/0" },
      { name = "SANDBOX_IMAGE", value = "${aws_ecr_repository.sandbox.repository_url}:latest" }
    ]
    secrets = [
      { name = "HF_TOKEN", valueFrom = aws_secretsmanager_secret.hf_token.arn },
      { name = "WANDB_API_KEY", valueFrom = aws_secretsmanager_secret.wandb_api_key.arn }
    ]
    mountPoints = [{
      sourceVolume  = "docker-sock"
      containerPath = "/var/run/docker.sock"
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.rollout.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "rollout"
      }
    }
    command = ["python", "-m", "swe_rl.cli", "rollout", "loop"]
  }])

  volume {
    name      = "docker-sock"
    host_path = "/var/run/docker.sock"
  }
}

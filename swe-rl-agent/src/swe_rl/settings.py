"""Process-wide settings loaded from environment.

Single source of truth for env-derived configuration. Never read os.environ
elsewhere — import `settings` and use it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Models / HF
    hf_token: str | None = None
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    model_revision: str = "main"
    tokenizer_name: str | None = None

    # vLLM
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_api_key: str = "local-dev"

    # Sandbox
    sandbox_image: str = "swe-rl/sandbox:latest"
    sandbox_mem_gb: int = 4
    sandbox_cpus: float = 2.0
    sandbox_wallclock_s: int = 600
    sandbox_disable_network_after_install: bool = True

    # Postgres
    database_url: str = "postgresql+psycopg://swe_rl:swe_rl@127.0.0.1:5432/swe_rl"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "swe_rl"
    postgres_user: str = "swe_rl"
    postgres_password: str = "swe_rl"

    # Redis / MinIO
    redis_url: str = "redis://127.0.0.1:6379/0"
    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio12345"
    minio_bucket: str = "swe-rl"

    # Observability
    sentry_dsn: str | None = None
    wandb_api_key: str | None = None
    wandb_project: str = "swe-rl-agent"
    wandb_entity: str | None = None
    prometheus_port: int = 9100
    log_dir: Path = Path("./logs")
    log_level: str = "INFO"

    # Cost
    max_dollars_per_run: float = 50.0
    dollar_per_1k_tokens_in: float = 0.0
    dollar_per_1k_tokens_out: float = 0.0

    # Train
    train_output_dir: Path = Path("./outputs")
    train_batch_size: int = 64
    grpo_group_size: int = 8
    grpo_kl_coef: float = 0.04
    grpo_lr: float = 1e-6

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.vllm_host}:{self.vllm_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

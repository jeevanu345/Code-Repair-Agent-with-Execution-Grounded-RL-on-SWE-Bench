"""SQLAlchemy schema + engine for persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from swe_rl.settings import settings


class Base(DeclarativeBase):
    pass


class Instance(Base):
    __tablename__ = "instances"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    repo: Mapped[str] = mapped_column(String, index=True)
    base_commit: Mapped[str] = mapped_column(String)
    problem_statement: Mapped[str] = mapped_column(Text)
    fail_to_pass: Mapped[list[str]] = mapped_column(JSON)
    pass_to_pass: Mapped[list[str]] = mapped_column(JSON)
    test_patch: Mapped[str | None] = mapped_column(Text, nullable=True)
    environment_setup_commit: Mapped[str | None] = mapped_column(String, nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"
    sha256: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    parent_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    path: Mapped[str] = mapped_column(String)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    eval_runs: Mapped[list["EvalRun"]] = relationship(back_populates="checkpoint")


class Trajectory(Base):
    __tablename__ = "trajectories"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    instance_id: Mapped[str] = mapped_column(ForeignKey("instances.id"), index=True)
    checkpoint_sha: Mapped[str | None] = mapped_column(
        ForeignKey("model_checkpoints.sha256"), nullable=True, index=True
    )
    seed: Mapped[int] = mapped_column(Integer)
    temperature: Mapped[float] = mapped_column(Float)
    top_p: Mapped[float] = mapped_column(Float)
    sandbox_image_digest: Mapped[str] = mapped_column(String)
    n_steps: Mapped[int] = mapped_column(Integer)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    final_patch: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_path: Mapped[str] = mapped_column(String)  # JSONL on disk / MinIO
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    rewards: Mapped[list["Reward"]] = relationship(back_populates="trajectory")


class Reward(Base):
    __tablename__ = "rewards"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trajectory_id: Mapped[str] = mapped_column(ForeignKey("trajectories.id"), index=True)
    kind: Mapped[str] = mapped_column(String)  # binary | shaped | prm
    value: Mapped[float] = mapped_column(Float)
    details: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    trajectory: Mapped[Trajectory] = relationship(back_populates="rewards")


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    checkpoint_sha: Mapped[str | None] = mapped_column(
        ForeignKey("model_checkpoints.sha256"), nullable=True, index=True
    )
    dataset: Mapped[str] = mapped_column(String)
    split: Mapped[str] = mapped_column(String)
    n_instances: Mapped[int] = mapped_column(Integer)
    resolved_at_1: Mapped[float] = mapped_column(Float)
    resolved_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    report_path: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    checkpoint: Mapped[ModelCheckpoint | None] = relationship(back_populates="eval_runs")


def make_engine(url: str | None = None) -> Any:
    return create_engine(url or settings.database_url, future=True, pool_pre_ping=True)


def init_db(url: str | None = None) -> None:
    """Create tables. For real deployments use alembic migrations under migrations/."""
    engine = make_engine(url)
    Base.metadata.create_all(engine)


def get_session(url: str | None = None) -> Session:
    return Session(make_engine(url))

-- Authoritative DDL for the swe_rl Postgres schema.
-- For dev: `make migrate` runs SQLAlchemy create_all from src/swe_rl/db.py.
-- For prod: apply this file or wire alembic on top.

CREATE TABLE IF NOT EXISTS instances (
    id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    base_commit TEXT NOT NULL,
    problem_statement TEXT NOT NULL,
    fail_to_pass JSONB NOT NULL,
    pass_to_pass JSONB NOT NULL,
    test_patch TEXT,
    environment_setup_commit TEXT,
    extra JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_instances_repo ON instances(repo);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    sha256 TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_sha TEXT,
    path TEXT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY,
    instance_id TEXT NOT NULL REFERENCES instances(id),
    checkpoint_sha TEXT REFERENCES model_checkpoints(sha256),
    seed INTEGER NOT NULL,
    temperature DOUBLE PRECISION NOT NULL,
    top_p DOUBLE PRECISION NOT NULL,
    sandbox_image_digest TEXT NOT NULL,
    n_steps INTEGER NOT NULL,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    final_patch TEXT,
    storage_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_traj_instance ON trajectories(instance_id);
CREATE INDEX IF NOT EXISTS idx_traj_ckpt ON trajectories(checkpoint_sha);

CREATE TABLE IF NOT EXISTS rewards (
    id SERIAL PRIMARY KEY,
    trajectory_id TEXT NOT NULL REFERENCES trajectories(id),
    kind TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    details JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rewards_traj ON rewards(trajectory_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    checkpoint_sha TEXT REFERENCES model_checkpoints(sha256),
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    n_instances INTEGER NOT NULL,
    resolved_at_1 DOUBLE PRECISION NOT NULL,
    resolved_at_k DOUBLE PRECISION,
    report_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

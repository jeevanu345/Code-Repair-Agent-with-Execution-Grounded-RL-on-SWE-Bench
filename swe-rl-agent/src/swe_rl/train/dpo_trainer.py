"""Offline DPO over preference pairs derived from rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_rl.observability.logging import get_logger
from swe_rl.settings import settings

_log = get_logger(__name__)


@dataclass
class DPOConfig:
    model_name: str
    output_dir: Path
    beta: float = 0.1
    learning_rate: float = 5e-7
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    max_length: int = 4096
    max_prompt_length: int = 2048
    bf16: bool = True
    seed: int = 17


def run_dpo(dataset: Any, config: DPOConfig) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    tok = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="bfloat16" if config.bf16 else "auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="bfloat16" if config.bf16 else "auto",
    )

    args = TRLDPOConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        beta=config.beta,
        bf16=config.bf16,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        seed=config.seed,
        save_strategy="epoch",
        logging_steps=10,
        report_to=["wandb"] if settings.wandb_api_key else "none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=dataset,
        tokenizer=tok,
    )
    trainer.train()
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    _log.info("dpo.saved", path=str(out))
    return out

"""SFT warm-start using TRL's SFTTrainer over HumanEvalPack + curated trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_rl.observability.logging import get_logger

_log = get_logger(__name__)


@dataclass
class SFTConfig:
    model_name: str
    output_dir: Path
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 4096
    bf16: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_peft: bool = True
    seed: int = 17
    extra: dict[str, Any] = field(default_factory=dict)


def run_sft(dataset: Any, config: SFTConfig) -> Path:
    """Train a model with SFT and save to config.output_dir.

    Lazy imports keep CPU-only smoke tests viable.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    tok = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    peft_config = None
    if config.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="bfloat16" if config.bf16 else "auto",
    )

    args = TRLSFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        seed=config.seed,
        max_length=config.max_seq_length,
        save_strategy="epoch",
        logging_steps=10,
        report_to=["wandb"],
        dataset_text_field=None,  # we'll format manually
    )

    def _formatter(example: dict[str, Any]) -> str:
        return f"{example['prompt']}\n\n<assistant>\n{example['completion']}"

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tok,
        peft_config=peft_config,
        formatting_func=_formatter,
    )
    trainer.train()
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    _log.info("sft.saved", path=str(out))
    return out

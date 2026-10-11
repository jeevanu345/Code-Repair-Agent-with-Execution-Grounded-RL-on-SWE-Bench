"""Optional training callbacks emitting only metrics reported by the trainer."""

import time
from collections.abc import Callable
from typing import Any


def make_callbacks(
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    control_callback: Callable[[], str | None] | None = None,
) -> list[Any]:
    if not event_callback and not control_callback:
        return []
    from transformers import TrainerCallback

    class TelemetryCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if event_callback:
                event_callback(
                    {
                        "type": "state",
                        "status": "running",
                        "step": state.global_step,
                        "max_steps": state.max_steps,
                    }
                )

        def on_log(self, args, state, control, logs=None, **kwargs):
            if event_callback:
                event_callback(
                    {
                        "type": "training_metrics",
                        "step": state.global_step,
                        "epoch": state.epoch,
                        "metrics": dict(logs or {}),
                    }
                )

        def on_save(self, args, state, control, **kwargs):
            if event_callback:
                event_callback(
                    {
                        "type": "checkpoint",
                        "step": state.global_step,
                        "path": f"{args.output_dir}/checkpoint-{state.global_step}",
                    }
                )

        def on_step_begin(self, args, state, control, **kwargs):
            if control_callback:
                action = control_callback()
                while action == "pause":
                    time.sleep(0.1)
                    action = control_callback()
                if action == "cancel":
                    control.should_training_stop = True
                    if event_callback:
                        event_callback({"type": "cancelled", "step": state.global_step})
            return control

    return [TelemetryCallback()]

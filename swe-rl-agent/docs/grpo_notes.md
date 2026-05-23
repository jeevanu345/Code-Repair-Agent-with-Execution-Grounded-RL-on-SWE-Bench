# GRPO notes

## Algorithm

For each prompt (one SWE-bench instance), sample K trajectories with different
seeds. Compute reward `r_i` per rollout. Advantage:

```
a_i = (r_i - mean_k(r)) / (std_k(r) + ε)
```

Policy update minimizes a clipped surrogate (PPO-style) on token log-probs of
the assistant turns, anchored to a reference policy via KL.

## Hyperparameters

| Param | Default | Why |
|---|---|---|
| group size K | 8 | Enough variance signal without exploding rollout cost |
| KL coef β | 0.04 | Standard; raise to 0.1+ if reward gaming appears |
| clip ratio | 0.2 | PPO default |
| lr | 1e-6 → 1e-7 | Conservative for 14B + LoRA; ramp warmup over 50 steps |
| global batch | 64 trajectories | Tradeoff: stability vs. per-step rollout cost |
| weight sync | every 50 steps | Tighter sync = fresher rollouts but more vLLM downtime |

## Failure modes

- **Reward variance collapse** in a group → advantage zero → no signal.
  Mitigation: curriculum (start with easier instances), shaped reward floor,
  occasional teacher-forced sample to seed gradient.
- **KL blowup** when reward jumps non-monotonically → clip + KL coef saves
  you. Watch the `kl_to_ref` metric in W&B.
- **Length collapse** (model outputs always-empty patches) → penalized via
  `no_patch_penalty` and via `tokens_out_total` regularization.

## Combining with DPO

Run a single DPO epoch on offline preference pairs (best vs worst rollout per
instance) before GRPO. Cheaper than the full RL loop, and warmstarts the
policy with directional preference info.

## What to log

- per-step: reward mean/std/min/max per group, KL to ref, clip fraction,
  optimizer step count, learning rate
- per-rollout: tokens_in/out, $ cost, n_steps, n_test_runs, resolved bool
- per-eval: resolved@1 on a held-out 50-instance dev split

# Reward design

## Primary: binary execution-grounded

```
resolved ⇔ all FAIL_TO_PASS pass  AND  all PASS_TO_PASS still pass
reward    = 1.0 if resolved else 0.0
```

This is the true objective. SWE-bench leaderboard uses the same definition.

## Auxiliary: dense shaped

```
shaped = 0.5 * f2p_pass_rate
       + 0.5 * p2p_pass_rate
       - λ * (lines_added + lines_removed)
       - μ * lint_errors
       + (-0.1 if patch is empty else 0)
       + (-0.05 if test runner timed out else 0)
```

Provides gradient when no rollout in a group fully resolves. Weight decays from
0.2 → 0 over `shaped_decay_steps` (default 2000).

## Why both?

- Binary alone collapses a GRPO group when all rollouts fail (advantage = 0).
- Shaped alone drifts toward minimal-edit local optima that flip a partial
  test set without solving the bug.

## Trainer-side combination

```
total_reward = binary + α(t) * shaped
```

with `α(t) = 0.2 * max(0, 1 - t / shaped_decay_steps)`.

## Reward integrity

- Pure function of `(test_outcomes, patch, instance)` — no network, no LLM.
- Reproducible from the saved trajectory.
- Sanity tests:
  - Gold patch → reward 1.0 (covered by smoke + e2e).
  - Empty patch → reward 0.0 + no-patch penalty.
  - Mixed outcomes → reward 0.0; partial f2p_rate reflects progress.

## PRM (optional, off by default)

A frozen judge that scores intermediate (state, action) pairs. Blends with
terminal reward at weight `prm.weight`. Off because (a) it adds latency,
(b) miscalibrated PRMs can dominate the binary signal. Enable only after
the binary objective has plateaued.

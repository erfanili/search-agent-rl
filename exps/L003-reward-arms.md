# L003: Reward arms em_hit, em_multi, f1 (60 steps)

- **Question:** With the L001 recipe and only the reward changed (em_hit, em_multi, f1), what validation EM and search counts do the runs reach at step 60?
- **Parent:** L001
- **Status:** done (2026-10-30/31)
- **Source:** reconstructed from `$DATA_ROOT/logs` (run dirs, `reward_arms_*.log`) and `scripts/compare_reward_arms.py`.

## Setup

Same recipe as L001 (64 × 5, mini 32, lr 1e-6, 10% warmup, KL 0.001, 2,048-token episodes, 60 steps). Only `REWARD` changes (`agent/reward.py`, `hit_bonus=0.2`, `multi_bonus=0.2`).

| arm | reward | run (`$DATA_ROOT/logs/`) |
|---|---|---|
| `em` | EM (the L001 run, not rerun) | `grpo_qwen2.5-3b_full_20261019_183501` |
| `em_hit` | EM + 0.2 × [a gold answer appears in a retrieved passage, whole-word] | `grpo_qwen2.5-3b_full_em_hit_20261030_022132` |
| `em_multi` | EM + 0.2 × [EM = 1 and ≥ 2 distinct queries] | `grpo_qwen2.5-3b_full_em_multi_20261031_061227` |
| `f1` | token F1 | `grpo_qwen2.5-3b_full_f1_20261031_102801` |

- **Hardware:** the three new arms ran on 4 GPUs (0, 1, 6, 7; retrieval encoder on CPU) at the same batch size. L001 used 5 GPUs.
- **Eval:** verl validation on the sub300 sets (Bamboogle = 125), greedy, steps 0/20/40/60.
- `n_distinct_search`, `evidence_hit` and `correct_multi` were added after L001, so the `em` arm has none of them.

## Commands

```bash
nohup bash scripts/run_reward_arms.sh > $DATA_ROOT/logs/reward_arms_<stamp>.log 2>&1 &     # ARMS="em_hit em_multi f1"
.venv/bin/python scripts/compare_reward_arms.py                                           # tables below
```

## Execution

- **Commit:** 2d88b78 in each run's `config.txt`. The reward arms were committed later (ccd9ff6, 2026-11-09), so the runs used uncommitted changes. Dirty state not recorded.
- **Queue logs:** `$DATA_ROOT/logs/reward_arms_20261030_022132.log`, `reward_arms_20261031_061227.log`.
- **Wall time (incl. validation):** em_hit 4 h 20 min (2026-10-30 02:21–06:41); em_multi 4 h 15 min (2026-10-31 06:12–10:27); f1 4 h 05 min (10:28–14:33).
- **Failure:** em_multi's first attempt (2026-10-30 06:41) failed at launch because leftover Ray workers held GPU memory. It was rerun on 2026-10-31.
- **Checkpoints:** none kept. Each `ckpt/` holds only `latest_checkpointed_iteration.txt`.

## Result

### Validation EM (%), steps 0 / 20 / 40 / 60

| arm | HotpotQA | 2Wiki | MuSiQue | Bamboogle | avg 3 multi-hop @60 |
|---|---|---|---|---|---|
| em | 8.7 / 23.7 / 24.0 / 23.7 | 7.7 / 16.7 / 24.7 / 29.0 | 2.3 / 5.0 / 5.7 / 7.0 | 7.2 / 14.4 / 6.4 / 4.0 | 19.9 |
| em_hit | 6.7 / 17.7 / 23.7 / 24.3 | 8.0 / 20.0 / 23.7 / 25.0 | 3.3 / 4.0 / 4.7 / 6.0 | 9.6 / 8.8 / 11.2 / 9.6 | 18.4 |
| em_multi | 8.0 / 18.0 / 23.7 / 26.3 | 7.7 / 18.0 / 25.7 / 25.0 | 3.0 / 2.7 / 3.7 / 5.3 | 12.0 / 8.0 / 7.2 / 8.0 | 18.9 |
| f1 | 8.3 / 18.0 / 25.3 / 25.0 | 7.7 / 21.7 / 25.3 / 24.7 | 3.0 / 4.0 / 5.7 / 6.7 | 11.2 / 9.6 / 9.6 / 8.8 | 18.8 |

### Validation F1, step 60 (multi-hop avg)

em 0.270, em_hit 0.251, em_multi 0.262, f1 0.264. Answered rate ≥ 0.997 on every set and arm at step 60.

### Search behaviour, step 60

| arm | searches/q: HotpotQA | 2Wiki | MuSiQue | Bamboogle | distinct queries/q, multi-hop avg | evidence hit, multi-hop avg | correct with ≥ 2 queries: HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|---|---|---|---|---|---|
| em | 1.01 | 1.01 | 1.01 | 1.00 | — | — | — | — | — |
| em_hit | 1.00 | 1.01 | 1.00 | 1.00 | 1.00 | 0.28 | 0% | 0% | 0% |
| em_multi | 1.09 | 1.31 | 1.03 | 1.00 | 1.14 | 0.28 | 3.0% | 12.3% | 0% |
| f1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.28 | 0% | 0% | 0% |

em_multi on 2Wiki, steps 0/20/40/60: searches/q 0.75 → 1.10 → 1.30 → 1.31; correct with ≥ 2 queries 0.7% → 3.7% → 11.7% → 12.3%.

Entropy, step 1 → 60: em_hit 1.88 → 0.27, em_multi 2.22 → 0.24, f1 1.83 → 0.41 (em, L001: 1.8 → 0.25).

### Summary

One seed per arm, n = 300 per set (SE ≈ 2.6 EM), Bamboogle n = 125 (SE ≈ 2.5). Step-60 multi-hop average EM: em 19.9, em_hit 18.4, em_multi 18.9, f1 18.8. Step-0 EM on identical weights differs across arms by up to 2.0 EM on the multi-hop sets and 4.8 on Bamboogle. Searches/q at step 60 is 1.00–1.01 for em, em_hit and f1. For em_multi it is 1.09 (HotpotQA), 1.31 (2Wiki), 1.03 (MuSiQue) and 1.00 (Bamboogle).

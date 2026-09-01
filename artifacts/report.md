# Agent Run Report

- Total iterations: 26 (scored: 24)
- Total LLM tokens: 275,742 in + 306,809 out = 582,551
- Total cost: $6.11
- GPU-hours: 0 (numpy-only pipeline, CPU only, no GPU used at any stage)
- Manual interventions: 0


## Results

| | valid GAUC | valid nDCG@5 | valid primary | test GAUC | test nDCG@5 | test primary | Δ primary vs baseline |
|---|---|---|---|---|---|---|---|
| random | - | - | 0.4834 | - | - | 0.4753 | -0.1193 |
| item popularity | - | - | 0.5807 | - | - | 0.5715 | -0.0231 |
| official FM baseline | 0.6674 | 0.5357 | 0.6016 | 0.6610 | 0.5282 | 0.5946 | 0.0000 |
| **our converged best (n0011)** | 0.6682 | 0.5359 | 0.6020 | 0.6630 | 0.5294 | 0.5962 | +0.0016 |

% of oracle headroom captured: 0.6% (oracle ceiling: test primary 0.8645)

## Score trajectory

| scored # | node | direction | valid primary | vs baseline |
|---|---|---|---|---|
| 1 | n0001 | loss | 0.5920 | -0.0096 |
| 2 | n0002 | multitask | 0.5967 | -0.0049 |
| 3 | n0003 | time_shift | 0.5865 | -0.0151 |
| 4 | n0004 | loss | 0.5992 | -0.0024 |
| 5 | n0005 | arch | 0.6014 | -0.0002 |
| 6 | n0006 | arch | 0.6007 | -0.0009 |
| 7 | n0007 | ensemble | 0.6015 | -0.0001 |
| 8 | n0008 | time_shift | 0.6017 | +0.0001 |
| 9 | n0009 | sequence | 0.5914 | -0.0102 |
| 10 | n0010 | watchtime | 0.5797 | -0.0219 |
| 11 | n0011 | loss | 0.6020 | +0.0004 |
| 12 | n0012 | calibration | 0.5929 | -0.0087 |
| 13 | n0013 | other | 0.5945 | -0.0071 |
| 14 | n0014 | loss | 0.5958 | -0.0058 |
| 15 | n0015 | loss | 0.5963 | -0.0053 |
| 16 | n0016 | loss | 0.6019 | +0.0003 |
| 17 | n0017 | loss | 0.6022 | +0.0006 |
| 18 | n0018 | loss | 0.6019 | +0.0003 |
| 19 | n0019 | loss | 0.6024 | +0.0008 |
| 20 | n0020 | loss | 0.6019 | +0.0003 |
| 21 | n0021 | loss | 0.6024 | +0.0008 |
| 22 | n0022 | loss | 0.6023 | +0.0007 |
| 23 | n0023 | loss | 0.6025 | +0.0009 |
| 24 | n0024 | loss | 0.6021 | +0.0005 |

## Directions explored

| direction | attempts | best primary | Δ vs baseline |
|---|---|---|---|
| loss | 14 | 0.6025 | +0.0009 |
| time_shift | 2 | 0.6017 | +0.0001 |
| ensemble | 1 | 0.6015 | -0.0001 |
| arch | 2 | 0.6014 | -0.0002 |
| multitask | 1 | 0.5967 | -0.0049 |
| other | 1 | 0.5945 | -0.0071 |
| calibration | 1 | 0.5929 | -0.0087 |
| sequence | 1 | 0.5914 | -0.0102 |
| watchtime | 1 | 0.5797 | -0.0219 |

## Errors and recovery events

| node | event | detail |
|---|---|---|
| n0001 | guard_or_smoke_reject | 'Split' object is not subscriptable |
| n0001 | debug_repair_attempt |  |
| n0002 | guard_or_smoke_reject | 'Split' object is not subscriptable |
| n0002 | debug_repair_attempt |  |
| n0003 | guard_or_smoke_reject | 'Split' object is not subscriptable |
| n0003 | debug_repair_attempt |  |
| n0009 | guard_or_smoke_reject | 'Split' object is not subscriptable |
| n0009 | debug_repair_attempt |  |
| n0010 | guard_or_smoke_reject | 'Split' object is not subscriptable |
| n0010 | debug_repair_attempt |  |
| n0012 | guard_or_smoke_reject | 'Split' object is not subscriptable |
| n0012 | debug_repair_attempt |  |
| n0012 | guard_or_smoke_reject | 'Split' object is not subscriptable |
| n0012 | debug_repair_attempt |  |
| n0013 | guard_or_smoke_reject | 'Split' object is not subscriptable |
| n0013 | debug_repair_attempt |  |
| n0013 | guard_or_smoke_reject | The weights and list don't have the same length. |
| n0013 | debug_repair_attempt |  |
| n0013 | guard_or_smoke_reject | 'Split' object is not subscriptable |
| n0013 | debug_repair_attempt |  |
| n0013 | guard_or_smoke_reject | '(' was never closed (<unknown>, line 88) |
| n0013 | debug_repair_attempt |  |
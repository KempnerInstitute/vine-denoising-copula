# Paper Sweep Status

- Jobs listed: `16`
- Completed (artifact found): `15`
- Pending/missing artifact: `1`

## E4 (Imputation) by config

| Config | Mean Δ% vs best baseline | Std | N |
|---|---:|---:|---:|
| `h002_pool100k` | +32.14% | 1.67 | 3 |
| `h005_pool50k` | +23.93% | 1.33 | 3 |
| `h01_pool50k` | +16.57% | 1.05 | 3 |

## E5 (Anomaly) by config

| Config | Ours AUROC mean | AUROC std | Gap vs best baseline (mean) | N |
|---|---:|---:|---:|---:|
| `score_best` | 0.582 | 0.000 | -0.260 | 3 |
| `score_neglogpdf` | 0.457 | 0.000 | -0.385 | 3 |

## Pending / Missing Artifacts

- `e3_var` job `59843463` label `daily_resim_short_horizon` seed `42`


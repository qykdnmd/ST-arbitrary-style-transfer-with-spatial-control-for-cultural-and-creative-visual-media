# Proposed-model repeated-stylization readout

The frozen v4-gate probe contains 160 observations: one proposed model, four contents, two styles, and 20 repeated-stylization rounds (eight fixed content-style pairs per round). The raw table has no duplicate or missing method-pair-round cells.

| Round | Mean LPIPS | Sample SD | Minimum | Maximum |
|---:|---:|---:|---:|---:|
| 1 | 0.382488 | 0.043557 | 0.346200 | 0.457300 |
| 5 | 0.507450 | 0.035719 | 0.467600 | 0.569000 |
| 10 | 0.611625 | 0.018121 | 0.579500 | 0.633700 |
| 15 | 0.670288 | 0.014888 | 0.641600 | 0.683400 |
| 20 | 0.702500 | 0.014732 | 0.679000 | 0.720600 |

Mean LPIPS from the original content increases monotonically over the 20 rounds, with an absolute round-1-to-round-20 increase of 0.320013. This supports a narrow descriptive claim that repeated application of the proposed model accumulates content deviation on this eight-pair probe.

This archive does not support a cross-method comparison. The existing five-method `results/leakage/leakage_summary.csv` and `results/leakage/leakage_curve.png` were generated before the v4 proposed-model raw CSV and are stale with respect to it. They must not be presented as summaries of the v4 rerun.

No model inference was run while producing this archive.

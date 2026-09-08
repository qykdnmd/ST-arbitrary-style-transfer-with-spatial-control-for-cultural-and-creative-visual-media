# General benchmark: 40 contents × 20 styles

| Method | SSIM ↑ | LPIPS ↓ | E-SSIM ↑ | Content loss ↓ | Style loss ↓ |
|---|---|---|---|---|---|
| CC-StyTr (Main) | 0.5704 | 0.4817 | 0.5640 | 1.5508 | 0.9800 |
| StyTr² | 0.4878 | 0.5859 | 0.4289 | 2.2424 | 1.4303 |
| AdaIN | 0.2965 | 0.6553 | 0.2707 | 2.6595 | 1.2774 |
| SANet | 0.3257 | 0.6483 | 0.2975 | 2.5195 | 1.2951 |
| CAST | 0.4413 | 0.6295 | 0.3897 | 2.4560 | 2.4069 |

## Paired comparisons

Two-sided Wilcoxon tests, Bonferroni correction across 20 comparisons. Positive rank-biserial effects favor CC-StyTr.

| Baseline | Metric | Adjusted p | Rank-biserial effect |
|---|---|---|---|
| StyTr² | ssim | 4.641e-131 | 0.9991 |
| StyTr² | lpips | 2.828e-131 | 1.0000 |
| StyTr² | e_ssim | 4.116e-130 | 0.9955 |
| StyTr² | l_c | 2.775e-131 | 1.0000 |
| StyTr² | l_s | 2.292e-117 | 0.9452 |
| AdaIN | ssim | 2.775e-131 | 1.0000 |
| AdaIN | lpips | 2.775e-131 | 1.0000 |
| AdaIN | e_ssim | 2.775e-131 | 1.0000 |
| AdaIN | l_c | 2.775e-131 | 1.0000 |
| AdaIN | l_s | 6.691e-60 | 0.6736 |
| SANet | ssim | 4.946e-131 | 0.9990 |
| SANet | lpips | 2.828e-131 | 1.0000 |
| SANet | e_ssim | 4.728e-131 | 0.9991 |
| SANet | l_c | 2.775e-131 | 1.0000 |
| SANet | l_s | 3.653e-89 | 0.8231 |
| CAST | ssim | 2.524e-126 | 0.9808 |
| CAST | lpips | 4.965e-131 | 0.9990 |
| CAST | e_ssim | 2.228e-120 | 0.9573 |
| CAST | l_c | 2.817e-131 | 1.0000 |
| CAST | l_s | 3.159e-123 | 0.9687 |

Dependence-aware intervals and auxiliary recognition tests are supplied separately in paper/evidence/.
ArtFID/CSD uses the separate 5000-pair protocol; see results/artfid5000/summary.csv.

The current 5,000-pair table uses finite-sample FID/ArtFID; see `results/artfid5000/README.md` for its source lineage and superseded historical results.

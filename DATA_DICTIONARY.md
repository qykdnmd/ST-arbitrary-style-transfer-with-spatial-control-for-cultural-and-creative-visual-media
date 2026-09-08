# Data dictionary

- `method`: a method or functional configuration identifier; see RESULTS_MAP. In general comparisons, `ccstytr` denotes CC-StyTr (Main). Ablations use the full functional identifiers.
- `cell`, `pair_id`, `output`: pair or output identifiers. Identically named IDs must not be joined across different evaluation datasets without checking their provenance.
- `content`, `content_id`, `style`, `style_id`: sample keys, with sources documented in the corresponding manifests.
- `ssim`, `e_ssim`: similarity measures; higher is better. `lpips`, `l_c`, `l_s`: perceptual content distance, content loss, and style loss; lower is better.
- `ocr_sim`, `n_gt`: OCR consistency and the number of ground-truth text items. Only eligible text-bearing samples enter the corresponding denominator.
- `lpips_content`, `csd_style_similarity`: content distance (lower is better) and style similarity (higher is better) in the control experiment.
- `alpha`, `condition`: global intensity or a protected/inverted regional intervention. The spatial interventions have mean α=0.5.
- `text_boxes`, `logo_boxes`: coordinates in the original poster. Scaling and coordinate mapping must follow the dataset-building scripts.

The full-precision general-benchmark metrics, full-precision AesFA metrics, and spatial-control records retain their respective export precision. Featured-baseline and ablation per-pair measurements are rounded to four decimal places; discarded digits cannot be recovered.

`results/featured/` contains 14,400 image-metric rows and 14,280 OCR rows for the four baselines. The proposed model's 3,600 image-metric rows and 3,570 OCR rows are in `results/ablation/`. The reader matches content-style identifiers to form the five-method comparison while preserving the numerical strings in the input records.

Blank statistical cells indicate an uncomputed or inapplicable result, not zero. A p-value rounded to zero is not an exact zero probability. Primary significance claims should be based on the full-precision statistical evidence.

## Independent 5,000-pair evaluation

`fid_finite` is ordinary finite-sample FID using the official art-trained Inception features. `lpips_content` here uses AlexNet LPIPS, unlike the primary benchmark's VGG LPIPS. `artfid_finite = (1 + lpips_content) * (1 + fid_finite)`. All three are lower-is-better. `csd_similarity_mean` is higher-is-better; `csd_ci95_low/high` are the archived 95% bootstrap intervals from 10,000 resamples. CSD is retained, not recomputed in the finite ArtFID run. The legacy `*_infinity` fields occur only in historical evidence and are not the current reported estimator.

# T037 Spatial-alpha controllability — manuscript readout

## Evidence status

The locked T037 experiment was reviewed PASS in PR #9. The proposed `residual_gating_spatial_alpha` checkpoint was evaluated on 60 PKU PosterLayout posters × 10 WikiArt styles (600 content-style cells). Seven conditions were rendered per cell: five global alpha values (0, 0.25, 0.5, 0.75, 1) and two complementary equal-mean spatial maps (`protected` and `inverted`), for exactly 4,200 outputs.

## Results suitable for the manuscript

Across the five global-alpha conditions, the mean per-cell Spearman correlation between alpha and LPIPS-to-content was 0.39067 (95% interval 0.35133–0.43083), while the correlation between alpha and target-style CSD similarity was 0.67183 (0.64550–0.69767). Thus increasing the global control generally moved the operating point away from the content image and toward the target style. The response was not strictly monotone for every cell: adjacent-step violation rates were 0.32125 for LPIPS and 0.27208 for CSD. The global endpoints moved from LPIPS/CSD = 0.46955/0.07867 at alpha=0 to 0.55980/0.30777 at alpha=1.

For the two complementary spatial maps with the same mean alpha=0.5, the pre-defined local direction criterion succeeded in 365/600 cells (0.60833; interval 0.57000–0.64833), and mean locality contrast was positive at 0.014792 (0.013217–0.016387; median 0.01297). These results indicate a measurable but not perfectly deterministic regional response.

OCR provides an application-facing check for the protected text/logo regions. Mean OCR similarity was 0.558944 for global alpha=0.5, 0.564468 for the protected map and 0.534910 for the inverted map. The paired mean differences were +0.00552 for protected minus global alpha=0.5 and +0.02956 for protected minus inverted; the inverted map was -0.02403 below global alpha=0.5.

## Claim boundary

These measurements support describing spatial alpha as a functioning regional control interface. They do **not** support claiming that spatial alpha universally improves fixed-operating-point image-quality metrics, nor that control is strictly monotonic for every content-style pair. The strongest manuscript interpretation is therefore: global alpha produces the expected aggregate content-style trajectory, and the equal-mean spatial maps create a modest directionally useful regional effect that is also reflected in OCR preservation, while substantial per-cell variability remains.

## Suggested manuscript wording

The locked T037 spatial-controllability experiment covered 600 combinations of 60 PKU PosterLayout posters and 10 WikiArt styles. Each combination produced outputs at five global alpha levels (0, 0.25, 0.5, 0.75, 1) and with two complementary spatial control maps, both with mean alpha=0.5, yielding 4,200 outputs.

Across the five global-alpha conditions, the mean per-pair Spearman correlation of alpha with content LPIPS was 0.39067 (interval 0.35133–0.43083), and its correlation with target-style CSD similarity was 0.67183 (0.64550–0.69767). Increasing global alpha therefore generally shifted the operating point away from the content image and toward the target style. However, the response was not strictly monotonic for every pair: adjacent-level violation rates were 0.32125 for LPIPS and 0.27208 for CSD. As alpha increased from 0 to 1, mean LPIPS/CSD changed from 0.46955/0.07867 to 0.55980/0.30777.

For the complementary spatial conditions with the same mean intensity of 0.5, the predefined local-direction criterion succeeded in 365/600 pairs: a success rate of 0.60833 (0.57000–0.64833). Mean locality contrast was positive at 0.014792 (0.013217–0.016387; median 0.01297). Spatial control thus produced a measurable, but not fully deterministic, regional response.

OCR results further reflect the application-level effect on text/logo preservation. Mean OCR similarities for global alpha=0.5, protected, and inverted conditions were 0.558944, 0.564468, and 0.534910, respectively. Protected exceeded global alpha=0.5 by 0.00552 and inverted by 0.02956; inverted was 0.02403 below global alpha=0.5.

Spatial alpha is therefore best described as an empirically supported regional control interface, rather than a universal source of image-quality gains at a fixed operating point. The evidence supports a cautious conclusion: global alpha produces the expected aggregate content-style trajectory, and equal-mean spatial maps yield modest, directionally useful local control and OCR-preservation effects, with substantial variability across individual pairs.

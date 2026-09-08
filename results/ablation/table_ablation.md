# Ablation on featured set (120 posters x 30 museum styles, 80k-step equal budget)

Wilcoxon signed-rank p-values vs Saliency-routed attention + spatial α (paired per content x style cell).

| Variant | SSIM↑ | LPIPS↓ | E-SSIM↑ | L_c↓ | L_s↓ | OCR-sim↑ |
|---|---|---|---|---|---|---|
| Saliency-routed attention + spatial α | 0.5161 ± 0.0049 | 0.5493 ± 0.0029 | 0.4448 ± 0.0043 | 1.7747 ± 0.0109 | 1.7357 ± 0.0466 | 0.4910 (n=3570) |
| Residual saliency gating + spatial α (proposed model) | 0.5394 ± 0.0052*** | 0.5218 ± 0.0031*** | 0.4795 ± 0.0047*** | 1.6570 ± 0.0116*** | 1.7904 ± 0.0409*** | 0.5906 (n=3570) |
| No structure guidance | 0.5389 ± 0.0052*** | 0.5172 ± 0.0029*** | 0.4815 ± 0.0047*** | 1.6448 ± 0.0108*** | 1.7374 ± 0.0443** | 0.5449 (n=3570) |
| Saliency-routed attention + scalar α | 0.5180 ± 0.0051*** | 0.5481 ± 0.0029* | 0.4603 ± 0.0046*** | 1.7414 ± 0.0117*** | 1.6324 ± 0.0364*** | 0.5930 (n=3570) |
| Residual saliency gating + scalar α | 0.5256 ± 0.0051*** | 0.5426 ± 0.0029*** | 0.4647 ± 0.0046*** | 1.6836 ± 0.0109*** | 1.5871 ± 0.0398*** | 0.5671 (n=3570) |

Significance vs Saliency-routed attention + spatial α: *** p<1e-3, ** p<1e-2, * p<0.05.
Residual gating modulates the cross-attention residual; saliency-routed attention modulates attention competition. Scalar alpha is spatially uniform.

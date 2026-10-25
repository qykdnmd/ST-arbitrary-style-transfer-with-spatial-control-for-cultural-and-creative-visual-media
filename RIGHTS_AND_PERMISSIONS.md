# Third-party rights and permissions review

Metadata checked 2026-10-15; author agreement status updated 2026-10-16. These records distinguish source access, research use,
redistribution and publication of displayed images. They are not a grant of rights.

| Resource | Verified source / observation | Candidate treatment | Remaining action |
|---|---|---|---|
| COCO | Official historical metadata identifies displayed photos 094157 and 435003 as CC BY-NC-ND 2.0, and 226171 as CC BY-NC-SA 2.0; annotations have separate terms | No source photographs redistributed in this code archive; this does not clear manuscript figures | Resolve publication/adaptation rights and credits before submission; do not assume attribution alone permits stylized derivatives or CC BY reuse |
| WikiArt | All three displayed artwork pages were checked and label the works public domain; the site also contains protected works | No full artwork collection redistributed | Preserve per-work sources; exact digital-input provenance still needs confirmation, and these labels do not license the complete dataset |
| PosterLayout | Official Release Agreement limits annotation use to scientific research and disallows unauthorized commercial use or distribution; the author cannot currently confirm a signed agreement | Pair IDs, measurements and author-generated masks only | Resolve acquisition/use authorization and publication permission for displayed original/stylized poster crops; the blank agreement is not evidence of permission |
| MuralDH | Original data paper's Usage Notes restrict the dataset to academic/educational use and prohibit commercial use | Selected IDs and measurements, not the image archive | Clarify publication/adaptation permission for mural reference and stylized images; research-purpose wording alone is not a publication authorization |
| The Met | Official metadata for all ten selected objects reports public-domain status | Source links/metadata retained; no full source image archive | Retain per-object credit and public-domain evidence; do not imply museum endorsement |
| Cleveland | Official metadata for all ten selected objects reports CC0 | Source links/metadata retained; no full source image archive | Retain per-object source records; CC0 does not authorize unrelated trademarks |
| StyTr2 | Archived source has no top-level LICENSE/NOTICE file establishing a redistribution license | Upstream tree excluded; pinned revision and provider links retained | Clarify terms before redistributing upstream code or weights |
| ArtFID | Archived README links MIT while setup.py declares Apache License 2.0 | Upstream tree/snapshot excluded; evaluation evidence and source hashes retained | Request clarification for the evaluated revision before relying on redistribution permission |
| Earlier Zenodo archive | The record API currently reports v0.1.0 and CC-BY-4.0; the project also has a no-project-wide-license research-use statement | Earlier DOI retained only for that snapshot; no metadata edited | Verify the license's actual scope with the authors/deposit record; do not imply that a new statement revokes prior grants |

## Primary sources

- COCO terms: https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm
- WikiArt terms: https://www.wikiart.org/en/terms-of-use
- PosterLayout: https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023
- PosterLayout Release Agreement: https://mipl.pku.edu.cn/PosterLayout/Release_Agreement.pdf
- MuralDH Usage Notes: https://doi.org/10.1038/s41597-024-03785-0
- MuralDH dataset: https://doi.org/10.5061/dryad.bnzs7h4jd
- The Met policy: https://www.metmuseum.org/hubs/open-access
- Cleveland policy: https://www.clevelandart.org/open-access
- StyTr2: https://github.com/diyiiyiii/StyTR-2
- ArtFID: https://github.com/matthias-wright/art-fid

The MuralDH article's own publication license is distinct from dataset use terms;
neither is silently substituted for permission to reproduce adapted source images.
No entire-project open-source license is added. Academic use and citation do not
by themselves establish permission for every use. The candidate excludes vendored
upstream trees but this does not resolve the earlier public archive's permissions.

Official source URLs are provided for traceability. If an endpoint was unavailable,
the machine-readable audit preserves that status rather than treating it as verified.
The ArtFID revision string recorded in the finite-run provenance could not be
resolved through the upstream Git-tree endpoint during this audit (HTTP 422).
The archived setup.py hash was separately checked against the evaluated source
hash. File-hash identity does not resolve the upstream license conflict or make
an unavailable Git revision publicly retrievable.

Additional commit-pinned baseline license checks are recorded in
`UPSTREAM_LICENSE_REVIEW.md` and `provenance/upstream_license_audit.json`.
A source-code license observation is not a license for trained weights, datasets
or figures, and an API's inability to identify a license does not establish that
the source is unlicensed. No redistribution permission is inferred from silence.

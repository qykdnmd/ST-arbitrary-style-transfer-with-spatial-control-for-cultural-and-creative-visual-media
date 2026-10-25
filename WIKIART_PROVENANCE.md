# WikiArt collection, public release and sample trace

The original collection is https://www.wikiart.org/. A corresponding public
release is the refined WikiArt dataset provided by the ArtGAN authors:
https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

The maintainers list a Google Drive archive:
https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view
and an alternative university-server link in their README. These are provider-listed
download entry points; no complete image archive was downloaded in this check.

The project's evaluation manifests were compared with the complete public style
metadata at commit 27876da69ef538fd2bc90eb044e3231d4006f8e5. The three metadata files
were verified against the sizes and Git blob identifiers published by GitHub.
All 20 general-evaluation style filenames and all 10 spatial-control style filenames
match exactly. Of the 5,000 independent evaluation styles, 4,959 movement/path
names match exactly. The other 41 have unique same-movement, same-title candidates:
39 differ in non-ASCII artist-name representations, and two differ only in an
apostrophe in prud'hon/prudhon. These are name-level correspondences, not image
hash verification. No project files or manifests were renamed.

The retrieved style metadata list 57,025 training and 24,421 validation paths
(81,446 total; 27 movements). A project script comment says 81,444 images; the
reason for this two-record difference has not been established. The full original
training-image collection was not recounted in this packaging check.

See provenance/wikiart_release_comparison_20261025.json and
provenance/wikiart_filename_variants_20261025.json for exact evidence and mappings.
The original download service is not established by retained acquisition records.
The historical helpers print a Kaggle Painter by Numbers suggestion; they are
not evidence that this service supplied the experiment's images.

The release maintainers identify WikiArt.org as the original image source, state
non-commercial research use, and require compliance with WikiArt terms. Their
requested paper citation is Tan et al. (2019), Improved ArtGAN for Conditional
Synthesis of Natural Image and Artwork, IEEE Transactions on Image Processing,
28(1), 394-409, https://doi.org/10.1109/TIP.2018.2866698. This is a paper DOI,
not the DOI of this project's code archive. Full source images are not redistributed.

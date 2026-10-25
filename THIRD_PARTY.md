# Third-party sources and redistribution limits

No `external/` source tree or upstream ArtFID source snapshot is included in
this candidate. Obtain implementations and weights from original providers.
The earlier public snapshot is a different archive and is not relicensed here.

| Implementation | Upstream | Archived revision |
|---|---|---|
| StyTr² | https://github.com/diyiiyiii/StyTR-2 | c11ea32913d16dbf2f9a1f82ac077dbf90f143c6 |
| AdaIN | https://github.com/naoto0804/pytorch-AdaIN | 47950d0e6656a95a80a4b105c4c0f58d38ef785c |
| SANet | https://github.com/GlebSBrykin/SANET | 2dfec142cf9bb4c7915a28284097fd4ea9dca251 |
| CAST | https://github.com/zyxElsa/CAST_pytorch | 67a7d9c1e9c1931db5d535385eca9861875049a8 |
| AesFA | https://github.com/Sooyyoungg/AesFA | 00449adf205e93f430dd54799b44a26d740706e1 |
| CSD | https://github.com/learn2phoenix/CSD | 3a9df32605b869eceb704897839be80977a9f1ea |

Cross-check these recorded revisions against `provenance/environment.json`
and experiment-specific provenance before use. ArtFID implementation/file
hashes and Inception weight hashes are in the finite-run provenance; its
paper is https://arxiv.org/abs/2207.12280 . The bundled earlier StyTR-2 copy
had no standalone LICENSE; ArtFID README/setup license declarations disagreed.
These unresolved observations are not cured by a research-use statement.
Verify permissions directly with rights holders before redistribution.

Source images and author-trained checkpoints are excluded; see
`DATA_AVAILABILITY.md`. Public-domain eligibility must be checked per museum
item. Restricted MuralDH and PosterLayout acquisition terms still apply.

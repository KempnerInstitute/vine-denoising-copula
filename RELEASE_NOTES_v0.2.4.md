# Version 0.2.4

- Adds `projects/neurips2026/`: portable experiments, protocols, reference results
  and instructions for the NeurIPS paper, without a separate public paper repository.
- Distributes figure/table reproduction as a curated release archive with saved
  inputs, hashes and a full table inventory. No manuscript workspace or cluster
  launchers are included.
- Replaces inaccessible manuscript-repository links with the public project.
- Requires explicit checkpoint selection in the relocated MI runner; sampling
  replay names the historical CCPP data explicitly and accepts user-supplied paths.
- VDC runtime, pretrained checkpoint and published numerical results are unchanged
  from v0.2.3. The only runtime-file change is the version string.

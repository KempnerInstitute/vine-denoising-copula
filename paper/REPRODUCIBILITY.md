# Reproducibility scope

Replay release: VDC v0.2.6. The reusable runtime is identical to v0.2.3
(commit 9b2d33ca610b63282b9bdc493dcad607522d313b) except its version string.
That release was qualified against the earlier camera-ready runtime; generation
pins inside archived outputs identify the original runs and are not overwritten.
The frozen checkpoint hash is given in README.md. No new checkpoint is used.

The public runners were relocated from qualified paper workflows. Numerical
parameters and estimators are preserved. The MI runner now requires explicit
checkpoint selection. The sampling runner replaces personal default paths with
arguments and names the historical CCPP data explicitly. Reference outputs in
`reference/` are selected immutable inputs, not a second maintained experiment.
`reference/MANIFEST.json` records their source and released hashes.

The paper is not completely reproducible from a fresh download:

- The S&P100 returns matrix is not redistributed and its original collection
  metadata cannot reconstruct it exactly. The runner supports a rebuilt matrix
  but distinguishes that protocol replay from matching reported values.
- The deployed training source is only partially pinned. The public checkpoint
  permits evaluation without claiming an exact retraining replay.
- Alternative corruption checkpoints and exploratory retrained estimators are
  not released. Their appendix results must not be attributed to this checkpoint.
  The corruption runner is supplied with reference scores, but needs the listed
  alternative checkpoints for the full comparison.
- Table 1's portable runner covers VDC/TLL/parametric held-out likelihoods;
  it does not rerun every ISE, tail or timing column. The figure archive contains
  those saved scores and the table-generation procedure.
- MINDE needs its upstream implementation; ACNet runs require its authors' code.
  Both are identified in the paper. Historical MI files include a defective Joe
  case; the published MI summary excludes it, as does the released table generator.
- Figure/table regeneration uses saved outputs. The release qualification checks
  runtime identity, rendering, imports and reduced numerical workflows; it does
  not repeat the complete training and full-size experimental campaign.

Hardware, thread counts, seeds, precisions, structures and truncation choices
remain explicit in the runner manual and outputs. Timings are deployment
measurements on stated hardware, not portable performance guarantees.

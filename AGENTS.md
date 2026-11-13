# Repository maintenance

This repository owns reusable VDC algorithms, training, inference, downloads,
tests, documentation and examples. Keep changes focused and follow
`CONTRIBUTING.md`.

`paper/` holds the NeurIPS 2026 reproduction material: portable experiment
runners, the figure and table generators with their compact saved inputs,
reference results, ablation configurations and one reproduction guide. These
import VDC and never vendor its runtime; they are not installed with the
package. Manuscript sources, internal research records and historical cluster
launchers are maintained in the separate private paper repository.

Keep numerical changes separate from organization work. Preserve scientific
source pins, model hashes and historical tags. Do not add private paths,
scheduler accounts, credentials or internal notes to public files or assets.

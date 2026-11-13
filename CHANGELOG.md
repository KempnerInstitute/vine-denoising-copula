# Changelog

Published versions are immutable. The NeurIPS 2026 paper pins v0.2.6;
its checkpoints and reproduction archives remain available.

## Unreleased

- Gather all NeurIPS 2026 reproduction material in `paper/`: the experiment runners and
  reference outputs (formerly `projects/neurips2026/`), the figure and table generators with
  their saved inputs (formerly distributed only as a release archive) and the ablation training
  configurations (formerly in `configs/train/`). Saved outputs, runner defaults and numerical
  behavior are unchanged; [`paper/README.md`](paper/README.md) maps every figure and table to
  its script, inputs and runner.

## v0.2.7 — 2026-11-12

- Correct Joe/BB1/BB7 training samples and BB density targets; test against analytic CDFs.
- Correct Frank/Joe conditional CDFs, BB rotations and mixture counts; reject invalid conditional training pairs.
- Replace obsolete API examples with programs using the released model loader.
- Correct API signatures and document numerical limits and opt-in batching.
- Simplify installation, consolidate release documentation and expand CI coverage.
- Keep manuscript production and internal research records outside the library.

Details: [v0.2.7 release notes](docs/releases/RELEASE_NOTES_v0.2.7.md).

## Earlier releases

- [v0.2.6](docs/releases/RELEASE_NOTES_v0.2.6.md)
- [v0.2.5](docs/releases/RELEASE_NOTES_v0.2.5.md)
- [v0.2.4](docs/releases/RELEASE_NOTES_v0.2.4.md)
- [v0.2.3](docs/releases/RELEASE_NOTES_v0.2.3.md)
- [v0.2.2](docs/releases/RELEASE_NOTES_v0.2.2.md)
- [v0.2.1](docs/releases/RELEASE_NOTES_v0.2.1.md)
- [v0.2.0](docs/releases/RELEASE_NOTES_v0.2.0.md)
- [v0.1.1](docs/releases/RELEASE_NOTES_v0.1.1.md)
- [v0.1.0](docs/releases/RELEASE_NOTES_v0.1.0.md)

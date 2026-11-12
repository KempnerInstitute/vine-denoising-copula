# Contributing

Start from `main`, the maintained public library branch. Keep changes focused
and open a pull request with the problem, resulting behavior and verification.
Use tagged releases and explicit checkpoint hashes for scientific reproduction.

## Development environment

```bash
python -m pip install -e '.[dev,data]'
python -m pytest tests
```

A CPU is sufficient for the tests. Optional compiled-sampler tests require a
C++17/OpenMP compiler; tests that require unavailable artifacts report a skip.
Do not describe skipped checks as passed. Avoid downloading large checkpoints
in unit tests; use a separate integration check for the released model.

## Repository organization

- `vdc/`: reusable models, data utilities, training and inference.
- `scripts/`: software entry points and model/data tools.
- `examples/`: small programs using the public API.
- `projects/neurips2026/`: portable paper protocols and reference outputs.
- `docs/`: user and maintainer documentation.

Keep manuscript builds, figure-production scripts, cluster launchers and internal
research notes outside this repository. Do not commit credentials, personal
storage paths, scheduler accounts, raw datasets or model weights.

## Numerical changes

Keep estimator changes separate from organization and documentation work. Test
fixed inputs, model serialization, seeds and relevant numerical outputs. Do not
change a scientific experiment's source pin solely because a newer release
imports successfully. Preserve historical release tags and publish numerical
changes in the changelog.

Use markers, line styles, hatches or labels in addition to color in examples.
Document shapes, units, random-state behavior and unsupported combinations.
See the [release guide](docs/RELEASING.md) before publishing a version.

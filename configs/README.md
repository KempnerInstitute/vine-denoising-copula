# Configurations

For a new run, start with `examples/train_custom_copula.py` and
[the configuration guide](../docs/CONFIGURATION.md). That example uses the
maintained unified trainer and a small model suitable for checking the workflow.

The existing YAML files retain earlier training and inference configurations.
They are source records, not a list of recommended recipes. In particular, the
`*_enhanced*` and `enhanced_cnn_cond.yaml` recipes contain `conditional_*` examples
that the corrected generator rejects. Remove those entries and explicitly choose
a new family mixture for a new study; do not describe it as exact historical
retraining. The `data/`, `model/` and `inference/` fragments also serve older
configuration frontends and cannot all be passed directly to the unified trainer.

Paper reproduction uses the source pins and protocol-specific settings in
[`projects/neurips2026`](../projects/neurips2026/), not whichever configuration
happens to be present on `main`. Pretrained inference restores the model's
embedded configuration and does not require these YAML files.

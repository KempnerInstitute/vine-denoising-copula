# Training configuration

For inference, load a pretrained bundle: it restores the architecture and
configuration from the checkpoint. You do not need a training configuration to
use VDC. See [getting started](GETTING_STARTED.md).

For a small new training run from a source checkout:

```bash
python examples/train_custom_copula.py --output results/custom_denoiser.yaml
python scripts/train_unified.py --model-type denoiser --config results/custom_denoiser.yaml
```

The example writes a small configuration and prints the training command. It
does not launch training. Its model is for learning the workflow, not a substitute
for the published checkpoint. Increase capacity and training duration only after
checking validation performance on independent synthetic cases.

## Data

The unified trainer reads these keys under `data`:

| Key | Meaning |
|---|---|
| `m` | Density grid width and height. |
| `n_samples_per_copula` | An integer, a list of choices, or a mapping with `mode`, `min`, `max` (or `choices`). |
| `copula_families` | Mapping of supported family names to nonnegative sampling weights. |
| `param_ranges` | Optional parameter ranges, e.g. `gaussian_rho: [-0.8, 0.8]`. |
| `rotation_prob` | Probability of selecting a rotation for asymmetric families. |
| `mixture_prob` | Probability of a parametric mixture example. |
| `n_mixture_components` | Inclusive minimum and maximum component counts. |
| `num_workers` | DataLoader worker count; use zero for a small CPU run. |

Supported parametric names are `gaussian`, `student`, `clayton`, `gumbel`,
`frank`, `joe`, `bb1`, `bb7`, and `independence`. Grid-defined complex examples
are selected with `complex`, `complex:x`, `complex:ring`, or
`complex:double_banana`. Mixtures combine parametric families.

```yaml
data:
  m: 64
  n_samples_per_copula:
    mode: log_uniform
    min: 200
    max: 20000
  copula_families:
    gaussian: 1
    student: 1
    clayton: 1
    gumbel: 1
    frank: 1
    joe: 1
    bb1: 1
    bb7: 1
    independence: 1
  rotation_prob: 0.5
  mixture_prob: 0.3
  n_mixture_components: [2, 4]
  num_workers: 2
```

This illustrates configuration syntax; it is not a validated replacement recipe
for the paper's model. Varying sample counts returns fixed-size histograms rather
than raw sample tensors so batches can be collated. Options that require raw
samples, such as histogram reconstruction, should use a fixed count.

Read [training data](TRAINING_DATA.md) before reusing historical configurations.
The invalid `conditional_*` training options are rejected. Corrected sampling
changes future training streams, while the released checkpoint remains unchanged.

## Architecture

Choose the architecture with the command-line `--model-type`. For `denoiser`,
the `model` section accepts `base_channels`, `depth`, `blocks_per_level`,
`time_emb_dim`, `dropout`, and `output_mode`. Channel widths must be divisible
by eight for group normalization. `use_coordinates` adds the two coordinate
channels; `use_log_n` adds a sample-count channel. Match these settings when
loading checkpoints; the public loader does that automatically.

`diffusion_unet` uses a different architecture (`channel_mults`,
`num_res_blocks`, `attention_resolutions`) and a `diffusion` section for its
noise schedule and timesteps. `baseline_cnn` and `enhanced_cnn` are additional
architectures. Their settings are not interchangeable with the denoiser's.

Probit-coordinate and density-transform options are experimental. They do not
by themselves turn the uniform cell grid into a nonuniform grid. The public
checkpoint uses its embedded geometry settings.

## Optimization and outputs

The `training` section requires `batch_size`, `max_steps`, and `learning_rate`.
Useful options include `weight_decay`, `gradient_clip`, `use_amp`, and
`projection_iters`. Set `use_amp: false` for the CPU example.

`loss_weights` supports `ce` (cross-entropy), `ise` (integrated squared error),
`tail`, `ms` (multiscale), `tv` (total variation), and `marg_kl` (marginal
regularization). A zero weight disables a term. These losses are grid
approximations, and their scales differ; compare validation metrics when changing
their balance. Unknown keys are not a way to add a loss.

`log_interval`, `save_interval`, and `viz_interval` control reporting, checkpoint
saving and training diagnostics. `checkpoint_dir` is a top-level key. Use
`--resume path/to/checkpoint.pt` to resume a compatible trusted checkpoint.

The unified command seeds dataset workers from `42 + distributed rank`; record
worker count, source revision, full configuration and environment alongside a
run. Exact historical training replay requires the original generator revision.
The older `scripts/train.py` is a separate training frontend; do not assume its
checkpoints or defaults are interchangeable with `scripts/train_unified.py`.

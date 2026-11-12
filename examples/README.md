# Examples

Install the package first with `python -m pip install -e .` from the repository
root. Pretrained examples download and verify the released model, or accept
`--checkpoint /path/to/trusted-model.pt`. Start on a CPU with several GB of RAM.

| Program | Purpose |
|---|---|
| `use_pretrained_model.py` | Load the public model, estimate a pair density and optionally fit a vine. |
| `fit_vine_copula.py` | Fit a D-vine and score a held-out Gaussian-copula sample. |
| `estimate_mutual_information.py` | Estimate MI and compare it with the exact Gaussian reference. |
| `visualize_copulas.py` | Plot a supplied density `.npy` file. |
| `train_custom_copula.py` | Write a small configuration; does not launch training. |

```bash
python examples/fit_vine_copula.py --device cpu --dimension 5
python examples/estimate_mutual_information.py --device cpu --rho 0.6
vdc estimate-pair pair.npy --output results/density.npy
python examples/visualize_copulas.py --density results/density.npy
python examples/train_custom_copula.py --output results/custom.yaml
```

These are software demonstrations, not reproductions of the paper's reported
benchmarks. Fixed protocols and reference outputs live in `projects/neurips2026/`.

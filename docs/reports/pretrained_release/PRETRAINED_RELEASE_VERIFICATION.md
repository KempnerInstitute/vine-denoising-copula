# Pretrained model verification

- Model id: `vdc-denoiser-m64-v1`
- Checkpoint: `/n/holylfs06/LABS/kempner_project_b/Lab/vine_diffusion_copula/vdc_paper_denoiser_cond_enhanced_20260207_105852_59344687/checkpoints/model_step_190000.pt`
- Device: `cpu`
- Mean ISE: `5.572113e-07`
- Mean MI absolute error: `0.014998`
- Mean mass error: `6.953875e-08`

## Cases

| Case | ISE | MI true | MI est | abs error | mass |
|---|---:|---:|---:|---:|---:|
| Independence | 9.778638e-08 | 0.0000 | 0.0002 | 0.0002 | 1.000000 |
| Gaussian(ρ=0.7) | 1.436493e-07 | 0.3306 | 0.3361 | 0.0055 | 1.000000 |
| Gaussian(ρ=-0.7) | 1.200465e-06 | 0.3306 | 0.3585 | 0.0279 | 1.000000 |
| Clayton(θ=3.0) | 1.305508e-06 | 0.6302 | 0.5975 | 0.0328 | 1.000000 |
| Frank(θ=5.0) | 1.365651e-07 | 0.2579 | 0.2608 | 0.0029 | 1.000000 |
| Gumbel(θ=2.5) | 4.592939e-07 | 0.5545 | 0.5339 | 0.0207 | 1.000000 |

## Qualitative figure

- PNG: `/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula/docs/reports/pretrained_release/fig_copula_example_main_verify.png`
- PDF: `/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula/docs/reports/pretrained_release/fig_copula_example_main_verify.pdf`
- DCD ISE: `7.517040785387508e-07`
- pyvine ISE: `0.00030537464185091396`

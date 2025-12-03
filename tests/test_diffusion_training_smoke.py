"""Smoke tests for diffusion training after projection/input fixes."""
from __future__ import annotations

import torch
import pytest

from scripts.train_unified import build_model, training_step
from vdc.models.copula_diffusion import CopulaAwareDiffusion
from vdc.data.onthefly import OnTheFlyCopulaDataset


def _make_config() -> dict:
    return {
        "data": {
            "m": 8,
            "n_samples_per_copula": 128,
            "copula_families": {
                "gaussian": 0.3,
                "clayton": 0.2,
                "gumbel": 0.2,
                "frank": 0.15,
                "joe": 0.1,
                "student": 0.05,
            },
        },
        "model": {
            "transform_to_probit_space": False,
            "use_coordinates": False,
        },
        "training": {
            "batch_size": 1,
            "max_steps": 8,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "use_amp": False,
            "projection_iters": 2,
            "gradient_clip": 1.0,
            "hist_sigma": 0.5,
            "use_antialiased_hist": True,
            "input_hist_sigma": 0.5,
            "input_noise_std": 0.0,
            "loss_weights": {
                "ce": 1.0,
                "ise": 0.5,
                "tail": 0.0,
                "ms": 0.0,
                "marg_kl": 0.0,
            },
            "multi_scale": {"enable": False},
        },
        "diffusion": {
            "timesteps": 10,
            "noise_schedule": "linear",
        },
    }


def _make_batch(config: dict, batch_size: int = 2) -> dict:
    dataset = OnTheFlyCopulaDataset(
        n_samples_per_batch=config["data"]["n_samples_per_copula"],
        m=config["data"]["m"],
        families=config["data"]["copula_families"],
        transform_to_probit_space=config["model"].get("transform_to_probit_space", False),
        seed=123,
    )
    samples_list = []
    densities = []
    flags = []
    for idx in range(batch_size):
        sample = dataset[idx]
        samples_list.append(sample["samples"])
        densities.append(sample["density"])
        flags.append(sample["is_log_density"])
    samples = torch.stack(samples_list, dim=0)
    density = torch.stack(densities, dim=0)
    is_log_density = torch.tensor(flags, dtype=torch.bool)
    return {
        "samples": samples,
        "density": density,
        "is_log_density": is_log_density,
    }


def _setup_training_objects(config: dict, device: torch.device):
    model = build_model("diffusion_unet", config, device)
    diffusion = CopulaAwareDiffusion(
        timesteps=config["diffusion"]["timesteps"],
        beta_schedule=config["diffusion"].get("noise_schedule", "linear"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    training_step.optimizer = optimizer
    training_step.loss_log_vars = None
    return model, diffusion, optimizer


@pytest.mark.slow(reason="runs a lightweight training step")
def test_diffusion_training_step_smoke():
    config = _make_config()
    device = torch.device("cpu")
    batch = _make_batch(config, batch_size=config["training"]["batch_size"])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    model, diffusion, _ = _setup_training_objects(config, device)

    metrics = training_step(
        "diffusion_unet",
        model,
        batch,
        device,
        config,
        diffusion=diffusion,
        scaler=None,
        step=0,
        profiler=None,
    )

    assert "loss" in metrics and torch.isfinite(torch.tensor(metrics["loss"])), "loss should be finite"
    assert "grad_norm" in metrics and torch.isfinite(torch.tensor(metrics["grad_norm"])), "grad norm should be finite"


@pytest.mark.slow(reason="verifies parameters update under a training step")
def test_diffusion_parameter_updates():
    config = _make_config()
    device = torch.device("cpu")
    batch = _make_batch(config, batch_size=config["training"]["batch_size"])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    model, diffusion, optimizer = _setup_training_objects(config, device)

    initial_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

    metrics = training_step(
        "diffusion_unet",
        model,
        batch,
        device,
        config,
        diffusion=diffusion,
        scaler=None,
        step=0,
        profiler=None,
    )

    updated_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    diff = torch.norm(initial_params - updated_params)

    assert diff > 0, "parameters should change after one optimization step"
    assert torch.isfinite(torch.tensor(metrics["loss"])), "updated run should still report finite loss"

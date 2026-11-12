"""Exercise the documented denoiser configuration through an optimizer update."""
import runpy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_unified import build_model, training_step
from vdc.data.onthefly import OnTheFlyCopulaDataset


def test_custom_configuration_trains_the_declared_model():
    example = Path(__file__).resolve().parents[1] / "examples/train_custom_copula.py"
    config = runpy.run_path(str(example))["make_config"]()
    device = torch.device("cpu")
    model = build_model("denoiser", config, device)
    dataset = OnTheFlyCopulaDataset(m=config["data"]["m"], n_samples_per_batch=128,
                                  families=config["data"]["copula_families"], seed=42)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    training_step.optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    training_step.loss_log_vars = None
    before = model.in_conv.weight.detach().clone()
    metrics = training_step("denoiser", model, batch, device, config)
    assert np.isfinite(metrics["loss"])
    assert not torch.equal(before, model.in_conv.weight)

"""Pilot pretraining of the linear-grid models: bank, batches, losses and loop (CPU)."""
import json

import numpy as np
import pytest
import pyvinecopulib as pv
import torch

from vdc.train.linear_grid_pretraining import (
    FAMILY_WEIGHTS,
    Batcher,
    M,
    build_bank,
    d4,
    density_losses,
    exact_masses,
    normalize,
    project_like_training,
    run_pretraining,
    tail_mask,
)


def test_dihedral_transforms_are_distinct_permutations():
    x = torch.arange(16.).view(4, 4)
    images = [d4(x, code) for code in range(8)]
    assert len({tuple(image.flatten().tolist()) for image in images}) == 8
    for image in images:
        assert sorted(image.flatten().tolist()) == list(range(16))


def test_exact_masses_have_uniform_margins():
    for bicop in (pv.Bicop(), pv.Bicop(family=pv.BicopFamily.clayton, parameters=np.array([[3.]]))):
        mass = exact_masses(bicop)
        assert mass.shape == (M, M) and mass.min() > 0
        np.testing.assert_allclose(mass.sum(0), 1/M, atol=1e-9)
        np.testing.assert_allclose(mass.sum(1), 1/M, atol=1e-9)


def test_bank_is_reproducible():
    bank, labels = build_bank(12, seed=42, progress_every=0)
    again, _ = build_bank(12, seed=42, progress_every=0)
    np.testing.assert_array_equal(bank, again)
    assert bank.dtype == np.float32 and set(labels) <= set(FAMILY_WEIGHTS)
    np.testing.assert_allclose(bank.sum((1, 2)), 1, atol=1e-5)


def test_batches_and_losses_on_cpu():
    bank, _ = build_bank(6, seed=3, progress_every=0)
    batcher = Batcher(bank, torch.device("cpu"), seed=5)
    masses, hist, n = batcher.batch(4)
    np.testing.assert_allclose(masses.sum((-2, -1)).numpy(), 1, atol=1e-5)
    assert ((n >= 200) & (n <= 8000)).all()
    np.testing.assert_allclose((hist.sum((-2, -1)) / (M*M)).numpy(), 1, atol=1e-5)
    counts = hist * n[:, None, None] / (M*M)
    np.testing.assert_allclose(counts.numpy(), np.round(counts.numpy()), atol=1e-3)
    pred = normalize(torch.rand(4, 1, M, M) + .5)
    projected = normalize(project_like_training(pred))
    loss, parts = density_losses(projected, (masses * M * M)[:, None], tail_mask(torch.device("cpu")))
    assert torch.isfinite(loss) and set(parts) == {"ce", "tail", "marg"}


def pretrain(tmp_path, out, **extra):
    settings = {"output_dir": tmp_path / out, "bank_path": tmp_path / "bank.npy", "bank_size": 6,
                "steps": 3, "batch_size": 2, "warmup": 2, "save_every": 2, "base_channels": 8,
                "seed": 11, "device": "cpu"}
    return run_pretraining({**settings, **extra})


def test_pretraining_is_reproducible_and_stops_without_changing_the_schedule(tmp_path):
    first = pretrain(tmp_path, "a")
    assert (tmp_path / "bank.npy").exists() and (tmp_path / "bank.npy.labels.json").exists()
    second = pretrain(tmp_path, "b")
    stopped = pretrain(tmp_path, "c", stop_after=1, save_every=10)
    for name in ("pilot_raw_step2.pt", "pilot_ema_step3.pt"):
        a = torch.load(tmp_path / "a" / name, weights_only=False)
        b = torch.load(tmp_path / "b" / name, weights_only=False)
        assert all(torch.equal(a["model_state_dict"][k], b["model_state_dict"][k]) for k in a["model_state_dict"])
    assert first.name == "pilot_ema_step3.pt" and second.exists() and stopped.name == "pilot_ema_step1.pt"
    config = json.loads((tmp_path / "a" / "config.json").read_text())
    assert config["model"]["grid_size"] == 64 and config["training"]["steps"] == 3
    with pytest.raises(ValueError):
        pretrain(tmp_path, "d", bank_sha256="0" * 64)
    assert pretrain(tmp_path, "e", bank_only=True) == tmp_path / "bank.npy"

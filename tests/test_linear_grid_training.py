"""Fine-tuning data cache and training loop of the linear-grid models (CPU, tiny)."""
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import rankdata

from vdc.data.continuous_mixtures import ContinuousMixture, draw_spec
from vdc.models.copula_denoiser import CopulaDenoiser
from vdc.models.linear_grid_estimator import LinearGridSettings, grid_geometry
from vdc.pretrained import linear_grid_settings_for, load_checkpoint_bundle
from vdc.train.linear_grid_pretraining import PRETRAINING_DEFAULTS
from vdc.train.linear_grid_training import (
    CACHE_DEFAULTS,
    FINETUNING_DEFAULTS,
    CachedContinuousBatches,
    build_cache_batch,
    build_training_cache,
    load_config,
    run_finetuning,
    run_stage,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 202610051


def test_cache_batch_matches_direct_generation(tmp_path):
    sizes = [50, 1000]
    row = build_cache_batch(0, 2, SEED, sizes, tmp_path, grid_sizes=(8, 16), latent_histograms=True)
    with np.load(tmp_path / row["path"]) as z:
        n = int(z["n"][0])
        assert row["n"] == n and n in sizes
        np.testing.assert_array_equal(z["indices"], [0, 1])
        for k in range(2):
            population, sample, target = np.random.SeedSequence([SEED, k, 11]).generate_state(3)
            truth = ContinuousMixture(draw_spec(np.random.default_rng(int(population))))
            raw = truth.simulate(n, int(sample))
            ranks = rankdata(raw, axis=0, method="average") / (n + 1.)
            for m in (8, 16):
                edges = grid_geometry(m)[0]
                np.testing.assert_array_equal(z[f"u{m}"][k], np.histogram2d(*ranks.T, bins=(edges, edges))[0])
                np.testing.assert_array_equal(z[f"latent_u{m}"][k], np.histogram2d(*raw.T, bins=(edges, edges))[0])
            np.testing.assert_array_equal(z["target_points"][k], truth.simulate(512, int(target)).astype(np.float32))


def small_cache(path, workers=1, steps=3, latent=True):
    return build_training_cache({"output_dir": path, "steps": steps, "batch_size": 4, "seed": SEED,
                                 "sizes": [50, 100], "grid_sizes": [8], "latent_histograms": latent,
                                 "workers": workers})


def test_cache_contents_do_not_depend_on_workers(tmp_path):
    one, two = small_cache(tmp_path / "one", 1), small_cache(tmp_path / "two", 2)
    for step in range(3):
        name = f"batch_{step:05d}.npz"
        with np.load(one / name) as a, np.load(two / name) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def test_cached_batches_mix_latent_inputs_reproducibly(tmp_path):
    root = small_cache(tmp_path / "cache")
    edges = grid_geometry(8)[0]
    area = np.outer(np.diff(edges), np.diff(edges))
    rank_only = CachedContinuousBatches(root, 3, 4, SEED, 8)
    mixed = CachedContinuousBatches(root, 3, 4, SEED, 8, latent_probability=.5)
    for index in range(3):
        item, plain = mixed[index], rank_only[index]
        mask = np.random.default_rng(np.random.SeedSequence([SEED, index, 71])).random(4) < .5
        np.testing.assert_array_equal(item["latent_mask"], mask)
        assert not plain["latent_mask"].any()
        with np.load(root / f"batch_{index:05d}.npz") as z:
            counts = np.where(mask[:, None, None], z["latent_u8"], z["u8"])
            np.testing.assert_array_equal(item["hist"], (counts / (z["n"][:, None, None] * area)).astype(np.float32))
            np.testing.assert_array_equal(plain["hist"], (z["u8"] / (z["n"][:, None, None] * area)).astype(np.float32))
        np.testing.assert_array_equal(item["target_points"], plain["target_points"])
    with pytest.raises(ValueError):
        CachedContinuousBatches(root, 3, 4, SEED + 1, 8)
    with pytest.raises(ValueError):
        CachedContinuousBatches(root, 4, 4, SEED, 8)
    no_latent = small_cache(tmp_path / "rank", latent=False, steps=1)
    with pytest.raises(ValueError):
        CachedContinuousBatches(no_latent, 1, 4, SEED, 8, latent_probability=.5)
    shard = root / "batch_00001.npz"
    shard.write_bytes(shard.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        mixed[1]


@pytest.fixture
def tiny_pilot(tmp_path):
    torch.manual_seed(0)
    config = {"model": {"type": "denoiser", "grid_size": 8, "base_channels": 8, "depth": 2, "blocks_per_level": 1,
                        "dropout": 0.1, "time_emb_dim": 16, "output_mode": "log", "use_coordinates": True,
                        "use_probit_coords": True, "use_log_n": True},
              "data": {"m": 8}}
    model = CopulaDenoiser(m=8, input_channels=4, base_channels=8, depth=2, blocks_per_level=1, time_emb_dim=16)
    path = tmp_path / "pilot.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config, "step": 0}, path)
    return path


def finetune(output, pilot, cache, **extra):
    settings = {"output_dir": output, "initial_checkpoint": pilot, "cache_dir": cache, "grid_size": 8,
                "latent_probability": .5, "base_channels": 8, "steps": 3, "batch_size": 4, "microbatch": 2,
                "workers": 0, "seed": SEED, "sizes": [50, 100], "save_every": 3, "device": "cpu",
                "log_every": 1, "cpu_threads": 1}
    return run_finetuning({**settings, **extra})


def test_finetuning_is_deterministic_and_produces_loadable_models(tmp_path, tiny_pilot):
    cache = small_cache(tmp_path / "cache")
    first = finetune(tmp_path / "a", tiny_pilot, cache, save_steps=[2])
    second = finetune(tmp_path / "b", tiny_pilot, cache, save_steps=[2])
    stopped = finetune(tmp_path / "c", tiny_pilot, cache, stop_after=2)
    for name in ("ema_step2.pt", "ema_step3.pt"):
        a = torch.load(first / name, weights_only=False)["model_state_dict"]
        b = torch.load(second / name, weights_only=False)["model_state_dict"]
        assert all(torch.equal(a[k], b[k]) for k in a)
    a = torch.load(first / "ema_step2.pt", weights_only=False)["model_state_dict"]
    c = torch.load(stopped / "ema_step2.pt", weights_only=False)["model_state_dict"]
    assert all(torch.equal(a[k], c[k]) for k in a)
    assert not (stopped / "ema_step3.pt").exists()
    rows = [line for line in (first / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    bundle = load_checkpoint_bundle(first / "ema_step3.pt", device="cpu")
    assert linear_grid_settings_for(bundle) == LinearGridSettings(grid_size=8)
    with pytest.raises(ValueError):
        finetune(tmp_path / "d", tiny_pilot, cache, initial_sha256="0" * 64)
    with pytest.raises(ValueError):
        finetune(tmp_path / "e", tiny_pilot, cache, microbatch=3)


def test_configurations_contain_only_known_settings():
    known = {"pretrain": PRETRAINING_DEFAULTS, "cache": CACHE_DEFAULTS, "finetune": FINETUNING_DEFAULTS}
    expected = {"linear_grid_pilot.yaml": "pretrain", "linear_grid_cache.yaml": "cache",
                "linear_grid_vdc64_v2.yaml": "finetune", "linear_grid_vdc128_v2.yaml": "finetune"}
    for name, stage in expected.items():
        config = load_config(ROOT / "configs/train" / name)
        assert config.pop("stage") == stage
        assert set(config) <= set(known[stage]), name
    config = load_config(ROOT / "configs/train/linear_grid_vdc128_v2.yaml", ["steps=10", "save_steps=[5]"])
    assert config["steps"] == 10 and config["save_steps"] == [5] and config["latent_probability"] == .75
    with pytest.raises(ValueError):
        run_stage({"stage": "evaluate"})
    with pytest.raises(ValueError, match="Unknown"):
        run_stage({"stage": "finetune", "recipe": "fresh_nll_u64_t0"})

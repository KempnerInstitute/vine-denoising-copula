"""Linear-grid pretrained manifests, local checkpoint loading and pair estimation."""
import hashlib
import json

import numpy as np
import pytest
import torch

import vdc.pretrained as pretrained
from vdc.models.copula_denoiser import CopulaDenoiser
from vdc.models.linear_grid_copula import LinearGridCopula
from vdc.models.linear_grid_estimator import LinearGridEstimator, LinearGridSettings
from vdc.pretrained import (
    LoadedPretrainedModel,
    estimate_pair_copula_from_samples,
    estimate_pair_copulas_from_samples,
    estimate_pair_density_from_samples,
    list_pretrained_models,
    load_pretrained_manifest,
    resolve_pretrained_checkpoint,
)

V2 = {"vdc-denoiser-m64-v2": ("73097772a18eb4b02fb0608b4bfa87c0205b71667d97af6a39483fd44d0df0fc", 64),
      "vdc-denoiser-m128-v2": ("f0a753d50b3cfb089e6d7dec999cfe741c6342c188454ec83183a9b3514d4653", 128)}


@pytest.mark.parametrize("model_id", sorted(V2))
def test_v2_manifests(model_id):
    manifest = load_pretrained_manifest(model_id)
    sha, m = V2[model_id]
    assert manifest["sha256"] == sha and manifest["grid_size"] == m
    assert manifest["checkpoint_filename"] == model_id + ".pt"
    hf = manifest["sources"]["huggingface"]
    assert hf["repo_id"] is None and hf["proposed_repo_id"] == "hsafaai/" + model_id
    assert manifest["inference"]["pair_estimator"] == "linear_grid"
    assert "local_checkpoint" not in manifest["sources"]
    text = json.dumps(manifest)
    assert "/n/" not in text and "slurm" not in text.lower()
    assert model_id in {m["model_id"] for m in list_pretrained_models()}


def test_default_model_is_unchanged():
    manifest = load_pretrained_manifest()
    assert manifest["model_id"] == "vdc-denoiser-m64-v1" and "inference" not in manifest


def test_unpublished_model_explains_local_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match="not yet published"):
        resolve_pretrained_checkpoint("vdc-denoiser-m64-v2", cache_dir=tmp_path)


def test_checkpoint_path_is_verified(tmp_path, monkeypatch):
    good = tmp_path / "model.pt"
    good.write_bytes(b"weights")
    manifest = {"model_id": "test-v2", "sha256": hashlib.sha256(b"weights").hexdigest(),
                "checkpoint_filename": "test-v2.pt", "sources": {"huggingface": {"repo_id": None}}}
    monkeypatch.setattr(pretrained, "load_pretrained_manifest", lambda model_id=None: manifest)
    assert resolve_pretrained_checkpoint("test-v2", checkpoint_path=good, cache_dir=tmp_path) == good
    bad = tmp_path / "other.pt"
    bad.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        resolve_pretrained_checkpoint("test-v2", checkpoint_path=bad, cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_pretrained_checkpoint("test-v2", checkpoint_path=tmp_path / "missing.pt", cache_dir=tmp_path)

    staged = pretrained.stage_release_bundle("test-v2", out_dir=tmp_path / "release", repo_id="owner/test-v2",
                                             checkpoint_path=good)
    assert (staged / "test-v2.pt").read_bytes() == b"weights"
    public = json.loads((staged / "manifest.json").read_text())
    assert public["sources"]["huggingface"]["repo_id"] == "owner/test-v2"
    assert public["sources"]["local_checkpoint"] is None
    assert not (staged / "train_config.yaml").exists()


def bundle_for(manifest, m=16):
    torch.manual_seed(3)
    model = CopulaDenoiser(m=m, input_channels=4, base_channels=8, depth=2, blocks_per_level=1,
                           time_emb_dim=16).eval()
    return LoadedPretrainedModel("test", None, manifest, model, None, {"data": {"m": m}}, torch.device("cpu"))


def test_pair_estimation_dispatches_to_linear_grid():
    inference = LinearGridSettings(grid_size=16, forward_batch_size=4).to_dict()
    bundle = bundle_for({"inference": inference})
    pairs = [np.random.default_rng(seed).uniform(0, 1, (n, 2)) for seed, n in ((1, 200), (2, 500))]
    grid = estimate_pair_density_from_samples(bundle, pairs[0])
    reference = LinearGridEstimator(bundle.model, LinearGridSettings(grid_size=16, forward_batch_size=4),
                                    "cpu").estimate_grids(pairs[:1])[0]
    assert grid.shape == (16, 16) and grid.dtype == np.float32
    np.testing.assert_array_equal(grid, reference)
    copula = estimate_pair_copula_from_samples(bundle, pairs[0])
    assert isinstance(copula, LinearGridCopula)
    np.testing.assert_array_equal(copula.density_grid,
                                  LinearGridCopula(reference, copula.nodes, solver="hybrid").density_grid)
    copulas = estimate_pair_copulas_from_samples(bundle, pairs)
    np.testing.assert_array_equal(copulas[0].density_grid, copula.density_grid)
    fewer = estimate_pair_density_from_samples(bundle, pairs[0], projection_iters=3)
    assert not np.array_equal(fewer, grid)
    with pytest.raises(ValueError):
        estimate_pair_density_from_samples(bundle, pairs[0], m=32)
    with pytest.raises(ValueError):
        estimate_pair_density_from_samples(bundle, pairs[0], diffusion_steps=10)


def test_pair_copula_requires_linear_grid_model():
    bundle = bundle_for({})
    with pytest.raises(ValueError, match="not a linear-grid model"):
        estimate_pair_copula_from_samples(bundle, np.full((10, 2), .5))

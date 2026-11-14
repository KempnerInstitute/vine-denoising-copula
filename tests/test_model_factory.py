"""Checkpoint construction works without importing the training environment."""
import json
import subprocess
import sys

import pytest
import torch

from vdc.models.factory import build_model
from vdc.pretrained import load_checkpoint_bundle


@pytest.mark.parametrize("model_type", ["baseline_cnn", "enhanced_cnn", "denoiser", "diffusion_unet"])
def test_checkpoint_roundtrip_without_training_dependencies(tmp_path, model_type):
    config = {"model_type": model_type, "data": {"m": 16}, "model": {
        "base_channels": 8, "n_blocks": 2, "depth": 2, "blocks_per_level": 1,
        "time_emb_dim": 16, "channel_mults": [1, 2], "num_res_blocks": 1,
        "use_log_n": True, "use_probit_coords": True,
    }}
    torch.manual_seed(7)
    model = build_model(model_type, config, torch.device("cpu"))
    checkpoint = tmp_path / "small.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
    loaded = load_checkpoint_bundle(checkpoint, device="cpu")
    assert type(loaded.model) is type(model)
    assert not loaded.model.training
    assert (loaded.diffusion is not None) == (model_type == "diffusion_unet")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    for name in ("vdc_use_coordinates", "vdc_use_probit_coords", "vdc_use_log_n"):
        assert getattr(model, name, None) == getattr(loaded.model, name, None)

    # Blocking imports also catches accidental dependencies when the full
    # development environment happens to contain optional packages.
    code = '''
import importlib.abc
import sys
class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"matplotlib", "h5py", "pyvinecopulib", "hydra", "sklearn"}:
            raise AssertionError("Inference imported an optional dependency: " + fullname)
        if fullname == "vdc.train.unified_trainer":
            raise AssertionError("Inference imported the trainer")
sys.meta_path.insert(0, NoTrainingImports())
from vdc.pretrained import load_checkpoint_bundle
from vdc.train import build_model
from vdc.models.factory import build_model as factory
assert build_model is factory
bundle = load_checkpoint_bundle(CHECKPOINT, device="cpu")
assert not bundle.model.training
'''.replace("CHECKPOINT", json.dumps(str(checkpoint)))
    subprocess.run([sys.executable, "-c", code], check=True, timeout=90)


def test_legacy_trainer_reexports_the_factory():
    from vdc.train.unified_trainer import build_model as legacy
    assert legacy is build_model

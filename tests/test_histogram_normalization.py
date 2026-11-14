"""Keep mass histograms out of the pretrained model's density input."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vdc.data.hist import scatter_to_hist as legacy_mass_hist
from vdc.data.hist import scatter_to_mass_hist
from vdc.inference.density import scatter_to_hist as density_hist
from vdc.pretrained import estimate_pair_density_from_samples


@pytest.mark.parametrize("m", [8, 64])
@pytest.mark.parametrize("reflect", [False, True])
def test_histogram_units_and_compatibility(m, reflect):
    points = np.random.default_rng(42).uniform(0.001, 0.999, (200, 2))
    mass = scatter_to_mass_hist(points, m, reflect=reflect)
    density = density_hist(points, m, reflect=reflect)
    assert mass.sum() == pytest.approx(1.0)
    assert density.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(density, mass * m**2, rtol=1e-10, atol=0)
    np.testing.assert_array_equal(mass, legacy_mass_hist(points, m, reflect=reflect))


def test_public_pretrained_entry_point_uses_density_units():
    class InputRecorder(torch.nn.Module):
        def forward(self, x, t):
            self.histogram = x[:, :1].detach().clone()
            return {"density": torch.ones_like(self.histogram)}

    model = InputRecorder()
    bundle = SimpleNamespace(model=model, device=torch.device("cpu"), diffusion=None,
                             config={"data": {"m": 64}}, manifest={})
    points = np.random.default_rng(17).uniform(0.001, 0.999, (200, 2))
    result = estimate_pair_density_from_samples(bundle, points)
    assert float(model.histogram.mean()) == pytest.approx(1.0, abs=1e-6)
    expected = density_hist(points, 64).astype(np.float32)
    np.testing.assert_array_equal(model.histogram[0, 0].numpy(), expected)
    np.testing.assert_allclose(result, 1.0, rtol=0, atol=1e-6)

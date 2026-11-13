"""Utilities for versioned pretrained VDC model releases."""

from __future__ import annotations

import hashlib
import importlib.resources as resources
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from vdc.models.copula_diffusion import CopulaAwareDiffusion
    from vdc.models.linear_grid_copula import LinearGridCopula
    from vdc.models.linear_grid_estimator import LinearGridSettings

DEFAULT_PRETRAINED_MODEL_ID = "vdc-denoiser-m64-v1"
_LEGACY_MODEL_ALIASES = {
    "vdc-icml2026-paper-v1": DEFAULT_PRETRAINED_MODEL_ID,
}
_RESOURCE_PACKAGE = "vdc.resources.pretrained"


@dataclass(frozen=True)
class LoadedPretrainedModel:
    """Loaded model bundle ready for inference."""

    model_id: str
    checkpoint_path: Path
    manifest: Dict[str, Any]
    model: torch.nn.Module
    diffusion: Optional["CopulaAwareDiffusion"]
    config: Dict[str, Any]
    device: torch.device


def _manifest_paths() -> List[Path]:
    root = resources.files(_RESOURCE_PACKAGE)
    return sorted(
        Path(str(item))
        for item in root.iterdir()
        if item.is_file() and item.name.endswith(".json")
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def list_pretrained_models() -> List[Dict[str, Any]]:
    """Return packaged pretrained model manifests."""
    return [_read_json(path) for path in _manifest_paths()]


def load_pretrained_manifest(model_id: str = DEFAULT_PRETRAINED_MODEL_ID) -> Dict[str, Any]:
    """Load a packaged pretrained-model manifest by id."""
    model_id = _LEGACY_MODEL_ALIASES.get(model_id, model_id)
    for payload in list_pretrained_models():
        if str(payload.get("model_id", "")).strip() == model_id:
            return payload
    known = ", ".join(sorted(p.get("model_id", "") for p in list_pretrained_models()))
    raise KeyError(f"Unknown pretrained model id: {model_id}. Known ids: {known}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected: Optional[str]) -> None:
    if not expected:
        return
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected}, found {actual}"
        )


def _default_cache_dir() -> Path:
    raw = os.environ.get("VDC_PRETRAINED_CACHE", "~/.cache/vdc")
    return Path(raw).expanduser().resolve()


def _hf_download_url(repo_id: str, revision: str, filename: str) -> str:
    repo = repo_id.strip("/")
    rev = revision.strip("/") or "main"
    name = filename.lstrip("/")
    return f"https://huggingface.co/{repo}/resolve/{rev}/{name}?download=1"


def resolve_pretrained_checkpoint(
    model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    *,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    prefer_local: bool = True,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    checkpoint_path: Optional[Path | str] = None,
) -> Path:
    """Resolve a checkpoint path for a pretrained model.

    Resolution order:
      0. ``checkpoint_path`` if given (a local copy, checked against the
         manifest's SHA-256; required for models not yet published)
      1. packaged local checkpoint path if present and prefer_local=True
      2. cached downloaded artifact
      3. Hugging Face download if repo_id is configured
    """
    manifest = load_pretrained_manifest(model_id)
    expected_sha = str(manifest.get("sha256", "")).strip() or None
    sources = manifest.get("sources", {})

    if checkpoint_path is not None:
        explicit = Path(checkpoint_path).expanduser()
        if not explicit.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        if expected_sha is None:
            raise RuntimeError(f"Manifest for {model_id} has no SHA-256 to verify {explicit}")
        _verify_sha256(explicit, expected_sha)
        return explicit

    local_raw = str(sources.get("local_checkpoint", "") or "").strip()
    local_path = Path(local_raw).expanduser() if local_raw else None
    if prefer_local and local_path is not None and local_path.exists():
        _verify_sha256(local_path, expected_sha)
        return local_path

    cache_root = cache_dir or _default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    filename = str(manifest.get("checkpoint_filename", "")).strip() or "model.pt"
    cached_path = cache_root / model_id / filename
    if cached_path.exists() and not force_download:
        _verify_sha256(cached_path, expected_sha)
        return cached_path

    hf = sources.get("huggingface", {}) or {}
    resolved_repo_id = repo_id or str(hf.get("repo_id") or "").strip()
    resolved_revision = revision or str(hf.get("revision") or "main").strip()
    resolved_filename = str(hf.get("filename") or filename).strip()
    if not resolved_repo_id:
        proposed = str(hf.get("proposed_repo_id") or "").strip()
        if proposed:
            raise FileNotFoundError(
                f"{model_id} is not yet published (proposed host: {proposed}). Pass "
                f"checkpoint_path=... to a local copy (verified against SHA-256 {expected_sha}) "
                f"or place it at {cached_path}."
            )
        raise FileNotFoundError(
            f"No accessible checkpoint found for {model_id}. "
            f"Set repo_id=... after publishing to Hugging Face or keep local_path available."
        )

    cached_path.parent.mkdir(parents=True, exist_ok=True)
    url = _hf_download_url(resolved_repo_id, resolved_revision, resolved_filename)
    urllib.request.urlretrieve(url, cached_path)
    _verify_sha256(cached_path, expected_sha)
    return cached_path


def load_checkpoint_bundle(
    checkpoint_path: Path | str,
    *,
    device: Optional[str | torch.device] = None,
) -> LoadedPretrainedModel:
    """Load a VDC checkpoint into model + optional diffusion objects."""
    from vdc.train.unified_trainer import build_model

    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device_t = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(ckpt_path, map_location=device_t, weights_only=False)
    config = checkpoint.get("config", {})
    model_type = str(
        config.get("model_type")
        or config.get("model", {}).get("type")
        or "denoiser"
    )
    model = build_model(model_type, config, device_t)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", {}))
    model.load_state_dict(state, strict=False)
    model.eval()

    diffusion = None
    if model_type == "diffusion_unet":
        from vdc.models.copula_diffusion import CopulaAwareDiffusion

        diff_cfg = config.get("diffusion", {})
        diffusion = CopulaAwareDiffusion(
            timesteps=int(diff_cfg.get("timesteps", 1000)),
            beta_schedule=str(diff_cfg.get("noise_schedule", "cosine")),
        ).to(device_t)

    return LoadedPretrainedModel(
        model_id="custom-checkpoint",
        checkpoint_path=ckpt_path,
        manifest={},
        model=model,
        diffusion=diffusion,
        config=config,
        device=device_t,
    )


def load_pretrained_model(
    model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    *,
    device: Optional[str | torch.device] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    prefer_local: bool = True,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    checkpoint_path: Optional[Path | str] = None,
) -> LoadedPretrainedModel:
    """Load a packaged pretrained model by id.

    ``checkpoint_path`` loads a local copy after checking it against the
    manifest's SHA-256 (needed for models that are not yet published).
    """
    manifest = load_pretrained_manifest(model_id)
    ckpt = resolve_pretrained_checkpoint(
        model_id,
        cache_dir=cache_dir,
        force_download=force_download,
        prefer_local=prefer_local,
        repo_id=repo_id,
        revision=revision,
        checkpoint_path=checkpoint_path,
    )
    loaded = load_checkpoint_bundle(ckpt, device=device)
    return LoadedPretrainedModel(
        model_id=model_id,
        checkpoint_path=loaded.checkpoint_path,
        manifest=manifest,
        model=loaded.model,
        diffusion=loaded.diffusion,
        config=loaded.config,
        device=loaded.device,
    )


def linear_grid_settings_for(bundle: LoadedPretrainedModel) -> Optional["LinearGridSettings"]:
    """Return the linear-grid inference settings of a bundle, or None.

    Registered models declare them in the manifest's ``inference`` block.
    For a custom checkpoint, the embedded configuration's ``inference`` block
    (written by ``vdc.train.linear_grid_training``) or a research-style
    ``estimator_study`` record with ``representation: linear_nodes`` identifies
    a linear-grid network; the evaluated defaults are then used for its grid size.
    """
    from vdc.models.linear_grid_estimator import LINEAR_GRID_ESTIMATOR, LinearGridSettings

    manifest = bundle.manifest or {}
    inference = manifest.get("inference")
    if inference is None:
        inference = bundle.config.get("inference")
    if inference is not None:
        if inference.get("pair_estimator") != LINEAR_GRID_ESTIMATOR:
            return None
        return LinearGridSettings.from_manifest(inference)
    study = bundle.config.get("estimator_study", {}) or {}
    if study.get("representation") == "linear_nodes" and study.get("loss") == "nll":
        return LinearGridSettings(grid_size=int(bundle.config["data"]["m"]),
                                  grid_spacing=str(study.get("spacing", "uniform")))
    return None


def _require_linear_grid(bundle: LoadedPretrainedModel) -> "LinearGridSettings":
    settings = linear_grid_settings_for(bundle)
    if settings is None:
        raise ValueError(
            f"{bundle.model_id} is not a linear-grid model; use estimate_pair_density_from_samples "
            "and HFuncLookup for the original estimator."
        )
    return settings


def estimate_pair_copulas_from_samples(
    bundle: LoadedPretrainedModel,
    pair_data_list: Sequence[Any],
    *,
    projection_iters: Optional[int] = None,
) -> List["LinearGridCopula"]:
    """Estimate continuous pair copulas for several ``(n, 2)`` samples at once.

    Linear-grid models only. Returns ``LinearGridCopula`` objects (``pdf``,
    ``h_u_given_v``, ``h_v_given_u`` and inverses) built exactly as in the
    models' evaluation; batching does not change any single result for a
    fixed device and ``forward_batch_size``. Points must lie in [0, 1].
    """
    from vdc.models.linear_grid_estimator import LinearGridEstimator

    settings = _require_linear_grid(bundle)
    if projection_iters is not None:
        settings = replace(settings, projection_iters=int(projection_iters))
    pairs = [np.asarray(p, dtype=np.float64) for p in pair_data_list]
    for pair in pairs:
        if pair.ndim != 2 or pair.shape[1] != 2:
            raise ValueError(f"pair_data must have shape (n, 2), got {pair.shape}")
    return LinearGridEstimator(bundle.model, settings, bundle.device).copulas(pairs)


def estimate_pair_copula_from_samples(
    bundle: LoadedPretrainedModel,
    pair_data,
    *,
    projection_iters: Optional[int] = None,
) -> "LinearGridCopula":
    """Estimate one continuous pair copula with a linear-grid model."""
    return estimate_pair_copulas_from_samples(bundle, [pair_data], projection_iters=projection_iters)[0]


def estimate_pair_density_from_samples(
    bundle: LoadedPretrainedModel,
    pair_data,
    *,
    m: Optional[int] = None,
    diffusion_steps: Optional[int] = None,
    cfg_scale: float = 1.0,
    projection_iters: Optional[int] = None,
) -> Any:
    """Estimate a bivariate copula density from pseudo-observations using a loaded bundle.

    Returns an ``(m, m)`` density grid. For the original models this is the
    projected cell-average grid (``projection_iters`` defaults to 50). For
    linear-grid models (``vdc-denoiser-m64-v2``/``-m128-v2``) it is the
    float32 node-value grid of the evaluated inference (``projection_iters``
    defaults to the manifest value, 30); use
    :func:`estimate_pair_copula_from_samples` for its PDF and h-functions.
    """
    from vdc.inference.density import sample_density_grid, scatter_to_hist
    from vdc.models.projection import copula_project

    pair_data = np.asarray(pair_data, dtype=np.float64)
    if pair_data.ndim != 2 or pair_data.shape[1] != 2:
        raise ValueError(f"pair_data must have shape (n, 2), got {pair_data.shape}")

    settings = linear_grid_settings_for(bundle)
    if settings is not None:
        if m is not None and int(m) != settings.grid_size:
            raise ValueError(f"This model has a fixed grid size of {settings.grid_size}")
        if diffusion_steps is not None or cfg_scale != 1.0:
            raise ValueError("Linear-grid models are single-pass; diffusion options do not apply")
        from vdc.models.linear_grid_estimator import LinearGridEstimator

        if projection_iters is not None:
            settings = replace(settings, projection_iters=int(projection_iters))
        return LinearGridEstimator(bundle.model, settings, bundle.device).estimate_grids([pair_data])[0]

    projection_iters = 50 if projection_iters is None else projection_iters
    m_eff = int(m or bundle.config.get("data", {}).get("m", 64))
    if bundle.diffusion is not None:
        use_histogram_conditioning = False
        conv_in = getattr(bundle.model, "conv_in", None)
        if conv_in is not None and hasattr(conv_in, "in_channels"):
            use_histogram_conditioning = int(conv_in.in_channels) > 1
        return sample_density_grid(
            model=bundle.model,
            diffusion=bundle.diffusion,
            samples=pair_data,
            m=m_eff,
            device=bundle.device,
            num_steps=int(diffusion_steps or 50),
            cfg_scale=float(cfg_scale),
            use_histogram_conditioning=use_histogram_conditioning,
            projection_iters=int(projection_iters),
        )

    hist = scatter_to_hist(pair_data, m=m_eff, reflect=True)
    hist_t = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0).to(bundle.device)
    transform_to_probit_space = bool(getattr(bundle.model, "vdc_transform_to_probit_space", False))
    if transform_to_probit_space:
        from vdc.utils.probit_transform import copula_logdensity_to_probit_logdensity

        log_hist = torch.log(hist_t.clamp_min(1e-12))
        hist_t = torch.exp(copula_logdensity_to_probit_logdensity(log_hist, m_eff)).clamp(
            min=1e-12, max=1e6
        )

    in_ch: Optional[int] = None
    if hasattr(bundle.model, "in_conv"):
        in_ch = int(getattr(bundle.model, "in_conv").in_channels)
    elif hasattr(bundle.model, "input_conv"):
        try:
            in_ch = int(getattr(bundle.model, "input_conv")[0].in_channels)
        except Exception:
            in_ch = None
    elif hasattr(bundle.model, "conv_in"):
        in_ch = int(getattr(bundle.model, "conv_in").in_channels)

    x = hist_t
    use_log_n = bool(getattr(bundle.model, "vdc_use_log_n", False))
    use_coords = bool(getattr(bundle.model, "vdc_use_coordinates", False))
    use_probit_coords = bool(getattr(bundle.model, "vdc_use_probit_coords", False))
    probit_coord_eps = float(getattr(bundle.model, "vdc_probit_coord_eps", 1e-4))

    if in_ch is not None and not hasattr(bundle.model, "vdc_use_log_n") and in_ch in (2, 4):
        use_log_n = True
    if in_ch is not None and not hasattr(bundle.model, "vdc_use_coordinates") and in_ch in (3, 4):
        use_coords = True

    if use_log_n:
        ln = float(np.log(max(1, pair_data.shape[0])))
        ln_chan = torch.full((1, 1, m_eff, m_eff), ln, device=bundle.device, dtype=x.dtype)
        x = torch.cat([x, ln_chan], dim=1)

    if use_coords:
        u = torch.linspace(0.5 / m_eff, 1.0 - 0.5 / m_eff, m_eff, device=bundle.device, dtype=x.dtype)
        v = torch.linspace(0.5 / m_eff, 1.0 - 0.5 / m_eff, m_eff, device=bundle.device, dtype=x.dtype)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        if use_probit_coords:
            eps = max(probit_coord_eps, 1.0 / (m_eff * m_eff))
            uu = torch.erfinv(2 * uu.clamp(eps, 1 - eps) - 1) * (2.0 ** 0.5)
            vv = torch.erfinv(2 * vv.clamp(eps, 1 - eps) - 1) * (2.0 ** 0.5)
        coords = torch.stack([uu, vv], dim=0).unsqueeze(0)
        x = torch.cat([x, coords], dim=1)

    if in_ch is not None and x.shape[1] != in_ch:
        raise ValueError(
            f"Input channel mismatch: built x has C={x.shape[1]} but model expects C_in={in_ch}"
        )

    try:
        out = bundle.model(x, torch.zeros(1, device=bundle.device))
    except TypeError:
        out = bundle.model(x)

    if isinstance(out, dict):
        if "density" in out:
            d = out["density"]
        elif "log_density" in out:
            d = torch.exp(out["log_density"].clamp(min=-20, max=20))
        elif "residual" in out:
            d = torch.exp((torch.log(hist_t.clamp_min(1e-12)) + out["residual"]).clamp(min=-20, max=20))
        else:
            raise ValueError(f"Unknown model output keys: {list(out.keys())}")
    else:
        d = out

    if transform_to_probit_space:
        from vdc.utils.probit_transform import probit_logdensity_to_copula_logdensity

        logd = torch.log(d.clamp_min(1e-12))
        logc = probit_logdensity_to_copula_logdensity(logd, m_eff)
        d = torch.exp(logc.clamp(min=-20, max=20)).clamp(min=1e-12, max=1e6)

    d = torch.nan_to_num(d, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=1e-12, max=1e6)
    du = 1.0 / m_eff
    d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
    d = copula_project(d, iters=int(projection_iters))
    d = d / ((d * du * du).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
    return d[0, 0].detach().cpu().numpy()


def stage_release_bundle(
    model_id: str,
    *,
    out_dir: Path | str,
    repo_id: Optional[str] = None,
    revision: str = "main",
    include_local_paths: bool = False,
    checkpoint_path: Optional[Path | str] = None,
) -> Path:
    """Stage a publishable release bundle directory for a pretrained model.

    ``checkpoint_path`` stages a local copy (checked against the manifest's
    SHA-256) of a model that cannot yet be resolved from its public host.
    """
    manifest = load_pretrained_manifest(model_id)
    checkpoint_path = resolve_pretrained_checkpoint(model_id, prefer_local=True,
                                                    checkpoint_path=checkpoint_path)
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = str(manifest.get("checkpoint_filename", "")).strip() or "model.pt"
    staged_checkpoint = out_root / checkpoint_filename
    if staged_checkpoint.resolve() != checkpoint_path.resolve():
        shutil.copy2(checkpoint_path, staged_checkpoint)

    sources = manifest.get("sources", {})
    # An empty entry would otherwise resolve to the current directory.
    train_cfg_raw = str(sources.get("local_train_config", "") or "").strip()
    train_cfg_path = Path(train_cfg_raw).expanduser()
    if train_cfg_raw and train_cfg_path.is_file():
        shutil.copy2(train_cfg_path, out_root / "train_config.yaml")

    ms_raw = str(sources.get("local_model_selection_joint", "") or "").strip()
    ms_path = Path(ms_raw).expanduser()
    if ms_raw and ms_path.is_file():
        shutil.copy2(ms_path, out_root / "model_selection_joint_best.json")

    public_manifest = json.loads(json.dumps(manifest))
    public_manifest.setdefault("sources", {})
    public_manifest["sources"]["huggingface"] = {
        "repo_id": repo_id,
        "revision": revision,
        "filename": checkpoint_filename,
    }
    if not include_local_paths:
        public_manifest["sources"]["local_checkpoint"] = None
        public_manifest["sources"]["local_train_config"] = None
        public_manifest["sources"]["local_model_selection_joint"] = None
    (out_root / "manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n")
    return out_root

# ruff: noqa: E702  (verbatim port of the research implementation)
"""Fine-tuning data and training loop of the linear-grid models.

``vdc-denoiser-m64-v2`` and ``vdc-denoiser-m128-v2`` were fine-tuned for 8,000
steps from the pilot (:mod:`vdc.train.linear_grid_pretraining`) with fresh
continuous-mixture populations (:mod:`vdc.data.continuous_mixtures`):

* Data. Optimizer step ``s`` draws one sample size ``n`` from the palette with
  ``SeedSequence([seed, s, 23])``. Example ``i = s * batch + k`` draws its
  population, its ``n`` observations and 512 independent target points from
  the three words of ``SeedSequence([seed, i, 11])``. The input histogram
  counts rank pseudo-observations (average ranks / (n + 1)) on uniform cells;
  with probability ``latent_probability`` (``SeedSequence([seed, s, 71])``) the
  unranked latent uniforms of the same observations are counted instead.
  :func:`build_training_cache` stores these counts losslessly once for all
  grid sizes; :class:`CachedContinuousBatches` reads them back.
* Objective. The network output is clamped to [-15, 15], exponentiated,
  projected for 30 iterations with integrated basis weights and evaluated
  bilinearly at the target points; the loss is their mean negative log
  density. Corruption: ``x = (1 - .65 t) h + .65 t``, ``t ~ U(0, 1)``.
* Optimizer. AdamW (betas .9/.99, weight decay .01), peak learning rate 5e-5,
  200 warmup steps then cosine decay to 1% of the peak, gradient-norm clip 1,
  EMA decay .999, batch 128 in microbatches of 32, bf16 autocast with float32
  projection and loss, TF32 disabled.

Use :func:`run_finetuning` (``scripts/train_linear_grid_estimator.py`` with
``configs/train/linear_grid_vdc64_v2.yaml`` or ``..._vdc128_v2.yaml``). The
recorded numerics require CUDA; CPU execution is supported for tests only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset

from ..data.continuous_mixtures import ContinuousMixture, draw_spec
from ..models.linear_grid_copula import basis_weights
from ..models.linear_grid_estimator import (
    LinearGridSettings,
    bilinear_density_torch,
    grid_geometry,
    project_density_torch,
)

DEFAULT_SAMPLE_SIZES = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
TARGET_POINTS = 512


def _sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _grid_key(m: int) -> str:
    return f"u{int(m)}"


def build_cache_batch(step: int, batch: int, seed: int, sizes: Sequence[int], output_dir,
                      grid_sizes: Sequence[int] = (64, 128), latent_histograms: bool = True) -> Dict[str, Any]:
    """Generate optimizer step ``step`` and write ``batch_{step:05d}.npz``.

    Arrays: ``u{m}`` (and ``latent_u{m}``) uint16 counts per grid size,
    ``target_points`` (batch, 512, 2) float32, ``effective_components``,
    ``components``, ``n`` and global example ``indices``.
    """
    n = int(np.random.default_rng(np.random.SeedSequence([seed, step, 23])).choice(sizes))
    keys = [(_grid_key(m), int(m)) for m in grid_sizes]
    edges = {key: grid_geometry(m, "uniform")[0] for key, m in keys}
    counts = {key: np.empty((batch, m, m), dtype=np.uint16) for key, m in keys}
    if latent_histograms:
        counts.update({"latent_"+key: np.empty((batch, m, m), dtype=np.uint16) for key, m in keys})
    targets = np.empty((batch, TARGET_POINTS, 2), dtype=np.float32)
    effective = np.empty(batch); components = np.empty(batch, dtype=np.int16)
    for k in range(batch):
        index = step*batch+k
        population_seed, sample_seed, target_seed = np.random.SeedSequence([seed, index, 11]).generate_state(3)
        spec = draw_spec(np.random.default_rng(int(population_seed)))
        truth = ContinuousMixture(spec)
        raw = truth.simulate(n, int(sample_seed))
        if not np.isfinite(raw).all():
            raise ValueError(f"Invalid samples at index {index}")
        ranks = rankdata(raw, axis=0)/(n+1.)
        for key, _ in keys:
            h = np.histogram2d(ranks[:, 0], ranks[:, 1], bins=(edges[key], edges[key]))[0]
            if h.sum() != n or h.max() > 65535:
                raise ValueError("Histogram count loss")
            counts[key][k] = h.astype(np.uint16)
            if latent_histograms:
                h = np.histogram2d(raw[:, 0], raw[:, 1], bins=(edges[key], edges[key]))[0]
                if h.sum() != n or h.max() > 65535:
                    raise ValueError("Latent histogram count loss")
                counts["latent_"+key][k] = h.astype(np.uint16)
        targets[k] = truth.simulate(TARGET_POINTS, int(target_seed))
        if not np.isfinite(targets[k]).all():
            raise ValueError(f"Invalid target observations at index {index}")
        effective[k] = spec["effective_components"]; components[k] = len(spec["components"])
    path = Path(output_dir)/f"batch_{step:05d}.npz"
    np.savez_compressed(path, **counts, target_points=targets, effective_components=effective,
                        components=components, n=np.full(batch, n, dtype=np.int32),
                        indices=np.arange(step*batch, (step+1)*batch))
    return {"step": step, "n": n, "path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _build_cache_task(task):
    step, batch, seed, sizes, output_dir, grid_sizes, latent = task
    return build_cache_batch(step, batch, seed, sizes, output_dir, grid_sizes, latent)


CACHE_DEFAULTS: Dict[str, Any] = {
    "output_dir": None, "steps": 8000, "batch_size": 128, "seed": 202610091,
    "sizes": list(DEFAULT_SAMPLE_SIZES), "grid_sizes": [64, 128], "latent_histograms": True,
    "workers": 24,
}


def build_training_cache(settings: Dict[str, Any]) -> Path:
    """Write all fine-tuning batches and a ``receipt.json`` with their SHA-256.

    The cache contents depend only on ``seed``, ``batch_size``, ``sizes``,
    ``grid_sizes`` and ``latent_histograms``, not on ``workers``.
    """
    unknown = set(settings) - set(CACHE_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown cache settings: {sorted(unknown)}")
    a = {**CACHE_DEFAULTS, **settings}
    output = Path(a["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    import pyvinecopulib as pv
    import scipy
    steps, batch, seed = int(a["steps"]), int(a["batch_size"]), int(a["seed"])
    sizes = [int(s) for s in a["sizes"]]
    grid_sizes = [int(m) for m in a["grid_sizes"]]
    receipt = {"status": "running", "format": "vdc-linear-grid-cache-1", "seed": seed, "batch": batch,
               "steps": steps, "sizes": sizes, "grid_sizes": grid_sizes,
               "latent_histograms": bool(a["latent_histograms"]), "workers": int(a["workers"]),
               "batches": [],
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "pyvinecopulib": pv.__version__,
                               "dtype": "uint16 histogram counts; float32 targets; float64 generation"}}
    _write_json(output/"receipt.json", receipt)
    start = time.perf_counter()
    tasks = [(i, batch, seed, sizes, str(output), grid_sizes, bool(a["latent_histograms"])) for i in range(steps)]
    try:
        if int(a["workers"]) <= 1:
            for task in tasks:
                receipt["batches"].append(_build_cache_task(task))
        else:
            with ProcessPoolExecutor(max_workers=int(a["workers"])) as pool:
                iterator = iter(tasks)
                pending = {pool.submit(_build_cache_task, t) for _, t in zip(range(int(a["workers"])*2), iterator)}
                try:
                    while pending:
                        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            receipt["batches"].append(future.result())
                            task = next(iterator, None)
                            if task is not None:
                                pending.add(pool.submit(_build_cache_task, task))
                        if len(receipt["batches"]) % 20 < len(completed):
                            _write_json(output/"receipt.json", receipt)
                            print(len(receipt["batches"]), "batches", time.perf_counter()-start, "seconds", flush=True)
                except Exception:
                    for future in pending:
                        future.cancel()
                    raise
        receipt["batches"].sort(key=lambda r: r["step"])
        receipt["status"] = "complete"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter()-start
        _write_json(output/"receipt.json", receipt)
    return output


class CachedContinuousBatches(Dataset):
    """Optimizer batches read from a training cache, with latent-input mixing.

    Each item is one optimizer step: ``hist`` (B, m, m) float32 densities
    (counts divided by ``n`` times the cell area), ``target_points``, ``n``,
    ``effective_components`` and the boolean ``latent_mask``. Shards are
    verified against the receipt's SHA-256.
    """

    def __init__(self, root, steps: int, batch: int, seed: int, grid_size: int,
                 latent_probability: float = 0.):
        self.root = Path(root); self.receipt = json.loads((self.root/"receipt.json").read_text())
        r = self.receipt
        if r["status"] != "complete" or r["steps"] < steps or r["batch"] != batch or r["seed"] != seed:
            raise ValueError("Training cache does not match this experiment")
        self.steps = steps; self.key = _grid_key(grid_size)
        edges, _ = grid_geometry(grid_size, "uniform"); self.area = np.outer(np.diff(edges), np.diff(edges))
        self.seed = seed
        if not 0 <= latent_probability <= 1:
            raise ValueError("Invalid latent-input probability")
        if latent_probability and not r.get("latent_histograms", False):
            raise ValueError("Latent-input mixture requires a cache with latent histograms")
        self.latent_probability = latent_probability
        self._shards = {int(row["step"]): row for row in r["batches"]}

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        r = self._shards[index]; path = self.root/r["path"]
        if _sha256(path) != r["sha256"]:
            raise ValueError("Training shard hash mismatch")
        with np.load(path) as z:
            histogram = z[self.key].copy()
            latent_mask = np.zeros(len(z['n']), dtype=bool)
            if self.latent_probability:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 71]))
                latent_mask = rng.random(len(latent_mask)) < self.latent_probability
                histogram[latent_mask] = z['latent_'+self.key][latent_mask]
            result = {"hist": (histogram/(z["n"][:, None, None]*self.area)).astype(np.float32),
                      "target_points": z["target_points"], "n": z["n"].astype(np.int64),
                      "effective_components": z["effective_components"], "latent_mask": latent_mask}
        return result


FINETUNING_DEFAULTS: Dict[str, Any] = {
    "output_dir": None, "initial_checkpoint": None, "initial_sha256": None, "cache_dir": None,
    "grid_size": 64, "latent_probability": 0.5, "base_channels": 128, "steps": 8000,
    "batch_size": 128, "microbatch": 32, "workers": 2, "seed": 202610091,
    "sizes": list(DEFAULT_SAMPLE_SIZES), "save_every": 8000, "save_steps": [],
    "stop_after": None, "log_every": 50, "device": "cuda", "cpu_threads": 4,
}


def run_finetuning(settings: Dict[str, Any]) -> Path:
    """Fine-tune a pilot checkpoint with the continuous-mixture likelihood.

    Settings (defaults in ``FINETUNING_DEFAULTS``): ``output_dir`` (new),
    ``initial_checkpoint`` and optional ``initial_sha256``, ``cache_dir`` (from
    :func:`build_training_cache`, same ``seed``/``batch_size``/``sizes``),
    ``grid_size``, ``latent_probability``, ``base_channels``, ``steps``,
    ``batch_size``, ``microbatch``, ``workers`` (data-loader processes, at
    most 2), ``seed``, ``save_every``, ``save_steps``, ``stop_after`` (end
    early without changing the schedule), ``log_every``, ``device`` and
    ``cpu_threads`` (``torch.set_num_threads``; 4 in the recorded runs).
    Writes ``receipt.json``, ``train.jsonl`` and EMA checkpoints
    ``ema_step{N}.pt`` loadable with :func:`vdc.pretrained.load_checkpoint_bundle`.
    """
    from ..train.unified_trainer import build_model

    unknown = set(settings) - set(FINETUNING_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown fine-tuning settings: {sorted(unknown)}")
    a = {**FINETUNING_DEFAULTS, **settings}
    for key in ("output_dir", "initial_checkpoint", "cache_dir"):
        if a[key] is None:
            raise ValueError(f"{key} is required")
    output, initial, cache = Path(a["output_dir"]), Path(a["initial_checkpoint"]), Path(a["cache_dir"])
    m, steps, batch, microbatch = int(a["grid_size"]), int(a["steps"]), int(a["batch_size"]), int(a["microbatch"])
    seed, sizes = int(a["seed"]), [int(s) for s in a["sizes"]]
    latent_probability, width = float(a["latent_probability"]), int(a["base_channels"])
    save_steps = {int(s) for s in a["save_steps"]}
    stop_after = None if a["stop_after"] is None else int(a["stop_after"])
    log_every = int(a["log_every"])
    if batch % microbatch:
        raise ValueError("Microbatch must divide effective batch")
    if stop_after is not None and not 0 < stop_after <= steps:
        raise ValueError("stop_after must be within the full schedule")
    initial_sha = _sha256(initial)
    if a["initial_sha256"] and initial_sha != a["initial_sha256"]:
        raise ValueError("Initial checkpoint hash mismatch")
    device = torch.device(a["device"])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(a["cpu_threads"]))
    torch.manual_seed(seed)
    ds = CachedContinuousBatches(cache, steps, batch, seed, m, latent_probability)
    if ds.receipt["sizes"] != sizes:
        raise ValueError("Sample-size palette differs from cache")
    loader = DataLoader(ds, batch_size=None, num_workers=min(int(a["workers"]), 2),
                        pin_memory=device.type == "cuda")
    cache_receipt_sha = _sha256(cache/"receipt.json")
    import pyvinecopulib
    import scipy
    settings_record = {k: str(v) if isinstance(v, Path) else v for k, v in a.items()}
    settings_record["save_steps"] = sorted(save_steps)
    receipt: Dict[str, Any] = {
        "trainer": "vdc.train.linear_grid_training.run_finetuning", "initial_sha256": initial_sha,
        "settings": settings_record, "cache_receipt_sha256": cache_receipt_sha,
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                     "pyvinecopulib": pyvinecopulib.__version__, "python": platform.python_version()},
        "status": "preflight",
        "population_seed_rule": "SeedSequence([seed, example_index, 11]); first word population, second observations, third targets"}
    _write_json(output/"receipt.json", receipt)
    ck = torch.load(initial, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(ck["config"])
    cfg["data"] = {"m": m, "source": "continuous_mixtures", "population_definition": "vdc.data.continuous_mixtures",
                   "sample_sizes": sizes, "target_samples_per_population": TARGET_POINTS,
                   "pseudo_observations": "average ranks / (n+1)", "seed": seed}
    if latent_probability:
        cfg["data"].update(pseudo_observations="Paired rank or unranked latent uniforms; same population, observations and targets",
                           latent_probability=latent_probability)
    cfg["model"]["base_channels"] = width
    cfg["model"]["grid_size"] = m
    cfg["inference"] = LinearGridSettings(grid_size=m).to_dict()
    model = build_model("denoiser", cfg, torch.device("cpu"))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    del ck
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.to(device); ema = copy.deepcopy(model).eval()
    for par in ema.parameters():
        par.requires_grad_(False)
    peak_lr, ema_decay = 5e-5, .999
    cfg["training"] = {"steps": steps, "batch_size": batch, "microbatch": microbatch, "peak_lr": peak_lr,
                       "optimizer": "AdamW", "betas": [.9, .99], "weight_decay": .01, "warmup": 200, "schedule": "cosine",
                       "ema": ema_decay, "gradient_clip_norm": 1., "precision": "BF16 autocast with float32 projection/loss",
                       "objective": "nll", "initial_sha256": initial_sha}
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=.01, betas=(.9, .99))
    _, node_array = grid_geometry(m, "uniform")
    nodes = torch.tensor(node_array, dtype=torch.float32, device=device)
    w = torch.tensor(basis_weights(node_array), dtype=torch.float32, device=device)
    coord = torch.tensor(np.stack(np.meshgrid(ndtri(node_array), ndtri(node_array), indexing="ij")),
                         dtype=torch.float32, device=device)[None]
    rng = torch.Generator(device=device); rng.manual_seed(seed+123)
    counts = {str(n): 0 for n in sizes}
    input_counts = {"rank": 0, "latent": 0}
    input_counts_by_n = {str(n): {"rank": 0, "latent": 0} for n in sizes}
    receipt.update(status="running", device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                   slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                   parameters=sum(p.numel() for p in model.parameters()), config=cfg,
                   peak_lr=peak_lr, ema_decay=ema_decay, checkpoints=[])
    _write_json(output/"receipt.json", receipt)
    start = time.perf_counter(); last = start
    step = 0
    with (output/"train.jsonl").open("x") as log:
        for step, item in enumerate(loader, 1):
            data_seconds = time.perf_counter()-last
            hist = item["hist"].to(device, non_blocking=True); points = item["target_points"].to(device, non_blocking=True)
            n = item["n"].to(device, non_blocking=True)
            counts[str(int(item["n"][0]))] += len(n)
            latent_count = int(item["latent_mask"].sum())
            input_counts["latent"] += latent_count; input_counts["rank"] += len(n)-latent_count
            if not bool((item["n"] == item["n"][0]).all()):
                raise ValueError("Expected one sample size per cached batch")
            counts_for_n = input_counts_by_n[str(int(item["n"][0]))]
            counts_for_n["latent"] += latent_count; counts_for_n["rank"] += len(n)-latent_count
            t = torch.rand(len(n), generator=rng, device=device)
            inp_hist = (1-.65*t[:, None, None])*hist+.65*t[:, None, None]
            ln = n.float().log()[:, None, None, None].expand(-1, 1, m, m)
            x = torch.cat([inp_hist[:, None], ln, coord.expand(len(n), -1, -1, -1)], 1)
            lr = peak_lr*min(step/200, 1.)*(.01+.99*.5*(1+math.cos(math.pi*max(0, (step-200)/max(1, steps-200)))))
            for group in opt.param_groups:
                group["lr"] = lr
            model.train(); opt.zero_grad(set_to_none=True); value = 0.; nll_value = 0.
            for lo in range(0, batch, microbatch):
                hi = lo+microbatch
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    out = model(x[lo:hi], t[lo:hi])["log_density"]
                raw = out[:, 0].float().clamp(-15, 15).exp()
                pred = project_density_torch(raw, w)
                nll = -bilinear_density_torch(pred, nodes, points[lo:hi]).clamp_min(1e-12).log().mean()
                loss = nll
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                (loss*microbatch/batch).backward()
                value += float(loss.detach())*microbatch/batch
                nll_value += float(nll.detach())*microbatch/batch
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1-ema_decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            if step == 1 or step % log_every == 0:
                row = dict(step=step, n=int(item["n"][0]), loss=value, nll=nll_value, gradient_norm=float(grad), lr=lr,
                           data_wait_seconds=data_seconds, seconds_per_step=(time.perf_counter()-start)/step,
                           gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                           mean_effective_components=float(item["effective_components"].mean()),
                           input_type_counts=dict(input_counts), input_type_counts_by_sample_size=input_counts_by_n)
                log.write(json.dumps(row)+"\n"); log.flush(); print(json.dumps(row), flush=True)
            if step % int(a["save_every"]) == 0 or step in save_steps or step == steps or step == stop_after:
                path = output/f"ema_step{step}.pt"
                torch.save({"model_state_dict": ema.state_dict(), "config": cfg, "step": step}, path)
                receipt["checkpoints"].append({"path": path.name, "sha256": _sha256(path), "step": step})
                receipt["sample_size_counts"] = counts; receipt["input_type_counts"] = dict(input_counts)
                receipt["input_type_counts_by_sample_size"] = input_counts_by_n
                _write_json(output/"receipt.json", receipt)
            last = time.perf_counter()
            if step == stop_after:
                break
    receipt.update(status="complete" if step == steps else "stopped_at_requested_step", completed_step=step,
                   elapsed_seconds=time.perf_counter()-start)
    _write_json(output/"receipt.json", receipt)
    return output


STAGES = {"pretrain", "cache", "finetune"}


def run_stage(config: Dict[str, Any]) -> Path:
    """Dispatch a configuration dictionary with a ``stage`` key."""
    from .linear_grid_pretraining import run_pretraining
    config = dict(config)
    stage = config.pop("stage", None)
    config.pop("description", None)
    config.pop("expected", None)
    if stage == "pretrain":
        return run_pretraining(config)
    if stage == "cache":
        return build_training_cache(config)
    if stage == "finetune":
        return run_finetuning(config)
    raise ValueError(f"stage must be one of {sorted(STAGES)}, got {stage!r}")


def _parse_value(text: str):
    import yaml
    return yaml.safe_load(text)


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``key=value`` overrides (values parsed as YAML) to a flat configuration."""
    config = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        config[key.strip()] = _parse_value(value)
    return config


def load_config(path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    import yaml
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return apply_overrides(config, overrides)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    import argparse
    parser = argparse.ArgumentParser(
        description="Pretrain, cache data for, or fine-tune a linear-grid VDC model.",
        epilog="Example: train_linear_grid_estimator.py configs/train/linear_grid_vdc64_v2.yaml "
               "output_dir=runs/vdc64 initial_checkpoint=runs/pilot/pilot_ema_step60000.pt")
    parser.add_argument("config", type=Path, help="YAML file with a 'stage' of pretrain, cache or finetune.")
    parser.add_argument("overrides", nargs="*", help="key=value settings that replace configuration entries.")
    args = parser.parse_args(argv)
    result = run_stage(load_config(args.config, args.overrides))
    print(result, flush=True)
    return result

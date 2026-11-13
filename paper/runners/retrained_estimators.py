#!/usr/bin/env python3
"""Recompute the retrained-estimator appendix: Tables A40 and A41 and chronological S&P100.

Evaluates the exploratory linear-grid checkpoints vdc-denoiser-m64-v2 (VDC-64) and
vdc-denoiser-m128-v2 (VDC-128) with the settings of the appendix, through the library's
pretrained-model API (VineCopulaModel.from_pretrained and the linear-grid pair estimator):

  table_a40  nine Table 1 cases x five seeds, 2,000 training and 20,000 test observations
             per fit, all 45 fits estimated in one call; held-out KL per fit (simulated data)
  table_a41  greedy D-vines on the paper's UCI inputs (MAF splits) at 5,000 and 50,000
             training rows: NLL, PIT-KS over all test rows, 5,000 generated rows (--data-root)
  sp100      the chronological S&P100 split, 15 projection iterations (--returns: the paper's
             matrix, which is not redistributed); with --original-logpdf, paired differences
             from the original model by the appendix's moving-block bootstrap

Every recomputed value is compared with paper/reference/retrained/*.json, which holds the
appendix's values and SHA-256 digests of its per-row arrays. Summary values must agree within
the tolerances in TOLERANCES on any device; the exit status is 1 otherwise. On one H100 80GB
with the pinned numerical environment, the per-row arrays reproduced the appendix's bitwise
(reported per array; --require-bitwise also makes that a condition). The original model, TLL
and parametric-selection rows of the tables are the paper's records, not recomputed here.

Writes into a new --output directory: results.json (settings, versions, hashes, values and
comparisons), per-row arrays (.npz) and, when complete, the two tables in the paper's layout.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

MODELS = {"vdc-denoiser-m64-v2": "vdc64", "vdc-denoiser-m128-v2": "vdc128"}
CASES = [("indep", [], 0), ("gaussian", [[.7]], 0), ("gaussian", [[-.7]], 0), ("student", [[.7], [5.]], 0),
         ("clayton", [[3.]], 0), ("clayton", [[3.]], 90), ("gumbel", [[2.5]], 0), ("joe", [[3.]], 0),
         ("frank", [[5.]], 0)]
SEEDS = [101, 202, 303, 404, 505]
UCI_DATASETS = ["power", "gas", "hepmass", "miniboone"]
# SHA-256 of the float64 little-endian bytes of the paper's MAF train and test arrays,
# as checked by run_uci_maf.py (preprocessed MAF splits of Zenodo record 1161203).
MAF_ARRAY_SHA256 = {
    "power": {"train": "a3c6eee0ffee8de3059925f1778afc1e09a384c14503b1e5e1dc5f186c46dcad",
              "test": "68dab933584464a90156b733f20d907e83462fb06c100a1c2ee8b70068ec7d69"},
    "gas": {"train": "70136e10e909ed59bfdead8e8251ecf9925ec5215abd1e8852c1f2ebd3ddab05",
            "test": "c52361fc20ff1dc236cae616dc689f8f1789843bd461d4d89e22d5212459f211"},
    "hepmass": {"train": "7eb15894622c90eb1e1b4433521feb3d81ffae834e7e07030b6918f0601a9a96",
                "test": "613a470cdc390f3ca3bd4aa22c3d1b9eb3cf755e2fd6a8010b6370491134d423"},
    "miniboone": {"train": "aafbb8e1cb55685729ce36329d159f6b8f5bf869c4b61a1651cae41a0fba0f19",
                  "test": "ff61c966cd4f99049eecf6e2ecb880ba52afe3605f6b925f406efcf418ef30b7"},
}
SP100_RETURNS_SHA256 = "5a2accb5e518b498464ea11732fa6bead1b1c34a22cc28ae95d719964ca1e476"
# Absolute tolerances. An A100 40GB rerun differed from the appendix's H100 values by at most
# 4.4e-8 nats (per-fit KL), 3.5e-5 bits/dim (UCI NLL), 8.8e-5 (UCI PIT-KS) and 2.3e-4 bits/dim
# (S&P100 NLL); the smallest per-fit margin between a retrained model and the original was
# 4.2e-5 nats, so these tolerances cannot change a Table A40 count.
TOLERANCES = {"a40_fit_kl_nats": 1e-6, "a40_mean_kl_nats": 1e-7, "a41_nll_bits_dim": 2e-4,
              "a41_pit_ks": 5e-4, "sp100_nll_bits_dim": 1e-3, "sp100_pit_ks": 5e-4, "roundtrip": 1e-7}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(x, dtype="<f8"):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x), dtype=dtype).tobytes()).hexdigest()


class Checks:
    """Collects tolerance checks (pass/fail) and bitwise comparisons (reported)."""

    def __init__(self):
        self.items, self.bitwise = [], {}

    def value(self, name, actual, expected, tolerance):
        difference = abs(float(actual) - float(expected))
        self.items.append(dict(name=name, actual=float(actual), expected=float(expected),
                               abs_difference=difference, tolerance=tolerance, passed=bool(difference <= tolerance)))

    def equal(self, name, actual, expected):
        self.items.append(dict(name=name, actual=actual, expected=expected, passed=bool(actual == expected)))

    def digest(self, name, array, expected, dtype="<f8"):
        actual = array_sha256(array, dtype)
        self.bitwise[name] = bool(actual == expected)
        return actual

    @property
    def passed(self):
        return all(item["passed"] for item in self.items)


def nll_bits(logpdf, d):
    import numpy as np
    return float((-float(np.mean(logpdf))) / (max(1, int(d)) * np.log(2.0)))


def pit_ks(pit):
    import numpy as np
    from scipy.stats import kstest
    return float(np.mean([kstest(pit[:, j], "uniform").statistic for j in range(pit.shape[1])]))


def load_models(args):
    from vdc.pretrained import load_pretrained_model
    paths = {}
    for item in args.checkpoint:
        model_id, sep, path = item.partition("=")
        if not sep or model_id not in MODELS:
            raise SystemExit(f"--checkpoint expects MODEL_ID=PATH with MODEL_ID in {sorted(MODELS)}, got {item!r}")
        paths[model_id] = Path(path)
    bundles = {}
    for model_id in args.models:
        # The loader checks the file against the SHA-256 of the packaged manifest.
        bundles[model_id] = load_pretrained_model(model_id, checkpoint_path=paths.get(model_id), device=args.device)
    return bundles


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def run_table_a40(args, bundles, reference, table1, checks, out):
    import numpy as np
    import pyvinecopulib as pv

    from vdc.models.linear_grid_estimator import LinearGridEstimator
    from vdc.pretrained import linear_grid_settings_for
    pairs, tests, truths = [], [], []
    original = [r["methods"]["vdc"]["kl"] for r in table1["records"]]
    for family, params, rotation in CASES:
        truth = (pv.Bicop(family=pv.BicopFamily.indep) if family == "indep" else
                 pv.Bicop(family=getattr(pv.BicopFamily, family), rotation=rotation,
                          parameters=np.asarray(params, dtype=float)))
        for seed in SEEDS:
            record = table1["records"][len(pairs)]
            assert record["seed"] == seed and record["rotation"] == rotation
            train = np.clip(truth.simulate(2000, seeds=[seed, seed + 1]), 1e-10, 1 - 1e-10)
            test = np.clip(truth.simulate(20000, seeds=[seed + 777777, seed + 777778]), 1e-10, 1 - 1e-10)
            logtrue = np.log(np.maximum(truth.pdf(test), 1e-300))
            checks.value(f"a40 oracle NLL {record['case']} seed {seed}", -logtrue.mean(), record["nll_oracle"], 1e-12)
            pairs.append(train)
            tests.append(test)
            truths.append(logtrue)
    inputs = reference["inputs"]
    for name, array in (("train", np.stack(pairs)), ("test", np.stack(tests)), ("logtrue", np.stack(truths))):
        checks.equal(f"a40 input {name} sha256", array_sha256(array), inputs[f"{name}_sha256"])
    result = {"models": {}}
    for model_id, bundle in bundles.items():
        expected = reference["models"][model_id]
        estimator = LinearGridEstimator(bundle.model, linear_grid_settings_for(bundle), bundle.device)
        estimator.estimate_grids(pairs[:16])  # warm-up, as in the appendix's timing protocol
        timings, grids = [], None
        for _ in range(args.timing_repeats):
            synchronize(args.device)
            start = time.perf_counter()
            batch = estimator.estimate_grids(pairs)
            copulas = [estimator.copula(grid) for grid in batch]
            synchronize(args.device)
            timings.append(time.perf_counter() - start)
            if grids is None:
                grids = np.stack(batch)
            elif not np.array_equal(grids, np.stack(batch)):
                raise SystemExit(f"{model_id}: repeated estimates differ on this device")
        logpdf, kl = [], []
        for copula, test, logtrue in zip(copulas, tests, truths):
            lp = np.log(np.maximum(copula.pdf(test[:, 0], test[:, 1]), 1e-300))
            if not np.isfinite(lp).all():
                raise SystemExit(f"{model_id}: nonfinite log density")
            logpdf.append(lp)
            kl.append(float(np.mean(logtrue - lp)))
        logpdf = np.stack(logpdf)
        path = out / "table_a40" / f"{model_id}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, grids=grids, logpdf=logpdf)
        for i, (value, wanted) in enumerate(zip(kl, expected["kl_nats"])):
            checks.value(f"a40 {model_id} fit {i} KL", value, wanted, TOLERANCES["a40_fit_kl_nats"])
        mean = statistics.mean(kl)
        checks.value(f"a40 {model_id} mean KL", mean, expected["summary"]["mean_kl"], TOLERANCES["a40_mean_kl_nats"])
        wins = sum(value < base for value, base in zip(kl, original))
        checks.equal(f"a40 {model_id} fits below the original", wins, expected["summary"]["wins_vs_original"])
        result["models"][model_id] = dict(
            kl_nats=kl, mean_kl=mean, sd_across_fits=statistics.stdev(kl), wins_vs_original=wins,
            reduction_vs_original=1 - mean / statistics.mean(original),
            max_marginal_error=max(c.marginal_error for c in copulas),
            grids_sha256=checks.digest(f"a40 {model_id} grids", grids, expected["grids_float32_sha256"], "<f4"),
            logpdf_sha256=checks.digest(f"a40 {model_id} logpdf", logpdf, expected["logpdf_float64_sha256"]),
            batch_seconds=timings, mean_ms_per_edge=1000 * statistics.mean(timings) / len(pairs),
            arrays=str(path.relative_to(out)), arrays_sha256=file_sha256(path))
        print(model_id, "Table A40 mean KL", f"{mean:.10f}", "fits below original", wins, flush=True)
    result["baselines"] = {key: dict(mean_kl=statistics.mean(v), sd_across_fits=statistics.stdev(v),
                                     wins_vs_original=sum(x < o for x, o in zip(v, original)))
                           for key, v in (("original", original),
                                          ("tll", [r["methods"]["tll"]["kl"] for r in table1["records"]]),
                                          ("parametric", [r["methods"]["param"]["kl"] for r in table1["records"]]))}
    return result


def table_a40_tex(result):
    rows = [("original", "Original VDC"), ("vdc-denoiser-m64-v2", "Retrained VDC-64"),
            ("vdc-denoiser-m128-v2", "Retrained VDC-128"), ("tll", "pyvine-TLL"), ("parametric", "Parametric selection")]
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Estimator & Held-out KL (nats) & Lower KL than original \\", r"\midrule"]
    for key, label in rows:
        entry = result["models"].get(key) or result["baselines"][key]
        count = "--" if key == "original" else f"{entry['wins_vs_original']}/45"
        lines.append(f"{label} & ${entry['mean_kl']:.5f} \\pm {entry['sd_across_fits']:.5f}$ & {count} \\\\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def run_vine(model_id, bundle, train, test, projection_iters, simulation_seed, device, warmed):
    import numpy as np

    from vdc import VineCopulaModel
    from vdc.models.linear_grid_estimator import LinearGridEstimator
    from vdc.pretrained import linear_grid_settings_for
    if model_id not in warmed:
        LinearGridEstimator(bundle.model, linear_grid_settings_for(bundle), bundle.device).estimate_grids([train[:, :2]] * 16)
        synchronize(device)
        warmed.add(model_id)
    vine = VineCopulaModel.from_pretrained(bundle, vine_type="dvine", projection_iters=projection_iters)
    synchronize(device)
    start = time.perf_counter()
    vine.fit(train, diffusion_model=bundle.model, diffusion=None, verbose=False)
    synchronize(device)
    fit_seconds = time.perf_counter() - start
    logpdf = vine.logpdf(test)
    pit = vine.rosenblatt(test)
    uniforms = np.random.default_rng(simulation_seed).uniform(1e-6, 1 - 1e-6, (5000, train.shape[1]))
    start = time.perf_counter()
    generated = vine.inverse_rosenblatt(uniforms)
    simulation_seconds = time.perf_counter() - start
    recovered = vine.rosenblatt(generated)
    arrays = dict(logpdf=logpdf, pit=pit, uniforms=uniforms, generated=generated, recovered=recovered)
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise SystemExit(f"{model_id}: nonfinite vine output")
    summary = dict(nll_bits_dim=nll_bits(logpdf, train.shape[1]), pit_mean_ks=pit_ks(pit), pit_rows=len(pit),
                   order=[int(v) for v in vine.structure.order], roundtrip_max=float(np.max(np.abs(uniforms - recovered))),
                   max_edge_marginal_error=float(np.asarray(vine.normalization_diagnostics)[:, 0].max()),
                   fit_seconds=fit_seconds, simulation_seconds=simulation_seconds)
    return summary, arrays


def compare_vine(prefix, summary, arrays, expected, checks, nll_tolerance, ks_tolerance):
    checks.value(f"{prefix} NLL", summary["nll_bits_dim"], expected["nll_bits_dim"], nll_tolerance)
    checks.value(f"{prefix} PIT-KS", summary["pit_mean_ks"], expected["pit_mean_ks"], ks_tolerance)
    checks.equal(f"{prefix} D-vine order", summary["order"], expected["order"])
    checks.items.append(dict(name=f"{prefix} round trip", actual=summary["roundtrip_max"],
                             tolerance=TOLERANCES["roundtrip"], passed=summary["roundtrip_max"] < TOLERANCES["roundtrip"]))
    return {key: checks.digest(f"{prefix} {key}", array, expected[f"{key}_float64_sha256"]) for key, array in arrays.items()}


def run_table_a41(args, bundles, reference, checks, out):
    import numpy as np

    from vdc.data.tabular import EmpiricalMarginals, maybe_load_uci
    cases = {(c["dataset"], c["size_label"]): c for c in reference["cases"]}
    data = {}
    for name in UCI_DATASETS:
        if name not in args.datasets:
            continue
        train, test = (np.asarray(x, dtype=np.float64) for x in maybe_load_uci(name, args.data_root))
        hashes = {"train": array_sha256(train), "test": array_sha256(test)}
        if hashes != MAF_ARRAY_SHA256[name]:
            raise SystemExit(f"{args.data_root}/uci/{name}.npz does not hold the paper's MAF {name} arrays")
        data[name] = (train, test)
    result, warmed = {"cases": []}, set()
    for size in args.sizes:
        label = "5k" if size == 5000 else "50k"
        rng = np.random.default_rng(42)
        for name in UCI_DATASETS:  # the selection stream runs over all four datasets in order
            if name in data:
                train_all, test_all = data[name]
                n_train, n_test = len(train_all), len(test_all)
            else:
                n_train, n_test = {"power": (1659917, 204928), "gas": (852174, 105206),
                                   "hepmass": (315123, 174987), "miniboone": (29556, 3648)}[name]
            train_idx = rng.choice(n_train, size, replace=False) if n_train > size else np.arange(n_train)
            test_idx = rng.choice(n_test, 50000, replace=False) if n_test > 50000 else np.arange(n_test)
            if name not in data:
                continue
            expected = cases[(name, label)]
            marginal = EmpiricalMarginals.fit(train_all[train_idx])
            train, test = marginal.transform(train_all[train_idx]), marginal.transform(test_all[test_idx])
            prefix = f"a41 {name} {label}"
            checks.equal(f"{prefix} training rows sha256", array_sha256(train), expected["U_train_sha256"])
            checks.equal(f"{prefix} test rows sha256", array_sha256(test), expected["U_test_sha256"])
            case = dict(dataset=name, size_label=label, n_train=len(train), n_test=len(test), d=train.shape[1],
                        baseline_nll_bits_dim=expected["baseline_nll_bits_dim"], models={})
            for model_id, bundle in bundles.items():
                summary, arrays = run_vine(model_id, bundle, train, test, 30, 202610111, args.device, warmed)
                summary["arrays_sha256"] = compare_vine(f"{prefix} {model_id}", summary, arrays, expected["models"][model_id],
                                                        checks, TOLERANCES["a41_nll_bits_dim"], TOLERANCES["a41_pit_ks"])
                path = out / "table_a41" / f"{name}_{label}" / f"{model_id}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                summary["arrays"] = str(path.relative_to(out))
                case["models"][model_id] = summary
                print(prefix, model_id, f"NLL {summary['nll_bits_dim']:.6f} PIT-KS {summary['pit_mean_ks']:.5f}", flush=True)
            result["cases"].append(case)
    return result


def table_a41_tex(result):
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Dataset & Train & Original & TLL & VDC-64 & VDC-128 & KS-64 & KS-128 \\", r"\midrule"]
    for case in result["cases"]:
        m64, m128 = case["models"]["vdc-denoiser-m64-v2"], case["models"]["vdc-denoiser-m128-v2"]
        train = "5k" if case["size_label"] == "5k" else ("29,556" if case["dataset"] == "miniboone" else "50k")
        values = (case["baseline_nll_bits_dim"]["original"], case["baseline_nll_bits_dim"]["tll"], m64["nll_bits_dim"],
                  m128["nll_bits_dim"], m64["pit_mean_ks"], m128["pit_mean_ks"])
        lines.append(" & ".join([case["dataset"].upper(), train] + [f"{v:.4f}" for v in values]) + r" \\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def block_bootstrap(delta, block, seed, replicates):
    """Moving-block bootstrap 95% interval of the mean, as in the appendix's S&P100 protocol."""
    import numpy as np
    rng = np.random.default_rng(seed)
    count = len(delta)
    starts = rng.integers(0, count - block + 1, size=(replicates, int(np.ceil(count / block))))
    indices = (starts[:, :, None] + np.arange(block)).reshape(replicates, -1)[:, :count]
    return np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()


def run_sp100(args, bundles, reference, checks, out):
    import numpy as np
    from scipy.stats import rankdata
    protocol = reference["protocol"]
    if file_sha256(args.returns) != SP100_RETURNS_SHA256:
        raise SystemExit("--returns is not the paper's S&P100 matrix (sha256 differs); see paper/README.md")
    raw = np.load(args.returns, allow_pickle=False).astype(np.float64)
    cut = int(protocol["train_fraction"] * len(raw))
    train_raw, test_raw = raw[:cut], raw[cut:]
    train = rankdata(train_raw, method="average", axis=0) / (cut + 1)
    test = np.column_stack([(np.searchsorted(np.sort(train_raw[:, j]), test_raw[:, j], side="right") + .5) / (cut + 1)
                            for j in range(raw.shape[1])])
    train, test = np.clip(train, 1e-6, 1 - 1e-6), np.clip(test, 1e-6, 1 - 1e-6)
    checks.equal("sp100 training rows sha256", array_sha256(train), reference["U_train_sha256"])
    checks.equal("sp100 test rows sha256", array_sha256(test), reference["U_test_sha256"])
    original = None
    if args.original_logpdf is not None:
        with np.load(args.original_logpdf, allow_pickle=False) as z:
            if not np.array_equal(z["U_test"], test):
                raise SystemExit("--original-logpdf was computed on other test rows")
            original = np.asarray(z["vdc"], dtype=np.float64)
    result, warmed = {"models": {}, "historical_nll_bits_dim": reference["historical_nll_bits_dim"]}, set()
    for model_id, bundle in bundles.items():
        expected = reference["models"][model_id]
        summary, arrays = run_vine(model_id, bundle, train, test, protocol["projection_iters"],
                                   protocol["simulation_seed"], args.device, warmed)
        summary["arrays_sha256"] = compare_vine(f"sp100 {model_id}", summary, arrays, expected, checks,
                                                TOLERANCES["sp100_nll_bits_dim"], TOLERANCES["sp100_pit_ks"])
        path = out / "sp100" / f"{model_id}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        summary["arrays"] = str(path.relative_to(out))
        if original is not None:
            delta = -(arrays["logpdf"] - original) / (test.shape[1] * np.log(2))
            summary["paired_vs_supplied_original"] = dict(
                delta_bits_dim=float(delta.mean()),
                intervals={str(b): block_bootstrap(delta, b, protocol["bootstrap_seed"], protocol["bootstrap_replicates"])
                           for b in protocol["block_sizes"]},
                paper=expected["paired_vs_original"],
                scope="Against the supplied original-model log densities, not the paper's historical refit.")
        result["models"][model_id] = summary
        print("sp100", model_id, f"NLL {summary['nll_bits_dim']:.6f}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", default=[], metavar="MODEL_ID=PATH",
                   help="local checkpoint of a model (repeatable), checked against the library manifest's SHA-256; "
                        "without it the library resolves the model from its cache or public host")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                   help="models to evaluate (default: both)")
    p.add_argument("--parts", nargs="+", choices=["table_a40", "table_a41", "sp100"], default=["table_a40", "table_a41"],
                   help="results to recompute (default: table_a40 table_a41)")
    p.add_argument("--data-root", type=Path, help="table_a41: output of scripts/download_uci_datasets.py, holding uci/<name>.npz")
    p.add_argument("--datasets", nargs="+", choices=UCI_DATASETS, default=UCI_DATASETS)
    p.add_argument("--sizes", nargs="+", type=int, choices=[5000, 50000], default=[5000, 50000],
                   help="table_a41 training rows (default: both)")
    p.add_argument("--returns", type=Path, help="sp100: the paper's S&P100 returns matrix (sha256 checked)")
    p.add_argument("--original-logpdf", type=Path,
                   help="sp100: arrays.npz of runners/reproduce_sp100_likelihood.py, for paired differences")
    p.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "reference",
                   help="paper/reference folder with MANIFEST.json (default: %(default)s)")
    p.add_argument("--expected-module-root", type=Path, required=True,
                   help="stop unless vdc is imported from this directory")
    p.add_argument("--device", default="cuda", help="torch device (default: %(default)s)")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (appendix: 4)")
    p.add_argument("--timing-repeats", type=int, default=5, help="table_a40: warmed repeats of the 45-fit batch")
    p.add_argument("--require-bitwise", action="store_true",
                   help="also fail unless every per-row array matches the appendix's bitwise (H100 80GB check)")
    p.add_argument("--output", type=Path, required=True, help="new output directory")
    args = p.parse_args()
    if "table_a41" in args.parts and args.data_root is None:
        p.error("table_a41 needs --data-root")
    if "sp100" in args.parts and args.returns is None:
        p.error("sp100 needs --returns")
    if args.timing_repeats < 1:
        p.error("--timing-repeats must be positive")
    import numpy as np
    import pyvinecopulib as pv
    import scipy
    import torch

    import vdc
    if not Path(vdc.__file__).resolve().is_relative_to(args.expected_module_root.resolve()):
        raise SystemExit(f"vdc is imported from {vdc.__file__}, not from {args.expected_module_root}")
    manifest = json.loads((args.reference / "MANIFEST.json").read_text())["files"]
    references = {}
    for name in ("retrained/table_a40_bivariate.json", "retrained/table_a41_uci.json",
                 "retrained/sp100_chronological.json", "table1/table1_v2.json"):
        path = args.reference / name
        if file_sha256(path) != manifest[name]["sha256"]:
            raise SystemExit(f"{path} differs from reference/MANIFEST.json")
        references[name] = json.loads(path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    bundles = load_models(args)
    checks = Checks()
    results = dict(
        status="running", scope="Retrained-estimator appendix (Tables A40, A41, chronological S&P100); "
        "original-model and baseline rows are the paper's records.",
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        tolerances=TOLERANCES, source_sha256=file_sha256(__file__),
        models={m: dict(sha256=b.manifest["sha256"], checkpoint_bytes=Path(b.checkpoint_path).stat().st_size,
                        inference=b.manifest["inference"]) for m, b in bundles.items()},
        references={name: manifest[name]["sha256"] for name in references},
        module_path=str(Path(vdc.__file__).resolve()), library_version=vdc.__version__,
        host=platform.node(), job=os.environ.get("SLURM_JOB_ID"), device=args.device,
        gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      torch=torch.__version__, pyvinecopulib=pv.__version__),
        precision=dict(dtype="float32 network, float64 copulas", matmul_tf32=False, cudnn_tf32=False))
    if "table_a40" in args.parts:
        results["table_a40"] = run_table_a40(args, bundles, references["retrained/table_a40_bivariate.json"],
                                             references["table1/table1_v2.json"], checks, args.output)
    if "table_a41" in args.parts:
        results["table_a41"] = run_table_a41(args, bundles, references["retrained/table_a41_uci.json"], checks, args.output)
    if "sp100" in args.parts:
        results["sp100"] = run_sp100(args, bundles, references["retrained/sp100_chronological.json"], checks, args.output)
    complete_models = set(bundles) == set(MODELS)
    if "table_a40" in args.parts and complete_models:
        lines = table_a40_tex(results["table_a40"])
        (args.output / "tab_retrained_bivariate.tex").write_text("\n".join(lines) + "\n")
        results["table_a40"]["matches_paper_table"] = lines == references["retrained/table_a40_bivariate.json"]["paper_table"]
    if ("table_a41" in args.parts and complete_models and args.datasets == UCI_DATASETS
            and args.sizes == [5000, 50000]):
        lines = table_a41_tex(results["table_a41"])
        (args.output / "tab_retrained_uci.tex").write_text("\n".join(lines) + "\n")
        results["table_a41"]["matches_paper_table"] = lines == references["retrained/table_a41_uci.json"]["paper_table"]
    results["checks"] = checks.items
    results["bitwise_equal_to_appendix"] = checks.bitwise
    results["all_arrays_bitwise_equal"] = bool(checks.bitwise) and all(checks.bitwise.values())
    passed = checks.passed and (results["all_arrays_bitwise_equal"] or not args.require_bitwise)
    results["status"] = "complete" if passed else "mismatch"
    results["wall_seconds"] = time.perf_counter() - started
    (args.output / "results.json").write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    failed = [c["name"] for c in checks.items if not c["passed"]]
    print(json.dumps(dict(status=results["status"], checks=len(checks.items), failed=failed[:20],
                          arrays_compared=len(checks.bitwise), arrays_bitwise_equal=sum(checks.bitwise.values()),
                          tables_match_paper={k: results[k].get("matches_paper_table") for k in ("table_a40", "table_a41")
                                              if k in results}), indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

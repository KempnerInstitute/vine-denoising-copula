#!/usr/bin/env python3
"""Fixed 50-dataset paired TC replication (paper appendix, "Structural control for the TC benchmark").

Release copy of the authors' runner (paper workspace commit c9a9ae0). Numerics are unchanged; the
reference file is an argument, and the checks that pin the library commit and a clean checkout run only
with --require-pins (the authors' run used library commit 67c5b1b). Usage from the repository root:
  python paper/neurips2026/runners/tc_replication_paired.py --library . --checkpoint MODEL.pt \
    --reference paper/neurips2026/runners/saved_outputs/tc_tree1_control_results.json --output NEW_DIR
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

LIBRARY_SHA = '67c5b1bc6311214695d4c7fc044ea06546324906'
CHECKPOINT_SHA = '8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git(root, *args):
    try:
        return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL,
            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--qualification-only', action='store_true')
    p.add_argument('--reference', type=Path, default=None,
                   help='saved three-dataset control (default: the paper workspace record)')
    p.add_argument('--require-pins', action='store_true',
                   help='stop unless the library is at the authors\' commit and both checkouts are clean')
    a = p.parse_args()
    a.library = a.library.resolve()
    paper = Path(__file__).resolve().parents[2]
    if a.require_pins:
        assert git(a.library, 'rev-parse', 'HEAD') == LIBRARY_SHA
        assert not git(a.library, 'status', '--porcelain', '--untracked-files=no')
        assert not git(paper, 'status', '--porcelain', '--untracked-files=no')
    sys.path.insert(0, str(a.library))
    import numpy as np
    import scipy
    from scipy.special import ndtri
    import torch
    import vdc
    from vdc.data.generators import generate_gaussian_vine
    from vdc.train.unified_trainer import build_model
    from vdc.utils.information import ksg_mutual_information
    from vdc.vine.api import VineCopulaModel
    assert Path(vdc.__file__).resolve().is_relative_to(a.library)
    assert digest(a.checkpoint) == CHECKPOINT_SHA
    if a.device == 'cuda':
        assert torch.cuda.is_available(), 'No silent CPU fallback'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    assert cfg['model']['type'] == 'denoiser'
    model = build_model('denoiser', cfg, torch.device(a.device))
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval()
    del ck
    projection_iters = int(cfg.get('training', {}).get('projection_iters', 30))
    ref_path = a.reference or paper / 'analysis/tc_tree1_control/ksg_tree1_results.json'
    ref = json.loads(ref_path.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    cpu = next((x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines()
                if x.startswith('model name')), platform.processor())
    settings = dict(d=50, n=5000, rho=.7, k=5, n_train=3750, n_test=1250,
        n_datasets=50, seed_entropy=[20261019, 50, 'trial'], edge_batch=256,
        projection_iters=projection_iters, fast_fit=False, test_fraction=.25,
        threads=1, dtype='float32', matmul_tf32=False, cudnn_tf32=True)
    meta = dict(paper_commit=git(paper, 'rev-parse', 'HEAD'), library_commit=git(a.library, 'rev-parse', 'HEAD'),
        script_sha256=digest(__file__), protocol_sha256=next((digest(f) for f in (Path(__file__).with_name('tc_replication_PROTOCOL.md'), Path(__file__).with_name('PROTOCOL.md')) if f.exists()), None),
        checkpoint_sha256=CHECKPOINT_SHA, reference_sha256=digest(ref_path),
        module_path=str(vdc.__file__), hostname=platform.node(), cpu=cpu,
        gpu=torch.cuda.get_device_name() if a.device == 'cuda' else None,
        device=a.device, job=os.environ.get('SLURM_JOB_ID'),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, torch=torch.__version__, vdc=vdc.__version__),
        thread_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        settings=settings)
    save(a.output/'manifest.json', meta)

    def ahash(arr):
        arr = np.ascontiguousarray(arr)
        return hashlib.sha256(arr.dtype.str.encode() + str(arr.shape).encode() + arr.tobytes()).hexdigest()

    def sync():
        if a.device == 'cuda':
            torch.cuda.synchronize()

    def ksg(U, seed, pairs):
        start = time.perf_counter()
        vals = [max(0., float(ksg_mutual_information(U[:, [i]], U[:, j], k=5,
                 seed=seed + 10000 * j))) for i, j in pairs]
        return float(sum(vals)), time.perf_counter()-start, vals

    def vdc_fit(U, seed):
        perm = np.random.default_rng(seed).permutation(len(U))
        te, tr = perm[:1250], perm[1250:]
        vine = VineCopulaModel(vine_type='dvine', truncation_level=1, m=64,
            device=a.device, projection_iters=projection_iters, hfunc_use_spline=False,
            batch_edges=True, edge_batch_size=256, fast_fit=False)
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            vine.fit(U[tr], diffusion_model=model, diffusion=None, verbose=False)
            lp = np.asarray(vine.logpdf(U[te]))
        sync()
        elapsed = time.perf_counter()-start
        assert np.isfinite(lp).all()
        return float(lp.mean()), elapsed, lp, list(map(int, vine.structure.order)), tr, te

    # Explicit warmup; neither warmup sample contributes to qualification or analysis.
    warm = generate_gaussian_vine(n=5000, d=50, rho=.7, seed=987654321)
    vdc_fit(warm, 987654321)
    ksg(warm[:, :2], 987654321, [(0, 1)])
    pairs = [(i-1, i) for i in range(1, 50)]
    true = float(-.5 * 49 * np.log1p(-.7**2))

    def trial(seed, index, phase):
        U = generate_gaussian_vine(n=5000, d=50, rho=.7, seed=seed)
        if index % 2:
            kv, kt, edges = ksg(U, seed, pairs)
            vv, vt, lp, order, tr, te = vdc_fit(U, seed)
        else:
            vv, vt, lp, order, tr, te = vdc_fit(U, seed)
            kv, kt, edges = ksg(U, seed, pairs)
        learned_pairs = sorted(tuple(sorted((order[i-1], order[i]))) for i in range(1,50))
        matched = learned_pairs == pairs
        aligned = kv if matched else ksg(U, seed, learned_pairs)[0]
        # Oracle evaluation noise diagnostic, outside both methods' timing.
        z = ndtri(U[te])
        quad = (z[:,0]**2 + z[:,-1]**2 + (1+.7**2)*(z[:,1:-1]**2).sum(axis=1)
                - 2*.7*(z[:,:-1]*z[:,1:]).sum(axis=1))/(1-.7**2)
        oracle = true - .5*(quad-(z*z).sum(axis=1))
        artifact = a.output/f'{phase}_{index:03d}.npz'
        np.savez_compressed(artifact, vdc_logpdf=lp, oracle_logpdf=oracle,
                            ksg_edges=np.asarray(edges), train_indices=tr, test_indices=te)
        row = dict(phase=phase, index=index, seed=seed, data_sha256=ahash(U),
            train_sha256=ahash(U[tr]), test_sha256=ahash(U[te]), tc_true=true,
            vdc_tc=vv, ksg_tc=kv, vdc_seconds=vt, ksg_seconds=kt,
            vdc_order=order, chain_recovered=matched, ksg_vdc_order_tc=aligned,
            oracle_heldout_tc=float(oracle.mean()), heldout_log_density_gap=float((oracle-lp).mean()),
            method_order=['ksg','vdc'] if index%2 else ['vdc','ksg'],
            artifact=artifact.name, artifact_sha256=digest(artifact))
        save(a.output/f'{phase}_{index:03d}.json', row)
        print(json.dumps({k:row[k] for k in ['phase','index','seed','vdc_tc','ksg_tc','chain_recovered']}), flush=True)
        return row

    qualification = []
    for old in [r for r in ref['trials'] if r['d']==50]:
        r = trial(old['trial_seed'], old['trial'], 'qualification')
        r['vdc_reference_delta'] = r['vdc_tc']-old['vdc_tree1_tc']
        r['ksg_reference_delta'] = r['ksg_tc']-old['ksg_tree1_tc']
        qualification.append(r)
    qualified = all(abs(r['vdc_reference_delta']) <= .001 and abs(r['ksg_reference_delta']) <= 1e-10
                    for r in qualification)
    save(a.output/'qualification.json', dict(passed=qualified, trials=qualification,
         tolerances=dict(vdc=.001, ksg=1e-10)))
    assert qualified, 'Historical replay failed; fresh run not started'
    if a.qualification_only:
        return
    rows = [trial(int(np.random.SeedSequence([20261019,50,i]).generate_state(1)[0]), i, 'fresh')
            for i in range(50)]
    ve = np.asarray([r['vdc_tc']-true for r in rows])
    ke = np.asarray([r['ksg_tc']-true for r in rows])
    boot = np.random.default_rng(20261019).integers(0, 50, size=(10000,50))
    def interval(samples):
        return list(map(float, np.quantile(samples,[.025,.975])))
    summary = dict(status='complete', n_datasets=len(rows), tc_true=true,
        chain_recovered=sum(r['chain_recovered'] for r in rows),
        paired_mae_difference_vdc_minus_ksg=float(np.mean(abs(ve)-abs(ke))),
        paired_mae_difference_ci95=interval((abs(ve)-abs(ke))[boot].mean(axis=1)),
        paired_mse_difference=float(np.mean(ve**2-ke**2)),
        paired_mse_difference_ci95=interval((ve**2-ke**2)[boot].mean(axis=1)),
        vdc_absolute_error_smaller=int((abs(ve)<abs(ke)).sum()))
    for name, err in [('vdc',ve),('ksg',ke)]:
        times = np.asarray([r[name+'_seconds'] for r in rows])
        summary[name] = dict(mae=float(abs(err).mean()), rmse=float(np.sqrt((err**2).mean())),
            signed_bias=float(err.mean()), mae_ci95=interval(abs(err)[boot].mean(axis=1)),
            bias_ci95=interval(err[boot].mean(axis=1)), median_seconds=float(np.median(times)),
            median_seconds_ci95=interval(np.median(times[boot],axis=1)))
    ratios = np.asarray([r['ksg_seconds']/r['vdc_seconds'] for r in rows])
    summary['median_paired_timing_ratio_ksg_over_vdc'] = float(np.median(ratios))
    summary['median_paired_timing_ratio_ci95'] = interval(np.median(ratios[boot],axis=1))
    save(a.output/'summary.json', summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()

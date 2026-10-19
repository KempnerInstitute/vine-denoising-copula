#!/usr/bin/env python3
"""Portable original-checkpoint chronological S&P100 likelihood replay.

Requires the authenticated returns artifact; raw-price provenance is not
reconstructed here. Reports VDC/TLL only; does not rewrite any paper table.
With --rebuilt-returns, runs the same protocol on another (1695, 100) returns
matrix, such as one rebuilt from paper/neurips2026/data/sp100_symbols.json, and
skips the checks against the paper's matrix hash and reported values.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import os


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--returns', type=Path, required=True)
    p.add_argument('--expected-module-root', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rebuilt-returns', action='store_true',
                   help='accept a returns matrix other than the paper\'s; no comparison with reported values')
    a = p.parse_args()
    import numpy as np
    import scipy
    from scipy.stats import rankdata
    import torch
    import pyvinecopulib as pv
    import vdc
    from vdc.pretrained import load_checkpoint_bundle
    from vdc import VineCopulaModel
    from vdc.baselines.pyvinecopulib import fit_vinecop
    assert Path(vdc.__file__).resolve().is_relative_to(a.expected_module_root.resolve())
    assert sha(a.checkpoint)=='8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f'
    if not a.rebuilt_returns:
        assert sha(a.returns)=='5a2accb5e518b498464ea11732fa6bead1b1c34a22cc28ae95d719964ca1e476'
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    raw = np.load(a.returns, allow_pickle=False).astype(np.float64)
    assert raw.shape==(1695,100) and np.isfinite(raw).all()
    # Preserve E28's training ranks and its test-only half-rank offset.
    tr, te = raw[:1356], raw[1356:]
    u = rankdata(tr, method='average', axis=0)/(len(tr)+1)
    v = np.column_stack([(np.searchsorted(np.sort(tr[:,j]),te[:,j],side='right')+.5)/(len(tr)+1)
        for j in range(100)])
    u, v = np.clip(u,1e-6,1-1e-6), np.clip(v,1e-6,1-1e-6)
    bundle = load_checkpoint_bundle(a.checkpoint,device=a.device)
    ck = torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    bundle.model.load_state_dict(ck['model_state_dict'],strict=True)
    del ck
    vine = VineCopulaModel(vine_type='dvine',m=64,device=a.device,projection_iters=15,
        hfunc_use_spline=False,batch_edges=True,edge_batch_size=256)
    vine.fit(u,diffusion_model=bundle.model,diffusion=None,verbose=False)
    lp_vdc = np.asarray(vine.logpdf(v))
    tll = fit_vinecop(u,mode='nonparametric',num_threads=a.threads,nonparametric_grid_size=30)
    lp_tll = np.log(np.clip(tll.pdf(np.clip(v,1e-10,1-1e-10)),1e-30,None))
    assert np.isfinite(lp_vdc).all() and np.isfinite(lp_tll).all()
    nll = {k:float(-x.mean()/(100*np.log(2))) for k,x in [('vdc',lp_vdc),('tll',lp_tll)]}
    # Historical refits were not bitwise stable. Specify practical tolerances before this run.
    if not a.rebuilt_returns:
        np.testing.assert_allclose(nll['vdc'],-.41848794320420774,rtol=0,atol=.001)
        np.testing.assert_allclose(nll['tll'],-.292759674956995,rtol=0,atol=1e-8)
        assert nll['vdc'] < nll['tll']
    np.savez_compressed(a.output/'arrays.npz',U_train=u,U_test=v,vdc=lp_vdc,tll=lp_tll)
    result = dict(status='complete' if not a.rebuilt_returns else 'complete_rebuilt_returns',
        scope='Chronological S&P100 VDC/TLL reproduction only; no historical timing replacement.',
        n_train=len(u),n_test=len(v),d=100,nll_bits_dim=nll,
        tolerances_bits_dim=dict(vdc=.001,tll=1e-8),source_sha256=sha(__file__),
        checkpoint_sha256=sha(a.checkpoint),returns_sha256=sha(a.returns),arrays_sha256=sha(a.output/'arrays.npz'),
        module_path=str(Path(vdc.__file__).resolve()),host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,pyvinecopulib=pv.__version__),
        configuration=dict(threads=a.threads,device=a.device,grid=64,projection_iters=15,edge_batch=256,
        dtype='float32',matmul_tf32=False,cudnn_tf32=True),
        provenance_limit='Returns matrix hash verified; original raw-price collection was not reconstructed.')
    (a.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

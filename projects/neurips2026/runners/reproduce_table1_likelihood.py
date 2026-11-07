#!/usr/bin/env python3
"""Portable reproduction of the three fitted estimators' Table 1 likelihoods.

Correct pyvinecopulib samples, nine fixed cases, five fixed sample seeds.
Uses the public package APIs, never the historical defective generators.
Does not claim to reproduce the other Table 1 metrics or historical timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-module-root', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--reference', type=Path, help='Optional authenticated rerun_table1.json for per-fit checks')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import pyvinecopulib as pv
    import torch
    import vdc
    from vdc.pretrained import load_checkpoint_bundle, estimate_pair_density_from_samples
    from vdc.models.hfunc import HFuncLookup
    from vdc.baselines.pyvinecopulib import fit_bicop
    assert Path(vdc.__file__).resolve().is_relative_to(a.expected_module_root.resolve())
    checkpoint_sha = digest(a.checkpoint)
    assert checkpoint_sha == '8a1b71ad6d2012ed344b22a743a96e667d32a2ab54ade4444686e375e91f5c0f'
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    bundle = load_checkpoint_bundle(a.checkpoint, device=a.device)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    bundle.model.load_state_dict(ck['model_state_dict'], strict=True)
    del ck
    baseline = None
    if a.reference:
        assert digest(a.reference) == '62323ffa061fabf0c55b2366b975a654e5db7947fc8712bc6a58fb23ba392f5e'
        baseline = json.loads(a.reference.read_text())['part_b']['records']
    cases = [('indep',[],0),('gaussian',[[.7]],0),('gaussian',[[-.7]],0),
        ('student',[[.7],[5.]],0),('clayton',[[3.]],0),('clayton',[[3.]],90),
        ('gumbel',[[2.5]],0),('joe',[[3.]],0),('frank',[[5.]],0)]
    rows, logs, truths, grids = [], [], [], []
    started = time.perf_counter()
    for i,(family,params,rotation) in enumerate(cases):
        bc = pv.Bicop(family=getattr(pv.BicopFamily,family), rotation=rotation,
            **({'parameters':np.asarray(params)} if params else {}))
        for seed in [101,202,303,404,505]:
            tr = np.clip(bc.simulate(2000,seeds=[seed,seed+1]),1e-10,1-1e-10)
            te = np.clip(bc.simulate(20000,seeds=[seed+777777,seed+777778]),1e-10,1-1e-10)
            true = np.log(np.maximum(bc.pdf(te),1e-300))
            with torch.no_grad():
                grid = np.asarray(estimate_pair_density_from_samples(bundle,tr,m=64,projection_iters=30),dtype=float)
            grid /= grid.mean()
            cop = HFuncLookup(grid,use_spline=False)
            values = [np.log(np.maximum(cop.pdf(te[:,0],te[:,1]),1e-300))]
            for mode in ['nonparametric','parametric']:
                fitted = fit_bicop(tr,mode=mode,num_threads=1,selection_criterion='bic',allow_rotations=True)
                values.append(np.log(np.maximum(fitted.pdf(te),1e-300)))
            kl = [float(np.mean(true-lp)) for lp in values]
            if baseline is not None:
                ref = baseline[len(rows)]
                assert ref['seed']==seed and ref['rotation']==rotation
                np.testing.assert_allclose(-true.mean(),ref['nll_oracle'],rtol=0,atol=1e-12)
                expected = [ref['methods'][k][metric]-ref['nll_oracle'] for k,metric in
                    [('vdc','nll_grid_bilinear_lib'),('tll_paper','nll_native_pdf'),('param','nll_native_pdf')]]
                # Declared before qualification: absolute KL error <= 1e-5 nats/edge.
                np.testing.assert_allclose(kl,expected,rtol=0,atol=1e-5)
            rows.append(dict(case_index=i,family=family,parameters=params,rotation=rotation,seed=seed,
                kl=dict(zip(['vdc','tll','parametric'],kl))))
            logs.append(values); truths.append(true); grids.append(grid)
        print('completed',family,rotation,flush=True)
    arrays = a.output/'likelihoods.npz'
    np.savez_compressed(arrays,logpdf=np.asarray(logs),logtrue=np.asarray(truths),vdc_grids=np.asarray(grids))
    result = dict(status='complete', scope='Table 1 likelihood columns for VDC, TLL and parametric selection only.',
        source_sha256=digest(__file__), checkpoint_sha256=checkpoint_sha,
        module_path=str(Path(vdc.__file__).resolve()), hostname=platform.node(), job=os.environ.get('SLURM_JOB_ID'),
        versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,pyvinecopulib=pv.__version__),
        device=a.device,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,
        precision=dict(dtype='float32',matmul_tf32=False,cudnn_tf32=True,batch_size=1,projection_iters=30),
        reference_sha256=digest(a.reference) if a.reference else None, reference_tolerance_nats=1e-5,
        rows=rows,means={k:float(np.mean([r['kl'][k] for r in rows])) for k in ['vdc','tll','parametric']},
        wall_seconds=time.perf_counter()-started,arrays_sha256=digest(arrays))
    (a.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['means'],indent=2))


if __name__ == '__main__':
    main()

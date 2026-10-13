"""Experimental explicit CPU sampler for frozen, linear-grid D-vines.

This is a snapshot of h tables, not a mutable fitted model. It never changes
the estimator or global dispatch. Rebuild the snapshot after any model change.
An explicitly compiled shared library is required; there is no implicit build.
"""
import ctypes
from pathlib import Path
import subprocess
import numpy as np
from .recursion import VineRecursion
from ..models.hfunc import HFuncLookup


def build_kernel(output, compiler='g++'):
    """Build in the caller's external cache. Run on a compute node."""
    output=Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    command=[compiler,'-O3','-std=c++17','-shared','-fPIC','-fopenmp',
             '-fno-fast-math','-ffp-contract=off',
             str(Path(__file__).with_suffix('.cpp')), '-o',str(output)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    return command


class CompiledDVineSampler:
    """Explicit immutable copy of supported legacy h tables; float64 execution."""
    def __init__(self, vine, kernel):
        if type(vine) is not VineRecursion or vine.vine_type!='dvine':
            raise ValueError('Only the unmodified legacy D-vine recursion is supported')
        self.d=vine.d
        if self.d<2 or vine.structure.order is None:
            raise ValueError('A D-vine with an explicit order and at least two variables is required')
        self.order=np.array(vine.structure.order,dtype=np.int64)
        if sorted(self.order.tolist())!=list(range(self.d)):
            raise ValueError('Invalid variable order')
        pairs=[]
        for k in range(1,self.d):
            for i in range(k):
                root,var=int(self.order[i]),int(self.order[k])
                cop=vine._get_pair_copula_or_independence(root,var,frozenset(self.order[i+1:k]))
                pairs.append((cop,root))
        first=next((cop.hfunc for cop,_ in pairs if cop is not None),None)
        if first is None: raise ValueError('At least one fitted edge is required')
        self.m=first.m_u
        if self.m<2: raise ValueError('Grid must have at least two nodes')
        self.nodes=np.array(first.u_grid,dtype=np.float64,copy=True)
        expected=np.linspace(.5/self.m,1-.5/self.m,self.m)
        if not np.array_equal(self.nodes,expected): raise ValueError('Only default cell-center grids are supported')
        self.tables=np.zeros((len(pairs),2,self.m,self.m),dtype=np.float64)
        self.orientation=np.zeros(len(pairs),dtype=np.int64)
        for index,(cop,root) in enumerate(pairs):
            if cop is None: continue
            h=cop.hfunc
            if (type(h) is not HFuncLookup or h.use_spline or not h.vectorized_inverse or
                h.m_u!=self.m or h.m_v!=self.m or
                not np.array_equal(h.u_grid,self.nodes) or not np.array_equal(h.v_grid,self.nodes)):
                raise ValueError('Requires identical cell-center grids and corrected vectorized linear HFuncLookup')
            self.tables[index,0]=h.h_u_given_v_grid
            self.tables[index,1]=h.h_v_given_u_grid
            self.orientation[index]=1 if cop.edge[0]==root else -1
        if not np.isfinite(self.tables).all(): raise ValueError('Tables must be finite')
        if np.any(np.diff(self.tables[:,0],axis=1)<0) or np.any(np.diff(self.tables[:,1],axis=2)<0):
            raise ValueError('Tables must be monotone')
        self._dll=ctypes.CDLL(str(Path(kernel).resolve()))
        self._call=self._dll.vdc_dvine_inverse
        doubles=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        integers=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self._call.argtypes=[doubles,doubles,ctypes.c_int64,ctypes.c_int,ctypes.c_int,
                            integers,integers,doubles,doubles,ctypes.c_int,ctypes.c_int]
        self._call.restype=ctypes.c_int
        for array in (self.order,self.nodes,self.tables,self.orientation): array.flags.writeable=False

    def inverse_rosenblatt(self, uniforms, *, threads=1, block_size=64):
        w=np.ascontiguousarray(uniforms,dtype=np.float64)
        if w.ndim!=2 or w.shape[1]!=self.d: raise ValueError('Expected (n,d) uniforms')
        if not np.isfinite(w).all(): raise ValueError('Uniforms must be finite')
        if not isinstance(threads,int) or not 1<=threads<=1024: raise ValueError('Invalid thread count')
        if not isinstance(block_size,int) or not 1<=block_size<=65536: raise ValueError('Invalid block size')
        result=np.empty_like(w)
        code=self._call(w,result,len(w),self.d,self.m,self.order,self.orientation,
                        self.nodes,self.tables,threads,block_size)
        if code: raise RuntimeError(f'Compiled sampler failed with code {code}')
        return result

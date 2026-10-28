"""Optional CPU sampling for frozen, linear-grid D-vines.

This is a snapshot of h tables, not a mutable fitted model. It never changes
the estimator or global dispatch. Rebuild the snapshot after any model change.
The prepared interface builds on request and falls back to NumPy in auto mode.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
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
    # Atomic publication allows independent processes to share a kernel cache.
    with tempfile.TemporaryDirectory(prefix='build-',dir=output.parent) as temporary:
        built=Path(temporary)/output.name
        subprocess.run(command[:-1]+[str(built)],check=True,capture_output=True,text=True,timeout=120)
        os.replace(built,output)
    return command


class CompiledDVineSampler:
    """Explicit immutable copy of supported legacy h tables; float64 execution."""
    def __init__(self, vine, kernel, *, max_table_bytes=512*1024**2):
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
        required=len(pairs)*2*self.m*self.m*8
        if max_table_bytes is not None and required>max_table_bytes:
            raise MemoryError(f'Packed tables require {required} bytes; limit is {max_table_bytes}')
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


_validated_kernels=set()


def validate_kernel(kernel):
    """Check arithmetic on this build once per process, without touching RNG state."""
    path=Path(kernel).resolve()
    info=path.stat()
    identity=(str(path),info.st_mtime_ns,info.st_size)
    if identity in _validated_kernels: return
    from .vine_types import build_dvine_structure
    from .recursion import VinePairCopula
    rng=np.random.default_rng(61013)
    for m in (64,128):
        structure=build_dvine_structure(rng.random((24,3)),order=[2,0,1])
        vine=VineRecursion(structure,vine_type='dvine')
        for level,tree in enumerate(structure.trees):
            for index,(i,j,cond) in enumerate(tree.edges):
                density=rng.uniform(.05,2.,(m,m))
                density/=density.mean()
                if index%2: i,j=j,i
                h=HFuncLookup(density,use_spline=False,vectorized_inverse=True)
                vine.add_pair_copula(VinePairCopula((i,j,cond),density,h,level))
        w=rng.random((71,3));w[:4]=np.array([0.,1.,1e-12,1-1e-12])[:,None]
        answer=CompiledDVineSampler(vine,path).inverse_rosenblatt(w)
        if not np.array_equal(answer,vine.inverse_rosenblatt(w)):
            raise RuntimeError('Kernel build failed the NumPy identity check')
    _validated_kernels.add(identity)


def cached_kernel(cache_dir=None, compiler='g++'):
    """Compile and validate a source/compiler/platform-specific cached kernel."""
    executable=shutil.which(compiler)
    if executable is None: raise FileNotFoundError(f'C++ compiler unavailable: {compiler}')
    version=subprocess.check_output([executable,'--version'],text=True,timeout=10)
    source=Path(__file__).with_suffix('.cpp').read_bytes()
    key=hashlib.sha256(source+Path(__file__).read_bytes()+version.encode()+
                       executable.encode()+platform.system().encode()+platform.machine().encode()).hexdigest()[:24]
    root=Path(cache_dir or os.environ.get('VDC_KERNEL_CACHE',Path.home()/'.cache'/'vdc'/'sampler'))
    kernel=root/key/'sampler.so'
    if not kernel.exists(): build_kernel(kernel,executable)
    validate_kernel(kernel)
    return kernel


class PreparedVineSampler:
    """Reuse packed tables with an automatic NumPy fallback.

    Keep the fitted vine unchanged while using this handle. Preparing a new
    handle is required after refitting or manually changing fitted tables.
    Explicit ``backend='compiled'`` fails loudly; ``'auto'`` records build,
    platform, memory or compatibility failures and uses the original recursion.
    """
    def __init__(self,vine,*,backend='auto',kernel=None,cache_dir=None,compiler='g++',
                 num_threads=1,block_size=64,max_table_bytes=512*1024**2):
        if backend not in ('auto','numpy','compiled'): raise ValueError('Unknown sampler backend')
        if not isinstance(num_threads,int) or not 1<=num_threads<=1024: raise ValueError('Invalid thread count')
        if not isinstance(block_size,int) or not 1<=block_size<=65536: raise ValueError('Invalid block size')
        if max_table_bytes is not None and (not isinstance(max_table_bytes,int) or max_table_bytes<0):
            raise ValueError('max_table_bytes must be a nonnegative integer or None')
        self.vine=vine;self.d=vine.d;self.num_threads=num_threads;self.block_size=block_size
        self.backend='numpy';self.fallback_reason=None;self.compiled=None;self.requested_backend=backend
        if backend=='numpy': return
        try:
            # Reject obviously expensive snapshots before building a kernel or allocating tables.
            first=next((c.hfunc for level in vine.pair_copulas for c in level),None)
            if first is None: raise ValueError('No fitted h-function tables')
            required=vine.d*(vine.d-1)//2*2*first.m_u*first.m_v*8
            if max_table_bytes is not None and required>max_table_bytes:
                raise MemoryError(f'Packed tables require {required} bytes; limit is {max_table_bytes}')
            path=Path(kernel) if kernel is not None else cached_kernel(cache_dir,compiler)
            validate_kernel(path)
            self.compiled=CompiledDVineSampler(vine,path,max_table_bytes=max_table_bytes)
            self.backend='compiled'
        except (OSError,ValueError,RuntimeError,MemoryError,AttributeError,subprocess.SubprocessError) as exc:
            if backend=='compiled': raise
            self.fallback_reason=f'{type(exc).__name__}: {exc}'

    def inverse_rosenblatt(self,uniforms,*,threads=None,block_size=None):
        w=np.asarray(uniforms,dtype=np.float64)
        if w.ndim!=2 or w.shape[1]!=self.d: raise ValueError('Expected (n,d) uniforms')
        if not np.isfinite(w).all(): raise ValueError('Uniforms must be finite')
        if self.compiled is not None:
            try:
                return self.compiled.inverse_rosenblatt(w,threads=self.num_threads if threads is None else threads,
                    block_size=self.block_size if block_size is None else block_size)
            except (RuntimeError,MemoryError) as exc:
                if self.requested_backend=='compiled': raise
                self.backend='numpy';self.fallback_reason=f'{type(exc).__name__}: {exc}';self.compiled=None
        return self.vine.inverse_rosenblatt(w)

    def simulate(self,n,seed=None):
        """Preserve VineRecursion.simulate's seeded draws and global RNG behavior."""
        if seed is not None: np.random.seed(seed)
        uniforms=np.random.uniform(0,1,(n,self.d))
        return self.inverse_rosenblatt(uniforms)

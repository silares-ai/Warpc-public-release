"""A/B device time of warpc GEMM kernels in one process: each loaded once,
the same inputs for all (cublas_bench's), CUDA-event timing, the order
alternating. The name "cublas" among the kernels is cublasGemmEx on the
same buffers (f16 in, f32 out and compute, its default algorithm), its C
checked against the first kernel's. A kernel is named by f.cubin, ptxas's
build of f.ptx, which is f.sass lifted to PTX (warpc lift); f.sass gives
the launch.
With AB_EACH=1 every round's time of each kernel is printed as it is
taken, in the order the round ran them, with the SM clock and the clock's
limiting reasons NVML reports right after it, and the energy the device
used in it a call (NVML's counter, read around the round). AB_GAP=S idles S seconds
before each kernel's turn: launches short enough after a gap stay under
the power cap, at the full clock. AB_BLOCK=k.sass queues each kernel's
timed launches behind a long cluster kernel (below). AB_WARM=MS runs each
kernel back to back for about MS milliseconds before its timed launches,
not waited for: the steady state of repeated calls, the clock and power
the kernel's own. AB_SERIES=R: R rounds of REPEAT launches back to back, each
round's time printed (below).
usage: ab.py ROUNDS REPEAT a.cubin [b.cubin | cublas] ..."""
import os, re, sys, ctypes, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda as cu
from header import read_header, dynamic_smem
from clusterlaunch import time_cluster, launch_cluster, series_cluster
from run_tma import cuda, tensor_map_2d, shape_of, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, CU_TENSOR_MAP_DATA_TYPE_FLOAT32
rounds, repeat, paths = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
ctx = cu.Context(); lib = cuda()
sass_of = lambda p: p[:-len('.cubin')] + '.sass' if p.endswith('.cubin') else p
M, N, K = shape_of(sass_of(next(p for p in paths if p != 'cublas')))
# cublas_bench's inputs, element for element (cublas_q*, run_tma.py
# WARPC_INPUTS=hash): at the power cap the time depends on the values the
# tensor cores multiply, so a comparison with those programs takes theirs
def val(n, salt):
    i = np.arange(n, dtype=np.uint64)
    return ((((i * np.uint64(2654435761)) + np.uint64(salt)) >> np.uint64(13)) % np.uint64(7)).astype(np.int8) - 3
A = val(M * K, 1).astype(np.float16)
B = val(N * K, 7).astype(np.float16)
dA = ctx.alloc(A.nbytes); ctx.write_bytes(dA, A.tobytes())
dB = ctx.alloc(B.nbytes); ctx.write_bytes(dB, B.tobytes())
dC = ctx.alloc(M * N * 4)
arrays = {'a': dA, 'bt': dB, 'c': dC}
def cublas_entry():
    blas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so')
    h = ctypes.c_void_p(); assert blas.cublasCreate_v2(ctypes.byref(h)) == 0
    dC2 = ctx.alloc(M * N * 4)
    alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
    # row-major C[M,N] = A[M,K] Bt[N,K]^T is column-major C^T[N,M] = Bt^T' A: T, N
    def gemm():
        r = blas.cublasGemmEx(h, 1, 0, N, M, K, ctypes.byref(alpha), ctypes.c_void_p(dB.ptr), 2, K, ctypes.c_void_p(dA.ptr), 2, K,
                              ctypes.byref(beta), ctypes.c_void_p(dC2.ptr), 0, N, 68, 0)  # CUDA_R_16F=2, CUDA_R_32F=0, CUBLAS_COMPUTE_32F=68, default algo
        assert r == 0, r
    ev0, ev1 = ctypes.c_void_p(), ctypes.c_void_p()
    lib.cuEventCreate(ctypes.byref(ev0), 0); lib.cuEventCreate(ctypes.byref(ev1), 0)
    def t(n=repeat):
        lib.cuEventRecord(ev0, None)
        for _ in range(n): gemm()
        lib.cuEventRecord(ev1, None); lib.cuEventSynchronize(ev1)
        ms = ctypes.c_float(); lib.cuEventElapsedTime(ctypes.byref(ms), ev0, ev1)
        return ms.value / n
    return t, dC2

def load(p, arrays):
    """a kernel ready to launch: its function, grid, block, arguments, dynamic
    shared memory, cluster -- its tensor maps over [arrays], each made once
    by name at the size the kernel's header states"""
    if not p.endswith('.cubin'):
        sys.exit('%s: a kernel is named by its ptxas build, f.cubin' % p)
    hdr = read_header(sass_of(p))
    image = open(p, 'rb').read()
    # its shared memory is all dynamic, as the PTX's launch line states
    dyn = dynamic_smem(p[:-len('.cubin')] + '.ptx')
    mod = ctx.load_module(image); f = mod.get_function(hdr['kernel'])
    lib.cuFuncSetAttribute(f._handle, 8, dyn)
    args = []
    for e in hdr['tmap']:
        if e[0] == 'ptr':
            if e[1] not in arrays:
                # a buffer of the kernel's own (a workspace, flags): zeroed once
                nb = int(e[2]) * int(e[3]) * int(e[4])
                arrays[e[1]] = ctx.alloc(nb); ctx.write_bytes(arrays[e[1]], b'\x00' * nb)
            args.append(ctypes.c_uint64(arrays[e[1]].ptr)); continue
        name, rows, cols, elem, br, bc, swz = e[0], *map(int, e[1:])
        if name not in arrays:
            arrays[name] = ctx.alloc(rows * cols * elem); ctx.write_bytes(arrays[name], b'\x00' * (rows * cols * elem))
        tm = tensor_map_2d(arrays[name].ptr, rows, cols, elem, br, bc, swizzle=CU_TENSOR_MAP_SWIZZLE_128B,
                           dtype={2: CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4: CU_TENSOR_MAP_DATA_TYPE_FLOAT32}[elem])
        d = ctx.alloc(128); ctx.write_bytes(d, bytes(tm)); args.append(ctypes.c_uint64(d.ptr))
    if 'debug' in hdr:
        # a timing build's stamp buffer
        dbg = ctx.alloc(8 << 20); ctx.write_bytes(dbg, b'\x00' * (8 << 20)); args.append(ctypes.c_uint64(dbg.ptr))
    grid = tuple(int(x) for x in hdr['grid'].split()) + (1,)
    return mod, f, grid, (hdr['threads'], 1, 1), args, dyn, int(hdr.get('cluster', 1))

# AB_PDL=1: each cluster kernel's launches may begin while the one before
# runs (time_cluster's pdl)
pdl = os.environ.get('AB_PDL') == '1'
# AB_SERIES=R: each kernel alone, R rounds of REPEAT launches back to back
# with no wait between them, each round's time a launch printed -- a long
# run of calls from a cold start, as cublas_q7's CQ_ROUNDS times cuBLAS
series = int(os.environ.get('AB_SERIES', '0'))
if series:
    for p in paths:
        mod, f, grid, block, args, dyn, cl = load(p, arrays)
        for q, ms in enumerate(series_cluster(f, grid, block, args, shared_mem=dyn, cluster=(cl, 1, 1), repeat=repeat, rounds=series, pdl=pdl)):
            print('series %s round %d %.2f us' % (p, q, ms * 1000))
    sys.exit(0)
warm_ms = float(os.environ.get('AB_WARM', '0'))
ks = []
dC2 = None
for p in paths:
    if p == 'cublas':
        t, dC2 = cublas_entry(); t(3); ks.append((p, t, None)); continue
    mod, f, grid, block, args, dyn, cl = load(p, arrays)
    if pdl and 'ACQBULK' not in open(sass_of(p)).read():
        sys.exit('AB_PDL: %s has no ACQBULK, so the launch before it may still be writing what it reads' % p)
    def t(f=f, grid=grid, block=block, args=args, dyn=dyn, cl=cl, n=repeat, w=0):
        return time_cluster(f, grid, block, args, shared_mem=dyn, cluster=(cl, 1, 1), repeat=n, pdl=pdl, warm=w) if cl > 1 or pdl else f.launch(grid=grid, block=block, args=args, shared_mem=dyn, timed=True, repeat=n)
    once = t(n=3)  # warm
    if warm_ms:
        # about warm_ms of device time of this kernel ahead of each timed window
        t = (lambda t=t, w=max(1, int(warm_ms / once)): t(w=w))
    ks.append((p, t, mod))
# AB_BLOCK=k.cubin: a cluster kernel launched, not waited for, before each
# kernel's turn, on buffers of its own; its milliseconds give the host time
# to queue the timed launches behind it, so they run back to back on the
# device whatever launching from Python costs
blocker = None
if os.environ.get('AB_BLOCK'):
    bmod, bf, bgrid, bblock, bargs, bdyn, bcl = load(os.environ['AB_BLOCK'], {})
    assert bcl > 1, 'AB_BLOCK: a cluster kernel, whose launch need not wait'
    blocker = lambda: launch_cluster(bf, bgrid, bblock, bargs, shared_mem=bdyn, cluster=(bcl, 1, 1), sync=False)
res = {p: [] for p, _, _ in ks}
each = os.environ.get('AB_EACH') == '1'
gap = float(os.environ.get('AB_GAP', '0'))
def energy():
    """the device's energy counter in mJ, by NVML, or None"""
    try:
        nv = ctypes.CDLL('libnvidia-ml.so.1')
        if not hasattr(clock, 'h'):
            nv.nvmlInit_v2(); clock.h = ctypes.c_void_p(); nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(clock.h))
        mj = ctypes.c_ulonglong()
        return mj.value if nv.nvmlDeviceGetTotalEnergyConsumption(clock.h, ctypes.byref(mj)) == 0 else None
    except OSError:
        return None
def clock():
    """the SM clock in MHz and the reasons limiting it, by NVML, or None"""
    try:
        nv = ctypes.CDLL('libnvidia-ml.so.1')
        if not hasattr(clock, 'h'):
            nv.nvmlInit_v2(); clock.h = ctypes.c_void_p(); nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(clock.h))
        mhz, why = ctypes.c_uint(), ctypes.c_ulonglong()
        nv.nvmlDeviceGetClockInfo(clock.h, 1, ctypes.byref(mhz))
        nv.nvmlDeviceGetCurrentClocksEventReasons(clock.h, ctypes.byref(why))
        return mhz.value, why.value
    except OSError:
        return None
import time
for r in range(rounds):
    for p, t, _ in (ks if r % 2 == 0 else ks[::-1]):
        if gap:
            time.sleep(gap)
        if blocker:
            blocker()
        e0 = energy() if each else None
        res[p].append(t())
        if each:
            e1 = energy(); c = clock()
            print('round %d %s %.4f ms%s%s' % (r, p, res[p][-1], '' if c is None else '  clock %d MHz, reasons 0x%x' % c,
                  '' if e0 is None or e1 is None else '  energy %.1f mJ a call' % ((e1 - e0) / (repeat + 1))), flush=True)  # time_cluster's warming launch is in it
if dC2 is not None:
    c1 = np.frombuffer(ctx.read_bytes(dC, M * N * 4), dtype=np.float32)
    c2 = np.frombuffer(ctx.read_bytes(dC2, M * N * 4), dtype=np.float32)
    print('cublas C against the first kernel\'s: %d of %d differ' % (int(np.sum(np.abs(c1 - c2) > 1e-3)), M * N))
for p, v in res.items():
    print('%-40s median %.4f ms  min %.4f  max %.4f  (%d x %d)' % (p, statistics.median(v), min(v), max(v), rounds, repeat))

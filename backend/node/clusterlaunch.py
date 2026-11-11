"""Launch a kernel with a cluster dimension, via cuLaunchKernelEx."""
import ctypes, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda as cu

CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6

class LaunchAttrValue(ctypes.Union):
    _fields_ = [('pad', ctypes.c_byte * 64), ('clusterDim', ctypes.c_uint * 3)]

class LaunchAttribute(ctypes.Structure):
    _fields_ = [('id', ctypes.c_int), ('pad', ctypes.c_byte * 4), ('value', LaunchAttrValue)]

class LaunchConfig(ctypes.Structure):
    _fields_ = [('gridDimX', ctypes.c_uint), ('gridDimY', ctypes.c_uint), ('gridDimZ', ctypes.c_uint),
                ('blockDimX', ctypes.c_uint), ('blockDimY', ctypes.c_uint), ('blockDimZ', ctypes.c_uint),
                ('sharedMemBytes', ctypes.c_uint), ('hStream', ctypes.c_void_p),
                ('attrs', ctypes.POINTER(LaunchAttribute)), ('numAttrs', ctypes.c_uint)]

_keepalive = []


def _bind():
    l = cu.lib()
    if not hasattr(l, '_ex_bound'):
        l.cuLaunchKernelEx.argtypes = [ctypes.POINTER(LaunchConfig), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        l.cuLaunchKernelEx.restype = ctypes.c_int
        l.cuOccupancyMaxActiveClusters.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.POINTER(LaunchConfig)]
        l.cuOccupancyMaxActiveClusters.restype = ctypes.c_int
        l._ex_bound = True
    return l

def launch_cluster(func, grid, block, args, shared_mem=0, cluster=(1, 1, 1), sync=True):
    lib = _bind()
    params, keep = cu.build_params(args)
    _keepalive.append((params, keep))
    if len(_keepalive) > 4096:
        del _keepalive[:2048]
    attr = LaunchAttribute()
    attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attr.value.clusterDim[0], attr.value.clusterDim[1], attr.value.clusterDim[2] = cluster
    cfg = LaunchConfig(grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem, None,
                       ctypes.pointer(attr), 1)
    r = lib.cuLaunchKernelEx(ctypes.byref(cfg), func._handle, params, None)
    if r != 0:
        raise RuntimeError('cuLaunchKernelEx: %d' % r)
    if sync:
        r = lib.cuCtxSynchronize()
        if r != 0:
            raise RuntimeError('cuCtxSynchronize: %d' % r)


def time_cluster(func, grid, block, args, shared_mem=0, cluster=(1, 1, 1), repeat=20, pdl=False, warm=0):
    """Mean device time per launch, in ms: the launch is built once and issued
    [repeat] times back to back between two CUDA events, as the plain launch
    is timed -- so the host's cost per launch is not measured as kernel time.
    The warming launch is not waited for, so work queued before the call
    keeps the device busy while the timed launches are issued. [pdl]: each
    launch may begin while the one before runs, from its PREEXIT on (the
    programmatic stream serialization attribute); the kernel's ACQBULK waits
    for the grid before it. [warm]: launches of the same kind, not waited
    for, before the timed ones -- the device's clock and power then the
    kernel's own, and the timed launches queued behind them."""
    lib = _bind()
    launch_cluster(func, grid, block, args, shared_mem, cluster, sync=False)  # binds, warms
    params, keep = cu.build_params(args)
    attrs = (LaunchAttribute * 2)()
    attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attrs[0].value.clusterDim[0], attrs[0].value.clusterDim[1], attrs[0].value.clusterDim[2] = cluster
    attrs[1].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    ctypes.cast(ctypes.byref(attrs[1].value), ctypes.POINTER(ctypes.c_int))[0] = 1
    cfg = LaunchConfig(grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem, None,
                       attrs, 2 if pdl else 1)
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    lib.cuEventCreate(ctypes.byref(start), 0)
    lib.cuEventCreate(ctypes.byref(stop), 0)
    for _ in range(warm):
        r = lib.cuLaunchKernelEx(ctypes.byref(cfg), func._handle, params, None)
        if r != 0:
            raise RuntimeError('cuLaunchKernelEx: %d' % r)
    lib.cuEventRecord(start, None)
    for _ in range(repeat):
        r = lib.cuLaunchKernelEx(ctypes.byref(cfg), func._handle, params, None)
        if r != 0:
            raise RuntimeError('cuLaunchKernelEx: %d' % r)
    lib.cuEventRecord(stop, None)
    r = lib.cuEventSynchronize(stop)
    if r != 0:
        raise RuntimeError('cuEventSynchronize: %d' % r)
    ms = ctypes.c_float()
    lib.cuEventElapsedTime(ctypes.byref(ms), start, stop)
    lib.cuEventDestroy_v2(start)
    lib.cuEventDestroy_v2(stop)
    del keep
    return ms.value / repeat

def series_cluster(func, grid, block, args, shared_mem=0, cluster=(1, 1, 1), repeat=20, rounds=10, pdl=False):
    """Device time per launch of each of [rounds] rounds of [repeat] launches,
    in ms, every launch back to back: an event between rounds and no wait
    until the last -- the series a long run of calls goes through"""
    lib = _bind()
    params, keep = cu.build_params(args)
    attrs = (LaunchAttribute * 2)()
    attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attrs[0].value.clusterDim[0], attrs[0].value.clusterDim[1], attrs[0].value.clusterDim[2] = cluster
    attrs[1].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    ctypes.cast(ctypes.byref(attrs[1].value), ctypes.POINTER(ctypes.c_int))[0] = 1
    cfg = LaunchConfig(grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem, None,
                       attrs, 2 if pdl else 1)
    evs = [ctypes.c_void_p() for _ in range(rounds + 1)]
    for e in evs:
        lib.cuEventCreate(ctypes.byref(e), 0)
    lib.cuEventRecord(evs[0], None)
    for q in range(rounds):
        for _ in range(repeat):
            r = lib.cuLaunchKernelEx(ctypes.byref(cfg), func._handle, params, None)
            if r != 0:
                raise RuntimeError('cuLaunchKernelEx: %d' % r)
        lib.cuEventRecord(evs[q + 1], None)
    r = lib.cuEventSynchronize(evs[rounds])
    if r != 0:
        raise RuntimeError('cuEventSynchronize: %d' % r)
    out = []
    for q in range(rounds):
        ms = ctypes.c_float()
        lib.cuEventElapsedTime(ctypes.byref(ms), evs[q], evs[q + 1])
        out.append(ms.value / repeat)
    for e in evs:
        lib.cuEventDestroy_v2(e)
    del keep
    return out


def max_active_clusters(func, grid, block, shared_mem=0, cluster=(1, 1, 1)):
    """How many clusters of this shape the device holds at once
    (cuOccupancyMaxActiveClusters)."""
    lib = _bind()
    attr = LaunchAttribute()
    attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attr.value.clusterDim[0], attr.value.clusterDim[1], attr.value.clusterDim[2] = cluster
    cfg = LaunchConfig(grid[0], grid[1], grid[2], block[0], block[1], block[2], shared_mem, None,
                       ctypes.pointer(attr), 1)
    n = ctypes.c_int()
    r = lib.cuOccupancyMaxActiveClusters(ctypes.byref(n), func._handle, ctypes.byref(cfg))
    if r != 0:
        raise RuntimeError("cuOccupancyMaxActiveClusters: %d" % r)
    return n.value

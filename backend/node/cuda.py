"""The CUDA driver API calls the runners make, through ctypes: one context on
device 0, device memory, a module loaded from a cubin, a kernel launched
(timed between CUDA events when asked)."""
import ctypes

_lib = None


def lib():
    """libcuda, its signatures bound and cuInit called once"""
    global _lib
    if _lib is not None:
        return _lib
    l = ctypes.CDLL('libcuda.so.1')
    vp, u32, i32, u64, sz = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t
    sig = {
        'cuInit': [u32],
        'cuDeviceGet': [ctypes.POINTER(i32), i32],
        'cuDevicePrimaryCtxRetain': [ctypes.POINTER(vp), i32],
        'cuCtxSetCurrent': [vp],
        'cuCtxSynchronize': [],
        'cuMemAlloc_v2': [ctypes.POINTER(u64), sz],
        'cuMemcpyHtoD_v2': [u64, vp, sz],
        'cuMemcpyDtoH_v2': [vp, u64, sz],
        'cuModuleLoadData': [ctypes.POINTER(vp), vp],
        'cuModuleGetFunction': [ctypes.POINTER(vp), vp, ctypes.c_char_p],
        'cuLaunchKernel': [vp, u32, u32, u32, u32, u32, u32, u32, vp, ctypes.POINTER(vp), ctypes.POINTER(vp)],
        'cuEventCreate': [ctypes.POINTER(vp), u32],
        'cuEventRecord': [vp, vp],
        'cuEventSynchronize': [vp],
        'cuEventElapsedTime': [ctypes.POINTER(ctypes.c_float), vp, vp],
        'cuEventDestroy_v2': [vp],
        'cuFuncSetAttribute': [vp, i32, i32],
    }
    for name, args in sig.items():
        f = getattr(l, name)
        f.argtypes, f.restype = args, ctypes.c_int
    check(l.cuInit(0), 'cuInit')
    _lib = l
    return l


def check(r, what):
    if r != 0:
        raise RuntimeError('%s: CUDA error %d' % (what, r))


def build_params(args):
    """the kernelParams array for [args], ctypes values; returned with what
    must stay alive until the launch has been issued"""
    ptrs = (ctypes.c_void_p * max(1, len(args)))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    return ptrs, list(args)


class DevPtr:
    def __init__(self, ptr, nbytes):
        self.ptr, self.nbytes = ptr, nbytes


class Function:
    def __init__(self, handle, module):
        self._handle, self._module = handle, module

    def launch(self, grid=(1, 1, 1), block=(1, 1, 1), args=(), shared_mem=0, timed=False, repeat=1):
        """one launch, waited for; with [timed], [repeat] launches back to back
        between two CUDA events, the mean milliseconds a launch returned"""
        l = lib()
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        params, keep = build_params(list(args))

        def go():
            check(l.cuLaunchKernel(self._handle, g[0], g[1], g[2], b[0], b[1], b[2], shared_mem, None, params, None), 'cuLaunchKernel')

        if not timed:
            go()
            check(l.cuCtxSynchronize(), 'cuCtxSynchronize')
            return None
        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        check(l.cuEventCreate(ctypes.byref(start), 0), 'cuEventCreate')
        check(l.cuEventCreate(ctypes.byref(stop), 0), 'cuEventCreate')
        check(l.cuEventRecord(start, None), 'cuEventRecord')
        for _ in range(max(1, repeat)):
            go()
        check(l.cuEventRecord(stop, None), 'cuEventRecord')
        check(l.cuEventSynchronize(stop), 'cuEventSynchronize')
        ms = ctypes.c_float()
        check(l.cuEventElapsedTime(ctypes.byref(ms), start, stop), 'cuEventElapsedTime')
        l.cuEventDestroy_v2(start)
        l.cuEventDestroy_v2(stop)
        del keep
        return ms.value / max(1, repeat)


class Module:
    def __init__(self, handle, image):
        self._handle, self._image = handle, image

    def get_function(self, name):
        f = ctypes.c_void_p()
        check(lib().cuModuleGetFunction(ctypes.byref(f), self._handle, name.encode()), 'cuModuleGetFunction ' + name)
        return Function(f, self)


class Context:
    """the primary context of device 0, made current"""

    def __init__(self, device=0):
        l = lib()
        d = ctypes.c_int()
        check(l.cuDeviceGet(ctypes.byref(d), device), 'cuDeviceGet')
        self._ctx = ctypes.c_void_p()
        check(l.cuDevicePrimaryCtxRetain(ctypes.byref(self._ctx), d.value), 'cuDevicePrimaryCtxRetain')
        check(l.cuCtxSetCurrent(self._ctx), 'cuCtxSetCurrent')

    def alloc(self, nbytes):
        p = ctypes.c_uint64()
        check(lib().cuMemAlloc_v2(ctypes.byref(p), nbytes), 'cuMemAlloc')
        return DevPtr(p.value, nbytes)

    def write_bytes(self, dst, data):
        buf = ctypes.create_string_buffer(bytes(data), len(data))
        check(lib().cuMemcpyHtoD_v2(dst.ptr, buf, len(data)), 'cuMemcpyHtoD')

    def read_bytes(self, src, nbytes=None):
        n = nbytes or src.nbytes
        buf = ctypes.create_string_buffer(n)
        check(lib().cuMemcpyDtoH_v2(buf, src.ptr, n), 'cuMemcpyDtoH')
        return buf.raw

    def load_module(self, image):
        img = ctypes.create_string_buffer(bytes(image), len(image))
        m = ctypes.c_void_p()
        check(lib().cuModuleLoadData(ctypes.byref(m), img), 'cuModuleLoadData')
        return Module(m, img)

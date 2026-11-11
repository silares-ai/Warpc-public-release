"""Run a warpc GEMM kernel on the device: k.sass is warpc's output, k.ptx
its lift (warpc lift k.sass > k.ptx) and k.cubin ptxas's build of that
(ptxas -arch=sm_100a k.ptx -o k.cubin). Encode the tensor maps the header
describes, launch the kernel (clustered when it asks for a cluster), check
the result against numpy and time it.
usage: run_tma.py k.sass k.cubin [repeat]"""
import os, sys, re, ctypes, struct, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda as cu
from header import read_header, dynamic_smem
from clusterlaunch import launch_cluster, time_cluster

CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6
CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7
CU_TENSOR_MAP_SWIZZLE_128B = 3
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

def cuda():
    lib = cu.lib()
    if not hasattr(lib, '_tma_bound'):
        lib.cuTensorMapEncodeTiled.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                                               ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.cuTensorMapEncodeTiled.restype = ctypes.c_int
        lib.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.cuFuncSetAttribute.restype = ctypes.c_int
        lib._tma_bound = True
    return lib

def tensor_map_2d(dptr, rows, cols, elem_bytes, box_rows, box_cols, swizzle=CU_TENSOR_MAP_SWIZZLE_128B, dtype=CU_TENSOR_MAP_DATA_TYPE_FLOAT16):
    """A row-major rows x cols matrix; box = box_rows x box_cols; returns 128 bytes."""
    lib = cuda()
    buf = ctypes.create_string_buffer(256)
    addr = (ctypes.addressof(buf) + 63) & ~63
    dims = (ctypes.c_uint64 * 2)(cols, rows)
    strides = (ctypes.c_uint64 * 1)(cols * elem_bytes)
    box = (ctypes.c_uint32 * 2)(box_cols, box_rows)
    estr = (ctypes.c_uint32 * 2)(1, 1)
    r = lib.cuTensorMapEncodeTiled(ctypes.c_void_p(addr), dtype, 2, ctypes.c_void_p(int(dptr)),
                                   dims, strides, box, estr, 0, swizzle, int(os.environ.get("WARPC_L2PROMO", "2")), 0)
    if r != 0:
        raise RuntimeError('cuTensorMapEncodeTiled failed: %d' % r)
    return ctypes.string_at(addr, 128)

def run(path, cubin, M, N, K, repeat=20, seed=1):
    hdr = read_header(path)
    tm, tn, tk = map(int, hdr['tile'].split())
    image = open(cubin, 'rb').read()
    # the lifted kernel's shared memory is all dynamic, as its PTX states
    dyn = dynamic_smem(cubin[:-len('.cubin')] + '.ptx')
    ctx = cu.Context()
    lib = cuda()
    rng = np.random.default_rng(seed)
    if os.environ.get("WARPC_INPUTS") == "hash":
        # cublas_bench.cu's inputs, element for element: val(i, salt)
        def val(n, salt):
            i = np.arange(n, dtype=np.uint64)
            return ((((i * np.uint64(2654435761)) + np.uint64(salt)) >> np.uint64(13)) % np.uint64(7)).astype(np.int64) - 3
        A = val(M * K, 1).reshape(M, K).astype(np.float16)
        Bt = val(N * K, 7).reshape(N, K).astype(np.float16)
    else:
        A = rng.integers(-3, 4, size=(M, K)).astype(np.float16)
        Bt = rng.integers(-3, 4, size=(N, K)).astype(np.float16)
    # WARPC_NO_CHECK: a timing run of a kernel already checked on these inputs
    ref = None if os.environ.get("WARPC_NO_CHECK") else A.astype(np.float32) @ Bt.astype(np.float32).T
    dA = ctx.alloc(A.nbytes); ctx.write_bytes(dA, A.tobytes())
    dB = ctx.alloc(Bt.nbytes); ctx.write_bytes(dB, Bt.tobytes())
    dC = ctx.alloc(M * N * 4); ctx.write_bytes(dC, b'\x00' * (M * N * 4))
    cl = int(hdr.get('cluster', 1))
    mod = ctx.load_module(image)
    func = mod.get_function(hdr['kernel'])
    r = lib.cuFuncSetAttribute(func._handle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dyn)
    if r != 0:
        raise RuntimeError('cuFuncSetAttribute failed: %d' % r)
    # Every tensor map as the kernel states it: the box and the swizzle are read
    # off the layout of the shared tile each matrix moves through, so the host
    # encodes exactly the image the kernel's addresses assume. Nothing here is
    # a default.
    arrays = {'a': (dA, (M, K)), 'bt': (dB, (N, K)), 'c': (dC, (M, N))}
    dtypes = {2: CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4: CU_TENSOR_MAP_DATA_TYPE_FLOAT32}
    swizzles = {128: CU_TENSOR_MAP_SWIZZLE_128B}
    if len(hdr.get('tmap', [])) != len(hdr['params']) - (1 if 'debug' in hdr else 0):
        raise RuntimeError('every parameter must be a tensor map the kernel describes')
    ptrs = []
    for entry in hdr['tmap']:
        if entry[0] == 'ptr':
            # a plain pointer: the matrix itself
            name, rows, cols = entry[1], int(entry[2]), int(entry[3])
            if name not in arrays:
                # a buffer of the kernel's own (a workspace, flags): zeroed once
                nb = rows * cols * int(entry[4])
                buf = ctx.alloc(nb); ctx.write_bytes(buf, b"\x00" * nb)
                arrays[name] = (buf, (rows, cols))
            buf, shape = arrays[name]
            if (rows, cols) != shape:
                raise RuntimeError('%s: the kernel was compiled for %dx%d, the run is %dx%d' % ((name, rows, cols) + shape))
            ptrs.append(buf.ptr)
            continue
        name, rows, cols, elem, brows, bcols, swz = entry
        rows, cols, elem, brows, bcols, swz = map(int, (rows, cols, elem, brows, bcols, swz))
        buf, shape = arrays[name]
        if (rows, cols) != shape:
            raise RuntimeError('%s: the kernel was compiled for %dx%d, the run is %dx%d' % ((name, rows, cols) + shape))
        dmap = ctx.alloc(128)
        tm = bytearray(tensor_map_2d(buf.ptr, rows, cols, elem, brows, bcols, swizzle=swizzles[swz], dtype=dtypes[elem]))
        # WARPC_TMAP_XOR="byte:mask,...": flip bits of the operand maps, for measuring
        if name != 'c' and os.environ.get('WARPC_TMAP_XOR'):
            for item in os.environ['WARPC_TMAP_XOR'].split(','):
                bi, mk = item.split(':'); tm[int(bi)] ^= int(mk, 0)
        ctx.write_bytes(dmap, bytes(tm))
        ptrs.append(dmap.ptr)
    args = [ctypes.c_uint64(p) for p in ptrs]
    dbg = None
    if 'debug' in hdr:
        # a debugging build: each wait that gives up writes (wait, parity) here
        ndbg = 1 << 20
        dbg = ctx.alloc(8 * ndbg); ctx.write_bytes(dbg, b'\x00' * (8 * ndbg))
        args.append(ctypes.c_uint64(dbg.ptr))
    grid = (tuple(int(x) for x in hdr['grid'].split()) + (1,)) if 'grid' in hdr else (N // tn, M // tm, 1)
    block = (hdr['threads'], 1, 1)
    if cl > 1:
        from clusterlaunch import max_active_clusters
        print("clusters resident at once: %d of %d" % (max_active_clusters(func, grid, block, shared_mem=dyn, cluster=(cl, 1, 1)), grid[0] // cl), file=sys.stderr, flush=True)
        launch_cluster(func, grid, block, args, shared_mem=dyn, cluster=(cl, 1, 1))
    else:
        func.launch(grid=grid, block=block, args=args, shared_mem=dyn, timed=False)
    if dbg is not None and "stamps" not in hdr and "stagestamps" not in hdr:
        import collections
        w = np.frombuffer(ctx.read_bytes(dbg, 8 * (1 << 20)), dtype=np.uint32).reshape(-1, 2)
        hit = np.nonzero(w[:, 0])[0]
        sites = dict(l[2:].split(': ', 1) for l in open(path) if l.startswith('# wait '))
        print('waits that gave up: %d' % len(hit))
        for (wid, par), n in collections.Counter((int(w[i, 0]), int(w[i, 1])) for i in hit).most_common(12):
            ex = [int(i) for i in hit if w[i, 0] == wid][:4]
            print('   %4d x  %s  parity %#x  (cta.warp %s)' % (n, sites.get('wait %d' % wid, '?'), par, ', '.join('%d.%d' % (i // 8, i % 8) for i in ex)))
    if ref is None:
        ms = func.launch(grid=grid, block=block, args=args, shared_mem=dyn, timed=True, repeat=repeat) if cl == 1 else time_cluster(func, grid, block, args, shared_mem=dyn, cluster=(cl, 1, 1), repeat=repeat)
        print('%-28s %dx%dx%d grid=%s  UNCHECKED  %.4f ms' % (path, M, N, K, grid[:2], ms))
        return True
    C = np.frombuffer(ctx.read_bytes(dC, M * N * 4), dtype=np.float32).reshape(M, N)
    bad = np.argwhere(np.abs(C - ref) > 1e-3)
    if os.environ.get('WARPC_DUMP'):
        np.save(os.environ['WARPC_DUMP'], np.stack([C, ref]))
    # the first launch has finished and been checked; say so before timing,
    # which launches again, so a hang is attributed to the right launch
    print('first launch done, %d wrong' % len(bad), file=sys.stderr, flush=True)
    if os.environ.get('WARPC_TILES') and len(bad):
        # which output tiles are wrong, and whether they were never written
        wrong = np.abs(C - ref) > 1e-3
        t = wrong.reshape(M // tm, tm, N // tn, tn).any(axis=(1, 3))
        z = (C == 0).reshape(M // tm, tm, N // tn, tn).all(axis=(1, 3))
        print("   wrong tiles %d of %d, never written %d" % (int(t.sum()), t.size, int((t & z).sum())))
        rows = [''.join('x' if t[i, j] else '.' for j in range(t.shape[1])) for i in range(min(t.shape[0], 16))]
        print("   " + "\n   ".join(rows))
    if os.environ.get('WARPC_DIAG') and len(bad):
        import collections
        z = int(np.sum(C[bad[:,0], bad[:,1]] == 0.0))
        print("   zeros %d of %d" % (z, len(bad)))
        print("   tiles %s" % collections.Counter((int(r)//tm, int(c)//tn) for r,c in bad).most_common(5))
        print("   warp-of-tile %s" % sorted(collections.Counter((int(r)%tm)//32 for r,_ in bad).items()))
        print("   chunk %s" % sorted(collections.Counter((int(c)%tn)//32 for _,c in bad).items()))
        print("   piece %s" % sorted(collections.Counter((int(c)%32)//4 for _,c in bad).items()))
        print("   row-in-32 %s" % sorted(collections.Counter(int(r)%32 for r,_ in bad).items())[:10])
        nz = bad[C[bad[:,0], bad[:,1]] != 0.0]
        if len(nz):
            r0, c0 = nz[0]
            hit = np.argwhere(np.abs(ref - C[r0, c0]) < 1e-3)
            print("   nonzero bad (%d,%d) got %s ; ref has that value at %s" % (r0, c0, C[r0,c0], hit[:4].tolist()))
    if cl > 1:
        # device time, between CUDA events, as the plain launch below is timed
        ms = time_cluster(func, grid, block, args, shared_mem=dyn, cluster=(cl, 1, 1), repeat=repeat)
    else:
        ms = func.launch(grid=grid, block=block, args=args, shared_mem=dyn, timed=True, repeat=repeat)
    if "stamps" in hdr:
        # the last launch's phase stamps: 64 global-timer words a CTA, 16 events
        # for each of its first four tiles
        g = grid[0]
        s = np.frombuffer(ctx.read_bytes(dbg, 256 * g), dtype=np.uint32).reshape(g, 4, 16).astype(np.int64)
        names = dict((int(l.split()[2][:-1]), l.split(": ", 1)[1].strip()) for l in open(path) if l.startswith("# stamp "))
        ran = s[:, 0, 0] != 0
        t0 = s[ran, 0, 0].min()
        print("  %d CTAs ran" % ran.sum())
        for e, nm in sorted(names.items()):
            cells = []
            for t in range(4):
                col = s[ran, t, e]
                col = col[col != 0] - t0
                cells.append("%5d:%6d/%6d/%6d" % (len(col), col.min(), np.median(col), col.max()) if len(col) else " " * 26)
            print("  %2d %-30s %s" % (e, nm, "  ".join(cells)))
        np.save(os.environ.get("WARPC_STAMPS", "stamps.npy"), s)
    if "stagestamps" in hdr:
        g = grid[0]
        g = min(g, 256)
        s = np.frombuffer(ctx.read_bytes(dbg, 4 * 4096 * g), dtype=np.uint32).reshape(g, 4096).astype(np.int64)
        np.save(os.environ.get("WARPC_STAMPS", "stagestamps.npy"), s)
        d = np.diff(s, axis=1)
        ok = (s[:, 1:] != 0) & (s[:, :-1] != 0) & (d > 0) & (d < 10**7)
        v = d[ok]
        print("  stage gaps: n %d  p10 %d  p50 %d  p90 %d  p99 %d  max %d  mean %.1f ns" % (len(v), *np.percentile(v, [10, 50, 90, 99]), v.max(), v.mean()))
        big = np.sort(v)[::-1][:12]
        print("  largest", big.tolist())
    if os.environ.get("WARPC_VERIFY_LAST"):
        # the last timed launch's result too: what a kernel leaves for the
        # next launch (a flag, a workspace) must not change the answer
        C2 = np.frombuffer(ctx.read_bytes(dC, M * N * 4), dtype=np.float32).reshape(M, N)
        print("  last launch: %d wrong" % int(np.sum(np.abs(C2 - ref) > 1e-3)))
    flops = 2.0 * M * N * K
    print('%-28s %dx%dx%d grid=%s  %s bad=%d/%d  %.4f ms  %.1f TFLOP/s' % (path, M, N, K, grid[:2], 'MATCH' if len(bad) == 0 else 'MISMATCH', len(bad), M * N, ms, flops / ms / 1e9))
    if len(bad):
        i, j = bad[0]; print('   first bad (%d,%d): got %s ref %s' % (i, j, C[i, j], ref[i, j]))
        print('   row0 got', C[0, :6], 'ref', ref[0, :6])
        rows_bad = sorted(set(bad[:, 0].tolist())); print('   bad rows: %d, first %s' % (len(rows_bad), rows_bad[:8]))
        cols_bad = sorted(set(bad[:, 1].tolist())); print('   bad cols: %d, first %s' % (len(cols_bad), cols_bad[:8]))
    return len(bad) == 0

def shape_of(path):
    """M, N, K as the kernel's tensor maps state them: a is M x K, bt is N x K."""
    dims = {}
    for line in open(path):
        if line.startswith('.tmap '):
            name, rows, cols = line.split()[1:4]
            dims[name] = (int(rows), int(cols))
    (m, k), (n, _) = dims['a'], dims['bt']
    return m, n, k

if __name__ == '__main__':
    path, cubin = sys.argv[1], sys.argv[2]
    run(path, cubin, *shape_of(path), repeat=int(sys.argv[3]) if len(sys.argv) > 3 else 30)

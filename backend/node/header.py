"""The header warpc writes above a kernel's SASS text: what the launch needs.

    .kernel NAME   .sm sm_100a   .regs N   .barriers N   .params 8 8 ...
    .threads N     .smem BYTES   .tcgen05  .grid X Y      .cluster N
    .tmap name rows cols elem_bytes box_rows box_cols swizzle_bytes   (a tensor map)
    .ptr  name rows cols elem_bytes                                    (a plain pointer)
    .tile M N K    .dynsmem BYTES

one per parameter, .tmap and .ptr in parameter order."""


def read_header(path):
    hdr = {'kernel': 'k', 'sm': 'sm_100a', 'regs': 16, 'barriers': 0, 'params': [], 'threads': None, 'smem': 0,
           'tcgen05': False, 'mbarriers': 1}
    for raw in open(path):
        s = raw.split('#')[0].split('//')[0].strip()
        if not s.startswith('.'):
            continue
        key, _, rest = s[1:].partition(' ')
        if key == 'params':
            hdr['params'] = [int(x, 0) for x in rest.split()]
        elif key in ('regs', 'barriers', 'threads', 'smem', 'mbarriers'):
            hdr[key] = int(rest, 0)
        elif key == 'tcgen05':
            hdr['tcgen05'] = True
        elif key == 'tmap':
            hdr.setdefault('tmap', []).append(rest.split())
        elif key == 'ptr':
            hdr.setdefault('tmap', []).append(['ptr'] + rest.split())
        else:
            hdr[key] = rest.strip()
    return hdr


def dynamic_smem(ptx_path):
    """the dynamic shared memory a kernel lifted to PTX (warpc lift) asks for,
    as the PTX's launch line states it"""
    import re
    line = next(l for l in open(ptx_path) if l.startswith('// launch:'))
    return int(re.search(r'dynamic shared memory (\d+) bytes', line).group(1))

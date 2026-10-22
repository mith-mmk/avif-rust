"""Summarize local measurements without saving image planes in the repository.

Usage: python3 summarize.py /path/to/private-measurements > evidence.json
Requires numpy. Expected layout is documented in REPORT.md.
"""
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

b = Path(sys.argv[1])


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(a, r, shapes):
    assert len(a) == len(r)
    v = np.abs(a.astype(np.int32) - r.astype(np.int32))
    result = dict(samples=len(a), count=int(np.count_nonzero(v)),
                  max=int(v.max()), mae=float(v.mean()), planes=[])
    off = 0
    for plane, (w, h) in enumerate(shapes):
        n = w * h
        ii = np.flatnonzero(v[off:off+n])
        row = dict(plane=plane, width=w, height=h, count=len(ii))
        if len(ii):
            row['bbox_inclusive'] = [int((ii % w).min()), int((ii // w).min()),
                                     int((ii % w).max()), int((ii // w).max())]
            row['first_samples'] = [dict(x=int(i % w), y=int(i // w),
                                          actual=int(a[off+i]), expected=int(r[off+i]))
                                    for i in ii[:5]]
        result['planes'].append(row)
        off += n
    assert off == len(a)
    return result


libaom = ctypes.CDLL('libaom.so.3')
libaom.aom_codec_version_str.restype = ctypes.c_char_p
dav = ctypes.CDLL('libdav1d.so.7')
dav.dav1d_version.restype = ctypes.c_char_p
out = dict(main='070ad6509bad18f9ab76732d543659ef6d7f9129',
           pr3='d3ea4e602763951626d678291cc9c325e2c475b2',
           libaom=libaom.aom_codec_version_str().decode(),
           dav1d=dav.dav1d_version().decode(),
           packages=subprocess.check_output(['dpkg-query', '-W', 'libaom3', 'libdav1d7'], text=True),
           zip_sha256=digest(b/'avif.zip'), images=[])
for p in sorted((b/'images').glob('*.avif')):
    q = b/'results'/p.stem
    w, h, bd = map(int, Path(str(q)+'.meta').read_text().split()[:3])
    dt = 'u1' if bd == 8 else '<u2'
    read = lambda path: np.fromfile(path, dtype=dt)
    r = read(str(q)+'.davstage7')
    shapes = [(w, h)] * 3
    if 'yuv420' in p.name:
        shapes = [(w,h), ((w+1)//2,(h+1)//2), ((w+1)//2,(h+1)//2)]
    a = read(str(q)+'.native')
    main = read(b/'main-results'/(p.stem+'.native'))
    refs = [Path(str(q)+'.'+s) for s in ['libaom-av1','libdav1d','davstage7','containerdav']]
    assert len({digest(f) for f in refs}) == 1
    row = dict(name=p.name, sha256=digest(p), bit_depth=bd,
               reference_sha256=digest(refs[0]), reference_agreement=True,
               native=stats(a[:len(r)], r, shapes),
               main=stats(main[:len(r)], r, shapes),
               main_pr_changed_samples=int(np.count_nonzero(a != main)), stages={})
    for stage, mask in [('raw',0),('deblock',1),('cdef',3),('restoration',7)]:
        stage_file = b/'stages'/p.stem/(stage+'.yuv')
        if stage_file.exists() and stage_file.stat().st_size == r.nbytes:
            row['stages'][stage] = stats(read(stage_file), read(str(q)+f'.davstage{mask}'), shapes)
    if len(a) > len(r):
        ar = read(b/'plum-alpha-check.dav')
        aa = read(b/'plum-alpha-check.aom')
        assert np.array_equal(ar, aa)
        row['alpha'] = stats(a[len(r):], ar, [(w,h)])
    out['images'].append(row)
for name in ['probe-results','refined-results','all-probes-results']:
    path = b/(name+'.json')
    if path.exists():
        out[name] = json.loads(path.read_text())
print(json.dumps(out, indent=2))

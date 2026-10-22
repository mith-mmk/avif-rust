"""Private seven-image measurement runner; never writes image data into git.

python3 measure.py INPUT_DIR OUTPUT_DIR NATIVE_DUMP DAV1D_DUMP
Build native_dump with probes.patch applied only in a disposable checkout.
Build dav1d_dump using examples/investigate_dav1d.c and matching 1.5.1 headers.
Run once per baseline/probe configuration; AVIF_PROBE_* environment is inherited.
"""
import os
from pathlib import Path
import subprocess
import sys

NAMES = [
    'fox.profile0.8bpc.yuv420.avif',
    'fox.profile1.10bpc.yuv444.avif',
    'fox.profile2.12bpc.yuv444.avif',
    'fox.profile1.8bpc.yuv444.odd-width.odd-height.avif',
    'plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif',
    'colors_hdr_rec2020.avif',
    'paris_icc_exif_xmp.avif',
]
src, out, native, dav = map(Path, sys.argv[1:])
assert all((src/n).is_file() for n in NAMES)
out.mkdir(parents=True, exist_ok=True)


def run(args, **kw):
    return subprocess.run(list(map(str, args)), check=True, capture_output=True, **kw)


for name in NAMES:
    p = src/name
    q = out/'results'/p.stem
    q.parent.mkdir(exist_ok=True)
    stage = out/'stages'/p.stem
    stage.mkdir(parents=True, exist_ok=True)
    r = run([native, p, q], env={**os.environ, 'AVIF_STAGE_DIR': str(stage)}, text=True)
    Path(str(q)+'.meta').write_text(r.stdout)
    Path(str(q)+'.stderr').write_text(r.stderr)
    # FFmpeg's OBU demuxer requires a Temporal Delimiter; decoder input payload
    # otherwise stays byte-identical to the AVIF primary item extracted above.
    obu = Path(str(q)+'.ffobu')
    obu.write_bytes(b'\x12\x00'+Path(str(q)+'.obu').read_bytes())
    for decoder in ['libaom-av1', 'libdav1d']:
        args = ['ffmpeg','-y','-v','error','-threads','1','-c:v',decoder]
        if decoder == 'libdav1d':
            args += ['-filmgrain','0']
        r = run(args+['-f','obu','-i',obu,'-map','0:v:0','-frames:v','1',
                      '-c:v','rawvideo','-threads','1','-f','rawvideo',str(q)+'.'+decoder])
        Path(str(q)+'.'+decoder+'.stderr').write_bytes(r.stderr)
    for mask in [0,1,3,7]:
        r = run([dav, str(q)+'.obu', str(q)+f'.davstage{mask}', mask])
        Path(str(q)+f'.davstage{mask}.stderr').write_bytes(r.stderr)
    run(['ffmpeg','-y','-v','error','-threads','1','-c:v','libdav1d','-filmgrain','0',
         '-i',p,'-map','0:v:0','-frames:v','1','-c:v','rawvideo','-threads','1',
         '-f','rawvideo',str(q)+'.containerdav'])
    refs = [Path(str(q)+'.'+s).read_bytes() for s in
            ['libaom-av1','libdav1d','davstage7','containerdav']]
    assert refs[0] and all(v == refs[0] for v in refs), name
    print(name, 'independent references agree', flush=True)

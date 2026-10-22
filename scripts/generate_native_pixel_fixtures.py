"""Generate synthetic AVIF/oracles with FFmpeg/libaom; verify dav1d agrees.
Usage: python scripts/generate_native_pixel_fixtures.py OUTPUT_DIRECTORY
No user input images are used. See tests/FIXTURES.md.
"""
from pathlib import Path
import math, struct, subprocess, json
out=Path(__import__("sys").argv[1]); out.mkdir(parents=True, exist_ok=True)
rows=[]
for bd in (8,10,12):
 for fmt,sx,sy in [('420',1,1),('422',1,0),('444',0,0),('gray',0,0)]:
  for lossless in (False,True):
   w,h=85,73
   pix=('gray' if fmt=='gray' else 'yuv'+fmt+'p')+(str(bd)+'le' if bd>8 else '')
   name=f'{pix}-'+('lossless' if lossless else 'lossy')
   raw=bytearray()
   for p in range(1 if fmt=='gray' else 3):
    pw=(w+(1<<sx)-1)>>sx if p else w;ph=(h+(1<<sy)-1)>>sy if p else h
    for y in range(ph):
     for x in range(pw):
      v=round((100+28*math.sin(x/13)+19*math.cos(y/7)+((x//11+y//9+p)%3)*8)*(1<<(bd-8)))
      raw.extend(struct.pack('<H',v) if bd>8 else bytes([v]))
   inp=out/(name+'.source');inp.write_bytes(raw)
   dest=out/(name+'.avif')
   cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format',pix,'-video_size',f'{w}x{h}','-i',str(inp),'-frames:v','1','-c:v','libaom-av1','-cpu-used','4','-threads','1','-still-picture','1','-aom-params','enable-cdef=0:enable-restoration=0','-crf','0' if lossless else '26','-b:v','0',str(dest)]
   subprocess.run(cmd,check=True)
   refs=[]
   for dec in ('libaom-av1','libdav1d'):
    b=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-c:v',dec,'-i',str(dest),'-frames:v','1','-c:v','rawvideo','-threads','1','-f','rawvideo','-'])
    refs.append(b)
   assert refs[0]==refs[1]
   (out/(name+'.yuv')).write_bytes(refs[0])
   rows.append(dict(name=name,width=w,height=h,depth=bd,format=fmt,lossless=lossless,bytes=len(refs[0])))
   print(name,flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2))

"""Verify exported source and output AVIF samples with an independent FFmpeg decode.
Run shared_transcode tests with AVIF_SHARED_ARTIFACTS pointing to the directory first.
"""
import concurrent.futures, hashlib, json, pathlib, struct, subprocess, sys
root=pathlib.Path(sys.argv[1])
def check(path):
    bits,model,alpha,stage=path.stem.split('-');bits=int(bits);alpha=alpha=='true'
    fmt={'Monochrome':'gray','IdentityGbr':'gbrp','Yuv444':'yuv444p','Yuv422':'yuv422p','Yuv420':'yuv420p'}[model]
    if bits!=8:fmt+=str(bits)+'le'
    streams=[(0,fmt,range(1 if model=='Monochrome' else 3))]
    if alpha:streams.append((1,'gray' if bits==8 else 'gray'+str(bits)+'le',[3]))
    count=0
    for stream,pix_fmt,roles in streams:
        result=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:v:'+str(stream),'-frames:v','1','-pix_fmt',pix_fmt,'-f','rawvideo','-'],capture_output=True,check=True)
        values=list(result.stdout) if bits==8 else list(struct.unpack('<'+'H'*(len(result.stdout)//2),result.stdout))
        expected=[]
        for role in roles:
            w,h=5,3
            if role in (1,2) and model in ('Yuv422','Yuv420'):w=3
            if role in (1,2) and model=='Yuv420':h=2
            expected.extend(0 if role==3 and i%3==0 else (i*41+role*137)%(1<<bits) for i in range(w*h))
        assert values==expected,(path.name,stream,values,expected)
        count+=len(expected)
    return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples_verified':count}
paths=sorted(root.glob('*-source.avif'))+sorted(root.glob('*-output.avif'))
assert len(paths)==60,('expected 30 sources and 30 outputs',len(paths))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(check,paths))
report={'oracle':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0], 'files':len(reports),'samples':sum(r['samples_verified'] for r in reports),'all_exact':True,'results':reports}
(root.parent/'independent-native-oracles.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='results'}))

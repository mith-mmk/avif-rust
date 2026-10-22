#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dav1d/dav1d.h>
int main(int argc, char **argv) {
    assert(argc==4);
    FILE *in=fopen(argv[1],"rb"); assert(in);
    fseek(in,0,SEEK_END); size_t n=ftell(in); rewind(in);
    Dav1dData d={0}; uint8_t *buf=dav1d_data_create(&d,n); assert(buf);
    assert(fread(buf,1,n,in)==n); fclose(in);
    Dav1dSettings s; dav1d_default_settings(&s);
    s.n_threads=1; s.max_frame_delay=1; s.apply_grain=0; s.inloop_filters=atoi(argv[3]);
    Dav1dContext *c=NULL; assert(!dav1d_open(&c,&s));
    assert(!dav1d_send_data(c,&d));
    Dav1dPicture p={0}; assert(!dav1d_get_picture(c,&p));
    fprintf(stderr,"dav1d %s %dx%d bpc=%d layout=%d grain=%d\n",dav1d_version(),p.p.w,p.p.h,p.p.bpc,p.p.layout,p.frame_hdr->film_grain.present);
    FILE *out=fopen(argv[2],"wb"); assert(out);
    for(int plane=0;plane<(p.p.layout==DAV1D_PIXEL_LAYOUT_I400?1:3);plane++) {
        int sx=plane && p.p.layout!=DAV1D_PIXEL_LAYOUT_I444;
        int sy=plane && p.p.layout==DAV1D_PIXEL_LAYOUT_I420;
        int w=(p.p.w+sx)>>sx,h=(p.p.h+sy)>>sy,bytes=p.p.bpc>8?2:1;
        for(int y=0;y<h;y++) assert(fwrite((char*)p.data[plane]+y*p.stride[!!plane],bytes,w,out)==(size_t)w);
    }
    fclose(out); dav1d_picture_unref(&p); dav1d_close(&c); return 0;
}

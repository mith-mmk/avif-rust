# Required CI fixtures

These files are tracked in this repository so that `cargo test` works in a
standalone checkout. Tests must not depend on a sibling WML2 checkout or skip
the native allocation checks when these fixtures are missing. The fixtures
are also included in the crate package for the library's unit tests.

## WML2Viewer.avif

- Unmodified copy of the WML2 project sample, `samples/WML2Viewer.avif`.
- Source: https://github.com/mith-mmk/wml2-on-rust/blob/7328fcde2f0161694f63d8bec90d22be0135f69e/samples/WML2Viewer.avif
- SHA-256: `8b6846a35bc335a12cde9fa656b4add2c6a654f829bd996d228238b932897644`
- Used for palette, native header ownership, and allocation limit tests.

## plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif

- Author: Ryo Hirafuji (https://github.com/ledyba-z).
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/.
- Source: https://github.com/link-u/avif-sample-images/blob/c666a368b73006246694919b5dbcc078317af6cc/plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif
- Attribution: https://github.com/link-u/avif-sample-images/blob/c666a368b73006246694919b5dbcc078317af6cc/README.md
- Changes: none; the upstream AVIF file is copied byte for byte.
- SHA-256: `245a3dad6371dc702f29eb7e9735f843b63c525da871859728bedbe5bb274985`
- Used for native alpha header, split OBU, and allocation limit tests.

## AVIS regression fixtures

These synthetic `testsrc2` fixtures were generated locally with FFmpeg
7.1.5 and libaom 3.12.1. They do not require an encoder or any sibling
repository at test time. `avis_regression` runs explicitly in every CI OS job
and is included in the published-source package test inventory.

- `avis-subsampled-palette.avif`: 64x64 YUV420, 60 displayed samples,
  61 coded frames, with hidden Inter frames and show-existing samples.
  SHA-256: `a80eeeadf5275403b8bdf8f2dc284a12e4555f6a98fb13b1dd9fe74a20f2b20e`.
  This reproduces three successive failures on the pre-fix decoder: undersized
  chroma palette maps, incorrect inferred compound index, and rejection of
  a valid one-byte entropy tile. The required test checks complete decoding
  and agreement between batch, incremental, primary and indexed display APIs.
- `avis-subsampled-key-reference.yuv`: the first displayed frame of the same
  non-lossless AVIS fixture, as 6,144 native Y/U/V bytes. FFmpeg dav1d 1.5.1,
  FFmpeg libaom 3.12.1 and scalar AOM commit
  `9440aeedd097343b763dfe1c75af72c99f4a3175` agree exactly. This is a
  same-bitstream decoder oracle, not a comparison with the uncompressed input.
  It catches CDEF incorrectly using chroma subsampled bounds for the luma plane.
  SHA-256: `d1dcfdf234f15af9ebc4ff8952c687921def68c891032268921c8e22421e94b6`.
- `lossless-sub8x8-palette.avif`: the lossless first key frame, remuxed as a
  still item. It fails strict entropy termination on the pre-fix decoder.
  SHA-256: `8848e29f4d8f8289fe25011f61e5a11353f81538b3390266fad15bce6529ff62`.
- `lossless-sub8x8-palette.yuv`: 6,144 unmodified planar Y/U/V bytes from the
  lossless key frame. FFmpeg's independent dav1d 1.5.1 and libaom 3.12.1
  decoders give identical bytes. Every native sample is checked exactly.
  SHA-256: `6a6238036a0d304ab9960bac59d282eebffd09ed0f4daf0c444b5e72f31624b7`.

Regeneration (the encoded bytes can vary with FFmpeg/libaom versions):

```sh
mkdir -p .test-avis-regenerate
ffmpeg -y -v error -f lavfi -i testsrc2=size=64x64:rate=30 \
  -frames:v 60 -c:v libaom-av1 -still-picture 0 -g 60 \
  -lag-in-frames 25 -auto-alt-ref 1 -f avif \
  .test-avis-regenerate/avis-subsampled-palette.avif
ffmpeg -y -v error -f lavfi -i testsrc2=size=64x64:rate=30 \
  -frames:v 60 -c:v libaom-av1 -crf 0 -b:v 0 -still-picture 0 \
  -g 60 -lag-in-frames 25 -auto-alt-ref 1 -threads 2 -f avif \
  .test-avis-regenerate/lossless.avif
ffmpeg -y -v error -i .test-avis-regenerate/lossless.avif \
  -map 0:v:0 -c:v copy -f avif .test-avis-regenerate/lossless-sub8x8-palette.avif
ffmpeg -y -v error -c:v libdav1d \
  -i .test-avis-regenerate/lossless-sub8x8-palette.avif \
  -pix_fmt yuv420p -f rawvideo .test-avis-regenerate/lossless-sub8x8-palette.yuv
```

The non-lossless key-frame oracle is now checked exactly after CDEF. The full
60-frame fixture remains a decode/sequence-state regression, not a claim of
complete inter-frame pixel conformance. Independent full-sequence decoding
must select `-map 0:v:1`; FFmpeg's default stream is only the primary still.
The current decoder still differs from the independent oracle on this lossy
sequence. See `avis-validation.json` for measured differences and scope.

Regenerate the non-lossless key oracle from the tracked AVIS bitstream (not
by re-encoding the source):

```sh
ffmpeg -y -v error -c:v libdav1d -i tests/avis-subsampled-palette.avif \
  -map 0:v:1 -frames:v 1 -pix_fmt yuv420p -f rawvideo \
  tests/avis-subsampled-key-reference.yuv
```

The independent libaom decode uses the same command with `-c:v libaom-av1`.
Temporary stage instrumentation in AOM and Rust verified that the key frame
matched before CDEF; the instrumentation is not part of the shipping sources.

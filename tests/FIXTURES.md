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

The non-lossless 60-frame fixture is a decode/sequence-state regression,
not a claim of exact pixel conformance. Independent full-sequence decoding
must select `-map 0:v:1`; FFmpeg's default stream is only the primary still.
The current decoder still differs from the independent oracle on this lossy
sequence. See `avis-validation.json` for measured differences and scope.

## Native pixel conformance fixtures

`native-pixel/` contains eight synthetic images (85x73 or 93x77) and unconverted planar
YUV oracles, generated from the arithmetic pattern in
`scripts/generate_native_pixel_fixtures.py`. No attached/user images or their
pixels are included. The generated fixture data is provided under this
repository's MIT license. SHA-256 digests are in `native-pixel/manifest.json`.

Generation used FFmpeg 7.1.5, libaom 3.12.1 and dav1d 1.5.1. Both independent
decoders produced identical oracle bytes. 8-bit samples occupy one byte;
10/12-bit samples are little-endian u16. The test compares every native sample
and requires legacy/strict equality, with no external executables at test time.
Six fixtures disable CDEF/restoration to isolate reconstruction and deblock.
`yuv422p12le-lossy` checks filtering the complete four-lane edge unit.
`yuv420p10le-cdef` and `gray12le-sgr` use default filters at 93x77 and check
full-size luma CDEF and the high-bitdepth SGR affine offset respectively. Separate unit vectors exercise 10/12-bit flat
6/8/14-tap deblock against libaom's C implementation, and partial-frame CfL,
shared chroma smooth contexts, padded metadata, and strict allocation admission.

```sh
python scripts/generate_native_pixel_fixtures.py /tmp/native-pixel-regenerated
python scripts/generate_native_pixel_fixtures.py /tmp/native-pixel-default --default-filters
cargo test --locked --test native_pixel_conformance
```

The generator additionally produces unbundled feature combinations for broader
manual validation. Encoded bytes can vary across encoder versions; do not
replace oracles with native output. The eight bundled names are listed in the manifest. For the two default-filter
fixtures, rename generated `yuv420p10le-lossy` to `yuv420p10le-cdef` and
`gray12le-lossy` to `gray12le-sgr` (both AVIF and YUV). Broader validation and known remaining differences are documented in
`docs/native-pixel-validation.md`.

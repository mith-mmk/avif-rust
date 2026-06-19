# AVIF decoder implementation checklist

## Repository layout and current state

- Parent repository: `wml2`, branch `v0.0.24`.
- Decoder repository: `avif/`, an independently managed nested Git repository on branch `master`.
- The parent `.gitignore` intentionally ignores `/avif/`; inspect and commit decoder changes from inside `avif/`.
- Parent-side integration changes currently exist in `wml2/tests/avif_decode.rs`.
- The decoder currently targets 8-bit, full-resolution GBR still AVIF images. It is not generally AV1-conformant yet.
- `samples/WML2Viewer.avif` and `samples/WML2Viewer.png` are the current oracle pair.

When Git rejects the nested repository as dubious ownership, use:

```powershell
git -c safe.directory=C:/Users/misir/OneDrive/source/wmprojects/wml2/avif -C avif status --short --branch
```

## Completed in the current implementation

- [x] ISO BMFF/AVIF primary item parsing.
- [x] AV1 sequence, frame, tile-group and basic entropy parsing for the sample.
- [x] Full sample block-tree traversal without unsupported syntax.
- [x] Luma and full-resolution chroma reconstruction paths.
- [x] Callback integration with `wml2` (`init`, `draw`, metadata and `terminate`).
- [x] AVIF feature-gating tests for both enabled and disabled builds.
- [x] AOM smooth-prediction weights and rounding.
- [x] Directional prediction zone 1/2/3 fixed-point interpolation.
- [x] Directional angle deltas interpreted as three-degree steps.
- [x] Prediction generated per transform block so reconstructed neighbours are visible to later transforms.
- [x] Frame-edge defaults use AV1 values: above `base - 1`, left `base + 1`, corner `base`.
- [x] Missing above/left references are completed from the available side where required.
- [x] Directional reference arrays collect additional top-right and bottom-left frame samples.
- [x] Type-0 directional edge upsampling uses the AOM four-tap interpolation and signed edge indices.

Current FFmpeg oracle metric for `WML2Viewer.avif`:

- Average RGB absolute error: approximately `71.0573`.
- The active regression ceiling is `71.1` in `avif/tests/ffmpeg_conformance.rs`.
- The strict conformance test remains ignored because the target is `<= 0.5` average RGB error.

## Next tasks, in priority order

### 1. Finish directional reference-edge processing

- [ ] Complete partition-aware availability checks for extended top-right and bottom-left samples.
- [x] Implement type-0 `av1_use_intra_edge_upsample` behaviour.
  - Type-0 condition: angle delta is non-zero, absolute delta is below 40, and the two block dimensions sum to at most 16.
  - Use the AOM four-tap interpolation: `(-a + 9*b + 9*c - d + 8) >> 4`.
  - Support the `p[-2]`, `p[-1]`, `p[0...]` indexing required by directional zones.
- [x] Derive the intra-edge filter type for each plane from neighbouring luma/UV smooth modes.
- [x] Implement AOM intra-edge filter strength selection and 5-tap kernels for filter type 0.
- [x] Implement corner filtering when both edges are needed and the transform dimensions sum to at least 24.
- [x] Add unit tests for upsampled zone 1, zone 2 negative indices and zone 3.
- [ ] Re-run the FFmpeg metric and only keep changes that preserve syntax correctness and improve or explain the oracle result.

Relevant AOM reference files:

- `av1/common/reconintra.c`
- `av1/common/reconintra.h`
- `av1/common/blockd.h`

### 2. Replace approximate inverse transforms

- [ ] Replace floating-point orthonormal DCT/ADST with AV1 staged integer inverse transforms.
- [ ] Implement normative stage ranges, cosine constants, half-butterfly rounding and row/column shifts.
- [ ] Cover `DCT_DCT`, `ADST_DCT`, `DCT_ADST`, `ADST_ADST`, identity, vertical DCT and horizontal DCT.
- [x] Verify 4x4 DCT, ADST and identity stages against AOM integer rounding.
- [x] Verify 8x8 DCT, ADST and identity stages against AOM integer rounding.
- [ ] Verify 16x16, 32x32 and 64x64.
- [ ] Add known-vector tests derived from the AOM reference implementation.
- [ ] Avoid accepting output solely because dimensions and alpha are correct; validate pixel values against FFmpeg.

Likely high-impact file: `avif/src/av1/transform.rs`.

### 3. Audit coefficient decoding and scan order

- [ ] Replace the generic zig-zag scan with AV1 scan tables selected by transform size/type.
- [ ] Audit EOB, coefficient-base, base-range, sign and Golomb decoding against the specification.
- [ ] Audit coefficient contexts for all transform sizes, especially 32x32 and 64x64.
- [ ] Confirm dequantisation shifts and clipping for each transform size.
- [ ] Compare decoded coefficient vectors against a reference decoder for small test streams.

### 4. Complete reconstruction filters

- [ ] Implement CDEF when enabled by the frame.
- [ ] Implement loop restoration when enabled.
- [ ] Implement super-resolution upscaling.
- [ ] Ensure filter order matches AV1 reconstruction order.

### 5. Expand supported AVIF/AV1 formats

- [ ] YUV-to-RGBA conversion for non-identity matrix coefficients.
- [ ] 4:2:0 and 4:2:2 chroma subsampling with correct chroma sample positions.
- [ ] Monochrome images.
- [ ] 10-bit and 12-bit decode/output conversion.
- [ ] Alpha auxiliary items and AVIF item-property associations.
- [ ] Multiple tiles and tile groups.
- [ ] Additional still-frame header tools currently returning `Unsupported`.
- [ ] AVIF sequences/animation only after still-image conformance is stable.

### 6. Conformance corpus and fuzzing

- [ ] Add small, redistributable AVIF samples under `test_data` only; keep `test_data` ignored.
- [ ] Cover each prediction mode, transform type/size, quantiser range and chroma layout.
- [ ] Add malformed-container and truncated-OBU regression tests.
- [ ] Fuzz container, OBU, frame-header and tile entropy parsers.
- [ ] Keep external test artifacts and generated diagnostics under `.test*` and remove them after use.

## Required validation commands

From the parent repository:

```powershell
cargo fmt --all
cargo test -p avif-rust
cargo test -p wml2 --test avif_decode
cargo test -p wml2 --test avif_decode --features avif
cargo test --workspace
git diff --check
git -c safe.directory=C:/Users/misir/OneDrive/source/wmprojects/wml2/avif -C avif diff --check
```

To print the current strict FFmpeg comparison result:

```powershell
cargo test -p avif-rust --test ffmpeg_conformance pure_rust_decode_matches_ffmpeg_oracle_and_original_png -- --ignored --nocapture
```

The strict test is expected to fail until pixel conformance is reached. Record the numeric error before and after each reconstruction change.

## Completion criteria for the initial decoder

- [ ] Strict sample comparison passes with average RGB absolute error `<= 0.5` and maximum error within the test threshold.
- [ ] No ignored AVIF conformance test remains for supported input classes.
- [ ] `cargo test --workspace` passes.
- [ ] AVIF-enabled and AVIF-disabled `wml2` builds both pass integration tests.
- [ ] Unsupported AV1 tools return explicit errors instead of partial or misleading successful images.
- [ ] `wml2/todo.md` is checked off only after the supported subset and limitations are documented.

## Guardrails

- Do not add native `libaom`, `dav1d` or FFmpeg as runtime decoder dependencies; FFmpeg is an optional test oracle only.
- Do not weaken the pixel-error regression ceiling to make tests pass.
- Do not mark AVIF complete based only on successful callbacks or non-zero pixels.
- Preserve the parent repository's optional `avif` feature behaviour.
- Keep temporary files and browser/server profiles under `.test*`, ensure they are ignored, and clean them after use.

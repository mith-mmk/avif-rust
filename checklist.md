# Pure Rust AVIF/AV1 decoder implementation checklist

## Current vertical slice

The first completion target is an exact, usable still-image decoder for the
profile exercised by `samples/WML2Viewer.avif`:

- 8-bit AV1 intra key frame
- 4:4:4 identity GBR
- one frame and one tile group
- native decoded planes plus SDR RGBA8/RGBA16 output

`WML2Viewer.avif` is a regression fixture, not the whole implementation
target. Its raw native planes now match the filter-disabled AOM oracle exactly;
the current priority is the normative post-filter pipeline. No format expansion
or additional architecture refactor is promoted above this vertical slice
until its final filtered planes match the reference oracle.

### other avif sample images

- https://github.com/link-u/avif-sample-images
- https://colinbendell.github.io/webperf/animated-gif-decode/avif.html

## Guardrails

- AVIF and AV1 decoding remain pure Rust. FFmpeg/libaom are test or fixture
  oracles only.
- Keep the existing callback API, `wml2` AVIF feature and public exports.
- Keep `test_data/` ignored. Commit only reproducible scripts, manifests and
  schema tests; generated AVIF/reference bytes stay local.
- Keep temporary files under ignored `.test*` directories and remove them.
- Unsupported AV1/AVIF tools must return explicit `Unsupported` rather than a
  partially decoded image presented as valid output.
- `avif/` is an independently managed nested repository; inspect and test it
  directly.

## 1. Oracle gate and integration boundary

- [x] Keep reusable exact-plane and RGBA max-error assertions.
- [x] Keep manifest schema, duplicate-ID, dimensions, bit-depth, plane-count
      and overflow validation.
- [x] Add a local fixture generator for AVIF, native planes, RGBA8/RGBA16,
      source hash and `test_data/oracles.csv`.
- [x] Add `AVIF_REQUIRE_ORACLES=1` strict mode so conformance runs fail when
      the manifest is absent, while normal parser/safety tests remain runnable
      without local test data.
- [ ] Make strict mode reject a header-only/zero-entry manifest and require
      the approved fixture IDs; manifest absence is already a strict failure.
- [ ] Add the future `-RegisterInStrictManifest` path to
      `generate_oracles.ps1`; the default `WML2Viewer` output remains
      diagnostic-only, while filter-disabled generation must opt in explicitly.
- [x] Recompute source hashes and compare them with the source manifest using
      `scripts/verify_oracle_sources.ps1`.
- [x] Generate the first filter-disabled 8-bit 4:4:4 identity-GBR fixture.
- [x] Generate exact residual, partition and directional fixtures in that same
      profile.
- [x] Generate the palette fixture in that same profile and assert that the
      bitstream actually contains decoded palette blocks and color maps.
- [ ] Generate the `WML2Viewer.avif` native-plane and RGBA fixtures.
- [ ] Make diagnostic fixture generation opt-in for strict registration
      (for example, `generate_oracles.ps1 -RegisterInStrictManifest`), and keep
      the default `WML2Viewer` generation out of `oracles.csv`.
- [ ] Add one reproducible recipe/bootstrap for `BlackLossless` and the five
      filter-disabled fixtures from a fresh clone (`scripts/bootstrap_oracles.ps1`).
- [x] Make the strict oracle command part of the documented AVIF validation
      gate.
- [x] Capture and assert `wml2` draw callback bytes and dimensions in the
      AVIF integration test.
- [ ] Assert callback order as `init -> draw -> terminate` on the
      `filter-disabled-gbr` public fixture; callback bytes and dimensions are
      already covered.

## 2. Raw block reconstruction: exact pre-filter checkpoint

The following work is one atomic correctness track. Each item must pass an
exact native-plane fixture before the next feature is enabled.

- [x] Align partition recursion and block traversal with the filter-disabled
      partition and directional reference streams.
- [x] Route first-leaf traversal through the first child of vertical,
      horizontal, extended and four-way partitions.
- [x] Align transform-block placement for luma and chroma, including clipped
      frame edges and per-plane coordinates.
- [x] Verify transform-size selection and transform partition traversal for
      every transform used by the filter-disabled fixture set.
- [x] Verify entropy/CDF update state across blocks in the current vertical
      slice: partition, mode, txb skip,
      DC sign, EOB, coefficient base/range, signs and Golomb extension.
- [x] Verify palette-mode filter-intra exclusion, entropy termination and
      complete coded-tile coverage through the aligned MI edge for
      `WML2Viewer.avif` and the strict filter-disabled fixtures.
- [ ] Validate all 19 rectangular and square transform sizes, including
      32x32/64x64 paths, against standalone normative vectors.
- [x] Verify the default zig-zag scan used by the supported 2-D
      `ADST_DCT`/`DCT_ADST` classes with known vectors.
- [x] Apply AOM's inverse `1/sqrt(2)` input normalization when a rectangular
      transform has an odd log2 aspect ratio; pin the Tx4x8 ADST_DCT path with
      a fixed residual vector.
- [x] Match AOM inverse-transform `round_shift` semantics for negative values,
      including negative half-step anchors shared by `round2`.
- [x] Preserve the AOM `mrow`/`mcol` entropy scan for 1-D directional classes
      and transpose square ADST, 1-D, identity and lossy 4x4/8x8 `DctDct`
      coefficient storage at the inverse-transform boundary; preserve the
      existing 4x4 storage for coded-lossless WHT and cover both paths with
      decoder vectors.
- [x] Pin the dense 4x4 `ADST_ADST` block feeding the former first
      `WML2Viewer` mismatch against AOM prediction, dequant and output vectors.
- [x] Pin the 8x8 `DctDct` residual inside a palette block against AOM palette
      prediction, dequant input and all 64 output samples.
- [x] Pin a lossy 4x4 `DctDct` block against AOM prediction, dequant input and
      output while keeping the strict lossless fixture exact.
- [x] Route non-zero angle deltas on vertical/horizontal modes through the
      directional predictor and use AOM's zone-specific top-right/bottom-left
      edge lengths.
- [x] Track top-right and bottom-left reconstruction availability as exact
      sample lengths, including partial transform-edge coverage, and pad each
      directional edge from its last available sample.
- [x] Verify coefficient context selection with known vectors for the enabled
      transform sizes and plane types.
- [x] Verify palette and filter-intra syntax gating for the supported profile.
- [ ] Support all 19 AV1 transform sizes, including rectangular transform
      placement and per-plane coordinates.
- [x] Derive intra chroma transform types from the signalled UV mode and transform
      set for the supported sub-32 transform sizes.
- [x] Match AOM's staged 32-point inverse DCT and sample-count-based dequant
      shift for large rectangular transforms; pin the `Tx32x16` chroma DC
      reconstruction used by `WML2Viewer.avif`.
- [x] Replace the approximate 32x32 fixed-basis/DC shortcut with the staged
      32-point row/column transform and pin the `WML2Viewer` luma DC block;
      use `bit_depth + 8` for the row-stage range through 12-bit paths.
- [x] Match the two-stage rounding for DC-only `Tx64x64` and remove the
      square-transform `1/8` output scaling from the 64-point stage reused by
      rectangular transforms.
- [x] Port the normative staged 64-point inverse DCT, including intermediate
      range clamps and rounding, and pin the `WML2Viewer` `Tx16x64` path.
- [x] Route `Tx64x64 DctDct` through the staged row/column core and transpose
      its lossy coefficient storage at the inverse-transform boundary.
- [ ] Validate the remaining chroma transform derivation and 32x32/64x64
      transform paths against normative reference vectors.
- [x] Reject partial tile groups before accepting a still-image decode.
- [x] Require entropy termination/trailing-bit validation before accepting a
      decoded frame.
- [x] Interleave Y/U/V residual traversal inside each 64x64 coding unit for
      blocks larger than 64x64, matching AV1 syntax order instead of decoding
      one complete 128x128 plane at a time.
- [x] Use rounded restoration-unit counts at the frame edge so a remainder of
      at most half a unit is merged into the preceding unit instead of reading
      nonexistent restoration coefficients.
- [x] Replace diagnostic-only sample assertions with exact plane assertions
      for the generated filter-disabled fixtures.
- [x] Complete the filter-disabled fixture set with exact Y/U/V plane matches.
- [x] Complete the `WML2Viewer.avif` raw reconstruction comparison with exact
      Y/U/V equality against the filter-disabled AOM oracle.
- [x] Add a private, ignored `AOM_PREFILTER_ORACLE` diagnostic that compares
      all native planes with an AOM build whose deblock, CDEF and restoration
      stages are disabled.
- [x] Read AV1 angle deltas for every block size at or above `BLOCK_8X8`,
      including the rectangular `4x16` and `16x4` forms.
- [x] Record the true AOM pre-filter result (no mismatch in any plane) and keep
      this diagnostic fixture out of the passing strict manifest.

Raw reconstruction is the fixed next correctness track for any remaining
bitstream mismatch. Keep palette-mode filter-intra gating, all 19 rectangular
and square TxSizes, chroma TxType derivation, 32/64-point transforms, entropy
termination and complete tile coverage explicit in the vectors before moving
the public decode gate to final filtered pixels.

Completed prerequisites retained as stable code:

- [x] Tile decoder responsibilities are split into syntax, entropy,
      reconstruction, diagnostics and public API modules.
- [x] Existing private tests moved with their implementation modules.
- [x] Size/class-specific coefficient CDFs and neighbour txb state are wired.
- [x] DC-sign contexts, coefficient state propagation and scan helpers have
      known-vector coverage.
- [x] Partition-aware top-right/bottom-left availability is derived from live
      reconstructed-MI coverage.
- [x] CFL-allowed UV mode CDFs, joint alpha syntax and 4:4:4 CFL prediction are
      wired for the current non-subsampled reconstruction path.
- [x] Existing transform dispatch and 8-bit reference anchors are retained.

## 3. Reconstruction filters

Raw reconstruction now passes the filter-disabled fixtures and the complete
`WML2Viewer` pre-filter oracle. Post-filter implementation is the current main
work. Keep filter metadata in private reconstruction state; do not change the
public decoded-frame shape.

- [x] Collect CDEF indices within each tile during block traversal.
- [x] Aggregate CDEF indices into frame-private post-filter state during full/prefix traversal; retain the state for the filter-application stage.
- [ ] Retain transform boundaries and restoration unit type/coefficients in
      the frame-level filter state used by every stage.
- [ ] Retain transform-boundary skip/mode state and consume it when deriving
      normative deblock levels and edge lengths.
- [ ] Integrate frame-level post-filter state and apply filters in fixed order:
      deblock -> CDEF -> loop restoration; verify each stage with plane oracles.
- [ ] Implement deblocking in normative order with boundary and strength
      vectors.
- [x] Implement the scalar 4-tap deblock edge kernel and limit/HEV mask.
- [x] Connect deblock stage traversal to retained transform boundaries for
      4:4:4 scalar frames.
- [x] Implement the scalar CDEF constrain, direction search and 8x8 block
      kernel.
- [x] Collect tile-local CDEF indices during traversal.
- [ ] Integrate frame-level CDEF state and apply each decoded per-block index
      after deblock; verify all planes against a CDEF oracle.
- [ ] Match normative CDEF frame preparation and block semantics: luma
      direction/variance is shared with chroma, luma primary strength uses the
      directional-variance adjustment, secondary strength `3` maps to `4`,
      and frame-edge sentinel/clipping behavior remains to be made exact. The
      scalar direction table now follows AOM's axis order; 8-bit chroma damping
      applies the normative one-step reduction.
- [x] Generate a local diagnostic oracle with deblock enabled and restoration
      disabled; keep the CDEF result non-authoritative until all plane values
      are exact and add a reproducible bootstrap recipe before release.
- [ ] Implement loop restoration and restoration-unit boundary handling.
- [x] Implement the scalar Wiener restoration kernel with transmitted
      vertical/horizontal axis order, AOM's 3/11-bit intermediate rounding,
      chroma 5-tap outer-zero rule and vertical halo.
- [x] Retain the SGRPROJ parameter index and projection coefficients in
      frame-private restoration state.
- [ ] Implement SGRPROJ restoration and verify stripe/boundary behavior.
- [ ] Verify the complete reconstruction/filter order against plane oracles.
- [ ] Enable the `WML2Viewer.avif` final oracle only after the required filters
      are active and exact.

## 4. Format and composition backlog

Do not start these items until the current vertical slice reaches exact plane
and RGBA gates.

- [ ] Monochrome decoded planes.
- [ ] 4:2:0 with chroma sample position.
- [ ] 4:2:2 with chroma sample position.
- [x] Keep the limited SDR BT.601/BT.709 colour-conversion core.
- [ ] Verify the SDR BT.601/BT.709 conversion core with real AVIF plane/RGBA
      fixtures.
- [ ] Verify 4:4:4 non-identity colour paths with real AVIF plane/RGBA
      fixtures.
- [ ] 10-bit and 12-bit quantisation/reconstruction.
- [ ] Alpha auxiliary decode and composition.
- [ ] Grid image-cell composition.
- [ ] `clap`, then `irot`, then `imir`.
- [ ] Multiple tile-group composition.
- [ ] Super-resolution.
- [ ] Film grain for still images.
- [x] Keep HDR transfer characteristics and ICC profiles explicitly
      `Unsupported` until implemented.
- [ ] Implement HDR tone mapping and ICC display conversion.

## 5. Safety, performance and release gate

- [x] Add malformed/truncated coverage for the currently supported container,
      OBU, entropy and metadata paths.
- [ ] Add malformed/truncated cases for each future syntax path as it becomes
      supported.
- [x] Audit frame dimensions, item offsets/extents and frame-buffer allocations
      for overflow and resource limits.
- [ ] Audit post-filter scratch-buffer sizing once normative filters are
      enabled.
- [x] Reject active but unimplemented filters, film grain, qmatrix and other
      unsupported AV1 tools before public decode returns an image.
- [x] Keep public decode fail-closed (`Unsupported`) whenever an unimplemented
      filter, film grain or qmatrix is active; retain pre-filter diagnostics as
      private/test-only paths.
- [x] Validate primary-item info/location, `ipma` property indices and
      essential-property flags.
- [x] Validate `ispe` dimensions against the AV1 frame dimensions.
- [x] Validate SDR `nclx` colour description and range against the AV1
      sequence header, while accepting CICP value `2` as unspecified.
- [ ] Gate every AVIF integration target with `required-features =
      ["avif"]`; the full feature-off integration run currently fails to
      compile.
- [x] Add an explicit feature-off test target and verify the AVIF-disabled
      library with `cargo test -p wml2 --lib --no-default-features`.
- [ ] Gate every AVIF integration target so the full
      `cargo test -p wml2 --no-default-features` run compiles successfully.
- [x] Keep container, OBU, frame-header and entropy fuzz targets.
- [ ] Optimise allocations only after exact-plane conformance passes.
- [ ] Correct the nested crate repository URL to the independently maintained
      `avif-rust` repository and verify it from a fresh clone.
- [ ] Add SIMD/parallel paths only with scalar equivalence and Wasm fallback
      tests.

Required validation after every implementation step:

Run these commands from the parent workspace root (`wml2/`):

```powershell
cargo fmt --all -- --check
cargo test -p avif-rust
cargo check --manifest-path avif/fuzz/Cargo.toml --bins
cargo test -p wml2 --test avif_decode --no-default-features --features avif
cargo check -p wml2 --no-default-features
cargo test -p wml2 --lib --no-default-features
cargo check -p wml2 --target wasm32-unknown-unknown --no-default-features --features avif
$avifPath = (Resolve-Path -LiteralPath avif).Path.Replace('\', '/')
git -c "safe.directory=$avifPath" -C avif diff --check
```

Strict fixture validation additionally requires:

```powershell
$env:AVIF_REQUIRE_ORACLES = '1'
cargo test -p avif-rust --test oracle_fixtures
powershell -File avif/scripts/verify_oracle_sources.ps1
```

Release additionally requires `cargo test --workspace`, exact supported-stream
planes, RGBA8/RGBA16 maximum error 1, native and Wasm checks, AVIF-enabled and
feature-off `wml2` tests, and explicit errors for unsupported tools and
colour-management paths.

The current structural gate is green for the supported feature-on workspace,
the AVIF-disabled `wml2` library, fuzz-bin compilation, AVIF-enabled `wml2`
tests, and the Wasm check. The full feature-off integration run is not yet a
success criterion because un-gated AVIF integration targets still fail to
compile. Exact WML2Viewer pre-filter plane equality is complete; exact
deblock/CDEF/restoration output is now the release blocker.

Until post-filters are implemented, `decode`, `image_from_bytes`,
`decode_frame_bytes` and the `wml2` callback must fail with `Unsupported` for a
stream requiring an unavailable filter. Prefilter diagnostics stay private or
test-only and do not define the public decoded-frame contract.

## Diagnostic history

The local planes produced by the ordinary FFmpeg path are final filtered
output. They remain useful for end-to-end investigation, but must not be
described or registered as a pre-filter oracle. The retained RGB measurements
(`68.77400905349795` before the private filter pipeline and
`68.7115646090535` after it) compare against that final output and are
diagnostic only.

The authoritative raw-reconstruction checkpoint uses
`decoder::prefilter_diagnostic_tests::reports_wml2viewer_against_aom_prefilter_oracle`
with `AOM_PREFILTER_ORACLE` set to planar 8-bit GBR emitted by an AOM build in
which deblock, CDEF and restoration are disabled. The 900x900 fixture now
reports:

- plane 0: `first=None`, `mismatches=0`;
- plane 1: `first=None`, `mismatches=0`;
- plane 2: `first=None`, `mismatches=0`.

The `reports_wml2viewer_raw_against_generated_final_planes` test compares raw
Rust planes with final filtered FFmpeg planes and therefore does not define the
raw acceptance gate. The filter-disabled strict fixtures remain the stable,
reproducible conformance set; generating the diagnostic `WML2Viewer` fixture
does not register it in the strict manifest.

The former first luma difference in the `(632,16)` `Block4x8` was caused by an
upstream dense 4x4 `ADST_ADST` transform. Its AOM prediction, dequant input and
output now match through an explicit square-ADST storage transpose. The former
32x32 DC-only mismatch also matches after routing it through the staged
32-point transform. The following palette-block 8x8 `DctDct` mismatch matches
after a size-specific square storage transpose. The clipped right-edge
`(896,112)` difference was caused by decoding all luma transforms of the
preceding `(768,0)` `Block128x128` before chroma; AV1 interleaves Y/U/V inside
each 64x64 coding unit. A subsequent `(32,128)` difference came from treating
the four-pixel right-edge remainder as a new restoration unit at `x=896`; AV1
rounds the unit count to nearest and merges that remainder. Correcting both
advanced the investigation to `(523,142)`. Exact partial bottom-left padding
then fixed the `Tx16x4 DctAdst` block at `(448,152)`, while two-stage DC-only
rounding fixed the `Tx64x64` block at `(192,192)`. Decode-order transform
hashes agreed through index 2492; the former first differing transform was
index 2493, the `Tx16x64 DctDct` block at `(176,256)`. Its prediction already
agreed with AOM, while a direct cosine-basis 64-point core could not reproduce
AOM's intermediate rounding. The staged DCT64 now fixes that block and
advances the first raster-order luma mismatch from `(176,277)` to `(541,320)`.
The `(541,320)` mismatch was a non-DC `Tx64x64` block still using the old
fixed-basis square path plus untransposed coefficient storage. Reusing the
staged core for both dimensions and transposing lossy Tx64 storage advanced
the next luma investigation to `(560,448)`. The remaining differences were
resolved by using AV1's eight-pixel-aligned MI dimensions, traversing the full
coded edge area, retaining coded padding until reconstruction completes, and
mapping square Tx16/Tx32 plus rectangular Tx64 coefficient storage at the
inverse-transform boundary. Post-filter pixels can now be implemented without
masking a raw-plane difference.
Lossy 4x4 `DctDct` also uses the square transpose, while
coded-lossless 4x4 WHT explicitly retains its original storage so the exact
`filter-disabled-directional` oracle stays green.

The current post-filter diagnostic compares each private stage with FFmpeg's
final `gbrp` output. Raw RGB average absolute error is
`0.14965555555555554`; the deblock and CDEF stages now report
`0.1328135802469136` and `0.062083127572016464`. A dedicated AOM oracle with
deblock enabled and restoration disabled is generated locally. The Rust
deblock stage still differs from that oracle by plane as follows:

- plane 0: `mismatches=13944`, `average_abs=0.02335432098765432`;
- plane 1: `mismatches=12850`, `average_abs=0.026087654320987655`;
- plane 2: `mismatches=9471`, `average_abs=0.01992962962962963`.

Against the corresponding AOM deblock-plus-CDEF oracle, the Rust CDEF stage
now reports `18302/0.025276543209876542`, `18678/0.02910246913580247` and
`15338/0.02364567901234568` mismatches/average absolute error for planes
0/1/2 respectively. Applying the Rust CDEF implementation to the exact AOM
deblock input isolates the remaining CDEF differences to
`23/0.00002839506172839506`, `30/0.00003827160493827161` and
`39/0.00004814814814814815` for planes 0/1/2. The remaining CDEF
differences are confined to frame-edge or block-boundary pixels; exact
plane-oracle agreement is still required before this item is complete.

These are diagnostic checkpoints only; normative level derivation, boundary
coverage and exact kernel behavior remain unfinished. The current private
pipeline reports a final-filter RGB average absolute error of about `0.026`
against FFmpeg. The restoration-only AOM oracle reports mismatches/average
absolute error of `735/0.000908641975308642`, `2040/0.0025617283950617282` and
`189/0.00023333333333333333` for planes 0/1/2 after the chroma 5-tap fix.
Wiener-only residuals remain `5804/0.008323456790123456`,
`23438/0.03442592592592593` and `5167/0.006903703703703704`; these are the
next first mismatches to eliminate before enabling the public final oracle.
The public `WML2Viewer.avif` gate remains incomplete (the historical public
RGB error was `50.161736...`), and no diagnostic generation may change the
strict manifest.

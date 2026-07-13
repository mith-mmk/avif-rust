# Pure Rust AVIF/AV1 decoder implementation checklist

## Current vertical slice

The first completion target is an exact, usable still-image decoder for the
profile exercised by `samples/WML2Viewer.avif`:

- 8-bit AV1 intra key frame
- 4:4:4 identity GBR
- one frame and one tile group
- native decoded planes plus SDR RGBA8/RGBA16 output

`WML2Viewer.avif` is a regression fixture, not the whole implementation
target. No format expansion or additional architecture refactor is promoted
above this vertical slice until its native planes match the reference oracle.

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
- [x] Make strict mode reject a header-only/zero-entry manifest and require
      the approved fixture IDs.
- [x] Require approved fixture source-manifest entries and validate SHA-256
      hash format in strict mode.
- [x] Recompute source hashes and compare them with the source manifest using
      `scripts/verify_oracle_sources.ps1`.
- [x] Generate the first filter-disabled 8-bit 4:4:4 identity-GBR fixture.
- [x] Generate exact residual, partition and directional fixtures in that same
      profile.
- [x] Generate the palette fixture in that same profile and assert that the
      bitstream actually contains decoded palette blocks and color maps.
- [ ] Generate the `WML2Viewer.avif` native-plane and RGBA fixtures.
- [x] Make diagnostic fixture generation opt-in for strict registration
      (for example, `generate_oracles.ps1 -RegisterInStrictManifest`), and keep
      the default `WML2Viewer` generation out of `oracles.csv`.
- [x] Add one reproducible recipe/bootstrap for `BlackLossless` and the five
      filter-disabled fixtures from a fresh clone (`scripts/bootstrap_oracles.ps1`).
- [x] Make the strict oracle command part of the documented AVIF validation
      gate.
- [x] Capture and assert `wml2` draw callback bytes and dimensions in the
      AVIF integration test.
- [ ] Assert callback order as `init -> draw -> terminate` on the
      `filter-disabled-gbr` public fixture.

## 2. Raw block reconstruction: current priority

The following work is one atomic correctness track. Each item must pass an
exact native-plane fixture before the next feature is enabled.

- [x] Align partition recursion and block traversal with the filter-disabled
      partition and directional reference streams.
- [x] Route first-leaf traversal through the first child of vertical,
      horizontal, extended and four-way partitions.
- [ ] Align transform-block placement for luma and chroma, including clipped
      frame edges and per-plane coordinates.
- [x] Verify transform-size selection and transform partition traversal for
      every transform used by the filter-disabled fixture set.
- [ ] Verify entropy/CDF update state across blocks: partition, mode, txb skip,
      DC sign, EOB, coefficient base/range, signs and Golomb extension.
- [ ] Verify palette-mode filter-intra exclusion, all 19 rectangular and square
      transform sizes, 32x32/64x64 transforms,
      entropy termination and complete tile coverage against normative vectors.
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
- [ ] Validate the remaining chroma transform derivation and 32x32/64x64
      transform paths against normative reference vectors.
- [x] Reject partial tile groups before accepting a still-image decode.
- [x] Require entropy termination/trailing-bit validation before accepting a
      decoded frame.
- [x] Replace diagnostic-only sample assertions with exact plane assertions
      for the generated filter-disabled fixtures.
- [x] Complete the filter-disabled fixture set with exact Y/U/V plane matches.
- [ ] Complete the `WML2Viewer.avif` raw reconstruction comparison.
- [x] Add a private, ignored `AOM_PREFILTER_ORACLE` diagnostic that compares
      all native planes with an AOM build whose deblock, CDEF and restoration
      stages are disabled.
- [x] Read AV1 angle deltas for every block size at or above `BLOCK_8X8`,
      including the rectangular `4x16` and `16x4` forms.
- [x] Record the current first mismatch against the true AOM pre-filter output
      and keep this diagnostic fixture out of the passing strict manifest.

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

Implement only after raw reconstruction passes the filter-disabled fixtures.
Keep filter metadata in private reconstruction state; do not change the
public decoded-frame shape.

- [x] Collect CDEF indices within each tile during block traversal.
- [x] Aggregate CDEF indices into frame-private post-filter state during full/prefix traversal; retain the state for the filter-application stage.
- [ ] Retain transform boundaries and restoration unit type/coefficients
      during frame decode.
- [ ] Retain transform-boundary skip/mode state during frame decode.
- [ ] Integrate frame-level post-filter state and apply filters in fixed order:
      deblock -> CDEF -> loop restoration; verify each stage with plane oracles.
- [ ] Implement deblocking in normative order with boundary and strength
      vectors.
- [x] Implement the scalar 4-tap deblock edge kernel and limit/HEV mask.
- [x] Connect deblock stage traversal to retained transform boundaries for
      4:4:4 scalar frames.
- [x] Implement the scalar CDEF constrain, direction search and 8x8 block
      kernel.
- [ ] Implement CDEF and apply the decoded per-block CDEF index.
- [ ] Implement loop restoration and restoration-unit boundary handling.
- [x] Implement the scalar Wiener restoration kernel and apply retained
      Wiener units after CDEF.
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
- [x] Gate every AVIF integration target with `required-features =
      ["avif"]`.
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
compile. Exact WML2Viewer pre-filter plane equality remains the release
blocker.

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
which deblock, CDEF and restoration are disabled. After matching AOM's
positive-bias inverse-transform rounding, staged 32-point DCT,
large-rectangular dequant shift, directional angle-delta routing and zone edge
lengths, the 900x900 fixture reports:

- plane 0: first linear mismatch `101696` (`x=896,y=112`), `673272` mismatches;
- plane 1: first linear mismatch `28882` (`x=82,y=32`), `655105` mismatches;
- plane 2: first linear mismatch `28882` (`x=82,y=32`), `657529` mismatches.

The `reports_wml2viewer_raw_against_generated_final_planes` test compares raw
Rust planes with final filtered FFmpeg planes and therefore does not define the
raw acceptance gate. The diagnostic prefix still traverses 2110 luma blocks
with no `Unsupported` boundary, and the filter-disabled strict fixtures remain
the stable conformance set.

The former first luma difference in the `(632,16)` `Block4x8` was caused by an
upstream dense 4x4 `ADST_ADST` transform. Its AOM prediction, dequant input and
output now match through an explicit square-ADST storage transpose. The former
32x32 DC-only mismatch also matches after routing it through the staged
32-point transform. The following palette-block 8x8 `DctDct` mismatch matches
after a size-specific square storage transpose. The next raw unit is the first
luma mismatch at the clipped right edge `(896,112)`. Lossy 4x4 `DctDct` also
uses the square transpose, while coded-lossless 4x4 WHT explicitly retains its
original storage so the exact `filter-disabled-directional` oracle stays green.
Post-filter work must not be used to mask this raw-plane difference.

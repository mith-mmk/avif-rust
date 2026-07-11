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
- [ ] Make strict mode reject a header-only/zero-entry manifest and require
      the approved fixture IDs.
- [ ] Require approved fixture source-manifest entries and validate SHA-256
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
- [ ] Verify the normative `mrow`/`mcol` scan mapping for the 1-D directional
      transform classes against the AV1 scan table and decoder vectors.
- [x] Verify coefficient context selection with known vectors for the enabled
      transform sizes and plane types.
- [x] Verify palette and filter-intra syntax gating for the supported profile.
- [ ] Support all 19 AV1 transform sizes, including rectangular transform
      placement and per-plane coordinates.
- [x] Derive intra chroma transform types from the signalled UV mode and transform
      set for the supported sub-32 transform sizes.
- [ ] Validate chroma transform derivation and the 32x32/64x64 transforms against
      normative reference vectors.
- [x] Reject partial tile groups before accepting a still-image decode.
- [x] Require entropy termination/trailing-bit validation before accepting a
      decoded frame.
- [x] Replace diagnostic-only sample assertions with exact plane assertions
      for the generated filter-disabled fixtures.
- [x] Complete the filter-disabled fixture set with exact Y/U/V plane matches.
- [ ] Complete the `WML2Viewer.avif` raw reconstruction comparison.
- [x] Record the current first native-plane mismatch at `(128, 0)` on all
      planes after AOM-aligned palette filter-intra gating and rectangular tx
      context selection; keep this diagnostic fixture out of the passing strict
      manifest until raw reconstruction and enabled filters are separated.

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

The previous single-sample RGB measurements remain investigation notes only.
The latest retained `WML2Viewer.avif` average RGB absolute error is about
`50.161736`; it is not a completion criterion. After the AOM-aligned
filter-intra palette exclusion and rectangular transform-context correction,
the current pre-filter diagnostic has first mismatches at plane 0 `(128, 0)`,
plane 1 `(128, 0)` and plane 2 `(128, 0)`, with 790,792, 774,349 and 771,959
samples respectively. The reproducible test-only report is
`decoder::prefilter_diagnostic_tests::reports_wml2viewer_prefilter_mismatches`.
Its first luma mismatch is now in the large block `(128, 0)` with four
`Tx64x64` transforms; this remains an entropy/reconstruction investigation
note, not a conformance result.
The AOM/Rust anchor now agrees through the `(88,16)` palette block, including
palette-map pre/post entropy state and the `(80,24)` partition plus tx-size
symbol. The normative AOM `get_txb_ctx_general` skip/DC context table also
matches Rust for the supported Tx4x4 blocks. The next divergence is after the
four residual blocks of `(80,24)` and before the `(80,28)` mode; coefficient
block order, per-plane traversal, and rectangular transform representation
remain explicit raw-reconstruction tasks. The current implementation still
represents only square `TxSize` values, so all 19 rectangular transform forms
remain unfinished.
The corresponding AOM trace labels the `(80,24)` chroma residual as `TX_8X4`
(enum value 6), while the Rust path currently decodes that plane as `Tx4x4`;
this is the concrete next representation gap to close before revisiting
post-filter pixels.
The decoder now has a tested `Tx8x4` representation and inverse-transform
primitive, but it is intentionally not wired into frame traversal yet: the
The entropy reader now retains AOM's full 32-bit arithmetic `dif` and uses the
normative raw-bit probability for literals. The AOM-aligned probe still exposes
a one-symbol/CDF-state mismatch after the preceding luma coefficient block;
wiring `Tx8x4` before that state is fixed causes the next luma partition to
diverge earlier, so the integration remains an explicit follow-up rather than
a claimed conformance result.
  The ignored stage report currently measures the private pipeline at about
  `50.1617` RGB absolute error before filters and `50.1392` after
  deblock/CDEF/Wiener/SGRPROJ, confirming that raw reconstruction remains the
  dominant error. A direct FFmpeg `-skip_loop_filter all` diagnostic produces
  the same plane mismatch counts as the generated fixture, so it is retained
  as evidence only and is not promoted to a strict pre-filter oracle.
  The frame header has `disable_cdf_update=false`; a test-only forced-CDF-update
  run leaves the first mismatch and counts unchanged, so CDF update mode is
  not currently supported as the primary explanation for the first block.
  The next completion criterion is exact WML2Viewer
pre-filter native planes, followed by the final sample after normative filters
are implemented.

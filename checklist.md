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
  the approved fixture IDs plus their source hashes.
- [x] Generate the first filter-disabled 8-bit 4:4:4 identity-GBR fixture.
- [x] Generate exact residual, partition and directional fixtures in that same
  profile.
- [x] Generate the palette fixture in that same profile and assert that the
  bitstream actually contains decoded palette blocks and color maps.
- [x] Generate the `WML2Viewer.avif` native-plane and RGBA fixtures.
- [ ] Make diagnostic fixture generation opt-in for strict registration
  (for example, `generate_oracles.ps1 -RegisterInStrictManifest`), and keep
  the default `WML2Viewer` generation out of `oracles.csv`.
- [ ] Add one reproducible recipe/bootstrap for `BlackLossless` and the five
  filter-disabled fixtures from a fresh clone.
- [x] Make the strict oracle command part of the documented AVIF validation
  gate.
- [x] Capture and assert `wml2` draw callback bytes and dimensions in the
  AVIF integration test.
- [ ] Assert callback order as `init -> draw -> terminate`.

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
- [ ] Verify coefficient scan selection and coefficient context for every
  enabled transform type in the first profile.
- [ ] Fix palette/filter-intra syntax gating and support all 19 AV1 transform
  sizes, including rectangular transform placement and per-plane coordinates.
- [ ] Derive chroma transform types from the signalled UV mode and transform
  set; validate the 32x32/64x64 transforms against normative reference vectors.
- [ ] Require entropy termination/trailing-bit validation and complete tile
  coverage before accepting a decoded frame.
- [x] Replace diagnostic-only sample assertions with exact plane assertions
  for the generated filter-disabled fixtures.
- [x] Complete the filter-disabled fixture set with exact Y/U/V plane matches.
- [ ] Complete the `WML2Viewer.avif` raw reconstruction comparison.
- [x] Record the current first native-plane mismatch at plane 0, `(146, 0)`
  after wiring CFL syntax/prediction and non-lossless chroma transform sizing;
  plane 1 starts at `(104, 0)` and plane 2 at `(96, 0)`; keep this diagnostic
  fixture out of the passing strict manifest until raw reconstruction and
  enabled filters are separated.

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
- [ ] Aggregate CDEF indices into frame-private post-filter state.
- [ ] Retain transform boundaries, skip/mode state and restoration unit type
  and coefficient information during frame decode.
- [ ] Implement deblocking in normative order with boundary and strength
  vectors.
- [ ] Implement CDEF and apply the decoded per-block CDEF index.
- [ ] Implement loop restoration and restoration-unit boundary handling.
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

- [ ] Add malformed/truncated cases for each newly supported syntax path.
- [ ] Audit dimensions, offsets, allocations and filter scratch buffers for
  overflow and resource limits.
- [ ] Reject active but unimplemented filters, film grain, qmatrix and other
  unsupported AV1 tools before public decode returns an image.
- [ ] Validate AVIF primary-item property association, essential flags and
  AV1/container dimension and colour metadata consistency.
- [ ] Gate the AVIF-disabled integration target with `required-features =
  ["avif"]` and add an explicit feature-off test target.
- [x] Keep container, OBU, frame-header and entropy fuzz targets.
- [ ] Optimise allocations only after exact-plane conformance passes.
- [ ] Correct the nested crate repository metadata to point at the independent
  `avif-rust` repository.
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
```

Release additionally requires `cargo test --workspace`, exact supported-stream
planes, RGBA8/RGBA16 maximum error 1, native and Wasm checks, AVIF-enabled and
feature-off `wml2` tests, and explicit errors for unsupported tools and
colour-management paths.

Until post-filters are implemented, `decode`, `image_from_bytes`,
`decode_frame_bytes` and the `wml2` callback must fail with `Unsupported` for a
stream requiring an unavailable filter. Prefilter diagnostics stay private or
test-only and do not define the public decoded-frame contract.

## Diagnostic history

The previous single-sample RGB measurements remain investigation notes only.
The latest retained `WML2Viewer.avif` average RGB absolute error is about
`50.161736`; it is not a completion criterion. The current pre-filter
diagnostic has first mismatches at plane 0 `(146, 0)`, plane 1 `(104, 0)` and
plane 2 `(96, 0)`, with approximately 780,827, 792,329 and 788,480 mismatched
samples respectively. The next completion criterion is exact WML2Viewer
pre-filter native planes, followed by the final sample after normative filters
are implemented.

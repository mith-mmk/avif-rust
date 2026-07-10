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
  generated fixtures are absent, while normal parser/safety tests remain
  runnable without local test data.
- [ ] Generate the first filter-disabled 8-bit 4:4:4 identity-GBR fixture.
- [ ] Generate residual/partition/directional/palette fixtures in that same
  profile.
- [ ] Generate the `WML2Viewer.avif` native-plane and RGBA fixtures.
- [ ] Make the strict oracle command part of the documented AVIF validation
  gate.
- [x] Capture and assert `wml2` draw callback bytes, dimensions and callback
  order in the AVIF integration test.

## 2. Raw block reconstruction: current priority

The following work is one atomic correctness track. Each item must pass an
exact native-plane fixture before the next feature is enabled.

- [ ] Align partition recursion and block traversal with the reference stream.
- [x] Route first-leaf traversal through the first child of vertical,
  horizontal, extended and four-way partitions.
- [ ] Align transform-block placement for luma and chroma, including clipped
  frame edges and per-plane coordinates.
- [ ] Verify transform-size selection and transform partition traversal for
  every transform used by the first fixture set.
- [ ] Verify entropy/CDF update state across blocks: partition, mode, txb skip,
  DC sign, EOB, coefficient base/range, signs and Golomb extension.
- [ ] Verify coefficient scan selection and coefficient context for every
  enabled transform type in the first profile.
- [ ] Replace diagnostic-only sample assertions with exact plane assertions
  at the first failing fixture.
- [ ] Complete the filter-disabled fixture set with exact Y/U/V plane matches.
- [ ] Complete the `WML2Viewer.avif` raw reconstruction comparison.

Completed prerequisites retained as stable code:

- [x] Tile decoder responsibilities are split into syntax, entropy,
  reconstruction, diagnostics and public API modules.
- [x] Existing private tests moved with their implementation modules.
- [x] Size/class-specific coefficient CDFs and neighbour txb state are wired.
- [x] DC-sign contexts, coefficient state propagation and scan helpers have
  known-vector coverage.
- [x] Partition-aware top-right/bottom-left availability is derived from live
  reconstructed-MI coverage.
- [x] Existing transform dispatch and 8-bit reference anchors are retained.

## 3. Reconstruction filters

Implement only after raw reconstruction passes the filter-disabled fixtures.
Keep filter metadata in private reconstruction state; do not change the
public decoded-frame shape.

- [ ] Retain transform boundaries, skip/mode state, CDEF index and restoration
  unit information during frame decode.
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
- [ ] 4:4:4 non-identity colour paths.
- [ ] 10-bit and 12-bit quantisation/reconstruction.
- [ ] Alpha auxiliary decode and composition.
- [ ] Grid image-cell composition.
- [ ] `clap`, then `irot`, then `imir`.
- [ ] Multiple tile-group composition.
- [ ] Super-resolution.
- [ ] Film grain for still images.
- [ ] HDR tone mapping and ICC display conversion remain explicit
  `Unsupported` until implemented.

## 5. Safety, performance and release gate

- [ ] Add malformed/truncated cases for each newly supported syntax path.
- [ ] Audit dimensions, offsets, allocations and filter scratch buffers for
  overflow and resource limits.
- [x] Keep container, OBU, frame-header and entropy fuzz targets.
- [ ] Optimise allocations only after exact-plane conformance passes.
- [ ] Add SIMD/parallel paths only with scalar equivalence and Wasm fallback
  tests.

Required validation after every implementation step:

```powershell
cargo fmt --all
cargo test -p avif-rust
cargo check --manifest-path avif/fuzz/Cargo.toml --bins
cargo test -p wml2 --test avif_decode --no-default-features --features avif
cargo check -p wml2 --target wasm32-unknown-unknown --no-default-features --features avif
git -c safe.directory=C:/Users/misir/OneDrive/source/wmprojects/wml2/avif -C avif diff --check
```

Strict fixture validation additionally requires:

```powershell
$env:AVIF_REQUIRE_ORACLES = '1'
cargo test -p avif-rust --test oracle_fixtures
```

Release requires exact supported-stream planes, RGBA8/RGBA16 maximum error 1,
native and Wasm checks, AVIF-enabled/disabled `wml2` tests, and explicit
errors for unsupported tools and colour-management paths.

## Diagnostic history

The previous single-sample RGB measurements remain investigation notes only.
The latest retained `WML2Viewer.avif` average RGB absolute error was about
`65.7132`; it is not a completion criterion. The next completion criterion is
the first generated native-plane fixture, followed by the final sample after
the normative filters are implemented.

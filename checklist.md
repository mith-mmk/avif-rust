# Pure Rust AVIF/AV1 decoder implementation checklist

## Purpose and completion target

The goal is to provide `wml2` with a practical still-image AVIF decoder while implementing the AVIF container and AV1 decoding path in pure Rust.

- Do not delegate AVIF or AV1 decoding to native libraries or external decoder crates.
- General-purpose Rust dependencies are allowed when they do not implement AVIF/AV1 decoding.
- Preserve Windows, Linux, macOS and Wasm builds.
- Prioritise specification accuracy, malformed-input safety, then performance.
- Treat `samples/WML2Viewer.avif` as one regression sample, not as the implementation target.
- Validate normative changes against specification vectors, coefficient/plane output and a conformance corpus. Do not reject a normative change solely because one incomplete decode path temporarily produces a worse RGB metric.

Initial still-image completion requires:

- 8/10/12-bit AV1 intra decoding.
- Monochrome, 4:2:0, 4:2:2, 4:4:4 and identity GBR layouts.
- Multiple tiles and tile groups.
- Alpha auxiliary items, image grids, `clap`, `irot` and `imir` composition.
- High-precision decoded planes plus SDR RGBA8/RGBA16 conversion.
- Explicit `Unsupported` errors for HDR/ICC display conversion until colour management is implemented.

AVIF sequences, encoding, HDR tone mapping and ICC application are later milestones.

## Repository and validation rules

- `avif/` is an independently managed nested repository.
- Keep external corpus files in ignored `test_data/`; record source, licence and SHA-256 in a reproducible fetch script.
- Keep temporary/generated diagnostics under ignored `.test*` paths and remove them after use.
- FFmpeg/libaom may be used only as test oracles, never as runtime dependencies.
- Preserve the existing callback API and optional `wml2` `avif` feature.

## 1. Conformance harness and architecture

- [ ] Replace the single-sample average-error gate with layered conformance checks.
  - [x] Add reusable integration-test helpers for exact decoded-plane checks and SDR RGBA max-error checks.
  - [x] Syntax/CDF/context known-vector tests derived from the AV1 specification or AOM.
  - [ ] Exact decoded Y/U/V/alpha plane comparison for supported streams.
    - [x] Add a manifest-driven `test_data/oracles.csv` harness for exact source-plane fixtures.
    - [ ] Populate external supported-stream plane fixtures.
  - [ ] SDR RGBA8/RGBA16 comparison with maximum per-channel error of 1.
    - [x] Add RGBA8/RGBA16 max-error comparison gates that include alpha.
    - [ ] Apply the RGBA8/RGBA16 max-error gate to external supported-stream fixtures.
  - [x] Keep `WML2Viewer.avif` as a normal regression case until exact plane fixtures are available.
- [x] Add a reproducible corpus fetch/verification command for `test_data/`.
- [ ] Split oversized decode code by responsibility: block syntax, coefficient entropy, reconstruction state and diagnostics.
  - [x] Split palette cache/context helpers into `tile_decode::palette`.
  - [x] Split public and internal decode diagnostic probe structs into `tile_decode::diagnostic`.
  - [x] Split per-plane reconstruction and prediction helpers into `tile_decode::reconstruction`.
  - [ ] Split block syntax/reconstruction diagnostics out of the remaining oversized tile decoder.
- [ ] Move tests out of implementation modules as files are split.
  - [x] Move reusable conformance/oracle assertions into integration-test support code.
  - [x] Move public container/OBU sample and malformed-input checks into integration tests.
  - [x] Move public AV1 config/header/tile/decode-plan sample checks into integration tests.
  - [x] Move prediction/reconstruction private unit tests with the reconstruction helpers.
  - [ ] Continue moving private implementation tests as decode modules are split.

## 2. AV1 entropy and coefficient decoding

This is the current highest-priority implementation block. Entropy components that share decoder state must be integrated together rather than enabled independently.

- [x] Integrate transform-size-specific EOB and coefficient CDFs for 4x4 through 64x64, including EOB transform-class selection.
- [x] Integrate per-plane above/left transform-block entropy state.
- [x] Select luma/chroma `txb_skip` contexts from neighbouring levels and plane/block geometry.
- [x] Select DC-sign contexts from neighbouring DC signs.
- [x] Propagate capped coefficient level and DC sign after every zero/non-zero transform block.
- [x] Verify EOB, base, base-EOB, base-range, sign and Golomb decoding as one state machine.
  - [x] Freeze AOM-bitwriter Golomb payload vectors and EOB group-start vectors.
  - [x] Add a scripted full coefficient-token transcript covering every phase and context transition.
- [x] Compare decoded coefficient vectors against an AOM-CDF/bitwriter reference payload before relying on final RGB metrics.
  - [x] Decode the AOM-produced arithmetic payload to the exact 4x4 quantized coefficient vector, including base-range, signs and Golomb extension.
- [ ] Align block syntax and transform-block traversal with the AOM sample oracle before adding full-sample coefficient fixtures.
  - [x] Consume loop-restoration-unit syntax before each applicable superblock partition.
  - [x] Match the AOM root/first-leaf partition sequence and first luma coefficient (`-468`).
  - [x] Enforce the AV1 32x32 maximum chroma transform size for a signalled 64x64 luma transform.
  - [x] Implement selected palette block syntax and color-map token consumption; the sample prefix now traverses palette blocks without `Unsupported`.
  - [x] Implement palette prediction/reconstruction output for decoded palette maps.
  - [x] Implement cached palette color syntax and sorted cache/transmitted color merging for above/left palette reuse.
  - [x] Add deterministic plane-level unit fixtures for palette color-map expansion and cached-color merging.
  - [ ] Add external AOM/FFmpeg plane-level oracle fixtures for palette-coded blocks, including cached-color cases.
- [x] Implement normative ext-tx subset mapping, filter-intra tx mode mapping, directional scan selection and 1D coefficient contexts.
- [x] Apply the normative 20-bit coefficient magnitude clamp after Golomb extension.
- [x] Confirm dequant shifts: 4/8/16 no shift, 32 divide by 2, 64 divide by 4.
- [x] Add unit-tested AOM `get_txb_ctx` equivalents for skip selection, DC-sign aggregation and entropy-state propagation.

## 3. Integer inverse transforms

- [ ] Remove floating-point transform fallbacks.
- [ ] Implement normative staged integer transforms, stage ranges, cosine constants and row/column shifts.
- [x] Verify 4x4 DCT/ADST/identity vectors.
- [x] Verify 8x8 DCT/ADST/identity vectors.
- [ ] Verify all supported 16x16 transform types.
  - [x] Route 16x16 DCT_DCT through the staged integer transform and verify its AOM rounding vector.
  - [x] Port AOM's 16-point IADST and route ADST, identity and directional 16x16 types through integer stages.
  - [x] Add independent AOM vectors for 16-point IADST and identity outputs.
- [ ] Verify all supported 32x32 transform types.
- [ ] Verify 64x64 DCT and coded top-left coefficient limits.
- [ ] Add reference vectors for every enabled transform type/size pair.

## 4. Prediction and reconstruction pipeline

- [x] Implement smooth prediction weights and directional zone interpolation.
- [x] Implement directional angle deltas and type-0 edge upsampling.
- [x] Implement intra-edge filters and corner filtering.
- [ ] Complete partition-aware top-right and bottom-left availability.
- [ ] Implement deblocking.
- [ ] Implement CDEF.
- [ ] Implement loop restoration.
- [ ] Implement super-resolution.
- [ ] Implement film grain when signalled for still images.
- [ ] Verify normative reconstruction/filter order.

## 5. Formats, colour and AVIF composition

- [ ] Decode monochrome frames.
- [ ] Decode 4:2:0 and honour chroma sample position.
- [ ] Decode 4:2:2 and honour chroma sample position.
- [ ] Decode 4:4:4 and identity GBR.
- [ ] Support 8/10/12-bit quantisation and reconstruction.
- [ ] Add a public high-precision decoded-frame API.
  - [x] Expose u16 source planes, dimensions, stride, bit depth and pixel layout.
  - [x] Preserve CICP (`nclx`), raw ICC (`prof`/`rICC`) and alpha-premultiplication metadata on decoded frames.
  - [x] Provide straight-alpha `to_rgba8()` and `to_rgba16()` conversion methods for the currently supported identity-GBR subset.
- [x] Implement SDR nclx range/matrix conversion for non-subsampled identity GBR and BT.601/709/2020 non-constant-luminance matrices.
- [x] Return explicit `Unsupported` for RGBA conversion requiring unimplemented HDR/ICC colour management.
- [ ] Parse and compose alpha auxiliary items.
  - [x] Parse alpha `auxC` / `auxl` metadata and payloads, and return explicit `Unsupported` instead of dropping alpha.
  - [ ] Decode and compose alpha auxiliary planes into RGBA output.
- [ ] Parse and compose grid items.
  - [x] Parse primary `grid` derived item metadata/payload and return explicit `Unsupported` instead of treating it as AV1.
  - [ ] Compose grid image cells into a decoded frame.
- [ ] Apply `clap`, then `irot`, then `imir`.
  - [x] Parse primary item `clap`, `irot` and `imir` properties and return explicit `Unsupported` instead of ignoring them.
  - [ ] Apply clean aperture, rotation and mirror transforms in AVIF composition order.
- [ ] Support multiple tiles and tile groups.
  - [x] Decode every tile payload in a parsed tile group instead of only the first tile.
  - [x] Detect multiple tile-group OBUs for one frame and return explicit `Unsupported` instead of using only the first.
  - [ ] Compose multiple tile-group OBUs for one frame.

## 6. Safety and performance

- [ ] Add malformed/truncated container, OBU and entropy-stream tests.
  - [x] Cover truncated top-level box headers, invalid box sizes, truncated OBU headers/size fields, overlong OBU payloads and malformed entropy termination/CDF inputs.
  - [ ] Expand malformed regression fixtures as new supported parser paths are added.
- [ ] Check all dimensions, offsets, allocations and arithmetic for overflow and resource limits.
  - [x] Cover item extent length/offset overflow, out-of-file extents, large grid truncation and multi-tile payload bounds.
  - [x] Reject public frame-buffer allocation requests that exceed the decoder plane sample resource limit.
  - [ ] Audit newly supported composition/filter paths for the same overflow/resource-limit checks.
- [x] Add fuzz targets for container boxes, OBU headers, frame headers and tile entropy.
- [ ] Optimise allocations only after conformance and safety gates pass.
- [ ] Add SIMD/parallel paths only with scalar equivalence tests and Wasm-compatible fallbacks.

## Diagnostic history

These measurements document incomplete combinations; they are not acceptance criteria.

- The previous retained sample path decoded `1997` blocks with average RGB absolute error about `69.4847` against FFmpeg.
- Integrating size/class-specific EOB CDFs, size-specific coefficient CDFs and complete neighbour txb state atomically produces a deterministic `915`-block traversal and improves average RGB absolute error to about `67.0337`; this is the current retained path.
- Enabling the AOM-vector-verified 16x16 integer DCT changes the sample average RGB error to about `74.4911`; it remains enabled because the single sample is diagnostic rather than the conformance oracle.
- Routing all 16x16 transform types through integer DCT/IADST/identity stages changes the diagnostic sample error to about `75.8173`.
- Direct AOM coefficient instrumentation shows the sample starts with a non-zero 64x64 luma DC coefficient (`-468`), while the current Rust traversal first reaches a non-zero luma transform later in the frame; this identifies block syntax/traversal as an upstream conformance blocker rather than a coefficient-state-machine failure.
- Loop-restoration syntax and the chroma 32x32 transform cap aligned the initial partition/transform sequence with AOM exactly. AOM reports 3646 leaves for the sample; Rust previously reached the first selected luma palette block after 38 leaves.
- Palette size/color syntax and color-map token consumption now allow the sample prefix to traverse palette blocks without `Unsupported`; the retained prefix test decodes 2166 luma leaves with no syntax frontier.
- Palette prediction now expands decoded palette maps into reconstructed block prediction. The sample diagnostic average RGB absolute error improved to about `56.1582`.
- Cached palette color selection now reuses sorted above/left palette colors and merges them with transmitted colors. The sample diagnostic average RGB absolute error improved further to about `54.5251`; visual conformance remains pending because plane-level oracle fixtures are not covered yet.
- Normative ext-tx mapping alone produced `71.5238`; mapping plus incomplete 1D contexts changed block traversal to `1347`.
- Directional scan/context combinations produced `72.8957` and `82.0004` before the complete retained scan/mapping subset was wired.
- DC-sign neighbour contexts alone changed `1997` blocks to `1976` and produced `77.1615`.
- Size-specific coefficient CDFs alone changed `1997` blocks to `2629` and produced `70.7998`.
- Neighbour `txb_skip` plus DC-sign state without size-specific coefficient CDFs changed `1997` blocks to `1734` and produced `81.8257`.
- These results require the size-specific coefficient CDFs and full transform-block neighbour state to land atomically.

## Required validation

```powershell
cargo fmt --all
cargo test -p avif-rust
cargo test -p wml2 --test avif_decode
cargo test -p wml2 --test avif_decode --features avif
cargo test --workspace
cargo check --target wasm32-unknown-unknown -p avif-rust
cargo check --target wasm32-unknown-unknown -p wml2 --features avif
git diff --check
git -c safe.directory=C:/Users/misir/OneDrive/source/wmprojects/wml2/avif -C avif diff --check
```

## Release gate

- [ ] Supported corpus plane output matches the reference decoder exactly.
- [ ] SDR RGBA output stays within one code value per channel.
- [ ] No ignored conformance test remains for the documented supported subset.
- [ ] Native workspace and Wasm checks pass.
- [ ] AVIF-enabled and AVIF-disabled `wml2` integration tests pass.
- [ ] Unsupported tools and colour-management cases fail explicitly instead of returning misleading images.
- [ ] `wml2/todo.md` is checked off only after the supported subset and limitations are documented.

# Shared pixel model and optional encoding

This branch integrates the codec-independent `avif-image-buffer` package,
both codec adapters, AVIF source context, preservation preflight and optional
encoder delegation. WML2 PR #13 remains on hold and unchanged. JPEG2000,
release versions and decoder/encoder defaults are unchanged.

## Dependency and distribution boundary

The shared package lives in `crates/avif-image-buffer`, has no dependencies
and forbids unsafe code. It is unpublished (`publish = false`). Both codecs
select that package from the same immutable avif-rust Git commit
`9ed4ad242582c3375c8d8ee1365ef7d6ddc5b53a`, preserving Rust type identity.
The local copy is excluded from the root workspace and tested independently;
it is not a second model package in the codec dependency graph.

The encoder implementation is pinned to avifenc commit
`e8eec232ef9ef61f99df9ecdb5c7f8a4685d5bac`:

```text
avif-rust ------------------------> avif-image-buffer
avif-rust -- feature encode ------> avifenc-rust --> avif-image-buffer
```

The encoder selects only the shared package from the decoder repository; it
has no dependency on the decoder package. `encode` is optional and does not
change defaults. Encode-off selects neither encoder nor encoder backends.
These are Git development dependencies, not a crates.io distribution solution.
`cargo package` currently fails because the unpublished Git dependency has no
registry version. The existing CI packaging check consequently remains blocked;
no release or publication was performed or implied.

## Shared model and compatibility

`avif_rust::buffer` and `avif_rust::encoder::buffer` expose identical types:
owned `ImageBufferEx`, immutable `ImageViewEx`, packed Gray/RGBA and native
Monochrome/IdentityGbr/Yuv444/Yuv422/Yuv420. U8, numeric U16 and U16LeBytes
storage are separate from valid depths 8/10/12/16. Plane offsets and strides
are bytes; numeric U16 requires even values, while LE byte access handles odd
positions without casts. Final-row padding is not required or exported.
Validation snapshots cannot outlive their immutable borrow. Constructors and
conversion allocations use fallible reservations; native code values and
full-range normalized precision conversion are different operations.

Alpha absent/straight/premultiplied and its domain remain explicit. ICC bytes
and CICP are independent; borrowing, adoption and validation apply no color
transform. Exact normalized scaling rejects fractional results, including
8->10 sample 1. Nearest requires an explicit tie rule. Unsupported color,
native range/depth, chroma or alpha conversion rejects before encoding.

Legacy public fields, struct literals, entry points and defaults remain.
Additive `view_ex` methods revalidate mutable legacy fields, rather than caching
a validated flag. RGBA16 retains full-range16-bit meaning. Old display helpers
retain ICC/unpremultiply behavior, and the encoder's old RGBA16 default remains
12 bits. New packed encoding currently supports Gray and identity GBR mapping;
RGB/YUV conversion and resampling require a separate conversion layer.

Two correctness fixes are deliberate observable changes: the encoder writes
alpha `auxl` from alpha to primary, and the decoder retains fractional Q3 luma
precision in subsampled CfL (AV1 7.11.5). The latter fixed an existing 5x3
YUV420 native sample mismatch; FFmpeg independently confirms the expected
samples. Reference: https://aomediacodec.github.io/av1-spec/ .

## Native context and preservation planning

`decode_native_ex(data, limits)` uses the existing strict bounded native decode
once, without an unbounded or RGBA fallback. `NativeDecodedFrame::into_parts`
and older projections stay available. Adoption moves native sample allocations;
ICC may be copied into model metadata. The retained `AvifSourceContext` contains
the complete existing `NativeAvifInformation`, coded/render sizes, original AV1
color configuration and alpha association. Existing native limits keep their
existing scope; they are not a total allocation guarantee for adapters, encoder,
post-filter scratch or subsequent caller-owned data.

The existing bounded scan/parser also records presence of structures the pixel
projection cannot forward. This inventory adds no second input parse/decode and
stores no invented opaque payload. Source context is not a promise that the
writer can preserve every AVIF field.

With `features = ["encode"]`, `encode_ex`/`preflight_ex` delegate to the standalone
encoder. `transcode::preflight_native_transcode` and `transcode_native` accept an
already decoded `NativeImageEx`, explicit `NativeEncodeOptions` and
`PreservationPolicy`. A representative call is:

```rust,ignore
let source = avif_rust::decode_native_ex(input, &input_limits)?;
let preview = avif_rust::transcode::preflight_native_transcode(
    &source, &native_options, &Default::default())?;
assert!(!preview.encoded);
let output = avif_rust::transcode::transcode_native(
    &source, &native_options, &Default::default(), &output_limits)?;
```

Default planning rejects geometry which the writer cannot forward, unknown
properties/colr, unmodelled metadata/auxiliary/graph structures and incompatible
encoded precision. Permissions to omit information are individual features;
allowing a derived structure to flatten never permits depth reduction. A
container-structure preservation request rejects because the writer rebuilds
a primary/alpha document. Property order, original graph, unknown payload and
file-byte preservation are not claimed.

Original nclx and AV1 color description/range must agree by default. Explicit
`ColorAuthority::Av1` or `Container` can select an available authority, recorded
as signalling changes without a color transform. The existing strict decoder
already rejects some conflicting nclx/AV1 files before planning; this behavior
was not relaxed. Missing nclx can reach preflight, where explicit AV1 selection
is needed. ICC bytes and prof/rICC type are checked independently.

Previews have `encoded=false`. Success verifies output container metadata under
caller limits before returning complete bytes. Reports distinguish native
sample handoff, codec lossless flags, actual conversions/drops, verified output
signalling, and unsupported container-structure preservation. Production encode
does not re-decode output pixels: `pixel_roundtrip_verified` remains false.
Independent test-oracle success does not change that per-call flag.
The encode adapter copies logical samples once into compact legacy input
vectors; only native ownership adoption and borrowed-view boundaries avoid
sample copies. No whole-codec zero-copy guarantee is made.

## Reproducing verification

```sh
cargo test --locked --manifest-path crates/avif-image-buffer/Cargo.toml
cargo clippy --locked --manifest-path crates/avif-image-buffer/Cargo.toml --all-targets -- -D warnings
cargo test --locked --release --features encode --test shared_buffer --test shared_transcode
cargo test --locked --release --features encode --lib --tests --no-fail-fast
cargo tree --locked --no-default-features
cargo tree --locked --features encode
cargo check --locked --no-default-features
cargo clippy --locked --features encode --all-targets
cargo rustdoc --locked --features encode --lib -- -D warnings
```

For an independent native oracle, create an artifact directory and run:

```sh
AVIF_SHARED_ARTIFACTS=/absolute/artifact/directory cargo test --locked --release --features encode --test shared_transcode
python3 tests/shared_oracles.py /absolute/artifact/directory
```

This generates 30 source and 30 output AVIFs (8/10/12 x 400/GBR/444/422/420 x
alpha absent/present, all 5x3), then requires FFmpeg to decode every primary and
alpha plane exactly. It includes asymmetric channels and color under alpha0.

A standalone real WASM harness is in `tests/wasm_shared`. Build it with
`cargo build --locked --release --target wasm32-unknown-unknown --manifest-path tests/wasm_shared/Cargo.toml`.
Run wasm-bindgen CLI0.2.126 with `--target nodejs --out-name verify`, then
`node tests/wasm_shared/run.cjs /absolute/verify.js /absolute/verify_bg.wasm`.
The harness checks native encode/decode/transcode at three depths, ICC/rICC,
transparent RGB, limits, unaligned LE layout and legacy RGBA16 semantics.
No WASM imports are mocked.

Recorded results and limitations are in `shared-buffer-validation.json`.
The ordinary full decoder runner has one AVIS inter-frame entropy failure,
reproduced on unmodified ecc37b2. Conditional external corpus tests can return
early when fixtures are absent; runner pass counts are not a full corpus claim.
Strict Clippy also fails on baseline warnings. MSRV1.88, non-Linux runtime and
a full external corpus were not tested. WML2 core integration and release-ready
registry packaging remain separate work.

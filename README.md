[![crates.io](https://img.shields.io/crates/v/avif-rust.svg)](https://crates.io/crates/avif-rust)
[![docs.rs](https://img.shields.io/docsrs/avif-rust)](https://docs.rs/avif-rust)
[![license](https://img.shields.io/crates/l/avif-rust.svg)](https://opensource.org/license/mit)

[日本語](https://github.com/mith-mmk/avif-rust/blob/main/README.ja.md)

# avif-rust

`avif-rust` is an experimental, pure Rust AVIF container and AV1 image-sequence
decoder. It provides direct RGBA helpers, access to decoded AV1 source planes,
and a callback interface compatible with [`wml2`](https://github.com/mith-mmk/wml2-on-rust).

The current release is
[`avif-rust 0.0.2`](https://crates.io/crates/avif-rust/0.0.2).

The crate does not call FFmpeg, libaom, or another native codec at runtime.
Those implementations are used only to generate and verify test oracles.

## Status and supported input

The public decoder currently accepts tested single-frame AVIF profiles with
8-bit, 10-bit, and 12-bit source planes, including monochrome, 4:2:0, 4:2:2,
alpha/grid composition, and `clap`/`irot`/`imir` properties. A one-frame `avis`
primary item is also exposed as a still image. The external 12-bit fox sample
passes the FFmpeg RGB oracle (average absolute error about 0.075, maximum 6).

| Capability | Status |
| --- | --- |
| AVIF primary item containing one AV1 still frame | Supported |
| 8-bit, 10-bit, and 12-bit source planes | Supported (12-bit RGB oracle passes) |
| Native decoded planes, optional alpha plane, RGBA8, and RGBA16 | Supported (alpha is `buffers.planes[3]` when present) |
| Premultiplied alpha (`prem`) | Supported (RGBA8/RGBA16 outputs are unpremultiplied; native planes remain unchanged) |
| Deblock, CDEF, and loop restoration used by supported streams | Supported |
| Monochrome, 4:2:0, and 4:2:2 | Supported |
| Extended `pixi` channel descriptors | Supported (unsigned integer channels and subsampling type/location are parsed and exposed) |
| Alpha auxiliary images and grid composition | Supported (native alpha plane and aligned `clap`/`irot`/`imir` are applied) |
| `iloc` item payloads in file data and meta `idat` | Supported (construction methods 0, 1, and indexed method 2) |
| `clap`, `irot`, and `imir` composition | Supported |
| One-frame `avis` primary item | Supported |
| AVIS Key/IntraOnly/Inter/Switch/show-existing sample decode by index or batch | Supported for tested motion-compensated Inter and Switch samples |
| Animated AVIF multi-frame callback output | Supported for Key/IntraOnly/Inter/Switch/show-existing samples (`animation: true`) |
| Layered-image selectors (`a1op=0`, `lsel=0`/`0xffff`) | Parsed and accepted; specific non-default layer/operating-point selection remains fail-closed |
| `tmap` primary item base-image fallback | Supported (base `av01` decode); ISO 21496 gain-map metadata and the referenced AV1 gain-map item can be inspected/decoded; explicit base-colour-space HDR application supports CICP, matrix-shaper, and linear-affine ICC LUT/mAB alternates (non-linear/reverse profiles fail closed) |
| PQ/HLG transfer to bounded SDR RGBA16 | Supported (BT.709 luminance-axis gamut compression and bounded tone mapping) |
| B2A ICC display conversion | Supported for B2A0/B2A1/B2A2 `mBA` device-RGB profiles |
| Display-specific HDR calibration | Not supported; RGBA output uses the documented bounded BT.709 display policy |

Unsupported composition or AV1 tools return `DecoderError::Unsupported`. The
decoder intentionally fails closed instead of returning a partially decoded
image as valid output.

## Installation

```console
cargo add avif-rust
```

Or add the dependency manually:

```toml
[dependencies]
avif-rust = "0.0.2"
```

The minimum supported Rust version (MSRV) is Rust 1.88.

## Usage

Decode bytes directly to RGBA8:

```rust
use avif_rust::image_from_bytes;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoded = std::fs::read("image.avif")?;
    let image = image_from_bytes(&encoded)?;

    assert_eq!(image.rgba.len(), image.width * image.height * 4);
    println!("decoded {}x{}", image.width, image.height);
    Ok(())
}
```

Native targets can use the file helper:

```rust
let image = avif_rust::image_from_file("image.avif")?;
# Ok::<(), avif_rust::DecoderError>(())
```

Use `decode_frame_bytes` when exact source planes or RGBA16 conversion are
needed:

```rust
let encoded = std::fs::read("image.avif")?;
let frame = avif_rust::decode_frame_bytes(&encoded)?;
let rgba16 = frame.to_rgba16()?;

assert_eq!(frame.bit_depth, 8);
assert_eq!(rgba16.rgba.len(), rgba16.width * rgba16.height * 4);
# Ok::<(), Box<dyn std::error::Error>>(())
```

For an AVIS sequence, `decode_sequence_frame_bytes` decodes an individual
Key/IntraOnly, motion-compensated Inter or Switch, or `show_existing_frame`
sample by index. Use `decode_sequence_frames_bytes` when all supported samples
are needed in one pass.

The lower-level `parse_info` and `decode` functions accept a
`bin-rs::reader::BinaryReader`. `decode` preserves the `wml2` callback order:
`init -> next -> draw -> ... -> terminate`; supported multi-frame AVIS
sequences emit one full-canvas `next`/`draw` pair per frame with
`InitOptions { animation: true, .. }`.

For a `tmap` image, `decode_gain_map_frame_bytes` returns the referenced AV1
gain-map item and its `GainMapMetadata` without changing the default base-image
decode. Call `DecodedFrame::to_rgba16_with_gain_map` with a selected display
headroom to apply the explicit base-colour-space composition path. An
alternate-colour-space CICP maps, matrix-shaper, and linear-affine ICC
LUT/mAB alternates are supported; non-linear or reverse-direction ICC
alternates remain fail-closed because their tone curves are not valid for
scalar gain samples.

## Validation

The contributor validation files are not included in the published crate
archive. Clone the independent
[`avif-rust` repository](https://github.com/mith-mmk/avif-rust) and run the
normal crate gates from its root:

```powershell
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
cargo check --manifest-path fuzz/Cargo.toml --bins
```

The optional `wml2` adapter is tested from the parent `wml2` workspace root:

```powershell
cargo test -p wml2 --test avif_decode --no-default-features --features avif
cargo check -p wml2 --target wasm32-unknown-unknown --no-default-features --features avif
```

The exact local oracle set is ignored by Git. When it has been bootstrapped,
run the strict gate with:

```powershell
$env:AVIF_REQUIRE_ORACLES = '1'
cargo test --test oracle_fixtures
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/verify_oracle_sources.ps1
```

See the repository
[`checklist.md`](https://github.com/mith-mmk/avif-rust/blob/main/checklist.md)
for the current conformance status, fixture workflow, and remaining format
work. The
[oracle source verification script](https://github.com/mith-mmk/avif-rust/blob/main/scripts/verify_oracle_sources.ps1)
referenced above is also a repository-only development tool and is not a
runtime dependency.

## License

[MIT](https://github.com/mith-mmk/avif-rust/blob/main/LICENSE)

[![crates.io](https://img.shields.io/crates/v/avif-rust.svg)](https://crates.io/crates/avif-rust)
[![docs.rs](https://img.shields.io/docsrs/avif-rust)](https://docs.rs/avif-rust)
[![license](https://img.shields.io/crates/l/avif-rust.svg)](https://opensource.org/license/mit)

[日本語](./README.ja.md)

# avif-rust

`avif-rust` is an experimental, pure Rust AVIF container and AV1 still-image
decoder. It provides direct RGBA helpers, access to decoded AV1 source planes,
and a callback interface compatible with [`wml2`](https://github.com/mith-mmk/wml2-on-rust).

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
| Deblock, CDEF, and loop restoration used by supported streams | Supported |
| Monochrome, 4:2:0, and 4:2:2 | Supported |
| Alpha auxiliary images and grid composition | Supported (native alpha plane and aligned `clap`/`irot`/`imir` are applied) |
| `iloc` item payloads in file data and meta `idat` | Supported (construction methods 0 and 1) |
| `clap`, `irot`, and `imir` composition | Supported |
| One-frame `avis` primary item | Supported |
| AVIS Key/IntraOnly/show-existing sample decode by index or batch | Supported (Inter/Switch remain fail-closed) |
| Animated AVIF multi-frame callback output | Supported for Key/IntraOnly/show-existing samples (`animation: true`); Inter/Switch remain fail-closed |
| Layered-image selectors (`a1op=0`, `lsel=0`) | Parsed and accepted; non-default layer/operating-point selection remains fail-closed |
| `tmap` primary item base-image fallback | Supported (base `av01` decode); gain-map HDR application is not performed |
| HDR tone mapping and non-matrix ICC display conversion | Not yet supported |

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
Key/IntraOnly or `show_existing_frame` sample by index. Use
`decode_sequence_frames_bytes` when all supported samples are needed in one
pass. Inter and Switch samples return `DecoderError::Unsupported` until
motion-vector reconstruction is available.

The lower-level `parse_info` and `decode` functions accept a
`bin-rs::reader::BinaryReader`. `decode` preserves the `wml2` callback order:
`init -> draw -> terminate`; supported multi-frame AVIS sequences emit one
`draw` call per frame with `InitOptions { animation: true, .. }`.

## Validation

The contributor validation files are not included in the published crate
archive. From a complete repository checkout, the crate lives in the `avif/`
submodule of the parent `wml2` workspace. Run the normal gates from the parent
workspace root:

```powershell
cargo fmt --all -- --check
cargo check -p avif-rust --all-targets
cargo clippy -p avif-rust --all-targets -- -D warnings
cargo test -p avif-rust
cargo check --manifest-path avif/fuzz/Cargo.toml --bins
cargo test -p wml2 --test avif_decode --no-default-features --features avif
cargo check -p wml2 --target wasm32-unknown-unknown --no-default-features --features avif
```

The exact local oracle set is ignored by Git. When it has been bootstrapped,
run the strict gate with:

```powershell
$env:AVIF_REQUIRE_ORACLES = '1'
cargo test -p avif-rust --test oracle_fixtures
pwsh -NoProfile -ExecutionPolicy Bypass -File avif/scripts/verify_oracle_sources.ps1
```

See the repository
[`checklist.md`](https://github.com/mith-mmk/avif-rust/blob/main/checklist.md)
for the current conformance status, fixture workflow, and remaining format
work. The
[oracle source verification script](https://github.com/mith-mmk/avif-rust/blob/main/scripts/verify_oracle_sources.ps1)
referenced above is also a repository-only development tool and is not a
runtime dependency.

## License

[MIT](./LICENSE)

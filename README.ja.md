[![crates.io](https://img.shields.io/crates/v/avif-rust.svg)](https://crates.io/crates/avif-rust)
[![docs.rs](https://img.shields.io/docsrs/avif-rust)](https://docs.rs/avif-rust)
[![license](https://img.shields.io/crates/l/avif-rust.svg)](https://opensource.org/license/mit)

[English](./README.md)

# avif-rust

`avif-rust`は、Pure Rustで実装された実験的なAVIFコンテナ／AV1静止画
デコーダです。RGBAへの直接デコード、デコード済みAV1ソースプレーンへの
アクセス、[`wml2`](https://github.com/mith-mmk/wml2-on-rust)互換のcallback
インターフェースを提供します。

実行時にFFmpeg、libaom、その他のネイティブcodecは呼び出しません。これらは
テスト用oracleの生成と検証にのみ使用します。

## 対応状況

公開デコーダは、8-bit／10-bit／12-bitのsource plane、monochrome、4:2:0、
4:2:2、alpha/grid composition、`clap`／`irot`／`imir`を含む検証済みの単一
frame AVIF profileに対応します。1 frameの`avis` primary itemも静止画として
公開します。外部12-bit fox sampleはFFmpegのRGB oracleを通過し、最終RGB差分は
平均約0.075、最大6です。

| 機能 | 状態 |
| --- | --- |
| 1個のAV1静止frameを含むAVIF primary item | 対応 |
| 8-bit／10-bit／12-bit source plane | 対応（12-bit RGB oracle通過） |
| native decoded plane、optional alpha plane、RGBA8、RGBA16 | 対応（alphaがある場合は`buffers.planes[3]`） |
| premultiplied alpha（`prem`） | 対応（RGBA8／RGBA16出力ではunpremultiply、native planeは変更しない） |
| 対応streamで使用するdeblock、CDEF、loop restoration | 対応 |
| monochrome、4:2:0、4:2:2 | 対応 |
| 拡張`pixi`チャネル記述 | 対応（符号なし整数チャネルとサブサンプリング種別・位置を解析して公開） |
| alpha auxiliary image、grid composition | 対応（native alpha planeと整列した`clap`／`irot`／`imir`を適用） |
| file dataおよびmeta `idat`の`iloc` item payload | 対応（construction method 0／1／indexed method 2） |
| `clap`、`irot`、`imir` composition | 対応 |
| 1 frameの`avis` primary item | 対応 |
| AVISのKey／IntraOnly／Inter／show-existing frameをindex指定・一括でデコード | 対応（動き補償付きInterの検証サンプルを含む、Switchはfail-closed） |
| animated AVIFの複数frame callback出力 | Key／IntraOnly／Inter／show-existingに対応（`animation: true`、Switchはfail-closed） |
| layered image selector（`a1op=0`、`lsel=0`） | 解析・受理（既定外の層／operating point選択はfail-closed） |
| `tmap`主画像のbase画像フォールバック | 対応（base `av01`、ISO 21496 Gain Map、CICPおよびmatrix-shaper ICC alternateの明示的HDR合成に対応。LUT alternateはfail-closed） |
| PQ／HLG transferからbounded SDR RGBA16への変換 | 対応（bounded tone mapping、display固有のcalibrationは未適用） |
| display固有HDR gamut calibration、matrix-shaper以外のICC表示変換 | 未対応 |

未対応のcompositionやAV1 toolは`DecoderError::Unsupported`を返します。不完全な
画像を正常な出力として返さない、fail-closedの方針です。

## インストール

```console
cargo add avif-rust
```

または`Cargo.toml`へ直接追加します。

```toml
[dependencies]
avif-rust = "0.0.2"
```

最小サポートRustバージョン（MSRV）はRust 1.88です。

## 使用方法

byte列からRGBA8へデコードする例です。

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

native targetではfile helperも使用できます。

```rust
let image = avif_rust::image_from_file("image.avif")?;
# Ok::<(), avif_rust::DecoderError>(())
```

正確なsource planeやRGBA16変換が必要な場合は`decode_frame_bytes`を使用します。

```rust
let encoded = std::fs::read("image.avif")?;
let frame = avif_rust::decode_frame_bytes(&encoded)?;
let rgba16 = frame.to_rgba16()?;

assert_eq!(frame.bit_depth, 8);
assert_eq!(rgba16.rgba.len(), rgba16.width * rgba16.height * 4);
# Ok::<(), Box<dyn std::error::Error>>(())
```

AVIS sequenceでは、`decode_sequence_frame_bytes`でKey／IntraOnly／動き補償付き
Interまたは`show_existing_frame`のサンプルをindex指定、
`decode_sequence_frames_bytes`で対応サンプルを一括デコードできます。Switchは
独立した参照意味論の検証が残っているため、`DecoderError::Unsupported`を返します。

低水準の`parse_info`と`decode`は`bin-rs::reader::BinaryReader`を受け取ります。
`decode`は`wml2`互換の`init -> draw -> terminate` callback順序を維持し、
対応する複数frame AVISでは`InitOptions { animation: true, .. }`と各frameの
`draw`を出力します。

`tmap`画像では、`decode_gain_map_frame_bytes`によって参照されたAV1 gain-map
itemと`GainMapMetadata`を取得できます。既定のbase画像デコードは変更せず、
`DecodedFrame::to_rgba16_with_gain_map`へdisplay headroomを渡すことで、base色空間
の明示的なHDR合成を適用できます。CICP alternateとmatrix-shaper ICC alternateは
対応しますが、gain値へtone curveを適用できないLUT alternateはfail-closedです。

## 検証

開発者向けの検証ファイルは公開crate archiveに含まれません。完全なrepository
checkoutでは、crateは親`wml2` workspaceの`avif/` submoduleにあります。通常の
gateは親workspace rootから実行します。

```powershell
cargo fmt --all -- --check
cargo check -p avif-rust --all-targets
cargo clippy -p avif-rust --all-targets -- -D warnings
cargo test -p avif-rust
cargo check --manifest-path avif/fuzz/Cargo.toml --bins
cargo test -p wml2 --test avif_decode --no-default-features --features avif
cargo check -p wml2 --target wasm32-unknown-unknown --no-default-features --features avif
```

完全一致oracleはGitの管理外です。local fixtureをbootstrap済みの場合はstrict gateを
実行できます。

```powershell
$env:AVIF_REQUIRE_ORACLES = '1'
cargo test -p avif-rust --test oracle_fixtures
pwsh -NoProfile -ExecutionPolicy Bypass -File avif/scripts/verify_oracle_sources.ps1
```

現在のconformance状況、fixture workflow、未対応形式はrepositoryの
[`checklist.md`](https://github.com/mith-mmk/avif-rust/blob/main/checklist.md)を
参照してください。上記の
[oracle source検証script](https://github.com/mith-mmk/avif-rust/blob/main/scripts/verify_oracle_sources.ps1)
もrepository専用の開発ツールであり、runtime依存ではありません。

## ライセンス

[MIT](./LICENSE)

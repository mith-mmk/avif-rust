# Changelog

All notable changes to this crate are documented in this file.

## 0.0.2 - 2026-07-28

### Added

- Native 8-bit, 10-bit, and 12-bit AV1 planes plus RGBA8/RGBA16 conversion.
- Monochrome, YUV 4:2:0/4:2:2, alpha, grid, image transforms, gain maps, and
  bounded HDR display conversion for the documented supported profiles.
- AVIS sequence decoding for tested Key, IntraOnly, Inter, Switch, and
  show-existing samples, including per-frame alpha and duration callbacks.
- Strict native-plane/RGBA oracle, malformed-input, fuzz-target, Wasm, and
  FFmpeg conformance gates.

### Fixed

- YUV 4:2:0 chroma reconstruction that could tint the right half of tiled
  images blue.
- Bottom-right block placement in the tested 10-bit Seine HDR images.
- Compound/inter motion reconstruction and alpha synchronization in the
  five-frame animated AVIF fixtures.

### Compatibility

- The minimum supported Rust version is 1.88.
- Unsupported AVIF composition or AV1 tools fail closed with
  `DecoderError::Unsupported` instead of returning a partial image.
- `DecoderError` is now non-exhaustive so later releases can add error kinds
  without forcing an otherwise unnecessary breaking API change.

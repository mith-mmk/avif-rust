# Changelog

All notable changes to this crate are documented in this file.

## 0.0.3 - 2026-07-29

### Fixed

- Enabled strict entropy termination validation for tested Inter and Switch
  AVIS frames instead of bypassing the final arithmetic-reader check.
- Corrected Inter/Switch header consumption, skip-mode reference derivation,
  transform contexts, global-motion inheritance, and compound/single-reference
  motion-vector candidate construction to match the AV1 normative decoder.
- Kept motion-vector fallback candidates paired with AV1 reference types and
  ranked compound candidates only after all spatial and temporal scans, so DRL
  syntax is consumed exactly when signalled.

### Compatibility

- The public still-frame, sequence, `DecodedFrame`, and WML2 callback APIs are
  unchanged from 0.0.2.
- Valid tested 8-bit, 10-bit, and 12-bit Inter/Switch streams now pass strict
  termination checks; malformed trailing-one, padding, truncation, and overread
  cases remain fail-closed as `DecoderError::Bitstream`.
- The runtime dependency remains `bin-rs` only, and the MSRV remains Rust 1.88.

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

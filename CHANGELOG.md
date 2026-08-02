# Changelog

All notable changes to this crate are documented in this file.

## 0.0.6 - 2026-08-02

### Changed

- `AvifSequenceDecoder` now retains AV1 reference slots, CDF state, motion
  fields, and the sample cursor across calls instead of replaying the sequence
  prefix for every frame.
- Color and alpha tracks advance through independent synchronized states; a
  static alpha auxiliary image is decoded once and reused for later frames.
- Batch sequence decoding and WML2 animation callbacks now use the same linear
  forward decoder. Indexed decoding remains compatible and replays only the
  required prefix for each independent call.
- Sequence payloads have one canonical owner inside the incremental decoder,
  avoiding a second retained copy in its private container view.

### Compatibility

- Existing sequence types, constructors, indexed/batch helpers, frame timing,
  metadata, full-canvas callbacks, and callback ordering are unchanged.
- A failed color or alpha sample leaves the incremental decoder at the same
  sample, and repeated calls after the end continue to return `Ok(None)`.
- Grid sequences, additional AV1 tools, and encoding remain outside this
  release.

## 0.0.5 - 2026-08-01

### Added

- Added exact AVIS frame timing (`timescale`, PTS, duration, and rounded
  milliseconds) and finite, infinite, or unknown repetition metadata.
- Added `AvifAnimation`, `DecodedSequenceFrame`, and the incremental
  `AvifSequenceDecoder` API while preserving the existing sequence helpers.
- AVIS callback output now decodes and emits one full-canvas RGBA frame at a
  time, synchronizing color and alpha tracks and passing frame durations to
  `NextOptions`.
- Added animation metadata for frame count, loop count, timescale, total
  duration, PTS, and per-frame durations.

### Compatibility

- Existing `AvifSequence`, indexed/batch sequence APIs, public callback method
  shapes, fixed-canvas drawing, and alpha-plane behavior remain compatible.
- Callback `Abort` is a normal early completion and does not call `terminate`;
  callback errors continue to propagate immediately.

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

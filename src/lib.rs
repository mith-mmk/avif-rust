//! Pure Rust AVIF decode entry points.
//!
//! The crate owns AVIF/AV1 parsing so callers such as `wml2` only need a thin
//! draw-side adapter. Container parsing is implemented with `bin-rs`
//! `BinaryReader` inputs for compatibility with the surrounding codecs.

pub mod av1;
pub mod compat;
pub mod container;
pub mod decoder;
mod error;
mod icc;
mod image;
pub mod obu;
#[cfg(test)]
mod test_support;

pub use compat::{
    CallbackResponse, DataMap, DecodeOptions, DrawCallback, DrawOptions, ImageRect, InitOptions,
    Metadata, NextBlend, NextDispose, NextOptions, ResponseCommand, TerminateOptions,
    VerboseOptions,
};
pub use container::{
    AuxiliaryImage, AvifAnimation, AvifFrameTiming, AvifInfo, AvifRepetitionCount, AvifSequence,
    AvifSequenceSampleKind, CleanAperture, ColorInformation, GainMapChannel, GainMapMetadata,
    GainMapRational, GridCell, GridImage, ImageMirror, ImageRotation, ImageSpatialExtents,
    NclxColorInformation, PixelChannelInformation, PixelInformation, PixelSubsampling,
    classify_av1_sequence_sample, parse_avif_animation, parse_avif_sequence,
    parse_gain_map_metadata,
};
pub use decoder::{
    AvifSequenceDecoder, DecodedFrame, DecodedGainMapFrame, DecodedSequenceFrame, decode,
    decode_frame_bytes, decode_gain_map_frame_bytes, decode_sequence_frame_bytes,
    decode_sequence_frames_bytes, parse_info,
};
pub use error::DecoderError;
pub use image::{ImageBuffer, Rgba16ImageBuffer};

/// Decodes a still AVIF image from memory into an RGBA buffer.
///
/// The decoder supports the tested 8/10/12-bit still-image profiles and
/// returns the decoded image as RGBA8. Unsupported AV1 tools are reported
/// through [`DecoderError::Unsupported`].
pub fn image_from_bytes(data: &[u8]) -> Result<ImageBuffer, DecoderError> {
    decoder::decode_bytes(data)
}

/// Reads a still AVIF image from disk and decodes it to RGBA.
#[cfg(not(target_family = "wasm"))]
pub fn image_from_file<P: AsRef<std::path::Path>>(
    filename: P,
) -> Result<ImageBuffer, DecoderError> {
    let data = std::fs::read(filename).map_err(|err| DecoderError::Io(err.to_string()))?;
    image_from_bytes(&data)
}

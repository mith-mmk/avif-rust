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
mod image;
pub mod obu;

pub use compat::{
    CallbackResponse, DataMap, DecodeOptions, DrawCallback, DrawOptions, InitOptions, Metadata,
    ResponseCommand, TerminateOptions, VerboseOptions,
};
pub use container::{
    AuxiliaryImage, AvifInfo, ColorInformation, ImageSpatialExtents, NclxColorInformation,
    PixelInformation,
};
pub use decoder::{DecodedFrame, decode, decode_frame_bytes, parse_info};
pub use error::DecoderError;
pub use image::{ImageBuffer, Rgba16ImageBuffer};

/// Decodes a still AVIF image from memory into an RGBA buffer.
///
/// The decoder currently targets 8-bit, full-resolution GBR still images.
/// Unsupported AV1 tools are reported through [`DecoderError::Unsupported`].
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

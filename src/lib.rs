//! Pure Rust AVIF decode entry points.
//!
//! The crate owns AVIF/AV1 parsing so callers such as `wml2` only need a thin
//! draw-side adapter. Container parsing is implemented with `bin-rs`
//! `BinaryReader` inputs for compatibility with the surrounding codecs.

pub mod compat;
pub mod container;
pub mod decoder;
mod error;
mod image;

pub use compat::{
    CallbackResponse, DataMap, DecodeOptions, DrawCallback, DrawOptions, InitOptions, Metadata,
    ResponseCommand, TerminateOptions, VerboseOptions,
};
pub use container::{AvifInfo, ColorInformation, ImageSpatialExtents, PixelInformation};
pub use decoder::{decode, parse_info};
pub use error::DecoderError;
pub use image::ImageBuffer;

/// Decodes a still AVIF image from memory into an RGBA buffer.
///
/// This API is reserved for the full AV1 image decoder. The container parser is
/// available through [`parse_info`].
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

use crate::compat::{DataMap, DecodeOptions, InitOptions};
use crate::container::{AvifInfo, parse_avif};
use crate::{DecoderError, ImageBuffer};
use bin_rs::reader::{BinaryReader, BytesReader};
use std::io::SeekFrom;

type Error = Box<dyn std::error::Error>;

/// Parses AVIF container metadata from a `bin-rs` reader.
pub fn parse_info<B: BinaryReader>(reader: &mut B) -> Result<AvifInfo, DecoderError> {
    let data = read_to_end(reader)?;
    parse_avif(&data)
}

/// Decodes an AVIF image using a callback-based interface compatible with
/// `wml2`'s draw-side flow.
pub fn decode<B: BinaryReader>(
    reader: &mut B,
    option: &mut DecodeOptions<'_>,
) -> Result<(), Error> {
    let info = parse_info(reader)?;
    emit_metadata(&info, option)?;

    let width = info
        .width
        .ok_or_else(|| DecoderError::Bitstream("primary image width is missing".to_string()))?;
    let height = info
        .height
        .ok_or_else(|| DecoderError::Bitstream("primary image height is missing".to_string()))?;
    option.drawer.init(
        width as usize,
        height as usize,
        Some(InitOptions {
            loop_count: 1,
            animation: false,
        }),
    )?;

    Err(Box::new(DecoderError::Unsupported(
        "AV1 image bitstream decoding is not implemented yet".to_string(),
    )))
}

pub fn decode_bytes(data: &[u8]) -> Result<ImageBuffer, DecoderError> {
    let mut reader = BytesReader::new(data);
    let info = parse_info(&mut reader)?;
    let width = info
        .width
        .ok_or_else(|| DecoderError::Bitstream("primary image width is missing".to_string()))?;
    let height = info
        .height
        .ok_or_else(|| DecoderError::Bitstream("primary image height is missing".to_string()))?;
    Err(DecoderError::Unsupported(format!(
        "AV1 image bitstream decoding is not implemented yet ({width}x{height})"
    )))
}

fn emit_metadata(info: &AvifInfo, option: &mut DecodeOptions<'_>) -> Result<(), Error> {
    option
        .drawer
        .set_metadata("Format", DataMap::Ascii("AVIF".to_string()))?;
    if let Some(width) = info.width {
        option
            .drawer
            .set_metadata("width", DataMap::UInt(width as u64))?;
    }
    if let Some(height) = info.height {
        option
            .drawer
            .set_metadata("height", DataMap::UInt(height as u64))?;
    }
    if let Some(primary_item_id) = info.primary_item_id {
        option
            .drawer
            .set_metadata("AVIF primary item", DataMap::UInt(primary_item_id as u64))?;
    }
    if let Some(pixi) = &info.pixel_information {
        option.drawer.set_metadata(
            "AVIF bits per channel",
            DataMap::UIntAllay(
                pixi.bits_per_channel
                    .iter()
                    .map(|value| *value as u64)
                    .collect(),
            ),
        )?;
    }
    if let Some(colr) = &info.color_information {
        option.drawer.set_metadata(
            "AVIF color type",
            DataMap::Ascii(String::from_utf8_lossy(&colr.color_type).to_string()),
        )?;
    }
    if let Some(av1_config) = &info.av1_config {
        option
            .drawer
            .set_metadata("AV1 config", DataMap::Raw(av1_config.clone()))?;
    }

    if option.debug_flag > 0 {
        option.drawer.verbose(
            &format!(
                "AVIF {}x{} primary_item_payload={} bytes",
                info.width.unwrap_or(0),
                info.height.unwrap_or(0),
                info.primary_item_payload.len()
            ),
            None,
        )?;
    }
    Ok(())
}

fn read_to_end<B: BinaryReader>(reader: &mut B) -> Result<Vec<u8>, DecoderError> {
    let current = reader
        .offset()
        .map_err(|err| DecoderError::Io(err.to_string()))?;
    let end = reader
        .seek(SeekFrom::End(0))
        .map_err(|err| DecoderError::Io(err.to_string()))?;
    reader
        .seek(SeekFrom::Start(current))
        .map_err(|err| DecoderError::Io(err.to_string()))?;
    if end < current {
        return Err(DecoderError::Bitstream(
            "reader end is before current position".to_string(),
        ));
    }
    let len = usize::try_from(end - current)
        .map_err(|_| DecoderError::InvalidParam("input is too large".to_string()))?;
    reader
        .read_bytes_as_vec(len)
        .map_err(|err| DecoderError::Io(err.to_string()))
}

use crate::av1::ColorRange;
use crate::av1::PostFilterState;
#[cfg(test)]
use crate::av1::alloc_frame_buffers;
#[cfg(test)]
use crate::av1::decode_luma_root_block_prefix_with_post_filter_state_and_entropy;
use crate::av1::{
    Av1CodecConfiguration, BlockModeProbe, CdfContext, ChromaSamplePosition, ColorConfig,
    FilmGrainParams, FrameBuffers, FrameDecodePlan, FrameHeader, FrameType, GlobalMotionParams,
    LoopFilterParams, PartitionProbe, PlaneBuffer, PlaneLayout, QuantState, ReferenceFrameState,
    ResidualProbe, SegmentationParams, SequenceHeader, TileEntropyState, TileGroup,
    alloc_coded_frame_buffers, apply_film_grain, apply_superres_horizontal,
    build_still_decode_plan, cdef_adjust_primary_strength, cdef_chroma_direction,
    cdef_filter_block_region_with_edge_mode_into_bit_depth_visible_scaled,
    cdef_find_direction_with_variance_visible, convert_linear_rgb_primaries,
    crop_frame_buffers_to_plan, deblock_filter_edge_with_visible_bounds,
    decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf,
    frame_buffers_to_rgba_16, parse_av1_config, parse_frame_header,
    parse_frame_header_with_references, parse_sequence_header, parse_show_existing_frame_index,
    parse_tile_group, plan_transform_blocks_with_tx_size, prepare_tile_entropy,
    probe_first_block_residuals, probe_tile_block_modes, probe_tile_partitions,
    sgrproj_filter_unit_into_with_scratch_bit_depth_visible,
    wiener_filter_unit_into_with_scratch_bit_depth_visible,
};
use crate::compat::{DataMap, DecodeOptions, InitOptions};
use crate::container::{
    AvifInfo, AvifSequenceSampleInfo, AvifSequenceSampleKind, CleanAperture, ColorInformation,
    GridCell, ImageMirror, ImageRotation, PixelInformation, SampleTransformInput,
    SampleTransformToken, grid_cell_alpha_item_id, grid_has_cell_alpha,
    inspect_av1_sequence_sample, parse_avif, parse_gain_map_image, parse_sample_transform,
};
use crate::obu::{ObuType, find_obu_payloads_in_parts, parse_obu_stream};
use crate::{DecoderError, ImageBuffer, Rgba16ImageBuffer};
use bin_rs::reader::BinaryReader;
use std::io::SeekFrom;
use std::sync::Arc;

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
    let data = read_to_end(reader)?;
    let info = parse_avif(&data)?;
    validate_public_container_preflight(&info, true)?;
    if let Some(frame) = decode_sample_transform_frame(&data, &info)? {
        let mut image = frame.to_rgba8()?;
        apply_clean_aperture(&mut image, info.clean_aperture)?;
        apply_mirror(&mut image, info.mirror)?;
        apply_rotation(&mut image, info.rotation)?;
        emit_metadata(&info, None, option)?;
        option.drawer.init(
            image.width,
            image.height,
            Some(InitOptions {
                loop_count: 1,
                animation: false,
            }),
        )?;
        option
            .drawer
            .draw(0, 0, image.width, image.height, &image.rgba, None)?;
        option.drawer.terminate(None)?;
        return Ok(());
    }
    if info.primary_grid.is_some() {
        let image = decode_grid_image(&info)?;
        emit_metadata(&info, None, option)?;
        option.drawer.init(
            image.width,
            image.height,
            Some(InitOptions {
                loop_count: 1,
                animation: false,
            }),
        )?;
        option
            .drawer
            .draw(0, 0, image.width, image.height, &image.rgba, None)?;
        option.drawer.terminate(None)?;
        return Ok(());
    }
    if info.sequence_sample_payloads.len() > 1 {
        let mut headers = parse_av1_headers(&info)?;
        // The probes replay entropy traversal and are only consumed by the
        // diagnostic metadata below. Keep the normal draw path to one decode;
        // callers that request debug output explicitly still receive the probes.
        if option.debug_flag > 0 {
            populate_diagnostic_probes(&mut headers);
        }
        emit_metadata(&info, Some(&headers), option)?;

        // Decode the complete supported sequence before initializing the
        // callback. Unsupported prediction syntax therefore fails closed
        // without exposing a partial animation to the caller.
        let mut frames = decode_sequence_frames_from_info(&info)?;
        if !info.alpha_auxiliary_items.is_empty() {
            let alpha_frame = decode_alpha_auxiliary_frame(&info)?;
            for frame in &mut frames {
                append_alpha_plane(frame, &alpha_frame)?;
            }
        }
        let mut images = Vec::with_capacity(frames.len());
        for frame in frames {
            let mut image = frame.to_rgba8()?;
            apply_clean_aperture(&mut image, info.clean_aperture)?;
            apply_mirror(&mut image, info.mirror)?;
            apply_rotation(&mut image, info.rotation)?;
            images.push(image);
        }
        let first = images.first().ok_or_else(|| {
            DecoderError::Bitstream("AVIS sequence has no decodable samples".to_string())
        })?;
        if images
            .iter()
            .any(|image| image.width != first.width || image.height != first.height)
        {
            return Err(DecoderError::Unsupported(
                "AVIS sequence samples have different output dimensions".to_string(),
            )
            .into());
        }
        option.drawer.init(
            first.width,
            first.height,
            Some(InitOptions {
                loop_count: 1,
                animation: true,
            }),
        )?;
        for image in images {
            option
                .drawer
                .draw(0, 0, image.width, image.height, &image.rgba, None)?;
        }
        option.drawer.terminate(None)?;
        return Ok(());
    }
    let mut headers = parse_av1_headers(&info)?;
    // The probes replay entropy traversal and are only consumed by the
    // diagnostic metadata below. Keep the normal draw path to one decode;
    // callers that request debug output explicitly still receive the probes.
    if option.debug_flag > 0 {
        populate_diagnostic_probes(&mut headers);
    }
    emit_metadata(&info, Some(&headers), option)?;
    let image = decode_still_image(&headers, Some(&info))?;

    option.drawer.init(
        image.width,
        image.height,
        Some(InitOptions {
            loop_count: 1,
            animation: false,
        }),
    )?;
    option
        .drawer
        .draw(0, 0, image.width, image.height, &image.rgba, None)?;
    option.drawer.terminate(None)?;
    Ok(())
}

pub fn decode_bytes(data: &[u8]) -> Result<ImageBuffer, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, true)?;
    if let Some(frame) = decode_sample_transform_frame(data, &info)? {
        let mut image = frame.to_rgba8()?;
        apply_clean_aperture(&mut image, info.clean_aperture)?;
        apply_mirror(&mut image, info.mirror)?;
        apply_rotation(&mut image, info.rotation)?;
        return Ok(image);
    }
    if info.primary_grid.is_some() {
        return decode_grid_image(&info);
    }
    if let Some(frame) = decode_hidden_key_frame_show_existing(&info)? {
        let mut image = frame.to_rgba8()?;
        apply_clean_aperture(&mut image, info.clean_aperture)?;
        apply_mirror(&mut image, info.mirror)?;
        apply_rotation(&mut image, info.rotation)?;
        return Ok(image);
    }
    let headers = parse_av1_headers(&info)?;
    decode_still_image(&headers, Some(&info))
}

/// Decoded still-frame planes before colour conversion.
///
/// Samples are stored as native AV1 source planes in raster order. The first
/// three planes are Y/U/V (or the profile's native plane order); when an alpha
/// auxiliary or alpha grid is present, plane index three is the optional alpha
/// plane. The current decoder only supports a subset of still-image tools, but
/// this type is the conformance-test boundary for exact plane comparisons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    pub width: usize,
    pub height: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub bit_depth: u8,
    pub color_config: ColorConfig,
    pub color_information: Option<ColorInformation>,
    pub alpha_premultiplied: bool,
    pub buffers: FrameBuffers,
}

/// A decoded ISO 21496 gain-map image and its descriptor.
///
/// Gain-map pixels are returned as a normal native AV1 frame at the map's own
/// dimensions. Composition resamples them to the base frame dimensions when
/// needed; applications can still apply display-headroom policy themselves.
/// The default still-image API intentionally continues to return only the base
/// image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedGainMapFrame {
    pub metadata: crate::container::GainMapMetadata,
    pub frame: DecodedFrame,
}

impl DecodedFrame {
    pub fn to_rgba8(&self) -> Result<ImageBuffer, DecoderError> {
        if self
            .color_information
            .as_ref()
            .and_then(ColorInformation::icc_profile)
            .is_none()
        {
            let mut image = crate::av1::frame_buffers_to_rgba_8(&self.buffers, &self.color_config)?;
            if self.alpha_premultiplied {
                unpremultiply_rgba8(&mut image.rgba);
            }
            return Ok(image);
        }
        let rgba16 = self.to_rgba16()?;
        let rgba = rgba16
            .rgba
            .iter()
            .map(|sample| ((u32::from(*sample) * 255 + 32767) / 65535) as u8)
            .collect();
        Ok(ImageBuffer {
            width: rgba16.width,
            height: rgba16.height,
            rgba,
        })
    }

    pub fn to_rgba16(&self) -> Result<Rgba16ImageBuffer, DecoderError> {
        let mut image = frame_buffers_to_rgba_16(&self.buffers, &self.color_config)?;
        if let Some(profile) = self
            .color_information
            .as_ref()
            .and_then(ColorInformation::icc_profile)
        {
            crate::icc::apply_to_rgba16(&mut image.rgba, profile)?;
        }
        if self.alpha_premultiplied {
            unpremultiply_rgba16(&mut image.rgba);
        }
        Ok(image)
    }

    /// Applies an explicitly decoded ISO 21496 gain map to this frame.
    ///
    /// Gain-map frames may use a different native size and are resampled to
    /// the base dimensions during composition. Base-colour maps and alternate
    /// maps in supported CICP RGB primary sets are
    /// supported; matrix-shaper and linear-affine ICC LUT/mAB alternate
    /// conversions are supported while non-linear or reverse-direction
    /// profiles fail closed because applying their tone curves to scalar gain
    /// samples would change the gain semantics.
    /// `hdr_headroom` is expressed in log2 headroom units; a value at the base
    /// headroom returns the base RGBA16 image unchanged. The default AVIF
    /// decode path never applies this method implicitly.
    pub fn to_rgba16_with_gain_map(
        &self,
        gain_map: &DecodedGainMapFrame,
        hdr_headroom: f32,
    ) -> Result<Rgba16ImageBuffer, DecoderError> {
        if !hdr_headroom.is_finite() || hdr_headroom < 0.0 {
            return Err(DecoderError::InvalidParam(
                "gain-map HDR headroom must be finite and non-negative".to_string(),
            ));
        }
        let weight = gain_map_weight(hdr_headroom, &gain_map.metadata)?;
        let mut base = self.to_rgba16()?;
        if weight == 0.0 {
            return Ok(base);
        }
        let declared_base_primaries = self
            .color_config
            .color_description
            .map(|description| description.color_primaries)
            .unwrap_or(2);
        let declared_alternate_primaries = gain_map
            .frame
            .color_config
            .color_description
            .map(|description| description.color_primaries)
            .unwrap_or(declared_base_primaries);
        // A gain-map item carries scalar/log gain samples, and libavif files
        // commonly leave its CICP primaries unspecified (2). Treat an
        // unspecified side as inheriting the other side rather than trying
        // to construct a chromaticity matrix for CICP 2.
        let (base_primaries, alternate_primaries) =
            if declared_base_primaries == 2 && declared_alternate_primaries != 2 {
                (declared_alternate_primaries, declared_alternate_primaries)
            } else if declared_alternate_primaries == 2 {
                (declared_base_primaries, declared_base_primaries)
            } else {
                (declared_base_primaries, declared_alternate_primaries)
            };
        let convert_alternate =
            !gain_map.metadata.use_base_colour_space && base_primaries != alternate_primaries;
        let alternate_icc = gain_map
            .frame
            .color_information
            .as_ref()
            .and_then(ColorInformation::icc_profile);
        let decoded_map = gain_map.frame.to_rgba16()?;
        let map = if decoded_map.width == base.width && decoded_map.height == base.height {
            decoded_map
        } else {
            resample_gain_map(&decoded_map, base.width, base.height)?
        };
        if map.rgba.len() != base.rgba.len() {
            return Err(DecoderError::Bitstream(
                "gain-map and base RGBA buffers do not match".to_string(),
            ));
        }
        let channels = gain_map.metadata.channels.as_slice();
        if !matches!(channels.len(), 1 | 3) {
            return Err(DecoderError::Unsupported(format!(
                "gain-map channel count {} is not supported",
                channels.len()
            )));
        }
        let mut gamma = [0.0; 3];
        let mut minimum = [0.0; 3];
        let mut maximum = [0.0; 3];
        let mut base_offset = [0.0; 3];
        let mut alternate_offset = [0.0; 3];
        for channel in 0..3 {
            let metadata = channels[if channels.len() == 1 { 0 } else { channel }];
            gamma[channel] = rational_to_f64(metadata.gamma, "gain-map gamma")?;
            if gamma[channel] <= 0.0 {
                return Err(DecoderError::Bitstream(
                    "gain-map gamma must be positive".to_string(),
                ));
            }
            minimum[channel] = rational_to_f64(metadata.gain_map_min, "gain-map minimum")?;
            maximum[channel] = rational_to_f64(metadata.gain_map_max, "gain-map maximum")?;
            base_offset[channel] = rational_to_f64(metadata.base_offset, "base offset")?;
            alternate_offset[channel] =
                rational_to_f64(metadata.alternate_offset, "alternate offset")?;
        }
        for (base_pixel, map_pixel) in base.rgba.chunks_exact_mut(4).zip(map.rgba.chunks_exact(4)) {
            let mut base_linear = [
                srgb_to_linear(f64::from(base_pixel[0]) / f64::from(u16::MAX)),
                srgb_to_linear(f64::from(base_pixel[1]) / f64::from(u16::MAX)),
                srgb_to_linear(f64::from(base_pixel[2]) / f64::from(u16::MAX)),
            ];
            if convert_alternate {
                if let Some(profile) = alternate_icc {
                    crate::icc::convert_linear_srgb_with_profile(&mut base_linear, profile, true)?;
                } else {
                    convert_linear_rgb_primaries(
                        &mut base_linear,
                        base_primaries,
                        alternate_primaries,
                    )?;
                }
            }
            let mut tone_mapped = [0.0; 3];
            for channel in 0..3 {
                let map_value = f64::from(map_pixel[channel]) / f64::from(u16::MAX);
                let gain_map_log2 = minimum[channel]
                    + (maximum[channel] - minimum[channel]) * map_value.powf(1.0 / gamma[channel]);
                tone_mapped[channel] = (base_linear[channel] + base_offset[channel])
                    * (gain_map_log2 * f64::from(weight)).exp2()
                    - alternate_offset[channel];
            }
            if convert_alternate {
                if let Some(profile) = alternate_icc {
                    crate::icc::convert_linear_srgb_with_profile(&mut tone_mapped, profile, false)?;
                } else {
                    convert_linear_rgb_primaries(
                        &mut tone_mapped,
                        alternate_primaries,
                        base_primaries,
                    )?;
                }
            }
            for channel in 0..3 {
                base_pixel[channel] = (linear_to_srgb(tone_mapped[channel].max(0.0))
                    * f64::from(u16::MAX))
                .round()
                .clamp(0.0, f64::from(u16::MAX)) as u16;
            }
        }
        Ok(base)
    }
}

fn gain_map_weight(
    hdr_headroom: f32,
    metadata: &crate::container::GainMapMetadata,
) -> Result<f32, DecoderError> {
    let base = rational_to_f64(metadata.base_hdr_headroom, "base HDR headroom")?;
    let alternate = rational_to_f64(metadata.alternate_hdr_headroom, "alternate HDR headroom")?;
    if (alternate - base).abs() < f64::EPSILON {
        return Ok(0.0);
    }
    let normalized = ((f64::from(hdr_headroom) - base) / (alternate - base)).clamp(0.0, 1.0);
    Ok(if metadata.backward_direction {
        -(normalized as f32)
    } else {
        normalized as f32
    })
}

fn rational_to_f64(
    rational: crate::container::GainMapRational,
    name: &str,
) -> Result<f64, DecoderError> {
    if rational.denominator == 0 {
        return Err(DecoderError::Bitstream(format!(
            "{name} denominator is zero"
        )));
    }
    Ok(rational.numerator as f64 / f64::from(rational.denominator))
}

fn srgb_to_linear(encoded: f64) -> f64 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(linear: f64) -> f64 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

fn resample_gain_map(
    input: &Rgba16ImageBuffer,
    width: usize,
    height: usize,
) -> Result<Rgba16ImageBuffer, DecoderError> {
    if input.width == 0 || input.height == 0 || width == 0 || height == 0 {
        return Err(DecoderError::Bitstream(
            "gain-map resampling dimensions must be non-zero".to_string(),
        ));
    }
    let input_pixels = input
        .width
        .checked_mul(input.height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| DecoderError::InvalidParam("gain-map buffer size overflows".to_string()))?;
    if input.rgba.len() != input_pixels {
        return Err(DecoderError::Bitstream(
            "gain-map RGBA buffer length does not match dimensions".to_string(),
        ));
    }
    let output_pixels = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| DecoderError::InvalidParam("gain-map output size overflows".to_string()))?;
    let mut rgba = vec![0_u16; output_pixels];
    for y in 0..height {
        let source_y = (((y as f64 + 0.5) * input.height as f64 / height as f64) - 0.5)
            .clamp(0.0, (input.height - 1) as f64);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(input.height - 1);
        let fy = source_y - y0 as f64;
        for x in 0..width {
            let source_x = (((x as f64 + 0.5) * input.width as f64 / width as f64) - 0.5)
                .clamp(0.0, (input.width - 1) as f64);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(input.width - 1);
            let fx = source_x - x0 as f64;
            let top = (y0 * input.width + x0) * 4;
            let top_right = (y0 * input.width + x1) * 4;
            let bottom = (y1 * input.width + x0) * 4;
            let bottom_right = (y1 * input.width + x1) * 4;
            let destination = (y * width + x) * 4;
            for channel in 0..4 {
                let top_value = f64::from(input.rgba[top + channel])
                    + (f64::from(input.rgba[top_right + channel])
                        - f64::from(input.rgba[top + channel]))
                        * fx;
                let bottom_value = f64::from(input.rgba[bottom + channel])
                    + (f64::from(input.rgba[bottom_right + channel])
                        - f64::from(input.rgba[bottom + channel]))
                        * fx;
                rgba[destination + channel] = (top_value + (bottom_value - top_value) * fy)
                    .round()
                    .clamp(0.0, f64::from(u16::MAX))
                    as u16;
            }
        }
    }
    Ok(Rgba16ImageBuffer {
        width,
        height,
        rgba,
    })
}

fn unpremultiply_rgba8(rgba: &mut [u8]) {
    for pixel in rgba.chunks_exact_mut(4) {
        let alpha = u32::from(pixel[3]);
        if alpha == 0 {
            pixel[..3].fill(0);
            continue;
        }
        for channel in &mut pixel[..3] {
            *channel = ((u32::from(*channel) * 255 + alpha / 2) / alpha).min(255) as u8;
        }
    }
}

fn unpremultiply_rgba16(rgba: &mut [u16]) {
    for pixel in rgba.chunks_exact_mut(4) {
        let alpha = u64::from(pixel[3]);
        if alpha == 0 {
            pixel[..3].fill(0);
            continue;
        }
        for channel in &mut pixel[..3] {
            *channel = ((u64::from(*channel) * u64::from(u16::MAX) + alpha / 2) / alpha)
                .min(u64::from(u16::MAX)) as u16;
        }
    }
}

/// Decodes a still AVIF image from memory into high-precision source planes.
pub fn decode_frame_bytes(data: &[u8]) -> Result<DecodedFrame, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, false)?;
    if let Some(frame) = decode_sample_transform_frame(data, &info)? {
        return Ok(frame);
    }
    if info.primary_grid.is_some() {
        return decode_grid_frame(&info);
    }
    let mut frame = if let Some(frame) = decode_hidden_key_frame_show_existing(&info)? {
        frame
    } else {
        let headers = parse_av1_headers(&info)?;
        decode_still_frame(&headers, Some(&info))?
    };
    if !info.alpha_auxiliary_items.is_empty() {
        let alpha_frame = decode_alpha_auxiliary_frame(&info)?;
        append_alpha_plane(&mut frame, &alpha_frame)?;
    }
    Ok(frame)
}

/// Decodes the AV1 gain-map item referenced by a `tmap` derived image.
///
/// `Ok(None)` means that the input has no `tmap` item. Unsupported gain-map
/// item layouts fail closed while the ordinary [`decode_frame_bytes`] API
/// remains available for the base image.
pub fn decode_gain_map_frame_bytes(
    data: &[u8],
) -> Result<Option<DecodedGainMapFrame>, DecoderError> {
    let Some(gain_map) = parse_gain_map_image(data)? else {
        return Ok(None);
    };
    let info = AvifInfo {
        major_brand: *b"avif",
        compatible_brands: vec![*b"avif"],
        primary_item_id: None,
        width: Some(gain_map.width),
        height: Some(gain_map.height),
        pixel_information: gain_map.pixel_information,
        color_information: gain_map.color_information,
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: gain_map.grid,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: gain_map.av1_config,
        primary_item_payload: gain_map.payload,
        sequence_sample_payloads: Vec::new(),
    };
    validate_public_container_preflight(&info, false)?;
    let frame = if info.primary_grid.is_some() {
        decode_grid_frame(&info)?
    } else {
        let headers = parse_av1_headers(&info)?;
        decode_still_frame(&headers, Some(&info))?
    };
    Ok(Some(DecodedGainMapFrame {
        metadata: gain_map.metadata,
        frame,
    }))
}

/// Decodes one sample from an AVIS sequence into source planes.
///
/// Key and intra-only samples are decoded independently while sharing the
/// sequence header from the primary item. A `show_existing_frame` sample can
/// reuse a previously decoded reference slot, and inter/switch samples use the
/// same reference-slot state for reconstruction.
pub fn decode_sequence_frame_bytes(
    data: &[u8],
    frame_index: usize,
) -> Result<DecodedFrame, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, false)?;
    let mut frame = decode_sequence_frame_from_info(&info, frame_index)?;
    if !info.alpha_auxiliary_items.is_empty() {
        let alpha_frame = decode_alpha_auxiliary_frame(&info)?;
        append_alpha_plane(&mut frame, &alpha_frame)?;
    }
    Ok(frame)
}

/// Decodes every independently addressable AVIS sample into source planes.
///
/// This animation-oriented API accepts Key/IntraOnly, inter/switch, and
/// show-existing samples.
pub fn decode_sequence_frames_bytes(data: &[u8]) -> Result<Vec<DecodedFrame>, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, false)?;
    let mut frames = decode_sequence_frames_from_info(&info)?;
    if !info.alpha_auxiliary_items.is_empty() {
        let alpha_frame = decode_alpha_auxiliary_frame(&info)?;
        for frame in &mut frames {
            append_alpha_plane(frame, &alpha_frame)?;
        }
    }
    Ok(frames)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Av1Headers {
    config: Option<Av1CodecConfiguration>,
    sequence: SequenceHeader,
    frame: FrameHeader,
    tile_group: ParsedTileGroup,
    decode_plan: FrameDecodePlan,
    quant_state: QuantState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedTileGroup {
    from_frame_obu: bool,
    payload_len: usize,
    tile_data: Vec<u8>,
    group: TileGroup,
    entropy_states: Vec<TileEntropyState>,
    partition_probes: Vec<PartitionProbe>,
    block_mode_probes: Vec<BlockModeProbe>,
    residual_probes: Vec<ResidualProbe>,
}

fn decode_sequence_frame_from_info(
    info: &AvifInfo,
    frame_index: usize,
) -> Result<DecodedFrame, DecoderError> {
    if info.primary_grid.is_some() {
        if frame_index == 0 {
            return decode_grid_frame(info);
        }
        return Err(DecoderError::Unsupported(
            "AVIF grid sequences are not supported".to_string(),
        ));
    }
    let samples = if info.sequence_sample_payloads.is_empty() {
        std::slice::from_ref(&info.primary_item_payload)
    } else {
        info.sequence_sample_payloads.as_slice()
    };
    if frame_index >= samples.len() {
        return Err(DecoderError::InvalidParam(format!(
            "AVIS frame index {frame_index} is outside the {}-sample sequence",
            samples.len()
        )));
    }
    let stop_index = frame_index;
    let frames = decode_sequence_samples_from_info(info, stop_index, false)?;
    frames.into_iter().next().ok_or_else(|| {
        DecoderError::Bitstream("AVIS sequence sample could not be decoded".to_string())
    })
}

fn decode_sequence_frames_from_info(info: &AvifInfo) -> Result<Vec<DecodedFrame>, DecoderError> {
    if info.primary_grid.is_some() {
        if info.sequence_sample_payloads.len() > 1 {
            return Err(DecoderError::Unsupported(
                "AVIF grid sequences are not supported".to_string(),
            ));
        }
        return decode_grid_frame(info).map(|frame| vec![frame]);
    }
    let sample_count = if info.sequence_sample_payloads.is_empty() {
        1
    } else {
        info.sequence_sample_payloads.len()
    };
    decode_sequence_samples_from_info(info, sample_count - 1, true)
}

fn decode_sequence_samples_from_info(
    info: &AvifInfo,
    stop_index: usize,
    collect_all: bool,
) -> Result<Vec<DecodedFrame>, DecoderError> {
    let samples = if info.sequence_sample_payloads.is_empty() {
        std::slice::from_ref(&info.primary_item_payload)
    } else {
        info.sequence_sample_payloads.as_slice()
    };
    if stop_index >= samples.len() {
        return Err(DecoderError::InvalidParam(format!(
            "AVIS frame index {stop_index} is outside the {}-sample sequence",
            samples.len()
        )));
    }
    let primary_obus = parse_obu_stream(&info.primary_item_payload)?;
    let sequence_obu = primary_obus
        .iter()
        .find(|obu| obu.obu_type == ObuType::SequenceHeader)
        .ok_or_else(|| DecoderError::Bitstream("AV1 sequence header OBU is missing".to_string()))?;
    let sequence_prefix = encode_obu(sequence_obu.obu_type, sequence_obu.payload)?;
    // Classify the requested range before decoding any frame. Besides making
    // the unsupported boundary deterministic, this avoids doing full Key/
    // IntraOnly reconstruction only to discover a later Inter/Switch sample.
    let sample_infos: Vec<_> = samples
        .iter()
        .enumerate()
        .take(stop_index + 1)
        .map(|(index, sample)| {
            let info = inspect_av1_sequence_sample(sample)?;
            if info.kind.is_none() {
                Err(DecoderError::Bitstream(format!(
                    "AVIS sample {index} has no coded frame OBU"
                )))
            } else {
                Ok(info)
            }
        })
        .collect::<Result<_, _>>()?;
    let kinds: Vec<_> = sample_infos
        .iter()
        .map(|sample| sample.kind.expect("sample kind checked above"))
        .collect();
    let independent_samples = kinds.iter().all(|kind| {
        matches!(
            kind,
            AvifSequenceSampleKind::Key | AvifSequenceSampleKind::IntraOnly
        )
    });
    if independent_samples {
        if collect_all {
            return decode_independent_sequence_samples(
                info,
                &sequence_prefix,
                &samples[..=stop_index],
                &sample_infos,
            );
        }
        let decoded = decode_independent_sequence_sample(
            info,
            &sequence_prefix,
            &samples[stop_index],
            sample_infos[stop_index].has_sequence_header,
        )?;
        return Ok(vec![decoded]);
    }
    let mut references = FrameReferenceSlots::default();
    let mut cdf_states: Option<Vec<CdfContext>> = None;
    let frame_capacity = if collect_all {
        stop_index.saturating_add(1).min(samples.len())
    } else {
        1
    };
    let mut frames = Vec::with_capacity(frame_capacity);
    for (((index, sample), sample_info), kind) in samples
        .iter()
        .enumerate()
        .take(stop_index + 1)
        .zip(sample_infos)
        .zip(kinds)
    {
        match kind {
            AvifSequenceSampleKind::Key | AvifSequenceSampleKind::IntraOnly => {
                // A per-sample Sequence Header is authoritative when an AVIS
                // track changes its sample description.  For the common case
                // where it is omitted, inspect the shared prefix and sample
                // as separate OBU streams to avoid rebuilding a concatenated
                // temporary payload.
                let av1_config = (!sample_info.has_sequence_header)
                    .then_some(info.av1_config.as_deref())
                    .flatten();
                let headers = parse_av1_sequence_sample_headers(
                    info,
                    &sequence_prefix,
                    sample,
                    sample_info.has_sequence_header,
                    av1_config,
                )?;
                let (decoded_state, next_cdf_states) =
                    decode_still_frame_with_filter_policy_and_state_and_references_and_cdf(
                        &headers,
                        Some(info),
                        true,
                        std::array::from_fn(|_| None),
                        (!sample_info.has_sequence_header && headers.frame.primary_ref_frame != 7)
                            .then_some(cdf_states.as_deref())
                            .flatten(),
                        true,
                        true,
                    )?;
                let decoded = finish_decoded_still_frame(&headers, decoded_state, true)?;
                references.refresh_with_cdf(
                    headers.frame.refresh_frame_flags,
                    &decoded,
                    &headers.frame,
                    &next_cdf_states,
                );
                if !headers.frame.disable_frame_end_update_cdf {
                    cdf_states = Some(next_cdf_states);
                }
                if collect_all {
                    frames.push(decoded);
                } else if index == stop_index {
                    return Ok(vec![decoded]);
                }
            }
            AvifSequenceSampleKind::ShowExisting {
                frame_to_show_map_idx,
            } => {
                // `inspect_av1_sequence_sample` already consumed the
                // show-existing prefix while classifying this sample. Reuse
                // the decoded index instead of reparsing the OBU stream.
                let decoded = references.frame_to_show(frame_to_show_map_idx)?;
                if collect_all {
                    frames.push(decoded);
                } else if index == stop_index {
                    return Ok(vec![decoded]);
                }
            }
            AvifSequenceSampleKind::Inter | AvifSequenceSampleKind::Switch => {
                let av1_config = (!sample_info.has_sequence_header)
                    .then_some(info.av1_config.as_deref())
                    .flatten();
                let reference_states = references.states();
                let headers = parse_av1_sequence_sample_headers_with_references(
                    info,
                    &sequence_prefix,
                    sample,
                    sample_info.has_sequence_header,
                    av1_config,
                    &reference_states,
                )?;
                let initial_cdfs =
                    if !sample_info.has_sequence_header && headers.frame.primary_ref_frame != 7 {
                        references
                            .cdf_for_header(&headers.frame)
                            .or(cdf_states.as_deref())
                    } else {
                        None
                    };
                let (decoded_state, next_cdf_states) =
                    match decode_still_frame_with_filter_policy_and_state_and_references_and_cdf(
                        &headers,
                        Some(info),
                        true,
                        references.buffers(),
                        initial_cdfs,
                        false,
                        true,
                    ) {
                        Ok(decoded) => decoded,
                        Err(DecoderError::Bitstream(_) | DecoderError::Unsupported(_)) => {
                            return Err(DecoderError::Unsupported(format!(
                                "AVIS sample {index} uses {kind:?} frame prediction"
                            )));
                        }
                        Err(err) => return Err(err),
                    };
                let decoded = finish_decoded_still_frame(&headers, decoded_state, true)?;
                references.refresh_with_cdf(
                    headers.frame.refresh_frame_flags,
                    &decoded,
                    &headers.frame,
                    &next_cdf_states,
                );
                if !headers.frame.disable_frame_end_update_cdf {
                    cdf_states = Some(next_cdf_states);
                }
                if collect_all {
                    frames.push(decoded);
                } else if index == stop_index {
                    return Ok(vec![decoded]);
                }
            }
        }
    }
    Ok(frames)
}

fn decode_independent_sequence_samples(
    info: &AvifInfo,
    sequence_prefix: &[u8],
    samples: &[Vec<u8>],
    sample_infos: &[AvifSequenceSampleInfo],
) -> Result<Vec<DecodedFrame>, DecoderError> {
    #[cfg(not(target_family = "wasm"))]
    if samples.len() > 1 && avis_parallel_work_is_large_enough(info, samples.len()) {
        const MAX_WORKERS: usize = 8;
        let worker_count = samples.len().min(MAX_WORKERS);
        let chunk_size = samples.len().div_ceil(worker_count);
        return std::thread::scope(|scope| {
            let handles = samples
                .chunks(chunk_size)
                .zip(sample_infos.chunks(chunk_size))
                .map(|(chunk, chunk_infos)| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .zip(chunk_infos)
                            .map(|(sample, sample_info)| {
                                decode_independent_sequence_sample(
                                    info,
                                    sequence_prefix,
                                    sample,
                                    sample_info.has_sequence_header,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                })
                .collect::<Vec<_>>();
            let mut frames = Vec::with_capacity(samples.len());
            for handle in handles {
                let chunk = handle.join().map_err(|_| {
                    DecoderError::Bitstream("AVIS sequence decoder thread panicked".to_string())
                })??;
                frames.extend(chunk);
            }
            Ok(frames)
        });
    }

    samples
        .iter()
        .zip(sample_infos)
        .map(|(sample, sample_info)| {
            decode_independent_sequence_sample(
                info,
                sequence_prefix,
                sample,
                sample_info.has_sequence_header,
            )
        })
        .collect()
}

#[cfg(not(target_family = "wasm"))]
fn avis_parallel_work_is_large_enough(info: &AvifInfo, sample_count: usize) -> bool {
    const PARALLEL_AVIS_MIN_PIXELS: usize = 256 * 1024;
    let Some((width, height)) = info.width.zip(info.height) else {
        return true;
    };
    let frame_pixels = usize::try_from(width).ok().and_then(|width| {
        usize::try_from(height)
            .ok()
            .map(|height| width.saturating_mul(height))
    });
    frame_pixels
        .map(|pixels| pixels.saturating_mul(sample_count) >= PARALLEL_AVIS_MIN_PIXELS)
        .unwrap_or(true)
}

fn decode_independent_sequence_sample(
    info: &AvifInfo,
    sequence_prefix: &[u8],
    sample: &[u8],
    sample_has_sequence_header: bool,
) -> Result<DecodedFrame, DecoderError> {
    let av1_config = (!sample_has_sequence_header)
        .then_some(info.av1_config.as_deref())
        .flatten();
    let headers = parse_av1_sequence_sample_headers(
        info,
        sequence_prefix,
        sample,
        sample_has_sequence_header,
        av1_config,
    )?;
    decode_still_frame(&headers, Some(info))
}

fn parse_av1_sequence_sample_headers(
    info: &AvifInfo,
    sequence_prefix: &[u8],
    sample: &[u8],
    sample_has_sequence_header: bool,
    av1_config: Option<&[u8]>,
) -> Result<Av1Headers, DecoderError> {
    parse_av1_sequence_sample_headers_with_references(
        info,
        sequence_prefix,
        sample,
        sample_has_sequence_header,
        av1_config,
        &[None; 8],
    )
}

fn parse_av1_sequence_sample_headers_with_references(
    info: &AvifInfo,
    sequence_prefix: &[u8],
    sample: &[u8],
    sample_has_sequence_header: bool,
    av1_config: Option<&[u8]>,
    references: &[Option<ReferenceFrameState>; 8],
) -> Result<Av1Headers, DecoderError> {
    if sample_has_sequence_header {
        parse_av1_headers_from_parts_with_references(info, &[sample], av1_config, references)
    } else {
        parse_av1_headers_from_parts_with_references(
            info,
            &[sequence_prefix, sample],
            av1_config,
            references,
        )
    }
}

/// The eight AV1 reference slots used by a sequence.  Each slot keeps the
/// decoded planes together with the frame-header geometry needed by
/// `frame_size_with_refs` and motion-compensated prediction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct FrameReferenceSlots {
    slots: [Option<ReferenceFrame>; 8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReferenceFrameMetadata {
    width: usize,
    height: usize,
    render_width: usize,
    render_height: usize,
    bit_depth: u8,
    color_config: ColorConfig,
    color_information: Option<ColorInformation>,
    alpha_premultiplied: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReferenceFrame {
    metadata: Arc<ReferenceFrameMetadata>,
    buffers: Arc<FrameBuffers>,
    frame_width: u32,
    frame_height: u32,
    upscaled_width: u32,
    render_width: u32,
    render_height: u32,
    order_hint: u32,
    frame_type: FrameType,
    film_grain: Option<FilmGrainParams>,
    frame_id: Option<u16>,
    global_motion: GlobalMotionParams,
    cdf_states: Arc<Vec<CdfContext>>,
}

impl FrameReferenceSlots {
    fn refresh(&mut self, refresh_frame_flags: u8, frame: &DecodedFrame, header: &FrameHeader) {
        self.refresh_with_cdf(refresh_frame_flags, frame, header, &[]);
    }

    fn refresh_with_cdf(
        &mut self,
        refresh_frame_flags: u8,
        frame: &DecodedFrame,
        header: &FrameHeader,
        cdf_states: &[CdfContext],
    ) {
        if refresh_frame_flags == 0 {
            return;
        }
        let cdf_states = Arc::new(cdf_states.to_vec());
        // Keep the decoded planes shared between reference slots. Cloning a
        // full `DecodedFrame` here used to duplicate every plane once per
        // refresh, which is especially expensive for AVIS inter sequences.
        let metadata = Arc::new(ReferenceFrameMetadata {
            width: frame.width,
            height: frame.height,
            render_width: frame.render_width,
            render_height: frame.render_height,
            bit_depth: frame.bit_depth,
            color_config: frame.color_config,
            color_information: frame.color_information.clone(),
            alpha_premultiplied: frame.alpha_premultiplied,
        });
        let buffers = Arc::new(frame.buffers.clone());
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if refresh_frame_flags & (1 << index) != 0 {
                *slot = Some(ReferenceFrame {
                    metadata: Arc::clone(&metadata),
                    buffers: Arc::clone(&buffers),
                    frame_width: header.frame_width,
                    frame_height: header.frame_height,
                    upscaled_width: header.upscaled_width,
                    render_width: header.render_width,
                    render_height: header.render_height,
                    order_hint: header.order_hint,
                    frame_type: header.frame_type,
                    film_grain: header.film_grain,
                    frame_id: header.frame_id,
                    global_motion: header.global_motion,
                    cdf_states: Arc::clone(&cdf_states),
                });
            }
        }
    }

    fn get(&self, index: u8) -> Result<&ReferenceFrame, DecoderError> {
        self.slots
            .get(usize::from(index))
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 reference frame slot {index} has no decoded reference"
                ))
            })
    }

    fn states(&self) -> [Option<ReferenceFrameState>; 8] {
        std::array::from_fn(|index| {
            self.slots[index]
                .as_ref()
                .and_then(|_| self.get(index as u8).ok())
                .map(|reference| ReferenceFrameState {
                    frame_width: reference.frame_width,
                    frame_height: reference.frame_height,
                    upscaled_width: reference.upscaled_width,
                    render_width: reference.render_width,
                    render_height: reference.render_height,
                    order_hint: reference.order_hint,
                    film_grain: reference.film_grain,
                    frame_id: reference.frame_id,
                    global_motion: reference.global_motion,
                })
        })
    }

    fn buffers(&self) -> [Option<Arc<FrameBuffers>>; 8] {
        std::array::from_fn(|index| {
            self.slots[index]
                .as_ref()
                .map(|reference| Arc::clone(&reference.buffers))
        })
    }

    fn cdf_for_header(&self, header: &FrameHeader) -> Option<&[CdfContext]> {
        if header.primary_ref_frame == 7 {
            return None;
        }
        let reference_type = usize::from(header.primary_ref_frame);
        let slot = *header.reference_frame_indices.get(reference_type)?;
        let cdf_states = &self.slots.get(usize::from(slot))?.as_ref()?.cdf_states;
        (!cdf_states.is_empty()).then_some(cdf_states.as_slice())
    }

    fn frame_to_show(&self, index: u8) -> Result<DecodedFrame, DecoderError> {
        self.slots
            .get(usize::from(index))
            .and_then(Option::as_ref)
            .map(|reference| DecodedFrame {
                width: reference.metadata.width,
                height: reference.metadata.height,
                render_width: reference.metadata.render_width,
                render_height: reference.metadata.render_height,
                bit_depth: reference.metadata.bit_depth,
                color_config: reference.metadata.color_config,
                color_information: reference.metadata.color_information.clone(),
                alpha_premultiplied: reference.metadata.alpha_premultiplied,
                buffers: reference.buffers.as_ref().clone(),
            })
            .ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 show_existing_frame slot {index} has no decoded reference"
                ))
            })
    }
}

#[cfg(test)]
mod reference_frame_tests {
    use super::*;
    use crate::container::GridImage;

    fn frame(value: u16) -> DecodedFrame {
        DecodedFrame {
            width: 1,
            height: 1,
            render_width: 1,
            render_height: 1,
            bit_depth: 8,
            color_config: ColorConfig {
                high_bitdepth: false,
                twelve_bit: false,
                bit_depth: 8,
                monochrome: true,
                color_description: None,
                color_range: ColorRange::Full,
                subsampling_x: false,
                subsampling_y: false,
                chroma_sample_position: None,
                separate_uv_delta_q: false,
            },
            color_information: None,
            alpha_premultiplied: false,
            buffers: FrameBuffers {
                width: 1,
                height: 1,
                planes: vec![PlaneBuffer {
                    layout: PlaneLayout {
                        plane: 0,
                        width: 1,
                        height: 1,
                        subsampling_x: 0,
                        subsampling_y: 0,
                        sample_count: 1,
                    },
                    samples: vec![value],
                }],
            },
        }
    }

    #[test]
    fn refresh_flags_store_and_replace_reference_slots() {
        let mut slots = FrameReferenceSlots::default();
        let header = reference_header();
        slots.refresh(0, &frame(5), &header);
        assert!(slots.slots.iter().all(Option::is_none));
        slots.refresh(0b0000_0101, &frame(10), &header);
        assert_eq!(
            slots.frame_to_show(0).unwrap().buffers.planes[0].samples,
            [10]
        );
        assert_eq!(
            slots.frame_to_show(2).unwrap().buffers.planes[0].samples,
            [10]
        );
        assert!(slots.frame_to_show(1).is_err());

        slots.refresh(0b0000_0100, &frame(20), &header);
        assert_eq!(
            slots.frame_to_show(0).unwrap().buffers.planes[0].samples,
            [10]
        );
        assert_eq!(
            slots.frame_to_show(2).unwrap().buffers.planes[0].samples,
            [20]
        );
    }

    #[test]
    fn refreshed_reference_keeps_frame_geometry_for_future_motion_compensation() {
        let mut slots = FrameReferenceSlots::default();
        let mut header = reference_header();
        header.frame_width = 32;
        header.frame_height = 24;
        header.upscaled_width = 40;
        header.render_width = 30;
        header.render_height = 20;
        header.order_hint = 7;
        slots.refresh(1, &frame(3), &header);
        let reference = slots.get(0).unwrap();
        assert_eq!(reference.frame_width, 32);
        assert_eq!(reference.frame_height, 24);
        assert_eq!(reference.upscaled_width, 40);
        assert_eq!((reference.render_width, reference.render_height), (30, 20));
        assert_eq!(reference.order_hint, 7);
    }

    #[test]
    fn refreshed_reference_keeps_primary_ref_cdf_state() {
        let mut slots = FrameReferenceSlots::default();
        let mut header = reference_header();
        header.primary_ref_frame = 0;
        header.reference_frame_indices[0] = 3;
        let cdf = CdfContext::new(header.base_q_idx);

        slots.refresh_with_cdf(1 << 3, &frame(3), &header, std::slice::from_ref(&cdf));

        let restored = slots.cdf_for_header(&header).unwrap();
        assert_eq!(restored, [cdf].as_slice());
    }

    fn reference_header() -> FrameHeader {
        let data = include_bytes!("../test_data/images/WML2Viewer.avif");
        let info = parse_avif(data).unwrap();
        parse_av1_headers(&info).unwrap().frame
    }

    #[test]
    fn missing_show_existing_slot_is_rejected_without_partial_frame() {
        let slots = FrameReferenceSlots::default();
        let error = slots.frame_to_show(7).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::Unsupported(message)
                if message.contains("show_existing_frame slot 7")
        ));
    }

    #[test]
    fn rebuilt_reference_probe_obus_roundtrip_through_stream_parser() {
        let encoded = encode_obu(ObuType::SequenceHeader, &[0x01, 0x02, 0x03]).unwrap();
        let obus = parse_obu_stream(&encoded).unwrap();
        assert_eq!(obus.len(), 1);
        assert_eq!(obus[0].obu_type, ObuType::SequenceHeader);
        assert_eq!(obus[0].payload, [0x01, 0x02, 0x03]);
    }

    #[test]
    fn indexed_show_existing_sample_reuses_decoded_key_frame() {
        let data = include_bytes!("../test_data/images/WML2Viewer.avif");
        let mut info = parse_avif(data).unwrap();
        info.major_brand = *b"avis";
        info.sequence_sample_payloads = vec![
            info.primary_item_payload.clone(),
            encode_obu(ObuType::FrameHeader, &[0x80]).unwrap(),
        ];
        let expected = decode_frame_bytes(data).unwrap();
        let shown = decode_sequence_frame_from_info(&info, 1).unwrap();
        assert_eq!(shown, expected);
    }

    #[test]
    fn batch_show_existing_sample_reuses_decoded_key_frame() {
        let data = include_bytes!("../test_data/images/WML2Viewer.avif");
        let mut info = parse_avif(data).unwrap();
        info.major_brand = *b"avis";
        info.sequence_sample_payloads = vec![
            info.primary_item_payload.clone(),
            encode_obu(ObuType::FrameHeader, &[0x80]).unwrap(),
        ];
        let expected = decode_frame_bytes(data).unwrap();
        let frames = decode_sequence_frames_from_info(&info).unwrap();
        assert_eq!(frames, vec![expected.clone(), expected]);
    }

    #[test]
    fn small_avis_sequences_avoid_thread_pool_overhead() {
        let data = include_bytes!("../test_data/images/WML2Viewer.avif");
        let mut info = parse_avif(data).unwrap();
        info.width = Some(64);
        info.height = Some(64);
        assert!(!avis_parallel_work_is_large_enough(&info, 4));
        assert!(avis_parallel_work_is_large_enough(&info, 64));
        info.width = None;
        assert!(avis_parallel_work_is_large_enough(&info, 1));
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn post_filter_parallelism_skips_small_frames_but_keeps_large_frames() {
        let mut small = super::grid_composition_tests::native_frame(
            64,
            64,
            super::grid_composition_tests::native_plane(64, 64, vec![0; 64 * 64]),
        );
        small
            .buffers
            .planes
            .push(super::grid_composition_tests::native_plane(
                32,
                32,
                vec![0; 32 * 32],
            ));
        assert!(!post_filter_parallel_work_is_large_enough(&small));

        let mut large = super::grid_composition_tests::native_frame(
            512,
            512,
            super::grid_composition_tests::native_plane(512, 512, vec![0; 512 * 512]),
        );
        large
            .buffers
            .planes
            .push(super::grid_composition_tests::native_plane(
                256,
                256,
                vec![0; 256 * 256],
            ));
        assert!(post_filter_parallel_work_is_large_enough(&large));
    }

    #[test]
    fn indexed_sample_with_own_sequence_header_uses_that_header() {
        let data = include_bytes!("../test_data/images/WML2Viewer.avif");
        let mut info = parse_avif(data).unwrap();
        info.major_brand = *b"avis";
        info.av1_config = Some(vec![0xff, 0xee, 0xdd, 0xcc]);
        info.sequence_sample_payloads = vec![
            info.primary_item_payload.clone(),
            info.primary_item_payload.clone(),
        ];
        let frame = decode_sequence_frame_from_info(&info, 1).unwrap();
        assert_eq!((frame.width, frame.height), (900, 900));
    }

    #[test]
    fn evaluates_64_bit_sato_expression_with_saturation() {
        let tokens = [
            SampleTransformToken::Constant(i64::MAX),
            SampleTransformToken::Constant(2),
            SampleTransformToken::Binary(0),
        ];
        let mut stack = Vec::new();
        let value =
            evaluate_sample_transform_expression(&tokens, 64, 65_535, |_| Ok(0), &mut stack)
                .unwrap();
        assert_eq!(value, 65_535);

        let tokens = [
            SampleTransformToken::Constant(2),
            SampleTransformToken::Constant(63),
            SampleTransformToken::Binary(7),
        ];
        let value =
            evaluate_sample_transform_expression(&tokens, 64, 65_535, |_| Ok(0), &mut stack)
                .unwrap();
        assert_eq!(value, 65_535);
    }

    #[test]
    fn sato_grid_input_is_routed_through_grid_decoder() {
        let input = SampleTransformInput {
            item_id: 7,
            width: 1,
            height: 1,
            pixel_information: PixelInformation {
                bits_per_channel: vec![8],
                extended_channels: None,
            },
            color_information: None,
            av1_config: Vec::new(),
            payload: Vec::new(),
            grid: Some(GridImage {
                item_id: 7,
                rows: 1,
                columns: 1,
                output_width: 1,
                output_height: 1,
                payload: Vec::new(),
                cells: Vec::new(),
            }),
        };
        let error = decode_sample_transform_input(&input).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::Bitstream(message) if message.contains("grid has 0 cells")
        ));
    }
}

/// Decodes the narrow sequence shape that can be handled without inter-frame
/// motion compensation: a hidden coded `OBU_FRAME` followed by a
/// `show_existing_frame` header referring to one of its refreshed slots.
/// Returning `None` leaves the normal first-frame path unchanged.
fn decode_hidden_key_frame_show_existing(
    info: &AvifInfo,
) -> Result<Option<DecodedFrame>, DecoderError> {
    if info.major_brand != *b"avis" && !info.compatible_brands.iter().any(|brand| brand == b"avis")
    {
        return Ok(None);
    }
    let obus = parse_obu_stream(&info.primary_item_payload)?;
    let sequence = obus
        .iter()
        .find(|obu| obu.obu_type == ObuType::SequenceHeader)
        .map(|obu| obu.payload)
        .ok_or_else(|| DecoderError::Bitstream("AV1 sequence header OBU is missing".to_string()))?;

    let first_frame = obus.iter().find(|obu| obu.obu_type == ObuType::Frame);
    let Some(first_frame) = first_frame else {
        return Ok(None);
    };
    let sequence_header = parse_sequence_header(sequence)?;
    let hidden_header = parse_frame_header(first_frame.payload, &sequence_header)?;
    if hidden_header.show_frame {
        return Ok(None);
    }

    let first_frame_position = obus
        .iter()
        .position(|obu| std::ptr::eq(obu, first_frame))
        .expect("first frame must belong to the parsed OBU list");
    let mut show_existing_index = obus
        .iter()
        .skip(first_frame_position + 1)
        .filter(|obu| matches!(obu.obu_type, ObuType::Frame | ObuType::FrameHeader))
        .find_map(|obu| parse_show_existing_frame_index(obu.payload).transpose())
        .transpose()?;
    if show_existing_index.is_none() {
        for sample_payload in info.sequence_sample_payloads.iter().skip(1) {
            let sample_obus = parse_obu_stream(sample_payload)?;
            show_existing_index = sample_obus
                .iter()
                .filter(|obu| matches!(obu.obu_type, ObuType::Frame | ObuType::FrameHeader))
                .find_map(|obu| parse_show_existing_frame_index(obu.payload).transpose())
                .transpose()?;
            if show_existing_index.is_some() {
                break;
            }
        }
    }
    let Some(show_existing_index) = show_existing_index else {
        return Ok(None);
    };

    let sequence_obu = obus
        .iter()
        .find(|obu| obu.obu_type == ObuType::SequenceHeader)
        .expect("sequence payload was found above");
    let mut hidden_payload = encode_obu(sequence_obu.obu_type, sequence_obu.payload)?;
    hidden_payload.extend(encode_obu(first_frame.obu_type, first_frame.payload)?);
    let mut hidden_info = info.clone();
    hidden_info.primary_item_payload = hidden_payload;
    let hidden_headers = parse_av1_headers(&hidden_info)?;
    let hidden_frame = decode_still_frame(&hidden_headers, Some(&hidden_info))?;
    let mut references = FrameReferenceSlots::default();
    references.refresh(
        hidden_headers.frame.refresh_frame_flags,
        &hidden_frame,
        &hidden_headers.frame,
    );
    references.frame_to_show(show_existing_index).map(Some)
}

fn encode_obu(obu_type: ObuType, payload: &[u8]) -> Result<Vec<u8>, DecoderError> {
    let type_bits: u8 = match obu_type {
        ObuType::SequenceHeader => 1,
        ObuType::Frame => 6,
        ObuType::FrameHeader => 3,
        ObuType::TileGroup => 4,
        _ => {
            return Err(DecoderError::Unsupported(format!(
                "cannot rebuild AV1 {:?} OBU for reference probing",
                obu_type
            )));
        }
    };
    let mut encoded = vec![(type_bits << 3) | 0x02];
    let mut length = payload.len();
    loop {
        let mut byte = (length & 0x7f) as u8;
        length >>= 7;
        if length != 0 {
            byte |= 0x80;
        }
        encoded.push(byte);
        if length == 0 {
            break;
        }
    }
    encoded.extend_from_slice(payload);
    Ok(encoded)
}

fn parse_av1_headers(info: &AvifInfo) -> Result<Av1Headers, DecoderError> {
    parse_av1_headers_from_parts_with_references(
        info,
        &[&info.primary_item_payload],
        info.av1_config.as_deref(),
        &[None; 8],
    )
}

fn parse_av1_headers_from_parts_with_references(
    info: &AvifInfo,
    payloads: &[&[u8]],
    av1_config: Option<&[u8]>,
    references: &[Option<ReferenceFrameState>; 8],
) -> Result<Av1Headers, DecoderError> {
    let [
        sequence_payload,
        frame_payload,
        frame_header_payload,
        _tile_group_payload,
    ] = find_obu_payloads_in_parts(
        payloads,
        [
            ObuType::SequenceHeader,
            ObuType::Frame,
            ObuType::FrameHeader,
            ObuType::TileGroup,
        ],
    )?;
    let sequence_payload = sequence_payload
        .ok_or_else(|| DecoderError::Bitstream("AV1 sequence header OBU is missing".to_string()))?;
    let sequence = parse_sequence_header(sequence_payload)?;
    validate_extended_pixi(info.pixel_information.as_ref(), &sequence.color_config)?;
    validate_color_metadata(info.color_information.as_ref(), &sequence.color_config)?;
    let config = av1_config.map(parse_av1_config).transpose()?;
    if let Some(config) = config {
        validate_av1_config(&config, &sequence)?;
    }
    let (frame, tile_group_payload) = if let Some(frame_payload) = frame_payload {
        let frame = parse_frame_header_with_references(frame_payload, &sequence, references)?;
        if frame.payload_after_header_offset > frame_payload.len() {
            return Err(DecoderError::Bitstream(
                "AV1 frame payload offset points outside OBU_FRAME".to_string(),
            ));
        }
        let tile_group = parse_tile_group(
            frame_payload,
            frame.payload_after_header_offset * 8,
            &frame.tile_info,
        )?;
        let entropy_states = prepare_tile_entropy(frame_payload, &tile_group, &frame)?;
        let tile_group_payload_len = frame_payload
            .len()
            .checked_sub(tile_group.data_start_offset)
            .ok_or_else(|| {
                DecoderError::Bitstream(
                    "AV1 tile group data offset points outside OBU_FRAME".to_string(),
                )
            })?;
        (
            frame,
            ParsedTileGroup {
                from_frame_obu: true,
                payload_len: tile_group_payload_len,
                tile_data: frame_payload.to_vec(),
                group: tile_group,
                entropy_states,
                partition_probes: Vec::new(),
                block_mode_probes: Vec::new(),
                residual_probes: Vec::new(),
            },
        )
    } else {
        let frame_header_payload = frame_header_payload.ok_or_else(|| {
            DecoderError::Bitstream("AV1 frame header OBU is missing".to_string())
        })?;
        let frame =
            parse_frame_header_with_references(frame_header_payload, &sequence, references)?;
        // A primary item may carry an AV1 sequence. The still-image API
        // exposes the first frame, so stop collecting tiles at the next
        // frame boundary instead of mixing subsequent frames into it.
        let mut tile_group_payloads = Vec::new();
        let mut first_frame_started = false;
        'parts: for payload in payloads {
            for obu in parse_obu_stream(payload)? {
                match obu.obu_type {
                    ObuType::FrameHeader => {
                        if first_frame_started {
                            break 'parts;
                        }
                        first_frame_started = true;
                    }
                    ObuType::TemporalDelimiter if first_frame_started => break 'parts,
                    ObuType::TileGroup if first_frame_started => {
                        tile_group_payloads.push(obu.payload)
                    }
                    _ => {}
                }
            }
        }
        if tile_group_payloads.is_empty() {
            return Err(DecoderError::Bitstream(
                "AV1 tile group OBU is missing".to_string(),
            ));
        }
        let (tile_group_data, tile_group) =
            merge_tile_group_payloads(&tile_group_payloads, &frame.tile_info)?;
        let entropy_states = prepare_tile_entropy(&tile_group_data, &tile_group, &frame)?;
        (
            frame,
            ParsedTileGroup {
                from_frame_obu: false,
                payload_len: tile_group_data.len(),
                tile_data: tile_group_data,
                group: tile_group,
                entropy_states,
                partition_probes: Vec::new(),
                block_mode_probes: Vec::new(),
                residual_probes: Vec::new(),
            },
        )
    };
    if let Some(width) = info.width
        && width != frame.upscaled_width
    {
        return Err(DecoderError::Bitstream(format!(
            "AVIF ispe width {width} does not match AV1 upscaled width {}",
            frame.upscaled_width
        )));
    }
    if let Some(height) = info.height
        && height != frame.frame_height
    {
        return Err(DecoderError::Bitstream(format!(
            "AVIF ispe height {height} does not match AV1 frame height {}",
            frame.frame_height
        )));
    }
    let decode_plan = build_still_decode_plan(&sequence, &frame, &tile_group_payload.group)?;
    let quant_state =
        QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
    Ok(Av1Headers {
        config,
        sequence,
        frame,
        tile_group: tile_group_payload,
        decode_plan,
        quant_state,
    })
}

fn merge_tile_group_payloads(
    payloads: &[&[u8]],
    tile_info: &crate::av1::TileInfo,
) -> Result<(Vec<u8>, TileGroup), DecoderError> {
    let tile_count = tile_info
        .tile_cols
        .checked_mul(tile_info.tile_rows)
        .ok_or_else(|| DecoderError::InvalidParam("AV1 tile count overflows".to_string()))?;
    if tile_count == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 tile count is zero".to_string(),
        ));
    }
    let mut tile_bytes = vec![
        None;
        usize::try_from(tile_count).map_err(|_| {
            DecoderError::InvalidParam("AV1 tile count is too large".to_string())
        })?
    ];
    for payload in payloads {
        let group = parse_tile_group(payload, 0, tile_info)?;
        for tile in group.tiles {
            let index = usize::try_from(tile.tile_id)
                .map_err(|_| DecoderError::InvalidParam("AV1 tile ID is too large".to_string()))?;
            let bytes = payload
                .get(tile.offset..tile.offset + tile.len)
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 tile payload range is invalid".to_string())
                })?;
            let slot = tile_bytes.get_mut(index).ok_or_else(|| {
                DecoderError::Bitstream("AV1 tile ID is outside the frame".to_string())
            })?;
            if slot.is_some() {
                return Err(DecoderError::Bitstream(format!(
                    "AV1 tile {index} is present in multiple tile groups"
                )));
            }
            *slot = Some(bytes.to_vec());
        }
    }
    if tile_bytes.iter().any(Option::is_none) {
        return Err(DecoderError::Unsupported(
            "AV1 partial tile groups are not supported for still-image decode".to_string(),
        ));
    }
    let mut data = Vec::new();
    let mut tiles = Vec::with_capacity(tile_bytes.len());
    for (tile_id, bytes) in tile_bytes.into_iter().enumerate() {
        let bytes = bytes.expect("tile presence checked above");
        let offset = data.len();
        data.extend_from_slice(&bytes);
        tiles.push(crate::av1::TilePayload {
            tile_id: tile_id as u32,
            offset,
            len: bytes.len(),
        });
    }
    Ok((
        data,
        TileGroup {
            start_tile: 0,
            end_tile: tile_count - 1,
            data_start_offset: 0,
            tiles,
        },
    ))
}

fn populate_diagnostic_probes(headers: &mut Av1Headers) {
    let tile_data = &headers.tile_group.tile_data;
    let tile_group = &headers.tile_group.group;
    if let Ok(probes) = probe_tile_partitions(
        tile_data,
        tile_group,
        &headers.sequence,
        &headers.frame,
        &headers.decode_plan,
    ) {
        headers.tile_group.partition_probes = probes;
    }
    if let Ok(probes) = probe_tile_block_modes(
        tile_data,
        tile_group,
        &headers.sequence,
        &headers.frame,
        &headers.decode_plan,
    ) {
        headers.tile_group.block_mode_probes = probes;
    }
    if let Ok(probes) = probe_first_block_residuals(
        tile_data,
        tile_group,
        &headers.sequence,
        &headers.frame,
        &headers.decode_plan,
    ) {
        headers.tile_group.residual_probes = probes;
    }
}

fn decode_still_image(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
) -> Result<ImageBuffer, DecoderError> {
    let frame = decode_still_frame(headers, info)?;
    let mut image = frame.to_rgba8()?;
    if let Some(info) = info {
        if !info.alpha_auxiliary_items.is_empty() {
            apply_alpha_auxiliary(&mut image, info)?;
        }
        apply_clean_aperture(&mut image, info.clean_aperture)?;
        apply_mirror(&mut image, info.mirror)?;
        apply_rotation(&mut image, info.rotation)?;
    }
    Ok(image)
}

fn decode_grid_frame(info: &AvifInfo) -> Result<DecodedFrame, DecoderError> {
    let grid = info
        .primary_grid
        .as_ref()
        .ok_or_else(|| DecoderError::Bitstream("AVIF grid metadata is missing".to_string()))?;
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    let cell_count = rows
        .checked_mul(columns)
        .ok_or_else(|| DecoderError::Bitstream("grid cell count overflow".to_string()))?;
    if grid.cells.len() != cell_count {
        return Err(DecoderError::Bitstream(format!(
            "grid has {} cells, expected {cell_count}",
            grid.cells.len()
        )));
    }
    let width = usize::try_from(grid.output_width)
        .map_err(|_| DecoderError::InvalidParam("grid output width is too large".to_string()))?;
    let height = usize::try_from(grid.output_height)
        .map_err(|_| DecoderError::InvalidParam("grid output height is too large".to_string()))?;
    if width == 0 || height == 0 {
        return Err(DecoderError::Bitstream(
            "grid output dimensions must be non-zero".to_string(),
        ));
    }
    let mut column_widths = vec![0usize; columns];
    let mut row_heights = vec![0usize; rows];
    let mut decoded_cells = decode_grid_cells(info, &grid.cells)?;
    normalize_grid_cells(&mut decoded_cells)?;
    for (index, cell) in grid.cells.iter().enumerate() {
        let cell_width = usize::try_from(cell.width)
            .map_err(|_| DecoderError::InvalidParam("grid cell width is too large".to_string()))?;
        let cell_height = usize::try_from(cell.height)
            .map_err(|_| DecoderError::InvalidParam("grid cell height is too large".to_string()))?;
        if cell_width == 0 || cell_height == 0 {
            return Err(DecoderError::Bitstream(format!(
                "grid cell {} has zero dimensions",
                cell.item_id
            )));
        }
        let row = index / columns;
        let column = index % columns;
        if column_widths[column] != 0 && column_widths[column] != cell_width {
            return Err(DecoderError::Bitstream(
                "grid cells in one column have different widths".to_string(),
            ));
        }
        if row_heights[row] != 0 && row_heights[row] != cell_height {
            return Err(DecoderError::Bitstream(
                "grid cells in one row have different heights".to_string(),
            ));
        }
        column_widths[column] = cell_width;
        row_heights[row] = cell_height;
        let decoded = decoded_cells.get(index).ok_or_else(|| {
            DecoderError::Bitstream("grid cell decode result is missing".to_string())
        })?;
        if decoded.width != cell_width || decoded.height != cell_height {
            return Err(DecoderError::Bitstream(format!(
                "grid cell {} decoded as {}x{}, metadata declares {}x{}",
                cell.item_id, decoded.width, decoded.height, cell_width, cell_height
            )));
        }
    }
    if column_widths.iter().sum::<usize>() < width || row_heights.iter().sum::<usize>() < height {
        return Err(DecoderError::Bitstream(format!(
            "grid cell dimensions do not cover declared output {}x{} (columns {:?}, rows {:?})",
            width, height, column_widths, row_heights
        )));
    }
    let first = decoded_cells
        .first()
        .ok_or_else(|| DecoderError::Bitstream("grid has no cells".to_string()))?;
    let plane_count = first.buffers.planes.len();
    for (index, cell) in decoded_cells.iter().enumerate() {
        if cell.buffers.planes.len() != plane_count
            || cell.bit_depth != first.bit_depth
            || cell.color_config != first.color_config
        {
            return Err(DecoderError::Unsupported(format!(
                "grid cell {index} uses different AV1 plane/color configuration: planes={}, bit_depth={}, color_config={:?}; first planes={}, bit_depth={}, color_config={:?}",
                cell.buffers.planes.len(),
                cell.bit_depth,
                cell.color_config,
                plane_count,
                first.bit_depth,
                first.color_config
            )));
        }
    }
    let mut planes = Vec::with_capacity(plane_count);
    for source in &first.buffers.planes {
        let subsampling_x = source.layout.subsampling_x;
        let subsampling_y = source.layout.subsampling_y;
        let plane_width = width.div_ceil(1usize << subsampling_x);
        let plane_height = height.div_ceil(1usize << subsampling_y);
        let sample_count = plane_width.checked_mul(plane_height).ok_or_else(|| {
            DecoderError::InvalidParam("grid plane sample count overflows".to_string())
        })?;
        planes.push(PlaneBuffer {
            layout: PlaneLayout {
                plane: source.layout.plane,
                width: plane_width,
                height: plane_height,
                subsampling_x,
                subsampling_y,
                sample_count,
            },
            samples: vec![0; sample_count],
        });
    }
    let mut y_offset = 0usize;
    for row in 0..rows {
        let mut x_offset = 0usize;
        for column in 0..columns {
            let cell = &decoded_cells[row * columns + column];
            for (plane_index, source) in cell.buffers.planes.iter().enumerate() {
                let destination = planes.get_mut(plane_index).ok_or_else(|| {
                    DecoderError::Bitstream("grid cell plane count differs".to_string())
                })?;
                if source.layout.subsampling_x != destination.layout.subsampling_x
                    || source.layout.subsampling_y != destination.layout.subsampling_y
                {
                    return Err(DecoderError::Unsupported(
                        "grid cells use different chroma subsampling".to_string(),
                    ));
                }
                let scale_x = 1usize << source.layout.subsampling_x;
                let scale_y = 1usize << source.layout.subsampling_y;
                if x_offset % scale_x != 0 || y_offset % scale_y != 0 {
                    return Err(DecoderError::Unsupported(
                        "grid cell boundary is not aligned to chroma samples".to_string(),
                    ));
                }
                let destination_x = x_offset / scale_x;
                let destination_y = y_offset / scale_y;
                if destination_x >= destination.layout.width
                    || destination_y >= destination.layout.height
                {
                    continue;
                }
                let copy_width = source
                    .layout
                    .width
                    .min(destination.layout.width - destination_x);
                let copy_height = source
                    .layout
                    .height
                    .min(destination.layout.height - destination_y);
                for source_y in 0..copy_height {
                    let source_start = source_y * source.layout.width;
                    let destination_start =
                        (destination_y + source_y) * destination.layout.width + destination_x;
                    destination.samples[destination_start..destination_start + copy_width]
                        .copy_from_slice(&source.samples[source_start..source_start + copy_width]);
                }
            }
            x_offset += column_widths[column];
        }
        y_offset += row_heights[row];
    }
    let mut frame = DecodedFrame {
        width,
        height,
        render_width: width,
        render_height: height,
        bit_depth: first.bit_depth,
        color_config: first.color_config,
        color_information: info.color_information.clone(),
        alpha_premultiplied: info.alpha_premultiplied,
        buffers: FrameBuffers {
            width,
            height,
            planes,
        },
    };
    if let Some(alpha_grid) = info.alpha_grid.as_ref() {
        let (alpha_plane, alpha_bit_depth) = decode_alpha_grid_plane(alpha_grid)?;
        append_alpha_plane_buffer(&mut frame, alpha_plane, alpha_bit_depth)?;
    } else if !info.alpha_auxiliary_items.is_empty()
        && !info
            .primary_grid
            .as_ref()
            .is_some_and(|grid| grid_has_cell_alpha(&grid.payload))
    {
        let alpha_frame = decode_alpha_auxiliary_frame(info)?;
        append_alpha_plane(&mut frame, &alpha_frame)?;
    }
    apply_native_grid_geometry(&mut frame, info)?;
    Ok(frame)
}

fn decode_grid_cells(
    info: &AvifInfo,
    cells: &[GridCell],
) -> Result<Vec<DecodedFrame>, DecoderError> {
    decode_grid_items(cells, |cell| decode_grid_cell_frame(info, cell))
}

fn decode_grid_images(
    info: &AvifInfo,
    cells: &[GridCell],
) -> Result<Vec<ImageBuffer>, DecoderError> {
    decode_grid_items(cells, |cell| decode_grid_cell(info, cell))
}

#[cfg(not(target_family = "wasm"))]
const PARALLEL_GRID_MIN_PIXELS: usize = 256 * 1024;
#[cfg(not(target_family = "wasm"))]
const MAX_GRID_WORKERS: usize = 8;

fn decode_grid_items<T, F>(cells: &[GridCell], decode: F) -> Result<Vec<T>, DecoderError>
where
    T: Send,
    F: Fn(&GridCell) -> Result<T, DecoderError> + Sync,
{
    #[cfg(not(target_family = "wasm"))]
    {
        let total_pixels = cells.iter().fold(0usize, |total, cell| {
            total.saturating_add(
                usize::try_from(cell.width)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(usize::try_from(cell.height).unwrap_or(usize::MAX)),
            )
        });
        if cells.len() > 1 && total_pixels >= PARALLEL_GRID_MIN_PIXELS {
            let worker_count = cells.len().min(MAX_GRID_WORKERS);
            let chunk_size = cells.len().div_ceil(worker_count);
            return std::thread::scope(|scope| {
                let handles = cells
                    .chunks(chunk_size)
                    .map(|chunk| {
                        scope.spawn(|| {
                            chunk
                                .iter()
                                .map(|cell| decode(cell))
                                .collect::<Result<Vec<_>, _>>()
                        })
                    })
                    .collect::<Vec<_>>();
                let mut decoded = Vec::with_capacity(cells.len());
                for handle in handles {
                    let chunk = handle.join().map_err(|_| {
                        DecoderError::Bitstream(
                            "AVIF grid cell decoder thread panicked".to_string(),
                        )
                    })??;
                    decoded.extend(chunk);
                }
                Ok(decoded)
            });
        }
    }

    cells.iter().map(decode).collect()
}

fn apply_native_grid_geometry(
    frame: &mut DecodedFrame,
    info: &AvifInfo,
) -> Result<(), DecoderError> {
    if let Some(aperture) = info.clean_aperture {
        let (start_x, start_y, width, height) =
            clean_aperture_rect(frame.width, frame.height, aperture)?;
        for plane in &mut frame.buffers.planes {
            let scale_x = 1usize << plane.layout.subsampling_x;
            let scale_y = 1usize << plane.layout.subsampling_y;
            if start_x % scale_x != 0 || start_y % scale_y != 0 {
                return Err(DecoderError::Unsupported(
                    "AVIF native-plane clean aperture is not aligned to chroma samples".to_string(),
                ));
            }
            let plane_start_x = start_x / scale_x;
            let plane_start_y = start_y / scale_y;
            let plane_width = width.div_ceil(scale_x);
            let plane_height = height.div_ceil(scale_y);
            if plane_start_x + plane_width > plane.layout.width
                || plane_start_y + plane_height > plane.layout.height
            {
                return Err(DecoderError::Bitstream(
                    "AVIF native-plane clean aperture exceeds a source plane".to_string(),
                ));
            }
            let mut cropped = vec![0; plane_width * plane_height];
            for row in 0..plane_height {
                let source_start = (plane_start_y + row) * plane.layout.width + plane_start_x;
                let destination_start = row * plane_width;
                cropped[destination_start..destination_start + plane_width]
                    .copy_from_slice(&plane.samples[source_start..source_start + plane_width]);
            }
            plane.layout.width = plane_width;
            plane.layout.height = plane_height;
            plane.layout.sample_count = cropped.len();
            plane.samples = cropped;
        }
        frame.width = width;
        frame.height = height;
        frame.render_width = width;
        frame.render_height = height;
        frame.buffers.width = width;
        frame.buffers.height = height;
    }
    if let Some(mirror) = info.mirror {
        apply_native_mirror(&mut frame.buffers, mirror)?;
    }
    if let Some(rotation) = info.rotation {
        apply_native_rotation(frame, rotation)?;
    }
    Ok(())
}

fn clean_aperture_rect(
    image_width: usize,
    image_height: usize,
    aperture: CleanAperture,
) -> Result<(usize, usize, usize, usize), DecoderError> {
    if aperture.width_d == 0
        || aperture.height_d == 0
        || aperture.horizontal_offset_d == 0
        || aperture.vertical_offset_d == 0
    {
        return Err(DecoderError::Bitstream(
            "AVIF clean aperture has a zero denominator".to_string(),
        ));
    }
    let width = (u64::from(aperture.width_n) / u64::from(aperture.width_d)) as usize;
    let height = (u64::from(aperture.height_n) / u64::from(aperture.height_d)) as usize;
    let center_x = image_width as i64 / 2;
    let center_y = image_height as i64 / 2;
    let offset_x_n = i32::from_be_bytes(aperture.horizontal_offset_n.to_be_bytes());
    let offset_y_n = i32::from_be_bytes(aperture.vertical_offset_n.to_be_bytes());
    let offset_x = i64::from(offset_x_n) / i64::from(aperture.horizontal_offset_d);
    let offset_y = i64::from(offset_y_n) / i64::from(aperture.vertical_offset_d);
    let start_x = center_x + offset_x - width as i64 / 2;
    let start_y = center_y + offset_y - height as i64 / 2;
    if width == 0
        || height == 0
        || start_x < 0
        || start_y < 0
        || start_x as usize + width > image_width
        || start_y as usize + height > image_height
    {
        return Err(DecoderError::Bitstream(
            "AVIF clean aperture is outside the decoded image".to_string(),
        ));
    }
    Ok((start_x as usize, start_y as usize, width, height))
}

fn apply_native_mirror(
    buffers: &mut FrameBuffers,
    mirror: ImageMirror,
) -> Result<(), DecoderError> {
    if mirror.axis > 1 {
        return Err(DecoderError::Bitstream(format!(
            "AVIF mirror axis {} is invalid",
            mirror.axis
        )));
    }
    for plane in &mut buffers.planes {
        if mirror.axis == 0 {
            for row in plane.samples.chunks_exact_mut(plane.layout.width) {
                row.reverse();
            }
        } else {
            let width = plane.layout.width;
            let height = plane.layout.height;
            for row in 0..height / 2 {
                let opposite = height - 1 - row;
                for column in 0..width {
                    plane
                        .samples
                        .swap(row * width + column, opposite * width + column);
                }
            }
        }
    }
    Ok(())
}

fn apply_native_rotation(
    frame: &mut DecodedFrame,
    rotation: ImageRotation,
) -> Result<(), DecoderError> {
    if rotation.angle > 3 {
        return Err(DecoderError::Bitstream(format!(
            "AVIF rotation angle {} is invalid",
            rotation.angle
        )));
    }
    for _ in 0..rotation.angle {
        for plane in &mut frame.buffers.planes {
            let width = plane.layout.width;
            let height = plane.layout.height;
            let mut transformed = vec![0; plane.samples.len()];
            let new_width = height;
            for y in 0..height {
                for x in 0..width {
                    let destination_x = y;
                    let destination_y = width - 1 - x;
                    transformed[destination_y * new_width + destination_x] =
                        plane.samples[y * width + x];
                }
            }
            plane.layout.width = new_width;
            plane.layout.height = width;
            if plane.layout.plane != 0 {
                std::mem::swap(
                    &mut plane.layout.subsampling_x,
                    &mut plane.layout.subsampling_y,
                );
            }
            plane.layout.sample_count = transformed.len();
            plane.samples = transformed;
        }
        std::mem::swap(
            &mut frame.color_config.subsampling_x,
            &mut frame.color_config.subsampling_y,
        );
        std::mem::swap(&mut frame.width, &mut frame.height);
        std::mem::swap(&mut frame.render_width, &mut frame.render_height);
        std::mem::swap(&mut frame.buffers.width, &mut frame.buffers.height);
    }
    Ok(())
}

fn decode_grid_image(info: &AvifInfo) -> Result<ImageBuffer, DecoderError> {
    let grid = info
        .primary_grid
        .as_ref()
        .ok_or_else(|| DecoderError::Bitstream("AVIF grid metadata is missing".to_string()))?;
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    let cell_count = rows
        .checked_mul(columns)
        .ok_or_else(|| DecoderError::Bitstream("grid cell count overflow".to_string()))?;
    if grid.cells.len() != cell_count {
        return Err(DecoderError::Bitstream(format!(
            "grid has {} cells, expected {cell_count}",
            grid.cells.len()
        )));
    }
    let width = usize::try_from(grid.output_width)
        .map_err(|_| DecoderError::InvalidParam("grid output width is too large".to_string()))?;
    let height = usize::try_from(grid.output_height)
        .map_err(|_| DecoderError::InvalidParam("grid output height is too large".to_string()))?;
    if width == 0 || height == 0 {
        return Err(DecoderError::Bitstream(
            "grid output dimensions must be non-zero".to_string(),
        ));
    }
    let mut column_widths = vec![0usize; columns];
    let mut row_heights = vec![0usize; rows];
    let decoded_cells = decode_grid_images(info, &grid.cells)?;
    for (index, cell) in grid.cells.iter().enumerate() {
        let cell_width = usize::try_from(cell.width)
            .map_err(|_| DecoderError::InvalidParam("grid cell width is too large".to_string()))?;
        let cell_height = usize::try_from(cell.height)
            .map_err(|_| DecoderError::InvalidParam("grid cell height is too large".to_string()))?;
        if cell_width == 0 || cell_height == 0 {
            return Err(DecoderError::Bitstream(format!(
                "grid cell {} has zero dimensions",
                cell.item_id
            )));
        }
        let row = index / columns;
        let column = index % columns;
        if column_widths[column] != 0 && column_widths[column] != cell_width {
            return Err(DecoderError::Bitstream(
                "grid cells in one column have different widths".to_string(),
            ));
        }
        if row_heights[row] != 0 && row_heights[row] != cell_height {
            return Err(DecoderError::Bitstream(
                "grid cells in one row have different heights".to_string(),
            ));
        }
        column_widths[column] = cell_width;
        row_heights[row] = cell_height;
    }
    if column_widths.iter().sum::<usize>() < width || row_heights.iter().sum::<usize>() < height {
        return Err(DecoderError::Bitstream(format!(
            "grid cell dimensions do not cover declared output {}x{} (columns {:?}, rows {:?})",
            width, height, column_widths, row_heights
        )));
    }
    let mut image = compose_grid_images(grid, &decoded_cells, &column_widths, &row_heights)?;
    if let Some(alpha_grid) = info.alpha_grid.as_ref() {
        apply_alpha_grid(&mut image, alpha_grid)?;
    } else if !info.alpha_auxiliary_items.is_empty()
        && !info
            .primary_grid
            .as_ref()
            .is_some_and(|grid| grid_has_cell_alpha(&grid.payload))
    {
        apply_alpha_auxiliary(&mut image, info)?;
    }
    apply_clean_aperture(&mut image, info.clean_aperture)?;
    apply_mirror(&mut image, info.mirror)?;
    apply_rotation(&mut image, info.rotation)?;
    Ok(image)
}

fn compose_grid_images(
    grid: &crate::container::GridImage,
    decoded_cells: &[ImageBuffer],
    column_widths: &[usize],
    row_heights: &[usize],
) -> Result<ImageBuffer, DecoderError> {
    let width = usize::try_from(grid.output_width)
        .map_err(|_| DecoderError::InvalidParam("grid output width is too large".to_string()))?;
    let height = usize::try_from(grid.output_height)
        .map_err(|_| DecoderError::InvalidParam("grid output height is too large".to_string()))?;
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    if decoded_cells.len() != rows.saturating_mul(columns)
        || column_widths.len() != columns
        || row_heights.len() != rows
    {
        return Err(DecoderError::InvalidParam(
            "grid composition dimensions do not match cell count".to_string(),
        ));
    }
    let rgba_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| DecoderError::InvalidParam("grid output buffer is too large".to_string()))?;
    let mut image = ImageBuffer {
        width,
        height,
        rgba: vec![0; rgba_len],
    };
    let mut y_offset = 0;
    for row in 0..rows {
        let mut x_offset = 0;
        for column in 0..columns {
            let cell = &decoded_cells[row * columns + column];
            if cell.width != column_widths[column] || cell.height != row_heights[row] {
                return Err(DecoderError::Bitstream(
                    "grid cell image dimensions do not match metadata".to_string(),
                ));
            }
            if x_offset < width && y_offset < height {
                let copy_width = cell.width.min(width - x_offset);
                let copy_height = cell.height.min(height - y_offset);
                for y in 0..copy_height {
                    let destination = ((y_offset + y) * width + x_offset) * 4;
                    let source = y * cell.width * 4;
                    let row_len = copy_width * 4;
                    image.rgba[destination..destination + row_len]
                        .copy_from_slice(&cell.rgba[source..source + row_len]);
                }
            }
            x_offset += column_widths[column];
        }
        y_offset += row_heights[row];
    }
    Ok(image)
}

fn normalize_grid_cells(decoded_cells: &mut [DecodedFrame]) -> Result<(), DecoderError> {
    let Some(target) = decoded_cells
        .iter()
        .find(|cell| !cell.color_config.monochrome)
    else {
        return Ok(());
    };
    let target_config = target.color_config;
    let target_color_information = target.color_information.clone();
    let target_bit_depth = target.bit_depth;
    let target_has_alpha = decoded_cells.iter().any(|cell| {
        cell.buffers
            .planes
            .iter()
            .any(|plane| plane.layout.plane == 3)
    });
    for cell in decoded_cells {
        if cell.color_config.monochrome {
            let luma = cell
                .buffers
                .planes
                .iter()
                .find(|plane| plane.layout.plane == 0)
                .cloned()
                .ok_or_else(|| {
                    DecoderError::Bitstream(
                        "monochrome grid cell is missing its luma plane".to_string(),
                    )
                })?;
            let alpha = cell
                .buffers
                .planes
                .iter()
                .find(|plane| plane.layout.plane == 3)
                .cloned();
            let chroma_x = u8::from(target_config.subsampling_x);
            let chroma_y = u8::from(target_config.subsampling_y);
            let chroma_width = cell.width.div_ceil(1usize << chroma_x);
            let chroma_height = cell.height.div_ceil(1usize << chroma_y);
            let chroma_samples = chroma_width.checked_mul(chroma_height).ok_or_else(|| {
                DecoderError::InvalidParam("grid chroma plane size overflows".to_string())
            })?;
            let neutral = 1u16 << target_bit_depth.saturating_sub(1);
            let make_chroma = |plane: u8| PlaneBuffer {
                layout: PlaneLayout {
                    plane,
                    width: chroma_width,
                    height: chroma_height,
                    subsampling_x: chroma_x,
                    subsampling_y: chroma_y,
                    sample_count: chroma_samples,
                },
                samples: vec![neutral; chroma_samples],
            };
            let mut planes = vec![luma, make_chroma(1), make_chroma(2)];
            if let Some(alpha) = alpha {
                planes.push(alpha);
            }
            cell.buffers.planes = planes;
            cell.color_config = target_config;
            cell.color_information = target_color_information.clone();
        }
        if target_has_alpha
            && !cell
                .buffers
                .planes
                .iter()
                .any(|plane| plane.layout.plane == 3)
        {
            let alpha_samples = cell.width.checked_mul(cell.height).ok_or_else(|| {
                DecoderError::InvalidParam("grid alpha plane size overflows".to_string())
            })?;
            let opaque = ((1u32 << target_bit_depth) - 1) as u16;
            cell.buffers.planes.push(PlaneBuffer {
                layout: PlaneLayout {
                    plane: 3,
                    width: cell.width,
                    height: cell.height,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    sample_count: alpha_samples,
                },
                samples: vec![opaque; alpha_samples],
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod grid_composition_tests {
    use super::*;
    use crate::container::GridImage;

    fn cell(width: usize, height: usize, value: u8) -> ImageBuffer {
        ImageBuffer {
            width,
            height,
            rgba: vec![value; width * height * 4],
        }
    }

    #[test]
    fn compose_grid_places_cells_in_reference_order() {
        let grid = GridImage {
            item_id: 1,
            rows: 2,
            columns: 2,
            output_width: 3,
            output_height: 3,
            payload: Vec::new(),
            cells: Vec::new(),
        };
        let cells = vec![
            cell(2, 2, 10),
            cell(1, 2, 20),
            cell(2, 1, 30),
            cell(1, 1, 40),
        ];
        let image = compose_grid_images(&grid, &cells, &[2, 1], &[2, 1]).unwrap();
        assert_eq!((image.width, image.height), (3, 3));
        assert_eq!(image.rgba[0], 10);
        assert_eq!(image.rgba[8], 20);
        assert_eq!(image.rgba[2 * 3 * 4], 30);
        assert_eq!(image.rgba[(2 * 3 + 2) * 4], 40);
    }

    #[test]
    fn compose_grid_crops_oversized_edge_cells_to_output() {
        let grid = GridImage {
            item_id: 1,
            rows: 2,
            columns: 1,
            output_width: 2,
            output_height: 3,
            payload: Vec::new(),
            cells: Vec::new(),
        };
        let cells = vec![cell(2, 2, 10), cell(2, 2, 20)];
        let image = compose_grid_images(&grid, &cells, &[2], &[2, 2]).unwrap();
        assert_eq!((image.width, image.height), (2, 3));
        assert_eq!(image.rgba[0], 10);
        assert_eq!(image.rgba[(2 * 2) * 4], 20);
        assert_eq!(image.rgba[(2 * 2 + 1) * 4], 20);
    }

    #[test]
    fn irot_angle_one_rotates_counter_clockwise() {
        let mut image = ImageBuffer {
            width: 2,
            height: 1,
            rgba: vec![10, 0, 0, 255, 20, 0, 0, 255],
        };
        apply_rotation(&mut image, Some(ImageRotation { angle: 1 })).unwrap();
        assert_eq!((image.width, image.height), (1, 2));
        assert_eq!(image.rgba, vec![20, 0, 0, 255, 10, 0, 0, 255]);
    }

    #[test]
    fn alpha_grid_rejects_mismatched_output_dimensions() {
        let mut image = cell(2, 1, 90);
        let grid = GridImage {
            item_id: 2,
            rows: 1,
            columns: 1,
            output_width: 1,
            output_height: 1,
            payload: Vec::new(),
            cells: Vec::new(),
        };
        let error = apply_alpha_grid(&mut image, &grid).unwrap_err();
        assert!(error.to_string().contains("dimensions do not match"));
    }

    pub(super) fn native_frame(width: usize, height: usize, plane: PlaneBuffer) -> DecodedFrame {
        DecodedFrame {
            width,
            height,
            render_width: width,
            render_height: height,
            bit_depth: 8,
            color_config: ColorConfig {
                high_bitdepth: false,
                twelve_bit: false,
                bit_depth: 8,
                monochrome: true,
                color_description: None,
                color_range: ColorRange::Full,
                subsampling_x: false,
                subsampling_y: false,
                chroma_sample_position: None,
                separate_uv_delta_q: false,
            },
            color_information: None,
            alpha_premultiplied: false,
            buffers: FrameBuffers {
                width,
                height,
                planes: vec![plane],
            },
        }
    }

    pub(super) fn native_plane(width: usize, height: usize, samples: Vec<u16>) -> PlaneBuffer {
        PlaneBuffer {
            layout: PlaneLayout {
                plane: 0,
                width,
                height,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: width * height,
            },
            samples,
        }
    }

    #[test]
    fn native_grid_mirror_reverses_each_plane_without_reallocation() {
        let mut frame = native_frame(3, 2, native_plane(3, 2, vec![1, 2, 3, 4, 5, 6]));
        apply_native_mirror(&mut frame.buffers, ImageMirror { axis: 0 }).unwrap();
        assert_eq!(frame.buffers.planes[0].samples, vec![3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn native_grid_rotation_updates_frame_and_plane_dimensions() {
        let mut frame = native_frame(2, 3, native_plane(2, 3, vec![1, 2, 3, 4, 5, 6]));
        apply_native_rotation(&mut frame, ImageRotation { angle: 1 }).unwrap();
        assert_eq!((frame.width, frame.height), (3, 2));
        assert_eq!(frame.buffers.planes[0].samples, vec![2, 4, 6, 1, 3, 5]);
    }

    #[test]
    fn native_grid_rotation_swaps_422_subsampling_axes() {
        let mut frame = native_frame(4, 2, native_plane(4, 2, (10..18).collect()));
        frame.color_config.monochrome = false;
        frame.color_config.subsampling_x = true;
        frame.buffers.planes.push(PlaneBuffer {
            layout: PlaneLayout {
                plane: 1,
                width: 2,
                height: 2,
                subsampling_x: 1,
                subsampling_y: 0,
                sample_count: 4,
            },
            samples: (0..4).collect(),
        });

        apply_native_rotation(&mut frame, ImageRotation { angle: 1 }).unwrap();

        assert_eq!((frame.width, frame.height), (2, 4));
        assert_eq!(
            (
                frame.color_config.subsampling_x,
                frame.color_config.subsampling_y
            ),
            (false, true)
        );
        let chroma = &frame.buffers.planes[1];
        assert_eq!((chroma.layout.width, chroma.layout.height), (2, 2));
        assert_eq!(
            (chroma.layout.subsampling_x, chroma.layout.subsampling_y),
            (0, 1)
        );
        assert_eq!(chroma.samples, vec![1, 3, 0, 2]);
    }

    #[test]
    fn native_grid_alpha_plane_composes_ordered_cells() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root should exist")
            .join("test/images/external/avif/unsupported/sofa_grid1x5_420.avif");
        if !path.is_file() {
            eprintln!("external grid sample is unavailable; skipping alpha-grid unit oracle");
            return;
        }
        let info = crate::container::parse_avif(
            &std::fs::read(path).expect("external grid sample should be readable"),
        )
        .expect("external grid sample metadata should parse");
        let grid = info
            .primary_grid
            .expect("external sample should contain a grid");
        let (alpha, _) = decode_alpha_grid_plane(&grid).expect("grid alpha plane should compose");
        assert_eq!(alpha.layout.plane, 3);
        assert_eq!((alpha.layout.width, alpha.layout.height), (1024, 770));
        assert_eq!(alpha.samples.len(), 1024 * 770);
    }
}

fn grid_cell_info(info: &AvifInfo, cell: &GridCell) -> AvifInfo {
    let alpha_auxiliary_items = if info.alpha_grid.is_none() {
        info.primary_grid
            .as_ref()
            .and_then(|grid| grid_cell_alpha_item_id(&grid.payload, cell.item_id))
            .and_then(|alpha_id| {
                info.alpha_auxiliary_items
                    .iter()
                    .find(|item| item.item_id == alpha_id)
                    .cloned()
            })
            .into_iter()
            .collect()
    } else {
        Vec::new()
    };
    AvifInfo {
        major_brand: info.major_brand,
        compatible_brands: info.compatible_brands.clone(),
        primary_item_id: Some(cell.item_id),
        width: Some(cell.width),
        height: Some(cell.height),
        pixel_information: cell.pixel_information.clone(),
        color_information: cell
            .color_information
            .clone()
            .or_else(|| info.color_information.clone()),
        alpha_premultiplied: false,
        alpha_auxiliary_items,
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: cell.av1_config.clone(),
        primary_item_payload: cell.payload.clone(),
        sequence_sample_payloads: Vec::new(),
    }
}

fn decode_grid_cell(info: &AvifInfo, cell: &GridCell) -> Result<ImageBuffer, DecoderError> {
    let cell_info = grid_cell_info(info, cell);
    validate_public_container_preflight(&cell_info, true)?;
    let headers = parse_av1_headers(&cell_info)?;
    decode_still_image(&headers, Some(&cell_info))
}

fn decode_grid_cell_frame(info: &AvifInfo, cell: &GridCell) -> Result<DecodedFrame, DecoderError> {
    let cell_info = grid_cell_info(info, cell);
    validate_public_container_preflight(&cell_info, false)?;
    let headers = parse_av1_headers(&cell_info)?;
    decode_still_frame(&headers, Some(&cell_info))
}

fn append_alpha_plane(
    frame: &mut DecodedFrame,
    alpha_frame: &DecodedFrame,
) -> Result<(), DecoderError> {
    if alpha_frame.width != frame.width || alpha_frame.height != frame.height {
        return Err(DecoderError::Bitstream(
            "AVIF alpha auxiliary dimensions do not match the primary image".to_string(),
        ));
    }
    if alpha_frame.buffers.planes.len() != 1 {
        return Err(DecoderError::Unsupported(
            "AVIF alpha auxiliary image must be monochrome".to_string(),
        ));
    }
    let alpha_plane = alpha_frame.buffers.planes.first().ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha auxiliary plane is missing".to_string())
    })?;
    append_alpha_plane_buffer(frame, alpha_plane.clone(), alpha_frame.bit_depth)
}

fn append_alpha_plane_buffer(
    frame: &mut DecodedFrame,
    mut alpha_plane: PlaneBuffer,
    alpha_bit_depth: u8,
) -> Result<(), DecoderError> {
    if alpha_plane.layout.width == 0 || alpha_plane.layout.height == 0 {
        return Err(DecoderError::Bitstream(
            "AVIF alpha plane has zero dimensions".to_string(),
        ));
    }
    if frame
        .buffers
        .planes
        .iter()
        .any(|plane| plane.layout.plane == 3)
    {
        return Err(DecoderError::Bitstream(
            "AVIF primary image contains duplicate alpha planes".to_string(),
        ));
    }
    if alpha_bit_depth != frame.bit_depth {
        let source_max = (1u32 << alpha_bit_depth) - 1;
        let target_max = (1u32 << frame.bit_depth) - 1;
        for sample in &mut alpha_plane.samples {
            *sample = ((u32::from(*sample) * target_max + source_max / 2) / source_max) as u16;
        }
    }
    alpha_plane.layout.plane = 3;
    frame.buffers.planes.push(alpha_plane);
    Ok(())
}

fn decode_alpha_auxiliary_frame(info: &AvifInfo) -> Result<DecodedFrame, DecoderError> {
    let auxiliary = info.alpha_auxiliary_items.first().ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha auxiliary item is missing".to_string())
    })?;
    let alpha_info = AvifInfo {
        major_brand: info.major_brand,
        compatible_brands: info.compatible_brands.clone(),
        primary_item_id: None,
        width: None,
        height: None,
        pixel_information: None,
        color_information: None,
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: None,
        primary_item_payload: auxiliary.payload.clone(),
        sequence_sample_payloads: Vec::new(),
    };
    let headers = parse_av1_headers(&alpha_info)?;
    decode_still_frame(&headers, None)
}

fn decode_alpha_grid_plane(
    grid: &crate::container::GridImage,
) -> Result<(PlaneBuffer, u8), DecoderError> {
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    let cell_count = rows.checked_mul(columns).ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha grid cell count overflow".to_string())
    })?;
    if grid.cells.len() != cell_count {
        return Err(DecoderError::Bitstream(
            "AVIF alpha grid cell count does not match its dimensions".to_string(),
        ));
    }
    let width = usize::try_from(grid.output_width).map_err(|_| {
        DecoderError::InvalidParam("alpha grid output width is too large".to_string())
    })?;
    let height = usize::try_from(grid.output_height).map_err(|_| {
        DecoderError::InvalidParam("alpha grid output height is too large".to_string())
    })?;
    if width == 0 || height == 0 {
        return Err(DecoderError::Bitstream(
            "alpha grid output dimensions must be non-zero".to_string(),
        ));
    }
    let mut column_widths = vec![0usize; columns];
    let mut row_heights = vec![0usize; rows];
    let decoded_cells = decode_alpha_grid_cells(&grid.cells)?;
    for (index, cell) in grid.cells.iter().enumerate() {
        let cell_width = usize::try_from(cell.width).map_err(|_| {
            DecoderError::InvalidParam("alpha grid cell width is too large".to_string())
        })?;
        let cell_height = usize::try_from(cell.height).map_err(|_| {
            DecoderError::InvalidParam("alpha grid cell height is too large".to_string())
        })?;
        let row = index / columns;
        let column = index % columns;
        if column_widths[column] != 0 && column_widths[column] != cell_width {
            return Err(DecoderError::Bitstream(
                "alpha grid cells in one column have different widths".to_string(),
            ));
        }
        if row_heights[row] != 0 && row_heights[row] != cell_height {
            return Err(DecoderError::Bitstream(
                "alpha grid cells in one row have different heights".to_string(),
            ));
        }
        column_widths[column] = cell_width;
        row_heights[row] = cell_height;
        let decoded = decoded_cells.get(index).ok_or_else(|| {
            DecoderError::Bitstream("alpha grid cell decode result is missing".to_string())
        })?;
        if decoded.width != cell_width || decoded.height != cell_height {
            return Err(DecoderError::Bitstream(format!(
                "alpha grid cell {} decoded as {}x{}, metadata declares {}x{}",
                cell.item_id, decoded.width, decoded.height, cell_width, cell_height
            )));
        }
    }
    if column_widths.iter().sum::<usize>() < width || row_heights.iter().sum::<usize>() < height {
        return Err(DecoderError::Bitstream(
            "alpha grid cell dimensions do not cover the declared output".to_string(),
        ));
    }
    let first = decoded_cells
        .first()
        .ok_or_else(|| DecoderError::Bitstream("alpha grid has no cells".to_string()))?;
    let source = first
        .buffers
        .planes
        .first()
        .ok_or_else(|| DecoderError::Bitstream("alpha grid cell has no plane".to_string()))?;
    for cell in &decoded_cells {
        let plane =
            cell.buffers.planes.first().ok_or_else(|| {
                DecoderError::Bitstream("alpha grid cell has no plane".to_string())
            })?;
        if plane.layout.subsampling_x != source.layout.subsampling_x
            || plane.layout.subsampling_y != source.layout.subsampling_y
            || cell.bit_depth != first.bit_depth
        {
            return Err(DecoderError::Unsupported(
                "alpha grid cells use different plane configurations".to_string(),
            ));
        }
    }
    let subsampling_x = usize::from(source.layout.subsampling_x);
    let subsampling_y = usize::from(source.layout.subsampling_y);
    let scale_x = 1usize << subsampling_x;
    let scale_y = 1usize << subsampling_y;
    let plane_width = width.div_ceil(scale_x);
    let plane_height = height.div_ceil(scale_y);
    let sample_count = plane_width.checked_mul(plane_height).ok_or_else(|| {
        DecoderError::InvalidParam("alpha grid plane sample count overflows".to_string())
    })?;
    let mut output = PlaneBuffer {
        layout: PlaneLayout {
            plane: 3,
            width: plane_width,
            height: plane_height,
            subsampling_x: source.layout.subsampling_x,
            subsampling_y: source.layout.subsampling_y,
            sample_count,
        },
        samples: vec![0; sample_count],
    };
    let mut y_offset = 0usize;
    for row in 0..rows {
        let mut x_offset = 0usize;
        for column in 0..columns {
            let cell = &decoded_cells[row * columns + column];
            let source = cell.buffers.planes.first().ok_or_else(|| {
                DecoderError::Bitstream("alpha grid cell has no plane".to_string())
            })?;
            if x_offset % scale_x != 0 || y_offset % scale_y != 0 {
                return Err(DecoderError::Unsupported(
                    "alpha grid cell boundary is not aligned to chroma samples".to_string(),
                ));
            }
            let destination_x = x_offset / scale_x;
            let destination_y = y_offset / scale_y;
            if destination_x >= output.layout.width || destination_y >= output.layout.height {
                continue;
            }
            let copy_width = source.layout.width.min(output.layout.width - destination_x);
            let copy_height = source
                .layout
                .height
                .min(output.layout.height - destination_y);
            for source_y in 0..copy_height {
                let source_start = source_y * source.layout.width;
                let destination_start =
                    (destination_y + source_y) * output.layout.width + destination_x;
                output.samples[destination_start..destination_start + copy_width]
                    .copy_from_slice(&source.samples[source_start..source_start + copy_width]);
            }
            x_offset += column_widths[column];
        }
        y_offset += row_heights[row];
    }
    Ok((output, first.bit_depth))
}

fn decode_alpha_grid_cell_frame(cell: &GridCell) -> Result<DecodedFrame, DecoderError> {
    let info = AvifInfo {
        major_brand: *b"avif",
        compatible_brands: vec![*b"avif"],
        primary_item_id: Some(cell.item_id),
        width: Some(cell.width),
        height: Some(cell.height),
        pixel_information: cell.pixel_information.clone(),
        color_information: None,
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: cell.av1_config.clone(),
        primary_item_payload: cell.payload.clone(),
        sequence_sample_payloads: Vec::new(),
    };
    validate_public_container_preflight(&info, false)?;
    let headers = parse_av1_headers(&info)?;
    decode_still_frame(&headers, None)
}

fn decode_alpha_grid_cells(cells: &[GridCell]) -> Result<Vec<DecodedFrame>, DecoderError> {
    decode_grid_items(cells, decode_alpha_grid_cell_frame)
}

fn apply_alpha_grid(
    image: &mut ImageBuffer,
    grid: &crate::container::GridImage,
) -> Result<(), DecoderError> {
    if grid.output_width != image.width as u32 || grid.output_height != image.height as u32 {
        return Err(DecoderError::Bitstream(
            "AVIF alpha grid dimensions do not match the primary grid".to_string(),
        ));
    }
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    let cell_count = rows.checked_mul(columns).ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha grid cell count overflow".to_string())
    })?;
    if grid.cells.len() != cell_count {
        return Err(DecoderError::Bitstream(
            "AVIF alpha grid cell count does not match its dimensions".to_string(),
        ));
    }
    let mut column_widths = vec![0usize; columns];
    let mut row_heights = vec![0usize; rows];
    let mut decoded_cells = Vec::with_capacity(cell_count);
    for (index, cell) in grid.cells.iter().enumerate() {
        let width = usize::try_from(cell.width).map_err(|_| {
            DecoderError::InvalidParam("alpha grid cell width is too large".to_string())
        })?;
        let height = usize::try_from(cell.height).map_err(|_| {
            DecoderError::InvalidParam("alpha grid cell height is too large".to_string())
        })?;
        let row = index / columns;
        let column = index % columns;
        if column_widths[column] != 0 && column_widths[column] != width {
            return Err(DecoderError::Bitstream(
                "alpha grid cells in one column have different widths".to_string(),
            ));
        }
        if row_heights[row] != 0 && row_heights[row] != height {
            return Err(DecoderError::Bitstream(
                "alpha grid cells in one row have different heights".to_string(),
            ));
        }
        column_widths[column] = width;
        row_heights[row] = height;
        decoded_cells.push(decode_grid_cell_from_alpha(cell)?);
    }
    let alpha = compose_grid_images(grid, &decoded_cells, &column_widths, &row_heights)?;
    for (pixel, alpha_pixel) in image
        .rgba
        .chunks_exact_mut(4)
        .zip(alpha.rgba.chunks_exact(4))
    {
        pixel[3] = alpha_pixel[0];
    }
    Ok(())
}

fn decode_grid_cell_from_alpha(cell: &GridCell) -> Result<ImageBuffer, DecoderError> {
    let info = AvifInfo {
        major_brand: *b"avif",
        compatible_brands: vec![*b"avif"],
        primary_item_id: Some(cell.item_id),
        width: Some(cell.width),
        height: Some(cell.height),
        pixel_information: cell.pixel_information.clone(),
        color_information: cell.color_information.clone(),
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: cell.av1_config.clone(),
        primary_item_payload: cell.payload.clone(),
        sequence_sample_payloads: Vec::new(),
    };
    validate_public_container_preflight(&info, true)?;
    let headers = parse_av1_headers(&info)?;
    decode_still_image(&headers, Some(&info))
}

fn apply_alpha_auxiliary(image: &mut ImageBuffer, info: &AvifInfo) -> Result<(), DecoderError> {
    let frame = decode_alpha_auxiliary_frame(info)?;
    let alpha_plane = frame.buffers.planes.first().ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha auxiliary plane is missing".to_string())
    })?;
    if frame.width != image.width || frame.height != image.height {
        return Err(DecoderError::Bitstream(
            "AVIF alpha auxiliary dimensions do not match the primary image".to_string(),
        ));
    }
    apply_alpha_plane_rows(
        &mut image.rgba,
        image.width,
        image.height,
        alpha_plane,
        frame.bit_depth,
    );
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
const PARALLEL_ALPHA_MIN_PIXELS: usize = 256 * 1024;
#[cfg(not(target_family = "wasm"))]
const MAX_ALPHA_WORKERS: usize = 8;

fn apply_alpha_plane_rows(
    rgba: &mut [u8],
    width: usize,
    _height: usize,
    alpha_plane: &crate::av1::PlaneBuffer,
    bit_depth: u8,
) {
    #[cfg(not(target_family = "wasm"))]
    let workers = if width.saturating_mul(_height) < PARALLEL_ALPHA_MIN_PIXELS {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(MAX_ALPHA_WORKERS)
            .min(_height.max(1))
    };
    #[cfg(target_family = "wasm")]
    let workers = 1;
    if workers <= 1 {
        apply_alpha_rows(
            rgba,
            0,
            width,
            alpha_plane.layout.width,
            alpha_plane.layout.height,
            alpha_plane.layout.subsampling_x,
            alpha_plane.layout.subsampling_y,
            &alpha_plane.samples,
            bit_depth,
        );
        return;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let rows_per_worker = _height.div_ceil(workers);
        let samples_per_chunk = rows_per_worker.saturating_mul(width).saturating_mul(4);
        let alpha_width = alpha_plane.layout.width;
        let alpha_height = alpha_plane.layout.height;
        let alpha_subsampling_x = alpha_plane.layout.subsampling_x;
        let alpha_subsampling_y = alpha_plane.layout.subsampling_y;
        let alpha_samples = &alpha_plane.samples;
        std::thread::scope(|scope| {
            for (chunk_index, chunk) in rgba.chunks_mut(samples_per_chunk).enumerate() {
                let first_row = chunk_index * rows_per_worker;
                scope.spawn(move || {
                    apply_alpha_rows(
                        chunk,
                        first_row,
                        width,
                        alpha_width,
                        alpha_height,
                        alpha_subsampling_x,
                        alpha_subsampling_y,
                        alpha_samples,
                        bit_depth,
                    );
                });
            }
        });
    }
}

fn apply_alpha_rows(
    rgba: &mut [u8],
    first_row: usize,
    width: usize,
    alpha_width: usize,
    alpha_height: usize,
    alpha_subsampling_x: u8,
    alpha_subsampling_y: u8,
    alpha_samples: &[u16],
    bit_depth: u8,
) {
    let row_count = rgba.len() / width.max(1) / 4;
    for local_y in 0..row_count {
        let y = first_row + local_y;
        let alpha_y = (y >> usize::from(alpha_subsampling_y)).min(alpha_height.saturating_sub(1));
        let row = &mut rgba[local_y * width * 4..(local_y + 1) * width * 4];
        for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
            let alpha_x =
                (x >> usize::from(alpha_subsampling_x)).min(alpha_width.saturating_sub(1));
            pixel[3] =
                scale_sample_to_u8(alpha_samples[alpha_y * alpha_width + alpha_x], bit_depth);
        }
    }
}

fn scale_sample_to_u8(sample: u16, bit_depth: u8) -> u8 {
    let maximum = (1u32 << bit_depth) - 1;
    ((u32::from(sample) * 255 + maximum / 2) / maximum) as u8
}

fn apply_clean_aperture(
    image: &mut ImageBuffer,
    aperture: Option<CleanAperture>,
) -> Result<(), DecoderError> {
    let Some(aperture) = aperture else {
        return Ok(());
    };
    let (start_x, start_y, width, height) =
        clean_aperture_rect(image.width, image.height, aperture)?;
    let mut cropped = vec![0; width * height * 4];
    for row in 0..height {
        let src = ((start_y + row) * image.width + start_x) * 4;
        let dst = row * width * 4;
        cropped[dst..dst + width * 4].copy_from_slice(&image.rgba[src..src + width * 4]);
    }
    image.width = width;
    image.height = height;
    image.rgba = cropped;
    Ok(())
}

fn apply_mirror(image: &mut ImageBuffer, mirror: Option<ImageMirror>) -> Result<(), DecoderError> {
    let Some(mirror) = mirror else {
        return Ok(());
    };
    if mirror.axis > 1 {
        return Err(DecoderError::Bitstream(format!(
            "AVIF mirror axis {} is invalid",
            mirror.axis
        )));
    }
    let horizontal = mirror.axis == 0;
    let mut transformed = vec![0; image.rgba.len()];
    for y in 0..image.height {
        for x in 0..image.width {
            let (source_x, source_y) = if horizontal {
                (image.width - 1 - x, y)
            } else {
                (x, image.height - 1 - y)
            };
            let source = (source_y * image.width + source_x) * 4;
            let destination = (y * image.width + x) * 4;
            transformed[destination..destination + 4]
                .copy_from_slice(&image.rgba[source..source + 4]);
        }
    }
    image.rgba = transformed;
    Ok(())
}

fn apply_rotation(
    image: &mut ImageBuffer,
    rotation: Option<ImageRotation>,
) -> Result<(), DecoderError> {
    let Some(rotation) = rotation else {
        return Ok(());
    };
    if rotation.angle > 3 {
        return Err(DecoderError::Bitstream(format!(
            "AVIF rotation angle {} is invalid",
            rotation.angle
        )));
    }
    for _ in 0..rotation.angle {
        let mut transformed = vec![0; image.rgba.len()];
        let new_width = image.height;
        let new_height = image.width;
        for y in 0..image.height {
            for x in 0..image.width {
                // The AVIF `irot` angle is a counter-clockwise quarter-turn.
                // The output width/height are swapped for each turn.
                let destination_x = y;
                let destination_y = image.width - 1 - x;
                let source = (y * image.width + x) * 4;
                let destination = (destination_y * new_width + destination_x) * 4;
                transformed[destination..destination + 4]
                    .copy_from_slice(&image.rgba[source..source + 4]);
            }
        }
        image.width = new_width;
        image.height = new_height;
        image.rgba = transformed;
    }
    Ok(())
}

fn decode_still_frame(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
) -> Result<DecodedFrame, DecoderError> {
    decode_still_frame_with_filter_policy(headers, info, true)
}

fn decode_sample_transform_frame(
    data: &[u8],
    info: &AvifInfo,
) -> Result<Option<DecodedFrame>, DecoderError> {
    let Some(transform) = parse_sample_transform(data)? else {
        return Ok(None);
    };
    if transform.inputs.iter().any(|input| {
        input.width != transform.output_width || input.height != transform.output_height
    }) {
        return Err(DecoderError::Unsupported(
            "sato input and output dimensions do not match".to_string(),
        ));
    }
    let mut frames = Vec::with_capacity(transform.inputs.len());
    for input in &transform.inputs {
        frames.push(decode_sample_transform_input(input)?);
    }
    let first = frames
        .first()
        .ok_or_else(|| DecoderError::Bitstream("sato has no input frames".to_string()))?;
    if frames.iter().any(|frame| {
        frame.width != first.width
            || frame.height != first.height
            || frame.buffers.planes.len() != first.buffers.planes.len()
    }) {
        return Err(DecoderError::Unsupported(
            "sato inputs have incompatible decoded planes".to_string(),
        ));
    }
    let output_bit_depth = transform.output_bit_depth;
    let output_max = (1_u32 << output_bit_depth) - 1;
    let mut planes = Vec::with_capacity(first.buffers.planes.len());
    let mut stack = Vec::with_capacity(transform.tokens.len());
    for (plane_index, first_plane) in first.buffers.planes.iter().enumerate() {
        let mut samples = Vec::with_capacity(first_plane.samples.len());
        for sample_index in 0..first_plane.samples.len() {
            samples.push(evaluate_sample_transform_expression(
                &transform.tokens,
                transform.intermediate_bit_depth,
                output_max,
                |index| {
                    let frame = frames.get(index).ok_or_else(|| {
                        DecoderError::Bitstream(format!(
                            "sato input reference {index} is out of range"
                        ))
                    })?;
                    let plane = frame.buffers.planes.get(plane_index).ok_or_else(|| {
                        DecoderError::Unsupported(
                            "sato input plane count does not match".to_string(),
                        )
                    })?;
                    let value = *plane.samples.get(sample_index).ok_or_else(|| {
                        DecoderError::Bitstream(
                            "sato input plane dimensions do not match".to_string(),
                        )
                    })?;
                    Ok(i128::from(value))
                },
                &mut stack,
            )?);
        }
        planes.push(PlaneBuffer {
            layout: first_plane.layout,
            samples,
        });
    }
    let mut color_config = first.color_config;
    color_config.bit_depth = output_bit_depth;
    color_config.high_bitdepth = output_bit_depth > 8;
    color_config.twelve_bit = output_bit_depth == 12;
    let mut output = DecodedFrame {
        width: transform.output_width as usize,
        height: transform.output_height as usize,
        render_width: transform.output_width as usize,
        render_height: transform.output_height as usize,
        bit_depth: output_bit_depth,
        color_config,
        color_information: first.color_information.clone(),
        alpha_premultiplied: false,
        buffers: FrameBuffers {
            width: transform.output_width as usize,
            height: transform.output_height as usize,
            planes,
        },
    };
    if let Some(alpha_grid) = info.alpha_grid.as_ref() {
        let (alpha_plane, alpha_bit_depth) = decode_alpha_grid_plane(alpha_grid)?;
        append_alpha_plane_buffer(&mut output, alpha_plane, alpha_bit_depth)?;
    } else if !info.alpha_auxiliary_items.is_empty() {
        let alpha_frame = decode_alpha_auxiliary_frame(info)?;
        append_alpha_plane(&mut output, &alpha_frame)?;
    }
    Ok(Some(output))
}

fn decode_sample_transform_input(
    input: &SampleTransformInput,
) -> Result<DecodedFrame, DecoderError> {
    let mut input_info = AvifInfo {
        major_brand: *b"avif",
        compatible_brands: vec![*b"avif"],
        primary_item_id: Some(input.item_id),
        width: Some(input.width),
        height: Some(input.height),
        pixel_information: Some(input.pixel_information.clone()),
        color_information: input.color_information.clone(),
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: Some(input.av1_config.clone()),
        primary_item_payload: input.payload.clone(),
        sequence_sample_payloads: Vec::new(),
    };
    if let Some(grid) = input.grid.as_ref() {
        input_info.primary_grid = Some(grid.clone());
        decode_grid_frame(&input_info)
    } else {
        let headers = parse_av1_headers(&input_info)?;
        decode_still_frame(&headers, Some(&input_info))
    }
}

fn evaluate_sample_transform_expression<F>(
    tokens: &[SampleTransformToken],
    intermediate_bit_depth: u8,
    output_max: u32,
    mut input_value: F,
    stack: &mut Vec<i128>,
) -> Result<u16, DecoderError>
where
    F: FnMut(usize) -> Result<i128, DecoderError>,
{
    if !(8..=64).contains(&intermediate_bit_depth) {
        return Err(DecoderError::Unsupported(format!(
            "sato intermediate bit depth {intermediate_bit_depth} is not supported"
        )));
    }
    let intermediate_min = -(1_i128 << (u32::from(intermediate_bit_depth) - 1));
    let intermediate_max = (1_i128 << (u32::from(intermediate_bit_depth) - 1)) - 1;
    stack.clear();
    for token in tokens {
        match *token {
            SampleTransformToken::Constant(value) => stack.push(i128::from(value)),
            SampleTransformToken::Input(index) => stack.push(input_value(index)?),
            SampleTransformToken::Unary(op) => {
                let value = stack.pop().ok_or_else(|| {
                    DecoderError::Bitstream("sato unary stack underflow".to_string())
                })?;
                let value = match op {
                    0 => value.saturating_neg(),
                    1 => value.saturating_abs(),
                    2 => !value,
                    3 => {
                        if value <= 0 {
                            0
                        } else {
                            i128::from(127 - value.leading_zeros())
                        }
                    }
                    _ => unreachable!(),
                };
                stack.push(value.clamp(intermediate_min, intermediate_max));
            }
            SampleTransformToken::Binary(op) => {
                let right = stack.pop().ok_or_else(|| {
                    DecoderError::Bitstream("sato binary stack underflow".to_string())
                })?;
                let left = stack.pop().ok_or_else(|| {
                    DecoderError::Bitstream("sato binary stack underflow".to_string())
                })?;
                let value = match op {
                    0 => left.saturating_add(right),
                    1 => left.saturating_sub(right),
                    2 => left.saturating_mul(right),
                    3 => {
                        if right == 0 {
                            return Err(DecoderError::Bitstream(
                                "sato division by zero".to_string(),
                            ));
                        }
                        left / right
                    }
                    4 => left & right,
                    5 => left | right,
                    6 => left ^ right,
                    7 => {
                        if right < 0 {
                            return Err(DecoderError::Unsupported(
                                "negative sato exponent is not supported".to_string(),
                            ));
                        }
                        saturating_pow_i128(left, right)
                    }
                    8 => left.min(right),
                    9 => left.max(right),
                    _ => unreachable!(),
                };
                stack.push(value.clamp(intermediate_min, intermediate_max));
            }
        }
    }
    let value = stack
        .pop()
        .ok_or_else(|| DecoderError::Bitstream("sato expression produced no result".to_string()))?;
    if !stack.is_empty() {
        return Err(DecoderError::Bitstream(
            "sato expression produced multiple results".to_string(),
        ));
    }
    Ok(value.clamp(0, i128::from(output_max)) as u16)
}

fn saturating_pow_i128(mut base: i128, exponent: i128) -> i128 {
    let mut result = 1_i128;
    let mut exponent = exponent as u128;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result.saturating_mul(base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base.saturating_mul(base);
        }
    }
    result
}

#[cfg(test)]
fn decode_still_frame_prefilter_for_test(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
) -> Result<DecodedFrame, DecoderError> {
    decode_still_frame_with_filter_policy(headers, info, false)
}

fn decode_still_frame_with_filter_policy(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
    validate_filters: bool,
) -> Result<DecodedFrame, DecoderError> {
    let decoded = decode_still_frame_with_filter_policy_and_state(headers, info, validate_filters)?;
    finish_decoded_still_frame(headers, decoded, validate_filters)
}

fn finish_decoded_still_frame(
    headers: &Av1Headers,
    decoded: DecodedStillFrame,
    validate_filters: bool,
) -> Result<DecodedFrame, DecoderError> {
    let DecodedStillFrame {
        mut frame,
        post_filter_state,
        film_grain,
    } = decoded;
    if validate_filters {
        apply_deblock_stage(&mut frame, &headers.frame, &post_filter_state);
        let restoration_boundaries = headers
            .frame
            .restoration
            .uses_lr
            .then(|| capture_restoration_boundary_rows(&frame));
        apply_cdef_stage(&mut frame, &headers.frame, &post_filter_state);
        apply_loop_restoration_stage_with_boundaries(
            &mut frame,
            &post_filter_state,
            headers.decode_plan.superblock_size << headers.frame.restoration.unit_shift,
            &[1, 2],
            restoration_boundaries.as_deref(),
        );
        crop_frame_buffers_to_plan(&mut frame.buffers, &headers.decode_plan)?;
        apply_superres_horizontal(
            &mut frame.buffers,
            headers.decode_plan.upscaled_width,
            headers.decode_plan.bit_depth,
        )?;
        frame.width = headers.decode_plan.upscaled_width;
        if let Some(film_grain) = film_grain {
            apply_film_grain(&mut frame.buffers, &frame.color_config, &film_grain);
        }
    }
    Ok(frame)
}

fn apply_deblock_stage(
    frame: &mut DecodedFrame,
    frame_header: &FrameHeader,
    state: &PostFilterState,
) {
    if state.block_filter_states.is_empty() || state.transform_boundaries.is_empty() {
        return;
    }
    if !deblock_has_active_strengths(&frame_header.loop_filter, &frame_header.segmentation, state) {
        return;
    }
    let frame_width = frame.width;
    let frame_height = frame.height;
    let bit_depth = frame.bit_depth;
    let subsampling_x = frame.color_config.subsampling_x;
    let subsampling_y = frame.color_config.subsampling_y;
    #[cfg(not(target_family = "wasm"))]
    if post_filter_parallel_work_is_large_enough(frame) {
        std::thread::scope(|scope| {
            for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
                scope.spawn(move || {
                    apply_deblock_plane(
                        plane,
                        plane_index,
                        frame_width,
                        frame_height,
                        bit_depth,
                        subsampling_x,
                        subsampling_y,
                        frame_header,
                        state,
                    );
                });
            }
        });
        return;
    }
    for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
        apply_deblock_plane(
            plane,
            plane_index,
            frame_width,
            frame_height,
            bit_depth,
            subsampling_x,
            subsampling_y,
            frame_header,
            state,
        );
    }
}

fn apply_deblock_plane(
    plane: &mut PlaneBuffer,
    target_plane_index: usize,
    frame_width: usize,
    frame_height: usize,
    bit_depth: u8,
    subsampling_x: bool,
    subsampling_y: bool,
    frame_header: &FrameHeader,
    state: &PostFilterState,
) {
    const FILTER_GRID_STEP: usize = 8;
    let filter_grid_width = frame_width.div_ceil(FILTER_GRID_STEP);
    let filter_grid_height = frame_height.div_ceil(FILTER_GRID_STEP);
    let mut filter_grid = vec![None; filter_grid_width * filter_grid_height];
    for candidate in &state.block_filter_states {
        let start_x = (candidate.x / FILTER_GRID_STEP).min(filter_grid_width);
        let start_y = (candidate.y / FILTER_GRID_STEP).min(filter_grid_height);
        let end_x = candidate
            .x
            .saturating_add(candidate.block_size.width())
            .div_ceil(FILTER_GRID_STEP)
            .min(filter_grid_width);
        let end_y = candidate
            .y
            .saturating_add(candidate.block_size.height())
            .div_ceil(FILTER_GRID_STEP)
            .min(filter_grid_height);
        for grid_y in start_y..end_y {
            let row = grid_y * filter_grid_width;
            for grid_x in start_x..end_x {
                let slot = &mut filter_grid[row + grid_x];
                if slot.is_none() {
                    *slot = Some(*candidate);
                }
            }
        }
    }
    let block_filter_state_at = |x: usize, y: usize| {
        filter_grid
            .get((y / FILTER_GRID_STEP) * filter_grid_width + x / FILTER_GRID_STEP)
            .copied()
            .flatten()
    };
    let mut previous_vertical =
        std::collections::HashMap::<(usize, usize), Vec<(usize, usize, usize, bool, bool)>>::new();
    let mut previous_horizontal =
        std::collections::HashMap::<(usize, usize), Vec<(usize, usize, usize, bool, bool)>>::new();
    for boundary in &state.transform_boundaries {
        let block = boundary.block;
        previous_vertical
            .entry((block.plane, block.x + block.tx_size.width()))
            .or_default()
            .push((
                block.y,
                block.tx_size.height(),
                block.tx_size.width(),
                boundary.skip,
                boundary.is_inter,
            ));
        previous_horizontal
            .entry((block.plane, block.y + block.tx_size.height()))
            .or_default()
            .push((
                block.x,
                block.tx_size.width(),
                block.tx_size.height(),
                boundary.skip,
                boundary.is_inter,
            ));
    }
    let mut boundaries = state.transform_boundaries.iter().collect::<Vec<_>>();
    for vertical in [true, false] {
        let mut previous_edge = None;
        if vertical {
            boundaries
                .sort_by_key(|boundary| (boundary.block.plane, boundary.block.y, boundary.block.x));
        } else {
            boundaries
                .sort_by_key(|boundary| (boundary.block.plane, boundary.block.x, boundary.block.y));
        }
        for boundary in &boundaries {
            let block = boundary.block;
            let plane_index = block.plane;
            if plane_index != target_plane_index {
                continue;
            }
            let subsampling_x = usize::from(plane_index > 0 && subsampling_x);
            let subsampling_y = usize::from(plane_index > 0 && subsampling_y);
            let base_level = if plane_index == 0 {
                frame_header.loop_filter.levels[usize::from(!vertical)]
            } else if plane_index == 1 {
                frame_header.loop_filter.levels[2]
            } else {
                frame_header.loop_filter.levels[3]
            };
            let edge = if vertical { block.x } else { block.y };
            let edge_key = (plane_index, block.x, block.y);
            if edge == 0 || previous_edge == Some(edge_key) {
                continue;
            }
            previous_edge = Some(edge_key);
            let span = if vertical {
                block.tx_size.height()
            } else {
                block.tx_size.width()
            };
            for offset in (0..span).step_by(4) {
                let (edge_x, edge_y) = if vertical {
                    (block.x, block.y + offset)
                } else {
                    (block.x + offset, block.y)
                };
                let current_block = if plane_index == 0 {
                    None
                } else {
                    block_filter_state_at(edge_x << subsampling_x, edge_y << subsampling_y)
                };
                let luma_x = if plane_index == 0 {
                    edge_x
                } else {
                    edge_x << subsampling_x
                };
                let luma_y = if plane_index == 0 {
                    edge_y
                } else {
                    edge_y << subsampling_y
                };
                if !frame_header.tile_info.loop_filter_across_tiles
                    && is_tile_edge(frame_header, luma_x, luma_y, vertical)
                {
                    continue;
                }
                let filter_state = block_filter_state_at(luma_x, luma_y);
                // Select the per-block reference/motion deltas for both still
                // images and AVIS inter frames. Intra blocks use the
                // INTRA_FRAME reference slot and zero-MV mode delta.
                let delta_lf_index = if plane_index == 0 {
                    usize::from(!vertical)
                } else {
                    plane_index + 1
                };
                let block_delta = filter_state
                    .map(|state| state.delta_lf[delta_lf_index])
                    .unwrap_or(0);
                let segment_delta = filter_state
                    .and_then(|state| {
                        frame_header
                            .segmentation
                            .segment_delta_lf
                            .get(usize::from(state.segment_id))
                    })
                    .map(|deltas| deltas[delta_lf_index])
                    .unwrap_or(0);
                let reference_delta_index =
                    loop_filter_reference_delta_index(boundary.is_inter, boundary.reference_frame);
                let mode_delta_index =
                    loop_filter_mode_delta_index(boundary.is_inter, boundary.has_nonzero_mv);
                let level = apply_loop_filter_deltas(
                    base_level,
                    frame_header.loop_filter.delta_enabled,
                    frame_header.loop_filter.ref_deltas[reference_delta_index],
                    frame_header.loop_filter.mode_deltas[mode_delta_index],
                    block_delta,
                    segment_delta,
                );
                let dimension = if plane_index == 0 {
                    if vertical {
                        block.tx_size.width()
                    } else {
                        block.tx_size.height()
                    }
                } else {
                    current_block
                        .map(|current| {
                            if vertical {
                                ceil_shift(current.block_size.width(), subsampling_x).min(64)
                            } else {
                                ceil_shift(current.block_size.height(), subsampling_y).min(64)
                            }
                        })
                        .unwrap_or_else(|| {
                            if vertical {
                                block.tx_size.width()
                            } else {
                                block.tx_size.height()
                            }
                        })
                };
                let previous_block = if plane_index == 0 {
                    None
                } else if vertical {
                    edge_x.checked_sub(1).and_then(|x| {
                        block_filter_state_at(x << subsampling_x, edge_y << subsampling_y)
                    })
                } else {
                    edge_y.checked_sub(1).and_then(|y| {
                        block_filter_state_at(edge_x << subsampling_x, y << subsampling_y)
                    })
                };
                let previous_dimension = if plane_index != 0 {
                    previous_block
                        .map(|previous| {
                            if vertical {
                                ceil_shift(previous.block_size.width(), subsampling_x).min(64)
                            } else {
                                ceil_shift(previous.block_size.height(), subsampling_y).min(64)
                            }
                        })
                        .unwrap_or(dimension)
                } else if vertical {
                    previous_vertical
                        .get(&(plane_index, edge_x))
                        .into_iter()
                        .flat_map(|entries| entries.iter())
                        .find(|(y, height, _, _, _)| edge_y >= *y && edge_y < *y + *height)
                        .map(|(_, _, width, _, _)| *width)
                        .unwrap_or(dimension)
                } else {
                    previous_horizontal
                        .get(&(plane_index, edge_y))
                        .into_iter()
                        .flat_map(|entries| entries.iter())
                        .find(|(x, width, _, _, _)| edge_x >= *x && edge_x < *x + *width)
                        .map(|(_, _, height, _, _)| *height)
                        .unwrap_or(dimension)
                };
                let previous_skipped = if plane_index != 0 {
                    previous_block.is_some_and(|previous| previous.skip && previous.is_inter)
                } else if vertical {
                    previous_vertical
                        .get(&(plane_index, edge_x))
                        .into_iter()
                        .flat_map(|entries| entries.iter())
                        .find(|(y, height, _, _, _)| edge_y >= *y && edge_y < *y + *height)
                        .is_some_and(|(_, _, _, skip, is_inter)| *skip && *is_inter)
                } else {
                    previous_horizontal
                        .get(&(plane_index, edge_y))
                        .into_iter()
                        .flat_map(|entries| entries.iter())
                        .find(|(x, width, _, _, _)| edge_x >= *x && edge_x < *x + *width)
                        .is_some_and(|(_, _, _, skip, is_inter)| *skip && *is_inter)
                };
                let current_block_start = filter_state.is_some_and(|current| {
                    if vertical {
                        edge_x == current.x
                    } else {
                        edge_y == current.y
                    }
                });
                if boundary.skip && boundary.is_inter && previous_skipped && !current_block_start {
                    continue;
                }
                let dimension = dimension.min(previous_dimension);
                let filter_length = if plane_index == 0 {
                    if dimension <= 4 {
                        4
                    } else if dimension <= 8 {
                        8
                    } else {
                        14
                    }
                } else if dimension <= 4 {
                    4
                } else {
                    6
                };
                deblock_filter_edge_with_visible_bounds(
                    &mut plane.samples,
                    plane.layout.width,
                    plane.layout.height,
                    ceil_shift(frame_width, subsampling_x),
                    ceil_shift(frame_height, subsampling_y),
                    edge_x,
                    edge_y,
                    vertical,
                    level,
                    frame_header.loop_filter.sharpness,
                    bit_depth,
                    filter_length,
                );
            }
        }
    }
}

fn is_tile_edge(frame_header: &FrameHeader, x: usize, y: usize, vertical: bool) -> bool {
    let tile_info = &frame_header.tile_info;
    if tile_info.tile_cols * tile_info.tile_rows <= 1 {
        return false;
    }
    if vertical {
        if x == 0 {
            return false;
        }
        tile_id_at(tile_info, x - 1, y) != tile_id_at(tile_info, x, y)
    } else {
        if y == 0 {
            return false;
        }
        tile_id_at(tile_info, x, y - 1) != tile_id_at(tile_info, x, y)
    }
}

fn tile_id_at(tile_info: &crate::av1::TileInfo, x: usize, y: usize) -> usize {
    let mi_x = (x / 4) as u32;
    let mi_y = (y / 4) as u32;
    let col = tile_info
        .mi_col_starts
        .windows(2)
        .position(|range| mi_x >= range[0] && mi_x < range[1])
        .unwrap_or(tile_info.mi_col_starts.len().saturating_sub(2));
    let row = tile_info
        .mi_row_starts
        .windows(2)
        .position(|range| mi_y >= range[0] && mi_y < range[1])
        .unwrap_or(tile_info.mi_row_starts.len().saturating_sub(2));
    row * tile_info.tile_cols as usize + col
}

#[inline]
fn deblock_has_active_strengths(
    loop_filter: &LoopFilterParams,
    segmentation: &SegmentationParams,
    state: &PostFilterState,
) -> bool {
    if loop_filter.levels.iter().any(|&level| level != 0) {
        return true;
    }
    if loop_filter.delta_enabled
        && (loop_filter.ref_deltas.iter().any(|&delta| delta != 0)
            || loop_filter.mode_deltas.iter().any(|&delta| delta != 0))
    {
        return true;
    }
    if segmentation
        .segment_delta_lf
        .iter()
        .flatten()
        .any(|&delta| delta != 0)
    {
        return true;
    }
    state
        .block_filter_states
        .iter()
        .any(|block| block.delta_lf.iter().any(|&delta| delta != 0))
}

fn apply_loop_filter_deltas(
    level: u8,
    enabled: bool,
    ref_delta: i8,
    mode_delta: i8,
    block_delta: i8,
    segment_delta: i8,
) -> u8 {
    if !enabled {
        return (i16::from(level) + i16::from(segment_delta) + i16::from(block_delta)).clamp(0, 63)
            as u8;
    }
    let adjusted = i16::from(level)
        + i16::from(segment_delta)
        + i16::from(ref_delta)
        + i16::from(mode_delta)
        + i16::from(block_delta);
    adjusted.clamp(0, 63) as u8
}

fn loop_filter_reference_delta_index(is_inter: bool, reference_frame: Option<u8>) -> usize {
    reference_frame
        .filter(|_| is_inter)
        .map(usize::from)
        .filter(|index| *index < 8)
        .unwrap_or(0)
}

fn loop_filter_mode_delta_index(is_inter: bool, has_nonzero_mv: bool) -> usize {
    usize::from(is_inter && has_nonzero_mv)
}

fn ceil_shift(value: usize, shift: usize) -> usize {
    value.div_ceil(1usize << shift)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecodedStillFrame {
    frame: DecodedFrame,
    post_filter_state: PostFilterState,
    film_grain: Option<FilmGrainParams>,
}

fn apply_cdef_stage(frame: &mut DecodedFrame, frame_header: &FrameHeader, state: &PostFilterState) {
    if !frame_header.cdef.enabled
        || state.cdef_units.is_empty()
        || !cdef_has_active_strengths(&frame_header.cdef)
    {
        return;
    }
    let unit_mask = (1usize << frame_header.cdef.bits) - 1;
    let luma_width = frame.buffers.planes[0].layout.width;
    let luma_height = frame.buffers.planes[0].layout.height;
    let visible_luma_width = frame.width.min(luma_width);
    let visible_luma_height = frame.height.min(luma_height);
    let cdef_coeff_shift = frame.bit_depth.saturating_sub(8);
    let cdef_units_width = visible_luma_width.div_ceil(64);
    let cdef_units_height = visible_luma_height.div_ceil(64);
    // CDEF indices are at most three bits wide. Keep a compact dense table so
    // the per-8x8 hot loop does not repeatedly unwrap an `Option<usize>`.
    // `u8::MAX` denotes a unit that was not present in the decoded state;
    // those units retain the historical default index 0 when filtering.
    let mut cdef_indices = vec![u8::MAX; cdef_units_width * cdef_units_height];
    for unit in &state.cdef_units {
        let index = (unit.y / 64) * cdef_units_width + unit.x / 64;
        if let Some(slot) = cdef_indices.get_mut(index) {
            *slot = (unit.index as usize & unit_mask) as u8;
        }
    }
    for block in &state.cdef_blocks {
        let index = (block.y / 64) * cdef_units_width + block.x / 64;
        if let Some(slot) = cdef_indices.get_mut(index) {
            *slot = (block.index as usize & unit_mask) as u8;
        }
    }
    if !cdef_indices_have_active_strengths(&frame_header.cdef, &cdef_indices) {
        return;
    }
    let cdef = frame_header.cdef;
    let filtered_blocks = cdef_filtered_block_mask(
        &state.block_filter_states,
        visible_luma_width,
        visible_luma_height,
    );
    let filtered_blocks_width = visible_luma_width.div_ceil(8);
    let mut cdef_block_origins = Vec::new();
    for y in (0..visible_luma_height).step_by(8) {
        for x in (0..visible_luma_width).step_by(8) {
            if !filtered_blocks[(y / 8) * filtered_blocks_width + x / 8] {
                continue;
            }
            let unit_x = x & !63;
            let unit_y = y & !63;
            let index = cdef_indices[(unit_y / 64) * cdef_units_width + unit_x / 64];
            let index = if index == u8::MAX {
                0
            } else {
                usize::from(index)
            };
            cdef_block_origins.push((x, y, index));
        }
    }
    let mut cdef_blocks = Vec::with_capacity(cdef_block_origins.len());
    #[cfg(not(target_family = "wasm"))]
    if post_filter_parallel_work_is_large_enough(frame)
        && cdef_block_origins.len() >= PARALLEL_CDEF_DIRECTION_MIN_BLOCKS
    {
        let worker_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_CDEF_DIRECTION_WORKERS)
            .min(cdef_block_origins.len());
        let chunk_size = cdef_block_origins.len().div_ceil(worker_count);
        let luma_samples = &frame.buffers.planes[0].samples;
        std::thread::scope(|scope| {
            let mut workers = Vec::with_capacity(worker_count);
            for origins in cdef_block_origins.chunks(chunk_size) {
                workers.push(scope.spawn(move || {
                    origins
                        .iter()
                        .map(|&(x, y, index)| {
                            cdef_direction_block(
                                luma_samples,
                                luma_width,
                                luma_height,
                                luma_width,
                                luma_height,
                                cdef_coeff_shift,
                                x,
                                y,
                                index,
                            )
                        })
                        .collect::<Vec<_>>()
                }));
            }
            for worker in workers {
                cdef_blocks.extend(worker.join().expect("CDEF direction worker panicked"));
            }
        });
    } else {
        for (x, y, index) in cdef_block_origins {
            cdef_blocks.push(cdef_direction_block(
                &frame.buffers.planes[0].samples,
                luma_width,
                luma_height,
                luma_width,
                luma_height,
                cdef_coeff_shift,
                x,
                y,
                index,
            ));
        }
    }
    let subsampling_x = frame.color_config.subsampling_x;
    let subsampling_y = frame.color_config.subsampling_y;
    let cdef_blocks = cdef_blocks.as_slice();
    // Entropy reconstruction is complete and each plane owns an independent
    // source/output buffer, so the expensive directional filtering can run
    // concurrently without sharing mutable state.
    #[cfg(not(target_family = "wasm"))]
    if post_filter_parallel_work_is_large_enough(frame) {
        std::thread::scope(|scope| {
            for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
                scope.spawn(move || {
                    apply_cdef_plane(
                        plane,
                        plane_index,
                        subsampling_x,
                        subsampling_y,
                        cdef_coeff_shift,
                        cdef,
                        ceil_shift(visible_luma_width, usize::from(subsampling_x)),
                        ceil_shift(visible_luma_height, usize::from(subsampling_y)),
                        &cdef_blocks,
                    );
                });
            }
        });
        return;
    }
    for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
        apply_cdef_plane(
            plane,
            plane_index,
            subsampling_x,
            subsampling_y,
            cdef_coeff_shift,
            cdef,
            ceil_shift(visible_luma_width, usize::from(subsampling_x)),
            ceil_shift(visible_luma_height, usize::from(subsampling_y)),
            &cdef_blocks,
        );
    }
}

fn cdef_filtered_block_mask(
    block_filter_states: &[crate::av1::BlockFilterState],
    luma_width: usize,
    luma_height: usize,
) -> Vec<bool> {
    let filtered_blocks_width = luma_width.div_ceil(8);
    let filtered_blocks_height = luma_height.div_ceil(8);
    let mut filtered_blocks = vec![false; filtered_blocks_width * filtered_blocks_height];
    for block in block_filter_states {
        // AOM's cdef_compute_sb_list contains each 8x8 block unless every
        // mode block covered by it has skip_txfm. Skip is a block-state
        // property, not a property of a signalled CDEF strength table.
        if block.skip {
            continue;
        }
        let start_x = (block.x / 8).min(filtered_blocks_width);
        let start_y = (block.y / 8).min(filtered_blocks_height);
        let end_x = block
            .x
            .saturating_add(block.block_size.width())
            .div_ceil(8)
            .min(filtered_blocks_width);
        let end_y = block
            .y
            .saturating_add(block.block_size.height())
            .div_ceil(8)
            .min(filtered_blocks_height);
        for y in start_y..end_y {
            let row = y * filtered_blocks_width;
            filtered_blocks[row + start_x..row + end_x].fill(true);
        }
    }
    filtered_blocks
}

#[inline]
fn cdef_direction_block(
    luma_samples: &[u16],
    luma_width: usize,
    luma_height: usize,
    visible_luma_width: usize,
    visible_luma_height: usize,
    coeff_shift: u8,
    x: usize,
    y: usize,
    index: usize,
) -> (usize, usize, usize, usize, i32) {
    let (detected_direction, variance) = cdef_find_direction_with_variance_visible(
        luma_samples,
        luma_width,
        luma_height,
        visible_luma_width,
        visible_luma_height,
        x,
        y,
        coeff_shift,
        true,
    );
    (x, y, index, detected_direction, variance)
}

fn apply_cdef_plane(
    plane: &mut PlaneBuffer,
    plane_index: usize,
    subsampling_x: bool,
    subsampling_y: bool,
    cdef_coeff_shift: u8,
    cdef: crate::av1::CdefParams,
    visible_width: usize,
    visible_height: usize,
    cdef_blocks: &[(usize, usize, usize, usize, i32)],
) {
    let source = std::mem::take(&mut plane.samples);
    let plane_has_configured_strength = cdef_blocks.iter().any(|&(_, _, index, _, _)| {
        let strength = &cdef.strengths[index];
        if plane_index == 0 {
            strength.y_pri != 0 || strength.y_sec != 0
        } else {
            strength.uv_pri != 0 || strength.uv_sec != 0
        }
    });
    if !plane_has_configured_strength {
        plane.samples = source;
        return;
    }
    // Cloning preserves the source snapshot while avoiding a zero-fill pass
    // before the filtered blocks overwrite their regions.
    let mut output = source.clone();
    let width = plane.layout.width;
    let height = plane.layout.height;
    let visible_width = visible_width.min(width);
    let visible_height = visible_height.min(height);
    let plane_subsampling_x = usize::from(plane_index > 0 && subsampling_x);
    let plane_subsampling_y = usize::from(plane_index > 0 && subsampling_y);
    let scale_x = 1usize << plane_subsampling_x;
    let scale_y = 1usize << plane_subsampling_y;
    let mut filtered = vec![0u16; 64];
    for &(x, y, index, detected_direction, variance) in cdef_blocks {
        let plane_x = x / scale_x;
        let plane_y = y / scale_y;
        if plane_x >= width || plane_y >= height {
            continue;
        }
        let strength = &cdef.strengths[index];
        let primary_strength = if plane_index == 0 {
            strength.y_pri
        } else {
            strength.uv_pri
        };
        let secondary_strength = if plane_index == 0 {
            strength.y_sec
        } else {
            strength.uv_sec
        };
        let strength_scale = 1u8
            .checked_shl(u32::from(cdef_coeff_shift))
            .unwrap_or(u8::MAX);
        let configured_scaled_primary_strength = primary_strength.saturating_mul(strength_scale);
        let scaled_primary_strength = if plane_index == 0 {
            cdef_adjust_primary_strength(configured_scaled_primary_strength, variance)
        } else {
            configured_scaled_primary_strength
        };
        let scaled_secondary_strength = secondary_strength.saturating_mul(strength_scale);
        let direction = if configured_scaled_primary_strength == 0 {
            0
        } else {
            cdef_chroma_direction(
                detected_direction,
                plane_index > 0 && subsampling_x,
                plane_index > 0 && subsampling_y,
            )
        };
        if cdef_strengths_disabled(scaled_primary_strength, scaled_secondary_strength) {
            continue;
        }
        let scaled_damping = cdef
            .damping
            .saturating_sub(u8::from(plane_index != 0))
            .saturating_add(cdef_coeff_shift);
        let block_width = visible_width
            .saturating_sub(plane_x)
            .min(8usize.div_ceil(scale_x));
        let block_height = visible_height
            .saturating_sub(plane_y)
            .min(8usize.div_ceil(scale_y));
        if block_width == 0 || block_height == 0 {
            continue;
        }
        // The aligned coded plane remains the filter source domain until the
        // post-filter crop. Limit writes to the visible plane, but retain the
        // coded padding taps used by the existing AV1 reconstruction path.
        cdef_filter_block_region_with_edge_mode_into_bit_depth_visible_scaled(
            &source,
            width,
            height,
            width,
            height,
            plane_x,
            plane_y,
            block_width,
            block_height,
            direction,
            scaled_primary_strength,
            scaled_secondary_strength,
            scaled_damping,
            cdef_coeff_shift,
            true,
            &mut filtered,
        );
        for row in 0..block_height {
            let start = (plane_y + row) * width + plane_x;
            let block_start = row * block_width;
            output[start..start + block_width]
                .copy_from_slice(&filtered[block_start..block_start + block_width]);
        }
    }
    plane.samples = output;
}

#[inline]
fn cdef_strengths_disabled(primary_strength: u8, secondary_strength: u8) -> bool {
    primary_strength == 0 && secondary_strength == 0
}

#[inline]
fn cdef_has_active_strengths(cdef: &crate::av1::CdefParams) -> bool {
    let active_count = 1usize << cdef.bits;
    cdef.strengths
        .iter()
        .take(active_count.min(cdef.strengths.len()))
        .any(|strength| {
            strength.y_pri != 0
                || strength.y_sec != 0
                || strength.uv_pri != 0
                || strength.uv_sec != 0
        })
}

#[inline]
fn cdef_indices_have_active_strengths(cdef: &crate::av1::CdefParams, indices: &[u8]) -> bool {
    indices.iter().any(|&index| {
        if index == u8::MAX {
            return false;
        }
        let Some(strength) = cdef.strengths.get(usize::from(index)) else {
            return false;
        };
        strength.y_pri != 0 || strength.y_sec != 0 || strength.uv_pri != 0 || strength.uv_sec != 0
    })
}

#[allow(dead_code)]
fn apply_loop_restoration_stage(
    frame: &mut DecodedFrame,
    state: &PostFilterState,
    unit_size: usize,
    enabled_types: &[u8],
) {
    apply_loop_restoration_stage_with_boundaries(frame, state, unit_size, enabled_types, None);
}

fn apply_loop_restoration_stage_with_boundaries(
    frame: &mut DecodedFrame,
    state: &PostFilterState,
    unit_size: usize,
    enabled_types: &[u8],
    boundaries: Option<&[RestorationBoundaryRows]>,
) {
    let bit_depth = frame.bit_depth;
    if state.restoration_units.is_empty()
        || !state
            .restoration_units
            .iter()
            .any(|unit| enabled_types.contains(&unit.restoration_type))
    {
        return;
    }
    // Restoration units never cross planes; keep their source snapshots local
    // to each worker and retain the sequential path for Wasm/single-plane data.
    #[cfg(not(target_family = "wasm"))]
    if post_filter_parallel_work_is_large_enough(frame) {
        std::thread::scope(|scope| {
            for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
                let boundaries = boundaries.and_then(|boundaries| boundaries.get(plane_index));
                let visible_width = frame
                    .width
                    .div_ceil(1usize << usize::from(plane.layout.subsampling_x));
                let visible_height = frame
                    .height
                    .div_ceil(1usize << usize::from(plane.layout.subsampling_y));
                scope.spawn(move || {
                    apply_loop_restoration_plane(
                        plane,
                        plane_index,
                        state,
                        unit_size,
                        enabled_types,
                        bit_depth,
                        visible_width,
                        visible_height,
                        boundaries,
                    );
                });
            }
        });
        return;
    }
    for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
        let boundaries = boundaries.and_then(|boundaries| boundaries.get(plane_index));
        let visible_width = frame
            .width
            .div_ceil(1usize << usize::from(plane.layout.subsampling_x));
        let visible_height = frame
            .height
            .div_ceil(1usize << usize::from(plane.layout.subsampling_y));
        apply_loop_restoration_plane(
            plane,
            plane_index,
            state,
            unit_size,
            enabled_types,
            bit_depth,
            visible_width,
            visible_height,
            boundaries,
        );
    }
}

#[cfg(not(target_family = "wasm"))]
const PARALLEL_POST_FILTER_MIN_SAMPLES: usize = 128 * 1024;

#[cfg(not(target_family = "wasm"))]
const PARALLEL_CDEF_DIRECTION_MIN_BLOCKS: usize = 512;

#[cfg(not(target_family = "wasm"))]
const MAX_CDEF_DIRECTION_WORKERS: usize = 8;

#[cfg(not(target_family = "wasm"))]
fn post_filter_parallel_work_is_large_enough(frame: &DecodedFrame) -> bool {
    frame.buffers.planes.len() > 1
        && frame
            .buffers
            .planes
            .iter()
            .map(|plane| plane.samples.len())
            .sum::<usize>()
            >= PARALLEL_POST_FILTER_MIN_SAMPLES
}

#[cfg(target_family = "wasm")]
fn post_filter_parallel_work_is_large_enough(_frame: &DecodedFrame) -> bool {
    false
}

#[derive(Debug, Clone, Default)]
struct RestorationBoundaryRows {
    rows: Vec<(usize, Vec<u16>)>,
}

fn capture_restoration_boundary_rows(frame: &DecodedFrame) -> Vec<RestorationBoundaryRows> {
    frame
        .buffers
        .planes
        .iter()
        .map(|plane| {
            let visible_height = frame
                .height
                .div_ceil(1usize << usize::from(plane.layout.subsampling_y));
            let stripe_height = 64usize >> usize::from(plane.layout.subsampling_y);
            let stripe_offset = 8usize >> usize::from(plane.layout.subsampling_y);
            let mut rows = Vec::new();
            let mut stripe = 0usize;
            loop {
                let stripe_start = stripe
                    .saturating_mul(stripe_height)
                    .saturating_sub(stripe_offset);
                if stripe_start >= visible_height {
                    break;
                }
                let stripe_end = ((stripe + 1) * stripe_height)
                    .saturating_sub(stripe_offset)
                    .min(visible_height);
                if stripe > 0 {
                    for row in stripe_start.saturating_sub(2)..stripe_start {
                        let start = row * plane.layout.width;
                        rows.push((
                            row,
                            plane.samples[start..start + plane.layout.width].to_vec(),
                        ));
                    }
                }
                if stripe_end < visible_height {
                    for row in stripe_end..(stripe_end + 2).min(visible_height) {
                        let start = row * plane.layout.width;
                        rows.push((
                            row,
                            plane.samples[start..start + plane.layout.width].to_vec(),
                        ));
                    }
                }
                stripe += 1;
            }
            RestorationBoundaryRows { rows }
        })
        .collect()
}

fn patch_restoration_stripe_boundaries(
    source: &mut [u16],
    width: usize,
    height: usize,
    visible_width: usize,
    visible_height: usize,
    origin_x: usize,
    unit_width: usize,
    stripe_y: usize,
    stripe_height: usize,
    frame_stripe: usize,
    boundaries: &RestorationBoundaryRows,
    saved: &mut Vec<(usize, usize, Vec<u16>)>,
) {
    let start_x = origin_x.saturating_sub(3);
    let end_x = origin_x
        .saturating_add(unit_width)
        .saturating_add(3)
        .min(visible_width)
        .min(width);
    let mut patch_row = |target_row: usize, boundary_row: usize| {
        if target_row >= height {
            return;
        }
        let Some(samples) = boundaries
            .rows
            .iter()
            .find_map(|&(row, ref samples)| (row == boundary_row).then_some(samples))
        else {
            return;
        };
        let start = target_row * width + start_x;
        let end = target_row * width + end_x;
        saved.push((target_row, start_x, source[start..end].to_vec()));
        source[start..end].copy_from_slice(&samples[start_x..end_x]);
    };
    if frame_stripe > 0 {
        let boundary_start = frame_stripe * 64 - 8;
        patch_row(stripe_y.saturating_sub(3), boundary_start - 2);
        patch_row(stripe_y.saturating_sub(2), boundary_start - 2);
        patch_row(stripe_y.saturating_sub(1), boundary_start - 1);
    }
    let stripe_end = stripe_y + stripe_height;
    if stripe_end < visible_height {
        patch_row(stripe_end, stripe_end);
        patch_row(stripe_end + 1, stripe_end + 1);
        patch_row(stripe_end + 2, stripe_end + 1);
    }
}

fn restore_restoration_stripe_boundaries(
    source: &mut [u16],
    width: usize,
    saved: &[(usize, usize, Vec<u16>)],
) {
    for &(row, start_x, ref samples) in saved {
        let start = row * width + start_x;
        source[start..start + samples.len()].copy_from_slice(samples);
    }
}

fn apply_loop_restoration_plane(
    plane: &mut PlaneBuffer,
    plane_index: usize,
    state: &PostFilterState,
    unit_size: usize,
    enabled_types: &[u8],
    bit_depth: u8,
    visible_width: usize,
    visible_height: usize,
    boundaries: Option<&RestorationBoundaryRows>,
) {
    const RESTORATION_UNIT_OFFSET: usize = 8;
    if !state
        .restoration_units
        .iter()
        .any(|unit| unit.plane == plane_index && enabled_types.contains(&unit.restoration_type))
    {
        return;
    }
    let source = std::mem::take(&mut plane.samples);
    // Keep the unfiltered source available to restoration taps without first
    // zero-initializing an output buffer that is immediately overwritten.
    let mut output = source.clone();
    let mut source = source;
    let mut wiener_scratch = Vec::new();
    let mut sgrproj_scratch = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for unit in state
        .restoration_units
        .iter()
        .filter(|unit| unit.plane == plane_index && enabled_types.contains(&unit.restoration_type))
    {
        let remaining_width = visible_width.saturating_sub(unit.x);
        let unit_width = if remaining_width < unit_size + unit_size / 2 {
            remaining_width
        } else {
            unit_size
        };
        let origin_y = unit.y.saturating_sub(RESTORATION_UNIT_OFFSET);
        let remaining_height = visible_height.saturating_sub(unit.y);
        let unit_extent = if remaining_height < unit_size + unit_size / 2 {
            remaining_height
        } else {
            unit_size
        };
        let end_y = if unit.y + unit_extent < visible_height {
            unit.y + unit_extent - RESTORATION_UNIT_OFFSET
        } else {
            visible_height
        };
        let unit_height = end_y.saturating_sub(origin_y);
        if unit_width == 0 || unit_height == 0 {
            continue;
        }
        let mut stripe_y = origin_y;
        while stripe_y < end_y {
            let frame_stripe = (stripe_y + RESTORATION_UNIT_OFFSET) / 64;
            let nominal_height = 64 - usize::from(frame_stripe == 0) * RESTORATION_UNIT_OFFSET;
            let stripe_height = nominal_height.min(end_y - stripe_y);
            let procunit_width = 64;
            let mut patched_rows = Vec::new();
            if let Some(boundaries) = boundaries {
                patch_restoration_stripe_boundaries(
                    &mut source,
                    plane.layout.width,
                    plane.layout.height,
                    visible_width,
                    visible_height,
                    unit.x,
                    unit_width,
                    stripe_y,
                    stripe_height,
                    frame_stripe,
                    boundaries,
                    &mut patched_rows,
                );
            }
            let mut chunk_x = 0;
            while chunk_x < unit_width {
                let x = unit.x + chunk_x;
                let chunk_width = procunit_width.min(unit_width - chunk_x);
                match unit.restoration_type {
                    1 => {
                        let Some(mut filters) = unit.wiener else {
                            break;
                        };
                        if plane_index > 0 {
                            filters[0][0] = 0;
                            filters[1][0] = 0;
                        }
                        wiener_filter_unit_into_with_scratch_bit_depth_visible(
                            &source,
                            &mut output,
                            plane.layout.width,
                            plane.layout.height,
                            visible_width,
                            visible_height,
                            x,
                            stripe_y,
                            chunk_width,
                            stripe_height,
                            filters,
                            bit_depth,
                            &mut wiener_scratch,
                        )
                    }
                    2 => {
                        let (Some(index), Some(xqd)) = (unit.sgrproj_index, unit.sgrproj) else {
                            break;
                        };
                        sgrproj_filter_unit_into_with_scratch_bit_depth_visible(
                            &source,
                            &mut output,
                            plane.layout.width,
                            plane.layout.height,
                            visible_width,
                            visible_height,
                            x,
                            stripe_y,
                            chunk_width,
                            stripe_height,
                            index,
                            xqd,
                            bit_depth,
                            &mut sgrproj_scratch,
                        )
                    }
                    _ => break,
                }
                chunk_x += chunk_width;
            }
            restore_restoration_stripe_boundaries(&mut source, plane.layout.width, &patched_rows);
            stripe_y += stripe_height;
        }
    }
    plane.samples = output;
}

fn decode_still_frame_with_filter_policy_and_state(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
    validate_filters: bool,
) -> Result<DecodedStillFrame, DecoderError> {
    decode_still_frame_with_filter_policy_and_state_and_references(
        headers,
        info,
        validate_filters,
        std::array::from_fn(|_| None),
    )
}

fn decode_still_frame_with_filter_policy_and_state_and_references(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
    validate_filters: bool,
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
) -> Result<DecodedStillFrame, DecoderError> {
    decode_still_frame_with_filter_policy_and_state_and_references_and_cdf(
        headers,
        info,
        validate_filters,
        reference_buffers,
        None,
        true,
        false,
    )
    .map(|(decoded, _)| decoded)
}

fn decode_still_frame_with_filter_policy_and_state_and_references_and_cdf(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
    validate_filters: bool,
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    initial_cdfs: Option<&[CdfContext]>,
    validate_entropy: bool,
    collect_cdf: bool,
) -> Result<(DecodedStillFrame, Vec<CdfContext>), DecoderError> {
    if validate_filters {
        validate_public_decode_tools(headers)?;
    }
    let mut buffers = alloc_coded_frame_buffers(&headers.decode_plan)?;
    let (prefix, post_filter_state, final_cdfs) =
        decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf(
            &headers.tile_group.tile_data,
            &headers.tile_group.group,
            &headers.sequence,
            &headers.frame,
            &headers.decode_plan,
            &mut buffers,
            usize::MAX,
            validate_entropy,
            false,
            reference_buffers,
            initial_cdfs,
            collect_cdf,
        )?;
    if let Some(err) = prefix.next_unsupported {
        return Err(err);
    }
    if !validate_filters {
        crop_frame_buffers_to_plan(&mut buffers, &headers.decode_plan)?;
    }
    Ok((
        DecodedStillFrame {
            frame: DecodedFrame {
                width: headers.decode_plan.width,
                height: headers.decode_plan.height,
                render_width: headers.decode_plan.render_width,
                render_height: headers.decode_plan.render_height,
                bit_depth: headers.decode_plan.bit_depth,
                color_config: headers.sequence.color_config,
                color_information: info.and_then(|info| info.color_information.clone()),
                alpha_premultiplied: info.is_some_and(|info| info.alpha_premultiplied),
                buffers,
            },
            post_filter_state,
            film_grain: headers.frame.film_grain,
        },
        final_cdfs,
    ))
}

fn validate_public_container_preflight(
    info: &AvifInfo,
    rgba_output: bool,
) -> Result<(), DecoderError> {
    let _ = rgba_output;
    let config = info
        .av1_config
        .as_deref()
        .map(parse_av1_config)
        .transpose()?;
    if let Some(config) = config {
        if !matches!(config.bit_depth(), 8 | 10 | 12) {
            return Err(DecoderError::Unsupported(format!(
                "AV1 {}-bit quantization is not supported by public decode yet",
                config.bit_depth()
            )));
        }
        // Monochrome and sub-sampled YUV layouts are decoded by the AV1
        // plane/reconstruction path.
    }
    Ok(())
}

fn validate_public_decode_tools(headers: &Av1Headers) -> Result<(), DecoderError> {
    let quantization = &headers.frame.quantization;
    if quantization.has_unsupported_qmatrix() {
        return Err(DecoderError::Unsupported(
            "AV1 quantization matrix level is outside the 0..15 range".to_string(),
        ));
    }
    Ok(())
}

fn validate_av1_config(
    config: &Av1CodecConfiguration,
    sequence: &SequenceHeader,
) -> Result<(), DecoderError> {
    if config.seq_profile != sequence.seq_profile {
        return Err(DecoderError::Bitstream(format!(
            "av1C seq_profile {} does not match sequence header {}",
            config.seq_profile, sequence.seq_profile
        )));
    }
    if config.seq_level_idx_0 != sequence.seq_level_idx_0 {
        return Err(DecoderError::Bitstream(format!(
            "av1C seq_level_idx_0 {} does not match sequence header {}",
            config.seq_level_idx_0, sequence.seq_level_idx_0
        )));
    }
    if config.bit_depth() != sequence.color_config.bit_depth {
        return Err(DecoderError::Bitstream(format!(
            "av1C bit depth {} does not match sequence header {}",
            config.bit_depth(),
            sequence.color_config.bit_depth
        )));
    }
    if config.monochrome != sequence.color_config.monochrome {
        return Err(DecoderError::Bitstream(
            "av1C monochrome flag does not match sequence header".to_string(),
        ));
    }
    if config.chroma_subsampling_x != sequence.color_config.subsampling_x
        || config.chroma_subsampling_y != sequence.color_config.subsampling_y
    {
        return Err(DecoderError::Bitstream(
            "av1C chroma subsampling does not match sequence header".to_string(),
        ));
    }
    if let Some(sequence_position) = sequence.color_config.chroma_sample_position {
        if config.chroma_sample_position != sequence_position {
            return Err(DecoderError::Bitstream(format!(
                "av1C chroma sample position {:?} does not match sequence header {:?}",
                config.chroma_sample_position, sequence_position
            )));
        }
    }
    Ok(())
}

fn validate_color_metadata(
    color_information: Option<&ColorInformation>,
    color_config: &ColorConfig,
) -> Result<(), DecoderError> {
    let Some(color_information) = color_information else {
        return Ok(());
    };
    if &color_information.color_type != b"nclx" {
        return Ok(());
    }
    let nclx = color_information.nclx().ok_or_else(|| {
        DecoderError::Bitstream("AVIF nclx colour information is truncated".to_string())
    })?;
    if let Some(description) = color_config.color_description {
        let primaries_mismatch = nclx.color_primaries != 2
            && nclx.color_primaries != u16::from(description.color_primaries);
        let transfer_mismatch = nclx.transfer_characteristics != 2
            && nclx.transfer_characteristics != u16::from(description.transfer_characteristics);
        let matrix_mismatch = nclx.matrix_coefficients != 2
            && nclx.matrix_coefficients != u16::from(description.matrix_coefficients);
        if primaries_mismatch || transfer_mismatch || matrix_mismatch {
            return Err(DecoderError::Bitstream(format!(
                "AVIF nclx colour description does not match AV1 sequence header: nclx=({}, {}, {}), av1=({}, {}, {})",
                nclx.color_primaries,
                nclx.transfer_characteristics,
                nclx.matrix_coefficients,
                description.color_primaries,
                description.transfer_characteristics,
                description.matrix_coefficients,
            )));
        }
    }
    let av1_full_range = matches!(color_config.color_range, ColorRange::Full);
    if nclx.full_range_flag != av1_full_range {
        return Err(DecoderError::Bitstream(
            "AVIF nclx range does not match AV1 sequence header".to_string(),
        ));
    }
    Ok(())
}

fn validate_extended_pixi(
    pixel_information: Option<&PixelInformation>,
    color_config: &ColorConfig,
) -> Result<(), DecoderError> {
    let Some(channels) = pixel_information.and_then(|pixi| pixi.extended_channels.as_deref())
    else {
        return Ok(());
    };
    let expected_channels = if color_config.monochrome { 1 } else { 3 };
    if channels.len() != expected_channels {
        return Err(DecoderError::Bitstream(format!(
            "extended pixi has {} channels but AV1 signals {expected_channels}",
            channels.len()
        )));
    }
    for (index, channel) in channels.iter().enumerate() {
        let expected_type = if index == 0 {
            0
        } else if color_config.subsampling_x && color_config.subsampling_y {
            2
        } else if color_config.subsampling_x {
            1
        } else if color_config.subsampling_y {
            4
        } else {
            0
        };
        let Some(subsampling) = channel.subsampling else {
            continue;
        };
        if subsampling.subsampling_type != expected_type {
            return Err(DecoderError::Bitstream(format!(
                "extended pixi channel {index} subsampling type {} does not match AV1 type {expected_type}",
                subsampling.subsampling_type
            )));
        }
        let expected_location = match (index, color_config.chroma_sample_position) {
            (0, _) | (_, None) | (_, Some(ChromaSamplePosition::Unknown)) => None,
            (_, Some(ChromaSamplePosition::Vertical)) => Some(0),
            (_, Some(ChromaSamplePosition::Colocated)) => Some(2),
            (_, Some(ChromaSamplePosition::Reserved)) => {
                return Err(DecoderError::Bitstream(
                    "AV1 chroma sample position is reserved".to_string(),
                ));
            }
        };
        if let Some(expected_location) = expected_location
            && subsampling.subsampling_location != expected_location
        {
            return Err(DecoderError::Bitstream(format!(
                "extended pixi channel {index} location {} does not match AV1 location {expected_location}",
                subsampling.subsampling_location
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "decoder_post_filter_tests.rs"]
mod post_filter_tests;

fn emit_metadata(
    info: &AvifInfo,
    headers: Option<&Av1Headers>,
    option: &mut DecodeOptions<'_>,
) -> Result<(), Error> {
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
    if let Some(headers) = headers {
        if let Some(config) = headers.config {
            option
                .drawer
                .set_metadata("AV1 config version", DataMap::UInt(config.version as u64))?;
        }
        let sequence_header = &headers.sequence;
        let frame_header = &headers.frame;
        option.drawer.set_metadata(
            "AV1 profile",
            DataMap::UInt(sequence_header.seq_profile as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 bit depth",
            DataMap::UInt(sequence_header.color_config.bit_depth as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 level",
            DataMap::UInt(sequence_header.seq_level_idx_0 as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 max frame width",
            DataMap::UInt(sequence_header.max_frame_width as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 max frame height",
            DataMap::UInt(sequence_header.max_frame_height as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 frame width",
            DataMap::UInt(frame_header.frame_width as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 frame height",
            DataMap::UInt(frame_header.frame_height as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 render width",
            DataMap::UInt(frame_header.render_width as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 render height",
            DataMap::UInt(frame_header.render_height as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 uncompressed header bits",
            DataMap::UInt(frame_header.uncompressed_header_bits as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 payload after header offset",
            DataMap::UInt(frame_header.payload_after_header_offset as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 base q idx",
            DataMap::UInt(frame_header.base_q_idx as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 tx mode",
            DataMap::Ascii(format!("{:?}", frame_header.tx_mode)),
        )?;
        option.drawer.set_metadata(
            "AV1 reduced tx set",
            DataMap::UInt(frame_header.reduced_tx_set as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 y dc quant",
            DataMap::UInt(headers.quant_state.y.dc as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 y ac quant",
            DataMap::UInt(headers.quant_state.y.ac as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 u dc quant",
            DataMap::UInt(headers.quant_state.u.dc as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 u ac quant",
            DataMap::UInt(headers.quant_state.u.ac as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 v dc quant",
            DataMap::UInt(headers.quant_state.v.dc as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 v ac quant",
            DataMap::UInt(headers.quant_state.v.ac as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 superblock size",
            DataMap::UInt(headers.decode_plan.superblock_size as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 superblock columns",
            DataMap::UInt(headers.decode_plan.superblock_cols as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 superblock rows",
            DataMap::UInt(headers.decode_plan.superblock_rows as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 tile columns",
            DataMap::UInt(frame_header.tile_info.tile_cols as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 tile rows",
            DataMap::UInt(frame_header.tile_info.tile_rows as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 tile group payload bytes",
            DataMap::UInt(headers.tile_group.payload_len as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 tile group in frame OBU",
            DataMap::UInt(headers.tile_group.from_frame_obu as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 tile group start",
            DataMap::UInt(headers.tile_group.group.start_tile as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 tile group end",
            DataMap::UInt(headers.tile_group.group.end_tile as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 tile payload count",
            DataMap::UInt(headers.tile_group.group.tiles.len() as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 entropy init bits",
            DataMap::UInt(
                headers
                    .tile_group
                    .entropy_states
                    .first()
                    .map(|state| state.entropy_start_bits)
                    .unwrap_or(0) as u64,
            ),
        )?;
        option.drawer.set_metadata(
            "AV1 entropy tile count",
            DataMap::UInt(headers.tile_group.entropy_states.len() as u64),
        )?;
        if let Some(root_partition) = headers.tile_group.partition_probes.first() {
            option.drawer.set_metadata(
                "AV1 root partition symbol",
                DataMap::UInt(root_partition.symbol as u64),
            )?;
            option.drawer.set_metadata(
                "AV1 root partition",
                DataMap::Ascii(format!("{:?}", root_partition.partition)),
            )?;
        }
        if let Some(block_mode) = headers.tile_group.block_mode_probes.first() {
            option.drawer.set_metadata(
                "AV1 first block skip",
                DataMap::UInt(block_mode.skip as u64),
            )?;
            option.drawer.set_metadata(
                "AV1 first block y mode",
                DataMap::Ascii(format!("{:?}", block_mode.y_mode)),
            )?;
            if let Some(uv_mode) = block_mode.uv_mode {
                option.drawer.set_metadata(
                    "AV1 first block uv mode",
                    DataMap::Ascii(format!("{uv_mode:?}")),
                )?;
            }
            if let Some(cdef_idx) = block_mode.cdef_idx {
                option
                    .drawer
                    .set_metadata("AV1 first block cdef idx", DataMap::UInt(cdef_idx as u64))?;
            }
            let first_block_transforms = plan_transform_blocks_with_tx_size(
                0,
                0,
                0,
                block_mode.block_size,
                block_mode.tx_size,
                headers.decode_plan.width,
                headers.decode_plan.height,
            );
            option.drawer.set_metadata(
                "AV1 first block transform count",
                DataMap::UInt(first_block_transforms.len() as u64),
            )?;
            if let Some(first_transform) = first_block_transforms.first() {
                option.drawer.set_metadata(
                    "AV1 first transform size",
                    DataMap::Ascii(format!("{:?}", first_transform.tx_size)),
                )?;
            }
        }
        if let Some(residual) = headers.tile_group.residual_probes.first() {
            option.drawer.set_metadata(
                "AV1 first block residual skipped",
                DataMap::UInt(residual.skipped as u64),
            )?;
            option.drawer.set_metadata(
                "AV1 first block zero transform count",
                DataMap::UInt(residual.zero_transform_count as u64),
            )?;
            if let Some(context) = residual.txb_skip_context {
                option.drawer.set_metadata(
                    "AV1 first transform txb skip context",
                    DataMap::UInt(context as u64),
                )?;
            }
            if let Some(symbol) = residual.all_zero_symbol {
                option.drawer.set_metadata(
                    "AV1 first transform all zero symbol",
                    DataMap::UInt(symbol as u64),
                )?;
            }
            option.drawer.set_metadata(
                "AV1 first transform all zero",
                DataMap::UInt(residual.first_transform_all_zero as u64),
            )?;
            if let Some(index) = residual.first_non_zero_transform_index {
                option.drawer.set_metadata(
                    "AV1 first non-zero transform index",
                    DataMap::UInt(index as u64),
                )?;
            }
            if let Some(tx_size) = residual.first_non_zero_tx_size {
                option.drawer.set_metadata(
                    "AV1 first non-zero transform size",
                    DataMap::Ascii(format!("{tx_size:?}")),
                )?;
            }
            option.drawer.set_metadata(
                "AV1 first transform tx type read",
                DataMap::UInt(residual.tx_type_read as u64),
            )?;
            if let Some(set) = residual.tx_type_set {
                option
                    .drawer
                    .set_metadata("AV1 first transform tx type set", DataMap::UInt(set as u64))?;
            }
            if let Some(symbol) = residual.tx_type_symbol {
                option.drawer.set_metadata(
                    "AV1 first transform tx type symbol",
                    DataMap::UInt(symbol as u64),
                )?;
            }
            if let Some(tx_type) = residual.tx_type {
                option.drawer.set_metadata(
                    "AV1 first transform tx type",
                    DataMap::Ascii(format!("{tx_type:?}")),
                )?;
            }
            if let Some(eob_multisize) = residual.eob_multisize {
                option.drawer.set_metadata(
                    "AV1 first transform eob multisize",
                    DataMap::UInt(eob_multisize as u64),
                )?;
            }
            if let Some(eob_pt_symbol) = residual.eob_pt_symbol {
                option.drawer.set_metadata(
                    "AV1 first transform eob pt symbol",
                    DataMap::UInt(eob_pt_symbol as u64),
                )?;
            }
            if let Some(eob_pt) = residual.eob_pt {
                option
                    .drawer
                    .set_metadata("AV1 first transform eob pt", DataMap::UInt(eob_pt as u64))?;
            }
            if let Some(eob_base) = residual.eob_base {
                option.drawer.set_metadata(
                    "AV1 first transform eob base",
                    DataMap::UInt(eob_base as u64),
                )?;
            }
            if let Some(context) = residual.eob_extra_context {
                option.drawer.set_metadata(
                    "AV1 first transform eob extra context",
                    DataMap::UInt(context as u64),
                )?;
            }
            if let Some(symbol) = residual.eob_extra_symbol {
                option.drawer.set_metadata(
                    "AV1 first transform eob extra symbol",
                    DataMap::UInt(symbol as u64),
                )?;
            }
            if let Some(bits) = residual.eob_extra_literal_bits {
                option.drawer.set_metadata(
                    "AV1 first transform eob extra literal bits",
                    DataMap::UInt(bits as u64),
                )?;
            }
            if let Some(eob) = residual.eob {
                option
                    .drawer
                    .set_metadata("AV1 first transform eob", DataMap::UInt(eob as u64))?;
            }
            if let Some(context) = residual.coeff_base_eob_context {
                option.drawer.set_metadata(
                    "AV1 first transform coeff base eob context",
                    DataMap::UInt(context as u64),
                )?;
            }
            if let Some(symbol) = residual.coeff_base_eob_symbol {
                option.drawer.set_metadata(
                    "AV1 first transform coeff base eob symbol",
                    DataMap::UInt(symbol as u64),
                )?;
            }
            if let Some(level) = residual.coeff_base_eob_level {
                option.drawer.set_metadata(
                    "AV1 first transform coeff base eob level",
                    DataMap::UInt(level as u64),
                )?;
            }
            if let Some(count) = residual.regular_coeff_base_count {
                option.drawer.set_metadata(
                    "AV1 first transform regular coeff base count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(count) = residual.regular_coeff_base_decoded_count {
                option.drawer.set_metadata(
                    "AV1 first transform regular coeff base decoded count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(count) = residual.coeff_base_non_zero_count {
                option.drawer.set_metadata(
                    "AV1 first transform coeff base non-zero count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(count) = residual.coeff_base_range_count {
                option.drawer.set_metadata(
                    "AV1 first transform coeff base range count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(count) = residual.coeff_br_decoded_count {
                option.drawer.set_metadata(
                    "AV1 first transform coeff br decoded count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(scan_index) = residual.first_coeff_br_scan_index {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff br scan index",
                    DataMap::UInt(scan_index as u64),
                )?;
            }
            if let Some(position) = residual.first_coeff_br_position {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff br position",
                    DataMap::UInt(position as u64),
                )?;
            }
            if let Some(context) = residual.first_coeff_br_context {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff br context",
                    DataMap::UInt(context as u64),
                )?;
            }
            if let Some(symbol) = residual.first_coeff_br_symbol {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff br symbol",
                    DataMap::UInt(symbol as u64),
                )?;
            }
            if let Some(level) = residual.first_coeff_br_level {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff br level",
                    DataMap::UInt(level as u64),
                )?;
            }
            if let Some(count) = residual.sign_decoded_count {
                option.drawer.set_metadata(
                    "AV1 first transform coeff sign decoded count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(context) = residual.dc_sign_context {
                option.drawer.set_metadata(
                    "AV1 first transform dc sign context",
                    DataMap::UInt(context as u64),
                )?;
            }
            if let Some(symbol) = residual.dc_sign_symbol {
                option.drawer.set_metadata(
                    "AV1 first transform dc sign symbol",
                    DataMap::UInt(symbol as u64),
                )?;
            }
            if let Some(scan_index) = residual.first_ac_sign_scan_index {
                option.drawer.set_metadata(
                    "AV1 first transform first ac sign scan index",
                    DataMap::UInt(scan_index as u64),
                )?;
            }
            if let Some(bit) = residual.first_ac_sign_bit {
                option.drawer.set_metadata(
                    "AV1 first transform first ac sign bit",
                    DataMap::UInt(bit as u64),
                )?;
            }
            if let Some(count) = residual.golomb_decoded_count {
                option.drawer.set_metadata(
                    "AV1 first transform coeff golomb decoded count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(scan_index) = residual.first_golomb_scan_index {
                option.drawer.set_metadata(
                    "AV1 first transform first golomb scan index",
                    DataMap::UInt(scan_index as u64),
                )?;
            }
            if let Some(value) = residual.first_golomb_value {
                option.drawer.set_metadata(
                    "AV1 first transform first golomb value",
                    DataMap::UInt(value as u64),
                )?;
            }
            if let Some(count) = residual.signed_coeff_non_zero_count {
                option.drawer.set_metadata(
                    "AV1 first transform signed coeff non-zero count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(scan_index) = residual.first_signed_coeff_scan_index {
                option.drawer.set_metadata(
                    "AV1 first transform first signed coeff scan index",
                    DataMap::UInt(scan_index as u64),
                )?;
            }
            if let Some(position) = residual.first_signed_coeff_position {
                option.drawer.set_metadata(
                    "AV1 first transform first signed coeff position",
                    DataMap::UInt(position as u64),
                )?;
            }
            if let Some(value) = residual.first_signed_coeff_value {
                option.drawer.set_metadata(
                    "AV1 first transform first signed coeff value",
                    DataMap::Ascii(value.to_string()),
                )?;
            }
            if let Some(count) = residual.dequant_non_zero_count {
                option.drawer.set_metadata(
                    "AV1 first transform dequant non-zero count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(position) = residual.first_dequant_coeff_position {
                option.drawer.set_metadata(
                    "AV1 first transform first dequant coeff position",
                    DataMap::UInt(position as u64),
                )?;
            }
            if let Some(value) = residual.first_dequant_coeff_value {
                option.drawer.set_metadata(
                    "AV1 first transform first dequant coeff value",
                    DataMap::Ascii(value.to_string()),
                )?;
            }
            if let Some(tx_type) = residual.residual_preview_tx_type {
                option.drawer.set_metadata(
                    "AV1 first transform residual preview tx type",
                    DataMap::Ascii(format!("{tx_type:?}")),
                )?;
            }
            if let Some(count) = residual.residual_preview_sample_count {
                option.drawer.set_metadata(
                    "AV1 first transform residual preview sample count",
                    DataMap::UInt(count as u64),
                )?;
            }
            if let Some(value) = residual.first_residual_preview_sample {
                option.drawer.set_metadata(
                    "AV1 first transform first residual preview sample",
                    DataMap::Ascii(value.to_string()),
                )?;
            }
            if let Some(scan_index) = residual.first_coeff_base_scan_index {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff base scan index",
                    DataMap::UInt(scan_index as u64),
                )?;
            }
            if let Some(position) = residual.first_coeff_base_position {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff base position",
                    DataMap::UInt(position as u64),
                )?;
            }
            if let Some(context) = residual.first_coeff_base_context {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff base context",
                    DataMap::UInt(context as u64),
                )?;
            }
            if let Some(magnitude) = residual.first_coeff_base_reference_magnitude {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff base reference magnitude",
                    DataMap::UInt(magnitude as u64),
                )?;
            }
            if let Some(symbol) = residual.first_coeff_base_symbol {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff base symbol",
                    DataMap::UInt(symbol as u64),
                )?;
            }
            if let Some(level) = residual.first_coeff_base_level {
                option.drawer.set_metadata(
                    "AV1 first transform first coeff base level",
                    DataMap::UInt(level as u64),
                )?;
            }
            option.drawer.set_metadata(
                "AV1 first block residual bit position",
                DataMap::UInt(residual.bit_position_after as u64),
            )?;
        }
        option.drawer.set_metadata(
            "AV1 plane count",
            DataMap::UInt(headers.decode_plan.planes.len() as u64),
        )?;
        option.drawer.set_metadata(
            "AV1 decode tile count",
            DataMap::UInt(headers.decode_plan.tiles.len() as u64),
        )?;
        if let Some(first_tile) = headers.decode_plan.tiles.first() {
            option.drawer.set_metadata(
                "AV1 first tile pixel width",
                DataMap::UInt(first_tile.pixel_width as u64),
            )?;
            option.drawer.set_metadata(
                "AV1 first tile pixel height",
                DataMap::UInt(first_tile.pixel_height as u64),
            )?;
        }
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

#[cfg(test)]
mod color_metadata_tests {
    use super::*;
    use crate::av1::ColorDescription;

    fn config() -> ColorConfig {
        ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 0,
            }),
            color_range: ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        }
    }

    fn nclx(primaries: u16, transfer: u16, matrix: u16, full_range: bool) -> ColorInformation {
        ColorInformation {
            color_type: *b"nclx",
            payload: vec![
                (primaries >> 8) as u8,
                primaries as u8,
                (transfer >> 8) as u8,
                transfer as u8,
                (matrix >> 8) as u8,
                matrix as u8,
                if full_range { 0x80 } else { 0 },
            ],
        }
    }

    #[test]
    fn color_metadata_accepts_unspecified_nclx_codes() {
        validate_color_metadata(Some(&nclx(2, 2, 2, true)), &config()).unwrap();
    }

    #[test]
    fn extended_pixi_must_match_av1_subsampling() {
        let mut color = config();
        color.subsampling_x = true;
        color.subsampling_y = true;
        color.chroma_sample_position = Some(ChromaSamplePosition::Vertical);
        let pixi = PixelInformation {
            bits_per_channel: vec![8, 8, 8],
            extended_channels: Some(vec![
                crate::container::PixelChannelInformation {
                    channel_idc: 0,
                    component_format: 0,
                    subsampling: Some(crate::container::PixelSubsampling {
                        subsampling_type: 0,
                        subsampling_location: 0,
                    }),
                },
                crate::container::PixelChannelInformation {
                    channel_idc: 0,
                    component_format: 0,
                    subsampling: Some(crate::container::PixelSubsampling {
                        subsampling_type: 2,
                        subsampling_location: 0,
                    }),
                },
                crate::container::PixelChannelInformation {
                    channel_idc: 0,
                    component_format: 0,
                    subsampling: Some(crate::container::PixelSubsampling {
                        subsampling_type: 2,
                        subsampling_location: 0,
                    }),
                },
            ]),
        };
        validate_extended_pixi(Some(&pixi), &color).unwrap();

        let mut mismatch = pixi.clone();
        mismatch.extended_channels.as_mut().unwrap()[1]
            .subsampling
            .as_mut()
            .unwrap()
            .subsampling_location = 2;
        assert!(matches!(
            validate_extended_pixi(Some(&mismatch), &color),
            Err(DecoderError::Bitstream(message)) if message.contains("location")
        ));
    }

    #[test]
    fn color_metadata_rejects_explicit_nclx_mismatch() {
        let error = validate_color_metadata(Some(&nclx(9, 13, 0, true)), &config()).unwrap_err();
        assert!(
            matches!(error, DecoderError::Bitstream(message) if message.contains("does not match"))
        );
    }
}

#[cfg(test)]
mod premultiplied_alpha_tests {
    use super::*;
    use crate::av1::ColorDescription;

    fn plane(index: u8, sample: u16) -> PlaneBuffer {
        PlaneBuffer {
            layout: PlaneLayout {
                plane: index,
                width: 1,
                height: 1,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 1,
            },
            samples: vec![sample],
        }
    }

    fn frame() -> DecodedFrame {
        DecodedFrame {
            width: 1,
            height: 1,
            render_width: 1,
            render_height: 1,
            bit_depth: 8,
            color_config: ColorConfig {
                high_bitdepth: false,
                twelve_bit: false,
                bit_depth: 8,
                monochrome: false,
                color_description: Some(ColorDescription {
                    color_primaries: 1,
                    transfer_characteristics: 13,
                    matrix_coefficients: 0,
                }),
                color_range: ColorRange::Full,
                subsampling_x: false,
                subsampling_y: false,
                chroma_sample_position: None,
                separate_uv_delta_q: false,
            },
            color_information: None,
            alpha_premultiplied: true,
            buffers: FrameBuffers {
                width: 1,
                height: 1,
                // Native identity GBR order is G, B, R, alpha.
                planes: vec![plane(0, 64), plane(1, 32), plane(2, 16), plane(3, 128)],
            },
        }
    }

    #[test]
    fn rgba_outputs_unpremultiplied_primary_channels() {
        let frame = frame();
        let rgba16 = frame.to_rgba16().unwrap();
        assert_eq!(rgba16.rgba, vec![8192, 32768, 16384, 32896]);
        let rgba8 = frame.to_rgba8().unwrap();
        assert_eq!(rgba8.rgba, vec![32, 128, 64, 128]);
    }

    #[test]
    fn zero_alpha_clears_unpremultiplied_channels() {
        let mut rgba8 = vec![255, 127, 1, 0];
        unpremultiply_rgba8(&mut rgba8);
        assert_eq!(rgba8, vec![0, 0, 0, 0]);
        let mut rgba16 = vec![u16::MAX, 32768, 1, 0];
        unpremultiply_rgba16(&mut rgba16);
        assert_eq!(rgba16, vec![0, 0, 0, 0]);
    }
}

#[cfg(test)]
mod av1_config_tests {
    use super::*;
    use crate::av1::ChromaSamplePosition;

    fn av1_config(position: ChromaSamplePosition) -> Av1CodecConfiguration {
        Av1CodecConfiguration {
            version: 1,
            seq_profile: 0,
            seq_level_idx_0: 5,
            seq_tier_0: false,
            high_bitdepth: false,
            twelve_bit: false,
            monochrome: false,
            chroma_subsampling_x: true,
            chroma_subsampling_y: true,
            chroma_sample_position: position,
            initial_presentation_delay: None,
        }
    }

    fn sequence(position: ChromaSamplePosition) -> SequenceHeader {
        SequenceHeader {
            seq_profile: 0,
            still_picture: true,
            reduced_still_picture_header: true,
            seq_level_idx_0: 5,
            frame_width_bits: 8,
            frame_height_bits: 8,
            max_frame_width: 64,
            max_frame_height: 64,
            frame_id_numbers_present: false,
            frame_id_length: 0,
            delta_frame_id_length: 0,
            use_128x128_superblock: false,
            enable_filter_intra: false,
            enable_intra_edge_filter: false,
            enable_order_hint: false,
            enable_dual_filter: false,
            enable_masked_compound: false,
            enable_interintra_compound: false,
            enable_dist_wtd_comp: false,
            enable_warped_motion: false,
            order_hint_bits: 0,
            seq_force_screen_content_tools: 0,
            seq_force_integer_mv: 0,
            enable_ref_frame_mvs: false,
            enable_superres: false,
            enable_cdef: false,
            enable_restoration: false,
            color_config: ColorConfig {
                high_bitdepth: false,
                twelve_bit: false,
                bit_depth: 8,
                monochrome: false,
                color_description: None,
                color_range: ColorRange::Full,
                subsampling_x: true,
                subsampling_y: true,
                chroma_sample_position: Some(position),
                separate_uv_delta_q: false,
            },
            film_grain_params_present: false,
        }
    }

    #[test]
    fn av1_config_rejects_chroma_sample_position_mismatch() {
        let error = validate_av1_config(
            &av1_config(ChromaSamplePosition::Colocated),
            &sequence(ChromaSamplePosition::Vertical),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            DecoderError::Bitstream(message) if message.contains("chroma sample position")
        ));
    }

    #[test]
    fn av1_config_accepts_matching_chroma_sample_position() {
        validate_av1_config(
            &av1_config(ChromaSamplePosition::Vertical),
            &sequence(ChromaSamplePosition::Vertical),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod prefilter_diagnostic_tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    #[ignore = "diagnostic comparison of raw Rust planes against generated final FFmpeg planes"]
    fn reports_wml2viewer_raw_against_generated_final_planes() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let avif_path = root.parent().unwrap().join("samples/WML2Viewer.avif");
        let plane_paths = [
            root.join("test_data/planes/WML2Viewer.y.u16le"),
            root.join("test_data/planes/WML2Viewer.u.u16le"),
            root.join("test_data/planes/WML2Viewer.v.u16le"),
        ];
        if !avif_path.exists() || plane_paths.iter().any(|path| !path.exists()) {
            eprintln!("WML2Viewer diagnostic fixtures are unavailable");
            return;
        }
        let data = std::fs::read(avif_path).unwrap();
        let info = parse_avif(&data).unwrap();
        let headers = parse_av1_headers(&info).unwrap();
        eprintln!(
            "diagnostic frame base_q_idx={} disable_cdf_update={} quant={:?}",
            headers.frame.base_q_idx, headers.frame.disable_cdf_update, headers.frame.quantization
        );
        let mut diagnostic_buffers = alloc_frame_buffers(&headers.decode_plan).unwrap();
        let (prefix, _) = decode_luma_root_block_prefix_with_post_filter_state_and_entropy(
            &headers.tile_group.tile_data,
            &headers.tile_group.group,
            &headers.sequence,
            &headers.frame,
            &headers.decode_plan,
            &mut diagnostic_buffers,
            usize::MAX,
            false,
        )
        .unwrap();
        // Default to the current first luma mismatch against the AOM build
        // with all post-filters disabled, while allowing an upstream edge to
        // be selected without editing the diagnostic.
        let target_x = std::env::var("AVIF_DIAGNOSTIC_X")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(634);
        let target_y = std::env::var("AVIF_DIAGNOSTIC_Y")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(18);
        for block in prefix.blocks.iter().filter(|block| {
            block.x <= target_x
                && target_x < block.x + block.block_size.width()
                && block.y <= target_y
                && target_y < block.y + block.block_size.height()
        }) {
            eprintln!(
                "diagnostic block ({}, {}) size={:?} palette={:?} transforms={:?}",
                block.x,
                block.y,
                block.block_size,
                block.palette,
                block
                    .transforms
                    .iter()
                    .map(|transform| {
                        (
                            transform.transform.x,
                            transform.transform.y,
                            transform.transform.tx_size,
                            transform.tx_type,
                            transform
                                .coefficients
                                .iter()
                                .enumerate()
                                .filter_map(|(index, coefficient)| {
                                    (*coefficient != 0).then_some((index, *coefficient))
                                })
                                .collect::<Vec<_>>(),
                            transform
                                .coefficients
                                .iter()
                                .take(8)
                                .copied()
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            );
        }
        let decoded = decode_still_frame_prefilter_for_test(&headers, Some(&info)).unwrap();
        for (plane_index, path) in plane_paths.iter().enumerate() {
            let expected = std::fs::read(path).unwrap();
            let expected = expected
                .chunks_exact(2)
                .map(|sample| u16::from_le_bytes([sample[0], sample[1]]));
            let actual = &decoded.buffers.planes[plane_index].samples;
            let mut first = None;
            let mut mismatches = 0usize;
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if *actual != expected {
                    first.get_or_insert(index);
                    mismatches += 1;
                }
            }
            eprintln!("raw-vs-final plane {plane_index}: first={first:?}, mismatches={mismatches}");
            if let Some(index) = first {
                let width = decoded.buffers.planes[plane_index].layout.width;
                let row_start = index / width * width;
                let start = index.saturating_sub(4).max(row_start);
                let end = (index + 12).min(row_start + width);
                let expected = std::fs::read(path).unwrap();
                let expected = expected
                    .chunks_exact(2)
                    .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
                    .collect::<Vec<_>>();
                eprintln!(
                    "raw-vs-final plane {plane_index} window {start}..{end}: actual={:?}, expected={:?}",
                    &actual[start..end],
                    &expected[start..end]
                );
            }
        }
    }

    #[test]
    #[ignore = "requires AOM_PREFILTER_ORACLE from an AOM build with post-filters disabled"]
    fn reports_wml2viewer_against_aom_prefilter_oracle() {
        let Some(oracle_path) = std::env::var_os("AOM_PREFILTER_ORACLE") else {
            eprintln!("AOM_PREFILTER_ORACLE is unavailable");
            return;
        };
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let avif_path = root.parent().unwrap().join("samples/WML2Viewer.avif");
        if !avif_path.exists() {
            eprintln!("WML2Viewer sample is unavailable");
            return;
        }
        let data = std::fs::read(avif_path).unwrap();
        let info = parse_avif(&data).unwrap();
        let headers = parse_av1_headers(&info).unwrap();
        let decoded = decode_still_frame_prefilter_for_test(&headers, Some(&info)).unwrap();
        if let Some(output_path) = std::env::var_os("AVIF_RUST_PREFILTER_OUTPUT") {
            let output = decoded
                .buffers
                .planes
                .iter()
                .flat_map(|plane| {
                    plane.samples.iter().map(|sample| {
                        u8::try_from(*sample).expect("8-bit diagnostic sample should fit in u8")
                    })
                })
                .collect::<Vec<_>>();
            std::fs::write(output_path, output).unwrap();
        }
        let oracle = std::fs::read(oracle_path).unwrap();
        let sample_count = decoded.buffers.width * decoded.buffers.height;
        assert_eq!(oracle.len(), sample_count * decoded.buffers.planes.len());

        for (plane_index, plane) in decoded.buffers.planes.iter().enumerate() {
            let expected = &oracle[plane_index * sample_count..(plane_index + 1) * sample_count];
            let mut first = None;
            let mut mismatches = 0usize;
            for (index, (&actual, &expected)) in
                plane.samples.iter().zip(expected.iter()).enumerate()
            {
                if actual != u16::from(expected) {
                    first.get_or_insert(index);
                    mismatches += 1;
                }
            }
            eprintln!(
                "AOM prefilter plane {plane_index}: first={first:?}, mismatches={mismatches}"
            );
            if let Some(index) = first {
                let row_start = index / plane.layout.width * plane.layout.width;
                let start = index.saturating_sub(4).max(row_start);
                let end = (index + 12).min(row_start + plane.layout.width);
                eprintln!(
                    "AOM prefilter plane {plane_index} window {start}..{end}: actual={:?}, expected={:?}",
                    &plane.samples[start..end],
                    &expected[start..end]
                );
            }
        }
    }

    #[test]
    #[ignore = "diagnostic comparison against FFmpeg with loop filters disabled"]
    fn reports_wml2viewer_against_ffmpeg_prefilter() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let avif_path = root.parent().unwrap().join("samples/WML2Viewer.avif");
        if !avif_path.exists() {
            eprintln!("WML2Viewer sample is unavailable");
            return;
        }
        let data = std::fs::read(&avif_path).unwrap();
        let info = parse_avif(&data).unwrap();
        let headers = parse_av1_headers(&info).unwrap();
        let decoded = decode_still_frame_prefilter_for_test(&headers, Some(&info)).unwrap();
        let reference = Command::new("ffmpeg")
            .args(["-v", "error", "-skip_loop_filter", "all", "-nostdin", "-i"])
            .arg(&avif_path)
            .args([
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv444p",
                "-",
            ])
            .output()
            .expect("ffmpeg should be available for the diagnostic oracle");
        assert!(reference.status.success());
        let sample_count = decoded.buffers.width * decoded.buffers.height;
        assert_eq!(reference.stdout.len(), sample_count * 3);
        for plane_index in 0..3 {
            let expected =
                &reference.stdout[plane_index * sample_count..(plane_index + 1) * sample_count];
            let actual = &decoded.buffers.planes[plane_index].samples;
            let mismatches = actual
                .iter()
                .zip(expected)
                .filter(|(actual, expected)| **actual != u16::from(**expected))
                .count();
            eprintln!("ffmpeg prefilter plane {plane_index}: mismatches={mismatches}");
        }
    }

    #[test]
    #[ignore = "diagnostic comparison of external high-bit-depth filter stages"]
    fn reports_external_12bit_filter_stages() {
        let Some(avif_path) = std::env::var_os("AVIF_12BIT_PATH").map(PathBuf::from) else {
            eprintln!("AVIF_12BIT_PATH is unavailable");
            return;
        };
        if !avif_path.exists() {
            eprintln!("12-bit AVIF diagnostic sample is unavailable: {avif_path:?}");
            return;
        }
        let data = std::fs::read(&avif_path).unwrap();
        let info = parse_avif(&data).unwrap();
        let headers = parse_av1_headers(&info).unwrap();
        eprintln!(
            "12-bit headers: bit_depth={}, monochrome={}, subsampling=({}, {}), matrix={:?}, range={:?}, quant={:?}, film_grain={}",
            headers.sequence.color_config.bit_depth,
            headers.sequence.color_config.monochrome,
            headers.sequence.color_config.subsampling_x,
            headers.sequence.color_config.subsampling_y,
            headers
                .sequence
                .color_config
                .color_description
                .map(|description| description.matrix_coefficients),
            headers.sequence.color_config.color_range,
            headers.frame.quantization,
            headers.frame.film_grain.is_some()
        );
        let DecodedStillFrame {
            frame,
            post_filter_state,
            ..
        } = decode_still_frame_with_filter_policy_and_state(&headers, Some(&info), false).unwrap();
        let sample_count = frame.width * frame.height;
        let reference = |skip_loop_filter: bool| {
            let mut command = Command::new("ffmpeg");
            command.args(["-v", "error", "-nostdin"]);
            if skip_loop_filter {
                command.args(["-skip_loop_filter", "all"]);
            }
            command.arg("-i").arg(&avif_path).args([
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv444p12le",
                "-",
            ]);
            let output = command.output().expect("ffmpeg should be available");
            assert!(
                output.status.success(),
                "ffmpeg failed: {:?}",
                output.status
            );
            assert_eq!(output.stdout.len(), sample_count * 3 * 2);
            output
                .stdout
                .chunks_exact(2)
                .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
                .collect::<Vec<_>>()
        };
        let prefilter = reference(true);
        let filtered = reference(false);
        let report = |label: &str, current: &DecodedFrame, expected: &[u16]| {
            for (plane_index, plane) in current.buffers.planes.iter().enumerate() {
                let start = plane_index * sample_count;
                let expected = &expected[start..start + sample_count];
                let mut sum = 0u64;
                let mut max = 0u16;
                let mut actual_min = u16::MAX;
                let mut actual_max = 0u16;
                let mut expected_min = u16::MAX;
                let mut expected_max = 0u16;
                for (&actual, &expected) in plane.samples.iter().zip(expected) {
                    let difference = actual.abs_diff(expected);
                    sum += u64::from(difference);
                    max = max.max(difference);
                    actual_min = actual_min.min(actual);
                    actual_max = actual_max.max(actual);
                    expected_min = expected_min.min(expected);
                    expected_max = expected_max.max(expected);
                }
                eprintln!(
                    "12-bit {label} plane {plane_index}: average_abs={}, max={max}, actual={actual_min}..{actual_max}, expected={expected_min}..{expected_max}, first={:?}/{:?}",
                    sum as f64 / sample_count as f64,
                    &plane.samples[..4],
                    &expected[..4]
                );
            }
        };
        report("prefilter-vs-ffmpeg-prefilter", &frame, &prefilter);
        let mut deblock = frame.clone();
        apply_deblock_stage(&mut deblock, &headers.frame, &post_filter_state);
        report("deblock-vs-ffmpeg-final", &deblock, &filtered);
        let mut cdef = deblock.clone();
        apply_cdef_stage(&mut cdef, &headers.frame, &post_filter_state);
        report("cdef-vs-ffmpeg-final", &cdef, &filtered);
        apply_loop_restoration_stage(
            &mut cdef,
            &post_filter_state,
            headers.decode_plan.superblock_size << headers.frame.restoration.unit_shift,
            &[1, 2],
        );
        report("restoration-vs-ffmpeg-final", &cdef, &filtered);
    }

    #[test]
    fn wml2viewer_post_filter_stages_match_strict_plane_oracle() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let avif_path = root.join("test_data/images/WML2Viewer.avif");
        let data = std::fs::read(&avif_path).expect("WML2Viewer AVIF should be readable");
        let info = parse_avif(&data).expect("WML2Viewer AVIF should parse");
        let headers = parse_av1_headers(&info).expect("WML2Viewer AV1 headers should parse");
        let DecodedStillFrame {
            frame: raw,
            post_filter_state,
            ..
        } = decode_still_frame_with_filter_policy_and_state_and_references(
            &headers,
            Some(&info),
            true,
            std::array::from_fn(|_| None),
        )
        .expect("WML2Viewer raw reconstruction should decode");

        assert!(
            !post_filter_state.cdef_blocks.is_empty(),
            "the strict sample must retain decoded CDEF block indices"
        );
        assert!(
            !post_filter_state.restoration_units.is_empty(),
            "the strict sample must retain decoded restoration units"
        );

        let mut deblocked = raw.clone();
        apply_deblock_stage(&mut deblocked, &headers.frame, &post_filter_state);
        let deblock_changes = raw
            .buffers
            .planes
            .iter()
            .zip(&deblocked.buffers.planes)
            .map(|(before, after)| {
                before
                    .samples
                    .iter()
                    .zip(&after.samples)
                    .filter(|(before, after)| before != after)
                    .count()
            })
            .sum::<usize>();
        assert!(
            deblock_changes > 0,
            "deblock stage must change the strict sample"
        );

        let restoration_boundaries = headers
            .frame
            .restoration
            .uses_lr
            .then(|| capture_restoration_boundary_rows(&deblocked));
        let mut cdef = deblocked.clone();
        apply_cdef_stage(&mut cdef, &headers.frame, &post_filter_state);
        let cdef_changes = deblocked
            .buffers
            .planes
            .iter()
            .zip(&cdef.buffers.planes)
            .map(|(before, after)| {
                before
                    .samples
                    .iter()
                    .zip(&after.samples)
                    .filter(|(before, after)| before != after)
                    .count()
            })
            .sum::<usize>();
        assert!(
            cdef_changes > 0,
            "CDEF stage must apply retained block indices"
        );

        let before_restoration = cdef.clone();
        apply_loop_restoration_stage_with_boundaries(
            &mut cdef,
            &post_filter_state,
            headers.decode_plan.superblock_size << headers.frame.restoration.unit_shift,
            &[1, 2],
            restoration_boundaries.as_deref(),
        );
        crop_frame_buffers_to_plan(&mut cdef.buffers, &headers.decode_plan)
            .expect("WML2Viewer filtered buffers should crop");
        let restoration_changes = before_restoration
            .buffers
            .planes
            .iter()
            .zip(&cdef.buffers.planes)
            .map(|(before, after)| {
                before
                    .samples
                    .iter()
                    .zip(&after.samples)
                    .filter(|(before, after)| before != after)
                    .count()
            })
            .sum::<usize>();
        assert!(
            restoration_changes > 0,
            "loop restoration stage must change the strict sample"
        );

        for (plane_index, plane) in cdef.buffers.planes.iter().enumerate() {
            let plane_name = ["y", "u", "v"]
                .get(plane_index)
                .unwrap_or_else(|| panic!("unexpected WML2Viewer plane {plane_index}"));
            let path = root.join(format!("test_data/planes/WML2Viewer.{plane_name}.u16le"));
            let expected = std::fs::read(&path)
                .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
            let expected = expected
                .chunks_exact(2)
                .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
                .collect::<Vec<_>>();
            assert_eq!(
                plane.samples, expected,
                "WML2Viewer final plane {plane_index} must match the strict oracle"
            );
        }
    }

    #[test]
    fn truncated_post_filter_sample_fails_closed_without_partial_rgba() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let data = std::fs::read(root.join("test_data/images/WML2Viewer.avif"))
            .expect("WML2Viewer AVIF should be readable");
        let failures = (1..=64)
            .filter(|trim| crate::image_from_bytes(&data[..data.len() - trim]).is_err())
            .count();
        assert!(
            failures >= 32,
            "truncating the filtered AVIF should fail closed for most suffixes, got {failures}/64"
        );
    }

    #[test]
    #[ignore = "diagnostic execution of the private post-filter pipeline"]
    fn runs_wml2viewer_private_post_filter_pipeline() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let avif_path = root.parent().unwrap().join("samples/WML2Viewer.avif");
        if !avif_path.exists() {
            eprintln!("WML2Viewer sample is unavailable");
            return;
        }
        let data = std::fs::read(avif_path).unwrap();
        let info = parse_avif(&data).unwrap();
        let headers = parse_av1_headers(&info).unwrap();
        let DecodedStillFrame {
            mut frame,
            post_filter_state,
            ..
        } = decode_still_frame_with_filter_policy_and_state(&headers, Some(&info), false).unwrap();
        let reference = Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(root.parent().unwrap().join("samples/WML2Viewer.avif"))
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
            .output()
            .ok()
            .filter(|output| output.status.success());
        let plane_reference = Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(root.parent().unwrap().join("samples/WML2Viewer.avif"))
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gbrp", "-"])
            .output()
            .ok()
            .filter(|output| output.status.success());
        let cdef_oracle = std::env::var_os("AOM_CDEF_ORACLE")
            .map(PathBuf::from)
            .and_then(|path| std::fs::read(path).ok());
        let deblock_cdef_oracle = std::env::var_os("AOM_DEBLOCK_CDEF_ORACLE")
            .map(PathBuf::from)
            .and_then(|path| std::fs::read(path).ok());
        let deblock_oracle = std::env::var_os("AOM_DEBLOCK_ORACLE")
            .map(PathBuf::from)
            .and_then(|path| std::fs::read(path).ok());
        let restoration_oracle = std::env::var_os("AOM_RESTORATION_ORACLE")
            .map(PathBuf::from)
            .and_then(|path| std::fs::read(path).ok());
        let report = |label: &str, frame: &DecodedFrame| {
            let Some(reference) = reference.as_ref() else {
                return;
            };
            let rgba = frame.to_rgba8().unwrap();
            if reference.stdout.len() != rgba.rgba.len() {
                return;
            }
            let mut sum = 0u64;
            let mut max = 0u8;
            for (index, (&actual, &expected)) in
                rgba.rgba.iter().zip(reference.stdout.iter()).enumerate()
            {
                if index % 4 == 3 {
                    continue;
                }
                let difference = actual.abs_diff(expected);
                sum += u64::from(difference);
                max = max.max(difference);
            }
            eprintln!(
                "private {label} WML2Viewer vs ffmpeg: average_rgb_abs={}, max={max}",
                sum as f64 / (frame.width * frame.height * 3) as f64
            );
            let Some(plane_reference) = plane_reference.as_ref() else {
                return;
            };
            let sample_count = frame.width * frame.height;
            if plane_reference.stdout.len() != sample_count * 3 {
                return;
            }
            for (plane_index, plane) in frame.buffers.planes.iter().enumerate() {
                let expected = &plane_reference.stdout
                    [plane_index * sample_count..(plane_index + 1) * sample_count];
                let mut mismatches = 0usize;
                let mut sum = 0u64;
                let mut worst = (0usize, 0u16);
                for (index, (&actual, &expected)) in plane.samples.iter().zip(expected).enumerate()
                {
                    let difference = actual.abs_diff(u16::from(expected));
                    mismatches += usize::from(difference != 0);
                    sum += u64::from(difference);
                    if difference > worst.1 {
                        worst = (index, difference);
                    }
                }
                eprintln!(
                    "private {label} plane {plane_index}: mismatches={mismatches}, average_abs={}, max={} at ({},{})",
                    sum as f64 / sample_count as f64,
                    worst.1,
                    worst.0 % frame.width,
                    worst.0 / frame.width
                );
            }
        };
        let report_cdef_oracle = |label: &str, frame: &DecodedFrame| {
            let Some(expected) = cdef_oracle.as_ref() else {
                return;
            };
            let sample_count = frame.width * frame.height;
            if expected.len() != sample_count * 3 {
                return;
            }
            for (plane_index, plane) in frame.buffers.planes.iter().enumerate() {
                let oracle =
                    &expected[plane_index * sample_count..(plane_index + 1) * sample_count];
                let mut mismatches = 0usize;
                let mut sum = 0u64;
                for (&actual, &expected) in plane.samples.iter().zip(oracle) {
                    let difference = actual.abs_diff(u16::from(expected));
                    mismatches += usize::from(difference != 0);
                    sum += u64::from(difference);
                }
                eprintln!(
                    "AOM CDEF oracle {label} plane {plane_index}: mismatches={mismatches}, average_abs={}",
                    sum as f64 / sample_count as f64
                );
            }
        };
        let report_deblock_cdef_oracle = |label: &str, frame: &DecodedFrame| {
            let Some(expected) = deblock_cdef_oracle.as_ref() else {
                return;
            };
            let sample_count = frame.width * frame.height;
            if expected.len() != sample_count * 3 {
                return;
            }
            for (plane_index, plane) in frame.buffers.planes.iter().enumerate() {
                let oracle =
                    &expected[plane_index * sample_count..(plane_index + 1) * sample_count];
                let mut mismatches = 0usize;
                let mut sum = 0u64;
                let mut first = None;
                for (index, (&actual, &expected)) in plane.samples.iter().zip(oracle).enumerate() {
                    let difference = actual.abs_diff(u16::from(expected));
                    mismatches += usize::from(difference != 0);
                    sum += u64::from(difference);
                    if difference != 0 && first.is_none() {
                        first = Some((index, actual, expected));
                    }
                }
                eprintln!(
                    "AOM deblock+CDEF oracle {label} plane {plane_index}: mismatches={mismatches}, average_abs={}, first={first:?}",
                    sum as f64 / sample_count as f64
                );
            }
        };
        let report_deblock_oracle = |label: &str, frame: &DecodedFrame| {
            let Some(expected) = deblock_oracle.as_ref() else {
                return;
            };
            let sample_count = frame.width * frame.height;
            if expected.len() != sample_count * 3 {
                return;
            }
            for (plane_index, plane) in frame.buffers.planes.iter().enumerate() {
                let oracle =
                    &expected[plane_index * sample_count..(plane_index + 1) * sample_count];
                let mut mismatches = 0usize;
                let mut sum = 0u64;
                let mut first = None;
                for (index, (&actual, &expected)) in plane.samples.iter().zip(oracle).enumerate() {
                    let difference = actual.abs_diff(u16::from(expected));
                    mismatches += usize::from(difference != 0);
                    sum += u64::from(difference);
                    if difference != 0 && first.is_none() {
                        first = Some((index, actual, expected));
                    }
                }
                eprintln!(
                    "AOM deblock oracle {label} plane {plane_index}: mismatches={mismatches}, average_abs={}, first={first:?}",
                    sum as f64 / sample_count as f64
                );
            }
        };
        let report_restoration_oracle = |label: &str, frame: &DecodedFrame| {
            let Some(expected) = restoration_oracle.as_ref() else {
                return;
            };
            let sample_count = frame.width * frame.height;
            if expected.len() != sample_count * 3 {
                return;
            }
            for (plane_index, plane) in frame.buffers.planes.iter().enumerate() {
                let oracle =
                    &expected[plane_index * sample_count..(plane_index + 1) * sample_count];
                let mut mismatches = 0usize;
                let mut sum = 0u64;
                let mut first = None;
                for (index, (&actual, &expected)) in plane.samples.iter().zip(oracle).enumerate() {
                    let difference = actual.abs_diff(u16::from(expected));
                    mismatches += usize::from(difference != 0);
                    sum += u64::from(difference);
                    if difference != 0 && first.is_none() {
                        first = Some((index, actual, expected));
                    }
                }
                eprintln!(
                    "AOM restoration oracle {label} plane {plane_index}: mismatches={mismatches}, average_abs={}, first={first:?}",
                    sum as f64 / sample_count as f64
                );
            }
        };
        let report_cdef_on_aom_deblock = |frame: &mut DecodedFrame| {
            let Some(path) = std::env::var_os("AOM_DEBLOCK_INPUT") else {
                return;
            };
            let Ok(expected) = std::fs::read(path) else {
                return;
            };
            let sample_count = frame.width * frame.height;
            if expected.len() != sample_count * 3 {
                return;
            }
            for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
                let oracle =
                    &expected[plane_index * sample_count..(plane_index + 1) * sample_count];
                for (sample, &value) in plane.samples.iter_mut().zip(oracle) {
                    *sample = u16::from(value);
                }
            }
            apply_cdef_stage(frame, &headers.frame, &post_filter_state);
            report_deblock_cdef_oracle("cdef-on-aom-deblock", frame);
        };
        eprintln!(
            "private post-filter state: loop={:?}, cdef_units={}, cdef_blocks={}, boundaries={}, restoration={:?}, restoration_units={}",
            headers.frame.loop_filter.levels,
            post_filter_state.cdef_units.len(),
            post_filter_state.cdef_blocks.len(),
            post_filter_state.transform_boundaries.len(),
            headers.frame.restoration,
            post_filter_state.restoration_units.len()
        );
        eprintln!(
            "private block filter state: blocks={}, skip={}",
            post_filter_state.block_filter_states.len(),
            post_filter_state
                .block_filter_states
                .iter()
                .filter(|block| block.skip)
                .count()
        );
        eprintln!(
            "private restoration unit sample={:?}",
            &post_filter_state.restoration_units
                [..post_filter_state.restoration_units.len().min(12)]
        );
        report("raw", &frame);
        report_cdef_oracle("raw", &frame);
        let mut restoration_only = frame.clone();
        apply_loop_restoration_stage(
            &mut restoration_only,
            &post_filter_state,
            headers.decode_plan.superblock_size << headers.frame.restoration.unit_shift,
            &[1, 2],
        );
        report_restoration_oracle("raw-input", &restoration_only);
        apply_deblock_stage(&mut frame, &headers.frame, &post_filter_state);
        report("deblock", &frame);
        report_deblock_oracle("deblock", &frame);
        apply_cdef_stage(&mut frame, &headers.frame, &post_filter_state);
        report("cdef", &frame);
        report_deblock_cdef_oracle("cdef", &frame);
        let mut cdef_on_aom_deblock = frame.clone();
        report_cdef_on_aom_deblock(&mut cdef_on_aom_deblock);
        let mut wiener_frame = frame.clone();
        apply_loop_restoration_stage(
            &mut wiener_frame,
            &post_filter_state,
            headers.decode_plan.superblock_size << headers.frame.restoration.unit_shift,
            &[1],
        );
        report("wiener-only", &wiener_frame);
        apply_loop_restoration_stage(
            &mut frame,
            &post_filter_state,
            headers.decode_plan.superblock_size << headers.frame.restoration.unit_shift,
            &[1, 2],
        );
        report("restoration", &frame);
        let rgba = frame.to_rgba8().unwrap();
        eprintln!(
            "private filtered WML2Viewer: {}x{}, rgba bytes={}, first={:?}",
            frame.width,
            frame.height,
            rgba.rgba.len(),
            &rgba.rgba[..4]
        );
        assert_eq!(rgba.rgba.len(), frame.width * frame.height * 4);
    }
}

#[cfg(test)]
mod tile_group_merge_tests {
    use super::*;

    fn leb128(mut value: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            bytes.push(byte);
            if value == 0 {
                return bytes;
            }
        }
    }

    fn sized_obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut obu = vec![(obu_type << 3) | 0x02];
        obu.extend(leb128(payload.len()));
        obu.extend_from_slice(payload);
        obu
    }

    fn find_box(data: &[u8], box_type: &[u8; 4]) -> (usize, usize) {
        for offset in 0..data.len().saturating_sub(8) {
            let size = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            if &data[offset + 4..offset + 8] == box_type
                && size >= 8
                && offset
                    .checked_add(size)
                    .is_some_and(|end| end <= data.len())
            {
                return (offset, size);
            }
        }
        panic!("box {:?} is missing", box_type);
    }

    fn read_uint(data: &[u8], offset: &mut usize, width: usize) -> usize {
        let end = offset.checked_add(width).unwrap();
        let bytes = data.get(*offset..end).unwrap();
        *offset = end;
        bytes
            .iter()
            .fold(0usize, |value, byte| (value << 8) | usize::from(*byte))
    }

    fn write_uint(data: &mut [u8], offset: usize, width: usize, value: usize) {
        let end = offset.checked_add(width).unwrap();
        let target = data.get_mut(offset..end).unwrap();
        for (index, byte) in target.iter_mut().enumerate() {
            *byte = (value >> ((width - index - 1) * 8)) as u8;
        }
    }

    fn primary_extent_length_offset(data: &[u8], primary_item_id: u32) -> usize {
        let (iloc, _) = find_box(data, b"iloc");
        let mut offset = iloc + 8;
        let version = data[offset];
        offset += 4;
        let sizes = data[offset];
        offset += 1;
        let offset_size = usize::from(sizes >> 4);
        let length_size = usize::from(sizes & 0x0f);
        let sizes = data[offset];
        offset += 1;
        let base_offset_size = usize::from(sizes >> 4);
        let index_size = usize::from(sizes & 0x0f);
        let item_count_width = if version >= 2 { 4 } else { 2 };
        let item_count = read_uint(data, &mut offset, item_count_width);
        for _ in 0..item_count {
            let item_id = read_uint(data, &mut offset, item_count_width) as u32;
            if version == 0 {
                offset += 2;
            } else {
                offset += 4;
            }
            offset += base_offset_size;
            let extent_count = read_uint(data, &mut offset, 2);
            for _ in 0..extent_count {
                if version >= 1 {
                    offset += index_size;
                }
                offset += offset_size;
                let length_offset = offset;
                offset += length_size;
                if item_id == primary_item_id {
                    return length_offset;
                }
            }
        }
        panic!("primary item extent is missing");
    }

    fn split_primary_frame_obu(data: &[u8]) -> Vec<u8> {
        let info = crate::container::parse_avif(data).unwrap();
        let headers = parse_av1_headers(&info).unwrap();
        let frame_payload = crate::obu::parse_obu_stream(&info.primary_item_payload)
            .unwrap()
            .into_iter()
            .find(|obu| obu.obu_type == ObuType::Frame)
            .unwrap()
            .payload;
        let frame_header_len = headers.frame.payload_after_header_offset;
        let tile_group = parse_tile_group(
            frame_payload,
            frame_header_len * 8,
            &headers.frame.tile_info,
        )
        .unwrap();
        assert_eq!(tile_group.tiles.len(), 2);
        let tiles = tile_group
            .tiles
            .iter()
            .map(|tile| frame_payload[tile.offset..tile.offset + tile.len].to_vec())
            .collect::<Vec<_>>();

        let frame_payload_offset = info
            .primary_item_payload
            .windows(frame_payload.len())
            .position(|window| window == frame_payload)
            .unwrap();
        let mut cursor = 0;
        let (frame_obu_start, frame_obu_end) = loop {
            let obu_start = cursor;
            let header = info.primary_item_payload[cursor];
            cursor += 1;
            if header & 0x04 != 0 {
                cursor += 1;
            }
            let mut payload_len = 0usize;
            let mut shift = 0;
            while header & 0x02 != 0 {
                let byte = info.primary_item_payload[cursor];
                cursor += 1;
                payload_len |= usize::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            let payload_start = cursor;
            let payload_end = payload_start + payload_len;
            if payload_start == frame_payload_offset {
                break (obu_start, payload_end);
            }
            cursor = payload_end;
        };
        let mut primary_payload = Vec::new();
        primary_payload.extend_from_slice(&info.primary_item_payload[..frame_obu_start]);
        primary_payload.extend(sized_obu(3, &frame_payload[..frame_header_len]));
        let mut first_tile_group = vec![0x80];
        first_tile_group.extend_from_slice(&tiles[0]);
        primary_payload.extend(sized_obu(4, &first_tile_group));
        let mut second_tile_group = vec![0xe0];
        second_tile_group.extend_from_slice(&tiles[1]);
        primary_payload.extend(sized_obu(4, &second_tile_group));
        primary_payload.extend_from_slice(&info.primary_item_payload[frame_obu_end..]);

        let (mdat, mdat_size) = find_box(data, b"mdat");
        let old_payload_start = mdat + 8;
        assert_eq!(
            &data[old_payload_start..old_payload_start + info.primary_item_payload.len()],
            info.primary_item_payload.as_slice()
        );
        let mut output = Vec::with_capacity(data.len() + primary_payload.len());
        output.extend_from_slice(&data[..old_payload_start]);
        output.extend_from_slice(&primary_payload);
        output.extend_from_slice(&data[old_payload_start + info.primary_item_payload.len()..]);
        write_uint(
            &mut output,
            mdat,
            4,
            mdat_size - info.primary_item_payload.len() + primary_payload.len(),
        );
        let extent_length_offset = primary_extent_length_offset(
            data,
            info.primary_item_id.expect("generated AVIF primary item"),
        );
        write_uint(&mut output, extent_length_offset, 4, primary_payload.len());
        output
    }

    fn tile_info() -> crate::av1::TileInfo {
        crate::av1::TileInfo {
            uniform_tile_spacing: true,
            dependent_tiles: false,
            loop_filter_across_tiles: false,
            tile_cols: 2,
            tile_rows: 1,
            tile_cols_log2: 1,
            tile_rows_log2: 0,
            tile_size_bytes: 1,
            context_update_tile_id: 0,
            mi_col_starts: vec![0, 16, 32],
            mi_row_starts: vec![0, 16],
        }
    }

    #[test]
    fn merge_tile_groups_reorders_payloads_by_tile_id() {
        let info = tile_info();
        let first = [0x80, 0xaa];
        let second = [0xe0, 0xbb];
        let (data, group) = merge_tile_group_payloads(&[&second, &first], &info).unwrap();
        assert_eq!(data, vec![0xaa, 0xbb]);
        assert_eq!(group.start_tile, 0);
        assert_eq!(group.end_tile, 1);
        assert_eq!(group.tiles[0].tile_id, 0);
        assert_eq!(group.tiles[0].offset, 0);
        assert_eq!(group.tiles[1].tile_id, 1);
        assert_eq!(group.tiles[1].offset, 1);
    }

    #[test]
    fn merge_tile_groups_rejects_duplicate_tiles() {
        let info = tile_info();
        let first = [0x80, 0xaa];
        let duplicate = [0x80, 0xbb];
        let err = merge_tile_group_payloads(&[&first, &duplicate], &info).unwrap_err();
        assert!(
            matches!(err, DecoderError::Bitstream(message) if message.contains("multiple tile groups"))
        );
    }
    #[test]
    fn merge_tile_groups_rejects_holes() {
        let info = tile_info();
        let first = [0x80, 0xaa];
        let err = merge_tile_group_payloads(&[&first], &info).unwrap_err();
        assert!(
            matches!(err, DecoderError::Unsupported(message) if message.contains("partial tile"))
        );
    }

    #[test]
    fn generated_separate_tile_group_obus_decode_when_ffmpeg_present() {
        let root =
            std::env::temp_dir().join(format!(".test-avif-split-obu-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let source_path = root.join("source.avif");
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error"])
            .arg("-i")
            .arg(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join("samples/WML2Viewer.png"),
            )
            .args([
                "-frames:v",
                "1",
                "-c:v",
                "libaom-av1",
                "-still-picture",
                "1",
                "-crf",
                "30",
                "-cpu-used",
                "8",
                "-aom-params",
                "tile-columns=1:tile-rows=0:enable-cdef=0:enable-restoration=0",
                "-f",
                "avif",
            ])
            .arg(&source_path)
            .status();
        let Ok(status) = status else {
            eprintln!("ffmpeg is not available; skipping split TileGroup sample");
            let _ = std::fs::remove_dir_all(&root);
            return;
        };
        if !status.success() {
            eprintln!("libaom encoder is unavailable; skipping split TileGroup sample");
            let _ = std::fs::remove_dir_all(&root);
            return;
        }
        let source = std::fs::read(&source_path).unwrap();
        let split = split_primary_frame_obu(&source);
        let split_path = root.join("separate-tile-groups.avif");
        std::fs::write(&split_path, &split).unwrap();
        let ffmpeg_status = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin"])
            .arg("-i")
            .arg(&split_path)
            .args(["-frames:v", "1", "-f", "null", "-"])
            .status()
            .unwrap();
        assert!(
            ffmpeg_status.success(),
            "FFmpeg rejected split TileGroup AVIF"
        );
        let info = crate::container::parse_avif(&split).unwrap();
        let headers = parse_av1_headers(&info).unwrap();
        assert!(!headers.tile_group.from_frame_obu);
        let original = crate::image_from_bytes(&source).unwrap();
        let decoded = crate::image_from_bytes(&split).unwrap();
        assert_eq!(decoded, original);
        let _ = std::fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
mod gain_map_tests {
    use super::*;

    fn synthetic_matrix_shaper_profile() -> Vec<u8> {
        fn add_tag(profile: &mut Vec<u8>, index: usize, signature: &[u8; 4], payload: &[u8]) {
            let entry = 132 + index * 12;
            let offset = profile.len();
            profile[entry..entry + 4].copy_from_slice(signature);
            profile[entry + 4..entry + 8].copy_from_slice(&(offset as u32).to_be_bytes());
            profile[entry + 8..entry + 12].copy_from_slice(&(payload.len() as u32).to_be_bytes());
            profile.extend_from_slice(payload);
        }
        fn xyz(values: [f64; 3]) -> Vec<u8> {
            let mut payload = Vec::from(*b"XYZ \0\0\0\0");
            for value in values {
                payload.extend_from_slice(&((value * 65_536.0).round() as i32).to_be_bytes());
            }
            payload
        }
        let curve = || {
            let mut payload = Vec::from(*b"curv\0\0\0\0");
            payload.extend_from_slice(&1_u32.to_be_bytes());
            payload.extend_from_slice(&256_u16.to_be_bytes());
            payload
        };
        let mut profile = vec![0; 132 + 7 * 12];
        profile[12..16].copy_from_slice(b"mntr");
        profile[16..20].copy_from_slice(b"RGB ");
        profile[20..24].copy_from_slice(b"XYZ ");
        profile[128..132].copy_from_slice(&7_u32.to_be_bytes());
        add_tag(&mut profile, 0, b"wtpt", &xyz([0.9505, 1.0, 1.0890]));
        add_tag(&mut profile, 1, b"rXYZ", &xyz([0.4124, 0.2126, 0.0193]));
        add_tag(&mut profile, 2, b"gXYZ", &xyz([0.3576, 0.7152, 0.1192]));
        add_tag(&mut profile, 3, b"bXYZ", &xyz([0.1805, 0.0722, 0.9505]));
        add_tag(&mut profile, 4, b"rTRC", &curve());
        add_tag(&mut profile, 5, b"gTRC", &curve());
        add_tag(&mut profile, 6, b"bTRC", &curve());
        let profile_size = profile.len() as u32;
        profile[0..4].copy_from_slice(&profile_size.to_be_bytes());
        profile
    }

    fn identity_frame(value: u16) -> DecodedFrame {
        let color_config = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(crate::av1::ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 0,
            }),
            color_range: ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };
        let plane = |index| PlaneBuffer {
            layout: PlaneLayout {
                plane: index,
                width: 1,
                height: 1,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 1,
            },
            samples: vec![value],
        };
        DecodedFrame {
            width: 1,
            height: 1,
            render_width: 1,
            render_height: 1,
            bit_depth: 8,
            color_config,
            color_information: None,
            alpha_premultiplied: false,
            buffers: FrameBuffers {
                width: 1,
                height: 1,
                planes: vec![plane(0), plane(1), plane(2)],
            },
        }
    }

    fn metadata(use_base_colour_space: bool) -> crate::container::GainMapMetadata {
        crate::container::GainMapMetadata {
            minimum_version: 0,
            writer_version: 0,
            is_multichannel: false,
            use_base_colour_space,
            backward_direction: false,
            base_hdr_headroom: crate::container::GainMapRational {
                numerator: 0,
                denominator: 1,
            },
            alternate_hdr_headroom: crate::container::GainMapRational {
                numerator: 1,
                denominator: 1,
            },
            channels: vec![crate::container::GainMapChannel {
                gain_map_min: crate::container::GainMapRational {
                    numerator: 0,
                    denominator: 1,
                },
                gain_map_max: crate::container::GainMapRational {
                    numerator: 1,
                    denominator: 1,
                },
                gamma: crate::container::GainMapRational {
                    numerator: 1,
                    denominator: 1,
                },
                base_offset: crate::container::GainMapRational {
                    numerator: 0,
                    denominator: 1,
                },
                alternate_offset: crate::container::GainMapRational {
                    numerator: 0,
                    denominator: 1,
                },
            }],
        }
    }

    #[test]
    fn gain_map_resampling_preserves_constant_map_values() {
        let input = Rgba16ImageBuffer {
            width: 1,
            height: 1,
            rgba: vec![1234, 2345, 3456, u16::MAX],
        };
        let output = resample_gain_map(&input, 3, 2).unwrap();
        assert_eq!((output.width, output.height), (3, 2));
        for pixel in output.rgba.chunks_exact(4) {
            assert_eq!(pixel, &[1234, 2345, 3456, u16::MAX]);
        }
    }

    #[test]
    fn gain_map_base_headroom_is_an_exact_fast_path() {
        let base = identity_frame(128);
        let gain_map = DecodedGainMapFrame {
            metadata: metadata(true),
            frame: identity_frame(255),
        };
        let expected = base.to_rgba16().unwrap();
        let actual = base.to_rgba16_with_gain_map(&gain_map, 0.0).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn gain_map_applies_log2_gain_and_preserves_alpha() {
        let base = identity_frame(128);
        let gain_map = DecodedGainMapFrame {
            metadata: metadata(true),
            frame: identity_frame(255),
        };
        let mapped = base.to_rgba16_with_gain_map(&gain_map, 1.0).unwrap();
        assert!(mapped.rgba[0] > base.to_rgba16().unwrap().rgba[0]);
        assert_eq!(mapped.rgba[3], u16::MAX);
    }

    #[test]
    fn gain_map_composes_different_alternate_colour_space() {
        let base = identity_frame(128);
        let mut alternate_frame = identity_frame(255);
        alternate_frame.color_config.color_description = Some(crate::av1::ColorDescription {
            color_primaries: 9,
            transfer_characteristics: 13,
            matrix_coefficients: 0,
        });
        let gain_map = DecodedGainMapFrame {
            metadata: metadata(false),
            frame: alternate_frame,
        };
        let mapped = base.to_rgba16_with_gain_map(&gain_map, 1.0).unwrap();
        assert!(mapped.rgba[0] > base.to_rgba16().unwrap().rgba[0]);
        assert!(mapped.rgba[1] > base.to_rgba16().unwrap().rgba[1]);
        assert!(mapped.rgba[2] > base.to_rgba16().unwrap().rgba[2]);
    }

    #[test]
    fn gain_map_accepts_equivalent_alternate_colour_space() {
        let base = identity_frame(128);
        let gain_map = DecodedGainMapFrame {
            metadata: metadata(false),
            frame: identity_frame(255),
        };
        let mapped = base.to_rgba16_with_gain_map(&gain_map, 1.0).unwrap();
        assert!(mapped.rgba[0] > base.to_rgba16().unwrap().rgba[0]);
    }

    #[test]
    fn gain_map_composes_matrix_shaper_icc_alternate_space() {
        let base = identity_frame(128);
        let mut alternate_frame = identity_frame(255);
        alternate_frame.color_config.color_description = Some(crate::av1::ColorDescription {
            color_primaries: 9,
            transfer_characteristics: 13,
            matrix_coefficients: 0,
        });
        alternate_frame.color_information = Some(ColorInformation {
            color_type: *b"prof",
            payload: synthetic_matrix_shaper_profile(),
        });
        let gain_map = DecodedGainMapFrame {
            metadata: metadata(false),
            frame: alternate_frame,
        };
        let mapped = base.to_rgba16_with_gain_map(&gain_map, 1.0).unwrap();
        assert!(mapped.rgba[0] > base.to_rgba16().unwrap().rgba[0]);
    }
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

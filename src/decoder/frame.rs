use super::sequence::{SequenceDecodeState, decode_hidden_key_frame_show_existing};
use super::*;
use crate::Rgba16ImageBuffer;
use crate::av1::{convert_linear_rgb_primaries, frame_buffers_to_rgba_16};
use crate::container::{
    AvifAnimation, AvifFrameTiming, AvifSequence, parse_avif_animation, parse_avif_sequence,
    parse_gain_map_image,
};

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

/// One source-plane frame and the timing assigned to its color-track sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedSequenceFrame {
    pub frame: DecodedFrame,
    pub timing: AvifFrameTiming,
}

/// Incremental AVIS decoder.
///
/// The decoder exposes one color frame at a time and never retains decoded
/// RGBA output for later frames. Color and alpha tracks keep their AV1
/// reference, CDF, and motion state between calls, so a forward traversal
/// decodes every track sample exactly once.
pub struct AvifSequenceDecoder {
    animation: AvifAnimation,
    tracks: SequenceTracksDecoder,
    next_index: usize,
}

struct SequenceTracksDecoder {
    info: AvifInfo,
    color_state: SequenceDecodeState,
    alpha_info: Option<AvifInfo>,
    alpha_state: Option<SequenceDecodeState>,
    static_alpha_frame: Option<DecodedFrame>,
}

impl SequenceTracksDecoder {
    fn new(mut info: AvifInfo, sequence: &AvifSequence) -> Result<Self, DecoderError> {
        if sequence.color_samples.is_empty() {
            return Err(DecoderError::Bitstream(
                "AVIS color track has no samples".to_string(),
            ));
        }
        if !sequence.alpha_samples.is_empty()
            && sequence.alpha_samples.len() != sequence.color_samples.len()
        {
            return Err(DecoderError::Bitstream(format!(
                "AVIS alpha track has {} frames, expected {}",
                sequence.alpha_samples.len(),
                sequence.color_samples.len()
            )));
        }
        let color_state = SequenceDecodeState::new(&info)?;
        let (alpha_info, alpha_state) = if sequence.alpha_samples.is_empty() {
            (None, None)
        } else {
            let mut alpha_info = info.clone();
            alpha_info.primary_item_payload = sequence.alpha_samples[0].clone();
            alpha_info.sequence_sample_payloads.clear();
            alpha_info.alpha_auxiliary_items.clear();
            alpha_info.alpha_grid = None;
            alpha_info.av1_config = None;
            let alpha_state = SequenceDecodeState::new(&alpha_info)?;
            alpha_info.primary_item_payload.clear();
            (Some(alpha_info), Some(alpha_state))
        };
        // The caller's `AvifSequence` is the canonical owner of track samples.
        // Keep only container metadata in the private decode view rather than
        // a second retained copy of every color payload.
        info.primary_item_payload.clear();
        info.sequence_sample_payloads.clear();
        Ok(Self {
            info,
            color_state,
            alpha_info,
            alpha_state,
            static_alpha_frame: None,
        })
    }

    #[cfg(test)]
    fn decoded_sample_counts(&self) -> (usize, Option<usize>) {
        (
            self.color_state.decoded_sample_count(),
            self.alpha_state
                .as_ref()
                .map(SequenceDecodeState::decoded_sample_count),
        )
    }

    fn next_frame(
        &mut self,
        sequence: &AvifSequence,
        next_index: usize,
    ) -> Result<Option<DecodedFrame>, DecoderError> {
        let mut color_state = self.color_state.clone();
        let mut alpha_state = self.alpha_state.clone();
        let mut frame = color_state
            .next_sample(&self.info, &sequence.color_samples)?
            .ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AVIS color track ended before sample {}",
                    next_index
                ))
            })?;
        if let (Some(alpha_info), Some(alpha_state)) =
            (self.alpha_info.as_ref(), alpha_state.as_mut())
        {
            let alpha_frame = alpha_state
                .next_sample(alpha_info, &sequence.alpha_samples)?
                .ok_or_else(|| {
                    DecoderError::Bitstream(format!(
                        "AVIS alpha track ended before sample {}",
                        next_index
                    ))
                })?;
            append_alpha_plane(&mut frame, &alpha_frame)?;
        } else if !self.info.alpha_auxiliary_items.is_empty() {
            let new_static_alpha = if self.static_alpha_frame.is_none() {
                Some(decode_alpha_auxiliary_frame(&self.info)?)
            } else {
                None
            };
            let alpha_frame = self
                .static_alpha_frame
                .as_ref()
                .or(new_static_alpha.as_ref())
                .expect("static alpha frame should be cached or freshly decoded");
            append_alpha_plane(&mut frame, alpha_frame)?;
            if let Some(alpha_frame) = new_static_alpha {
                self.static_alpha_frame = Some(alpha_frame);
            }
        }
        self.color_state = color_state;
        self.alpha_state = alpha_state;
        Ok(Some(frame))
    }
}

impl AvifSequenceDecoder {
    pub fn new(data: &[u8]) -> Result<Self, DecoderError> {
        let info = parse_avif(data)?;
        validate_public_container_preflight(&info, false)?;
        let animation = parse_avif_animation(data)?;
        if animation.sequence.color_samples.len() != animation.color_timing.len() {
            return Err(DecoderError::Bitstream(
                "AVIS color timing does not match the sample count".to_string(),
            ));
        }
        let tracks = SequenceTracksDecoder::new(info, &animation.sequence)?;
        Ok(Self {
            animation,
            tracks,
            next_index: 0,
        })
    }

    pub fn animation(&self) -> &AvifAnimation {
        &self.animation
    }

    #[cfg(test)]
    pub(super) fn decoded_track_sample_counts(&self) -> (usize, Option<usize>) {
        self.tracks.decoded_sample_counts()
    }

    pub fn next_frame(&mut self) -> Result<Option<DecodedSequenceFrame>, DecoderError> {
        let Some(timing) = self.animation.color_timing.get(self.next_index).copied() else {
            return Ok(None);
        };
        let frame = self
            .tracks
            .next_frame(&self.animation.sequence, self.next_index)?
            .ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AVIS sequence ended before sample {}",
                    self.next_index
                ))
            })?;
        self.next_index += 1;
        Ok(Some(DecodedSequenceFrame { frame, timing }))
    }
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

pub(super) fn resample_gain_map(
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

pub(super) fn unpremultiply_rgba8(rgba: &mut [u8]) {
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

pub(super) fn unpremultiply_rgba16(rgba: &mut [u16]) {
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
    let sequence = parse_avif_sequence(data)?;
    let sample_count = sequence.color_samples.len();
    if frame_index >= sample_count {
        return Err(DecoderError::InvalidParam(format!(
            "AVIS frame index {frame_index} is outside the {sample_count}-sample sequence"
        )));
    }
    let mut tracks = SequenceTracksDecoder::new(info, &sequence)?;
    for index in 0..=frame_index {
        let frame = tracks.next_frame(&sequence, index)?.ok_or_else(|| {
            DecoderError::Bitstream(format!("AVIS sequence ended before sample {frame_index}"))
        })?;
        if index == frame_index {
            return Ok(frame);
        }
    }
    unreachable!("validated AVIS frame index should be returned from the decode loop")
}

/// Decodes every independently addressable AVIS sample into source planes.
///
/// This animation-oriented API accepts Key/IntraOnly, inter/switch, and
/// show-existing samples.
pub fn decode_sequence_frames_bytes(data: &[u8]) -> Result<Vec<DecodedFrame>, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, false)?;
    let sequence = parse_avif_sequence(data)?;
    let mut tracks = SequenceTracksDecoder::new(info, &sequence)?;
    let mut frames = Vec::with_capacity(sequence.color_samples.len());
    for index in 0..sequence.color_samples.len() {
        let frame = tracks.next_frame(&sequence, index)?.ok_or_else(|| {
            DecoderError::Bitstream(format!("AVIS sequence ended before sample {index}"))
        })?;
        frames.push(frame);
    }
    Ok(frames)
}

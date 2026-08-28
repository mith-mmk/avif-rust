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

/// Strict native still-image decode used by high-precision consumers.
///
/// The historical [`decode_frame_bytes`] path keeps its established alpha
/// normalization behaviour for compatibility.  This additive entry point
/// validates AVIF's master/auxiliary constraints before attaching alpha, so a
/// depth mismatch can never be rounded into a successful native frame.
pub fn decode_frame_bytes_strict(data: &[u8]) -> Result<DecodedFrame, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, false)?;
    decode_frame_bytes_strict_from_info(data, &info, None)
}

/// Bounded native still decode. The rich container parse is performed once;
/// the legacy unbounded entry point is not used as a fallback.
///
/// At this stage the limits cover the container/metadata projection and the
/// pre-tile geometry/header checks for one ordinary still `av01` item and its
/// selected auxiliary alpha. AVIS sequences and derived images are rejected
/// by this entry point. Deep tile/entropy traversal, reference-frame state,
/// post-filter scratch, and a total-live-allocation guarantee are not covered
/// yet; those are the planned C2/C3 bounded-decoding stages.
pub fn decode_frame_bytes_strict_with_limits(
    data: &[u8],
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<crate::native::NativeDecodedFrame, DecoderError> {
    let information = crate::native::parse_native_info(data, limits)?;
    let info = information.info();
    validate_public_container_preflight(info, false)?;
    if !info.sequence_sample_payloads.is_empty() {
        return Err(DecoderError::Unsupported(
            "bounded native decode accepts one still av01 item, not an AVIS sequence".to_string(),
        ));
    }
    if information.primary_item_type() != Some(*b"av01") {
        return Err(DecoderError::Unsupported(
            "bounded native decode accepts only a primary av01 item".to_string(),
        ));
    }
    let derived = information.primary_item_type() == Some(*b"sato") || info.primary_grid.is_some();
    if derived {
        limits.check_count(1, limits.max_derived_depth(), "derived image depth")?;
        return Err(DecoderError::Unsupported(
            "bounded native decode does not yet support derived images".to_string(),
        ));
    }
    let frame = decode_frame_bytes_strict_from_info(data, info, Some(limits))?;
    validate_native_frame_limits(&frame, limits)?;
    Ok(crate::native::NativeDecodedFrame::new(frame, information))
}

fn decode_frame_bytes_strict_from_info(
    data: &[u8],
    info: &AvifInfo,
    limits: Option<&crate::limits::NativeDecodeLimits>,
) -> Result<DecodedFrame, DecoderError> {
    if limits.is_none() {
        if let Some(frame) = decode_sample_transform_frame(data, info)? {
            reject_strict_derived_alpha(
                "sato",
                info.alpha_grid.is_some(),
                !info.alpha_auxiliary_items.is_empty(),
            )?;
            return Ok(frame);
        }
        if info.primary_grid.is_some() {
            reject_strict_derived_alpha(
                "grid",
                info.alpha_grid.is_some(),
                !info.alpha_auxiliary_items.is_empty(),
            )?;
            return decode_grid_frame_raw(info);
        }
    }
    let mut bounded_frame_prefix = None;
    let mut bounded_alpha_prefix = None;
    let bounded_sequence = if let Some(limits) = limits {
        let sequence = validate_native_sequence_limits(info, limits)?;
        let frame_prefix = validate_native_frame_prefix_limits(info, &sequence, limits)?;
        if !info.alpha_auxiliary_items.is_empty() {
            let alpha_header = super::still::validate_alpha_auxiliary_header_limits(
                info,
                limits,
                usize::try_from(frame_prefix.geometry().0).map_err(|_| {
                    DecoderError::InvalidParam("AV1 frame width is too large".to_string())
                })?,
                usize::try_from(frame_prefix.geometry().1).map_err(|_| {
                    DecoderError::InvalidParam("AV1 frame height is too large".to_string())
                })?,
                sequence.color_config.bit_depth,
            )?;
            validate_bounded_item_payload(
                &info.alpha_auxiliary_items[0].payload,
                &alpha_header.sequence,
                "alpha",
            )?;
            bounded_alpha_prefix = Some(alpha_header);
        }
        validate_bounded_item_payload(&info.primary_item_payload, &sequence, "primary")?;
        bounded_frame_prefix = Some(frame_prefix);
        Some(sequence)
    } else {
        None
    };
    let mut frame = if bounded_sequence.is_some() {
        let headers = parse_av1_headers_with_frame_prefix(
            info,
            bounded_frame_prefix
                .take()
                .expect("bounded native path has a validated frame prefix"),
        )?;
        validate_native_decode_plan_limits(&headers.decode_plan, limits.expect("bounded path"))?;
        decode_still_frame(&headers, Some(info))?
    } else if let Some(frame) = decode_hidden_key_frame_show_existing(info)? {
        frame
    } else {
        let headers = parse_av1_headers(info)?;
        decode_still_frame(&headers, Some(info))?
    };
    if !info.alpha_auxiliary_items.is_empty() {
        let alpha_frame = if limits.is_some() {
            let prefix = bounded_alpha_prefix
                .take()
                .expect("bounded native path has a validated alpha frame prefix");
            super::still::decode_alpha_auxiliary_frame_with_prefix(
                info,
                prefix.parts,
                prefix.prefix,
            )?
        } else {
            decode_alpha_auxiliary_frame(info)?
        };
        if limits.is_some() {
            append_native_alpha_plane(&mut frame, alpha_frame)?;
        } else {
            validate_strict_alpha(&frame, &alpha_frame)?;
            append_alpha_plane_buffer(
                &mut frame,
                alpha_frame.buffers.planes[0].clone(),
                alpha_frame.bit_depth,
            )?;
        }
    }
    Ok(frame)
}

/// Attaches the already-decoded alpha owner on the strict native path.
///
/// The ownership move is intentionally kept in the production helper used by
/// the bounded decoder.  The legacy path above continues to clone its plane,
/// preserving the historical callback/API behavior.
pub(super) fn append_native_alpha_plane(
    frame: &mut DecodedFrame,
    alpha_frame: DecodedFrame,
) -> Result<(), DecoderError> {
    validate_strict_alpha(frame, &alpha_frame)?;
    let alpha_bit_depth = alpha_frame.bit_depth;
    let alpha_plane = alpha_frame
        .buffers
        .planes
        .into_iter()
        .next()
        .ok_or_else(|| {
            DecoderError::Bitstream("AVIF alpha auxiliary plane is missing".to_string())
        })?;
    append_alpha_plane_buffer(frame, alpha_plane, alpha_bit_depth)
}

pub(super) fn validate_native_sequence_limits(
    info: &AvifInfo,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<crate::av1::SequenceHeader, DecoderError> {
    let sequence_payload = crate::obu::find_obu_payload(
        &info.primary_item_payload,
        crate::obu::ObuType::SequenceHeader,
    )?
    .ok_or_else(|| DecoderError::Bitstream("AV1 sequence header OBU is missing".to_string()))?;
    let sequence = crate::av1::parse_sequence_header(sequence_payload)?;
    limits.check_dimensions(
        usize::try_from(sequence.max_frame_width).map_err(|_| {
            DecoderError::InvalidParam("AV1 maximum width is too large".to_string())
        })?,
        usize::try_from(sequence.max_frame_height).map_err(|_| {
            DecoderError::InvalidParam("AV1 maximum height is too large".to_string())
        })?,
    )?;
    if sequence.enable_superres {
        return Err(DecoderError::Unsupported(
            "bounded native decode does not yet budget AV1 super-resolution".to_string(),
        ));
    }
    Ok(sequence)
}

fn validate_bounded_item_payload(
    payload: &[u8],
    sequence: &crate::av1::SequenceHeader,
    item_label: &str,
) -> Result<(), DecoderError> {
    let mut displayed_frames = 0usize;
    for obu in crate::obu::ObuIter::new(payload) {
        let obu = obu?;
        if matches!(
            obu.obu_type,
            crate::obu::ObuType::Frame | crate::obu::ObuType::FrameHeader
        ) {
            displayed_frames = displayed_frames.checked_add(1).ok_or_else(|| {
                DecoderError::InvalidParam("AV1 frame count overflows".to_string())
            })?;
            if !sequence.reduced_still_picture_header
                && crate::av1::parse_show_existing_frame_index(obu.payload)?.is_some()
            {
                return Err(DecoderError::Unsupported(
                    "bounded native decode does not yet support show_existing_frame".to_string(),
                ));
            }
        }
    }
    if displayed_frames > 1 {
        return Err(DecoderError::Unsupported(format!(
            "bounded native decode accepts one displayed {item_label} AV1 frame"
        )));
    }
    Ok(())
}

/// Parse only the borrowed AV1 frame header and account for every visible
/// geometry before the materializer is allowed to copy a tile payload.
///
/// This is deliberately separate from `parse_av1_headers`: that routine owns
/// the frame/tile data used by the legacy decoder.  Native callers need the
/// header decision first so a plane limit cannot be discovered after the
/// frame OBU has already been copied.
pub(super) fn validate_native_frame_prefix_limits<'a>(
    info: &'a AvifInfo,
    sequence: &crate::av1::SequenceHeader,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<crate::av1::FramePrefix<'a, 'static>, DecoderError> {
    let payload =
        crate::obu::find_obu_payload(&info.primary_item_payload, crate::obu::ObuType::Frame)?
            .or(crate::obu::find_obu_payload(
                &info.primary_item_payload,
                crate::obu::ObuType::FrameHeader,
            )?)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 frame header OBU is missing".to_string())
            })?;
    let sequence_payload = crate::obu::find_obu_payload(
        &info.primary_item_payload,
        crate::obu::ObuType::SequenceHeader,
    )?
    .ok_or_else(|| DecoderError::Bitstream("AV1 sequence header OBU is missing".to_string()))?;
    let (_, sequence_metadata) = crate::av1::parse_sequence_header_with_metadata(sequence_payload)?;
    let prefix = crate::av1::parse_frame_prefix(
        payload,
        sequence,
        &sequence_metadata,
        &crate::av1::NO_REFERENCES,
    )?;
    if !prefix.show_frame() {
        return Err(DecoderError::Unsupported(
            "bounded native decode requires a displayed AV1 frame".to_string(),
        ));
    }
    validate_native_frame_prefix_geometry_limits(sequence, &prefix, limits)?;
    Ok(prefix)
}

pub(super) fn validate_native_frame_prefix_geometry_limits(
    sequence: &crate::av1::SequenceHeader,
    prefix: &crate::av1::FramePrefix<'_, '_>,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(), DecoderError> {
    let (width, height, upscaled_width, render_width, render_height) = prefix.geometry();
    validate_native_frame_geometry_limits(
        sequence,
        width,
        height,
        upscaled_width,
        render_width,
        render_height,
        limits,
    )
}

fn validate_native_frame_geometry_limits(
    sequence: &crate::av1::SequenceHeader,
    frame_width: u32,
    frame_height: u32,
    upscaled_width: u32,
    render_width: u32,
    render_height: u32,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(), DecoderError> {
    let width = usize::try_from(frame_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
    let height = usize::try_from(frame_height)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
    let upscaled_width = usize::try_from(upscaled_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 upscaled width is too large".to_string()))?;
    let render_width = usize::try_from(render_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 render width is too large".to_string()))?;
    let render_height = usize::try_from(render_height)
        .map_err(|_| DecoderError::InvalidParam("AV1 render height is too large".to_string()))?;
    validate_native_geometry(sequence, width, height, limits, "visible")?;
    validate_native_geometry(sequence, upscaled_width, height, limits, "upscaled")?;
    validate_native_geometry(sequence, render_width, render_height, limits, "render")?;

    let coded_width = width
        .div_ceil(8)
        .checked_mul(8)
        .ok_or_else(|| DecoderError::InvalidParam("AV1 coded width overflows".to_string()))?;
    let coded_height = height
        .div_ceil(8)
        .checked_mul(8)
        .ok_or_else(|| DecoderError::InvalidParam("AV1 coded height overflows".to_string()))?;
    validate_native_geometry(sequence, coded_width, coded_height, limits, "coded")
}

fn validate_native_geometry(
    sequence: &crate::av1::SequenceHeader,
    width: usize,
    height: usize,
    limits: &crate::limits::NativeDecodeLimits,
    label: &str,
) -> Result<(), DecoderError> {
    limits.check_dimensions(width, height)?;
    for plane in 0..3 {
        if let Some(layout) = crate::av1::plane_layout_for_geometry(sequence, width, height, plane)?
        {
            let bytes = layout
                .sample_count
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| {
                    DecoderError::InvalidParam(format!("AV1 {label} plane size overflows"))
                })?;
            limits.check_plane_bytes(bytes)?;
        }
    }
    Ok(())
}

pub(super) fn validate_native_decode_plan_limits(
    plan: &FrameDecodePlan,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(), DecoderError> {
    limits.check_dimensions(plan.width, plan.height)?;
    limits.check_dimensions(plan.upscaled_width, plan.render_height)?;
    limits.check_dimensions(plan.render_width, plan.render_height)?;
    for layout in &plan.planes {
        let bytes = layout
            .sample_count
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| DecoderError::InvalidParam("native plane size overflows".to_string()))?;
        limits.check_plane_bytes(bytes)?;
        let coded_width = plan.width.div_ceil(8).checked_mul(8).ok_or_else(|| {
            DecoderError::InvalidParam("native coded width overflows".to_string())
        })?;
        let coded_height = plan.height.div_ceil(8).checked_mul(8).ok_or_else(|| {
            DecoderError::InvalidParam("native coded height overflows".to_string())
        })?;
        let coded_width = coded_width.div_ceil(1usize << layout.subsampling_x);
        let coded_height = coded_height.div_ceil(1usize << layout.subsampling_y);
        let coded_bytes = coded_width
            .checked_mul(coded_height)
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| {
                DecoderError::InvalidParam("native coded plane size overflows".to_string())
            })?;
        limits.check_plane_bytes(coded_bytes)?;
    }
    Ok(())
}

fn validate_native_frame_limits(
    frame: &DecodedFrame,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(), DecoderError> {
    limits.check_dimensions(frame.width, frame.height)?;
    for plane in &frame.buffers.planes {
        let bytes = plane
            .samples
            .len()
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| DecoderError::InvalidParam("native plane size overflows".to_string()))?;
        limits.check_plane_bytes(bytes)?;
    }
    Ok(())
}

pub(super) fn reject_strict_derived_alpha(
    derived_kind: &str,
    has_alpha_grid: bool,
    has_alpha_auxiliary: bool,
) -> Result<(), DecoderError> {
    if has_alpha_grid || has_alpha_auxiliary {
        return Err(DecoderError::Unsupported(format!(
            "strict native decode does not yet support alpha on a derived {derived_kind} image"
        )));
    }
    Ok(())
}

pub(super) fn validate_strict_alpha(
    master: &DecodedFrame,
    alpha: &DecodedFrame,
) -> Result<(), DecoderError> {
    if alpha.width != master.width || alpha.height != master.height {
        return Err(DecoderError::Bitstream(format!(
            "AVIF alpha dimensions {}x{} do not match master dimensions {}x{}",
            alpha.width, alpha.height, master.width, master.height
        )));
    }
    let alpha_plane = alpha.buffers.planes.first().ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha auxiliary plane is missing".to_string())
    })?;
    if alpha_plane.layout.width != alpha.width || alpha_plane.layout.height != alpha.height {
        return Err(DecoderError::Bitstream(
            "AVIF alpha plane geometry does not match its decoded header".to_string(),
        ));
    }
    if alpha.buffers.planes.len() != 1 || !alpha.color_config.monochrome {
        return Err(DecoderError::Unsupported(
            "AVIF alpha auxiliary image must be monochrome".to_string(),
        ));
    }
    if alpha.bit_depth != master.bit_depth {
        return Err(DecoderError::Unsupported(format!(
            "AVIF alpha bit depth {} does not match master bit depth {}",
            alpha.bit_depth, master.bit_depth
        )));
    }
    if alpha.color_config.color_range != ColorRange::Full {
        return Err(DecoderError::Unsupported(
            "AVIF alpha auxiliary image must use full range".to_string(),
        ));
    }
    Ok(())
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

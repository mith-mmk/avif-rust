use super::bitstream::BitReader;
use super::sequence::SequenceHeader;
use super::tile::{TileInfo, parse_tile_info};
use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Key,
    Inter,
    IntraOnly,
    Switch,
}

impl FrameType {
    fn from_bits(value: u32) -> Result<Self, DecoderError> {
        match value {
            0 => Ok(Self::Key),
            1 => Ok(Self::Inter),
            2 => Ok(Self::IntraOnly),
            3 => Ok(Self::Switch),
            _ => Err(DecoderError::Bitstream(format!(
                "frame_type {value} is reserved"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub show_existing_frame: bool,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: u8,
    pub frame_size_override_flag: bool,
    pub order_hint: u32,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    /// Reference slots signalled by an inter/switch frame.  The seven entries
    /// correspond to LAST..ALTREF in AV1 reference-frame order.
    pub reference_frame_indices: [u8; 7],
    pub frame_refs_short_signaling: bool,
    pub allow_high_precision_mv: bool,
    pub is_filter_switchable: bool,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    pub allow_warped_motion: bool,
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub allow_intrabc: bool,
    pub disable_frame_end_update_cdf: bool,
    pub tile_info: TileInfo,
    pub base_q_idx: u8,
    pub quantization: QuantizationParams,
    pub segmentation: SegmentationParams,
    pub delta_q: DeltaQParams,
    pub delta_lf: DeltaLfParams,
    pub loop_filter: LoopFilterParams,
    pub cdef: CdefParams,
    pub restoration: RestorationParams,
    pub tx_mode: TxMode,
    pub reduced_tx_set: bool,
    pub film_grain: Option<FilmGrainParams>,
    pub uncompressed_header_bits: usize,
    pub payload_after_header_offset: usize,
}

/// Geometry metadata retained for resolving AV1 `frame_size_with_refs`.
/// Pixel planes are owned by the sequence decoder; this small copy is kept in
/// the bitstream layer so frame-header parsing remains independent of buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReferenceFrameState {
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub order_hint: u32,
}

impl FrameHeader {
    pub(crate) fn coded_lossless(&self) -> bool {
        self.quantization.coded_lossless()
            && (!self.segmentation.enabled
                || (self.segmentation.delta_q == 0
                    && self
                        .segmentation
                        .segment_delta_q
                        .iter()
                        .all(|delta| *delta == 0)))
    }
}

/// Frame-level segmentation signalling. The still-image decoder accepts the
/// no-op form, `ALT_Q`/`ALT_LF` deltas and the still-image-safe `SKIP` feature
/// on the current segmentation map. Reference-frame and GLOBALMV feature
/// values are also consumed for still-image headers; inter-frame prediction is
/// rejected before reconstruction, so those values have no effect here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentationParams {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub preskip: bool,
    /// Segment 0's `SEG_LVL_ALT_Q` delta, kept as a compatibility alias for
    /// callers that only need the first segment.
    pub delta_q: i16,
    pub segment_delta_q: [i16; 8],
    /// Segment-level `SEG_LVL_ALT_LF_*` deltas in Y-vertical, Y-horizontal,
    /// U and V order.
    pub segment_delta_lf: [[i8; 4]; 8],
    pub segment_skip: [bool; 8],
    pub last_active_segment: u8,
}

impl SegmentationParams {
    pub(crate) fn effective_qindex(self, base_q_idx: u8) -> u8 {
        self.effective_qindex_for_segment(base_q_idx, 0)
    }

    pub(crate) fn effective_qindex_for_segment(self, base_q_idx: u8, segment: u8) -> u8 {
        let delta = self
            .segment_delta_q
            .get(usize::from(segment))
            .copied()
            .unwrap_or(self.delta_q);
        (i16::from(base_q_idx) + delta).clamp(0, 255) as u8
    }
}

pub fn parse_frame_header(
    data: &[u8],
    sequence: &SequenceHeader,
) -> Result<FrameHeader, DecoderError> {
    parse_frame_header_with_references(data, sequence, &[None; 8])
}

pub(crate) fn parse_frame_header_with_references(
    data: &[u8],
    sequence: &SequenceHeader,
    references: &[Option<ReferenceFrameState>; 8],
) -> Result<FrameHeader, DecoderError> {
    let mut reader = BitReader::new(data);

    if sequence.reduced_still_picture_header {
        let disable_cdf_update = reader.read_bool("disable_cdf_update")?;
        let allow_screen_content_tools = read_allow_screen_content_tools(&mut reader, sequence)?;
        let force_integer_mv =
            read_force_integer_mv(&mut reader, sequence, allow_screen_content_tools)?;
        let frame_size = parse_frame_size(&mut reader, sequence, false)?;
        let render_size = parse_render_size(&mut reader, frame_size.width, frame_size.height)?;
        let allow_intrabc =
            if allow_screen_content_tools && frame_size.upscaled_width == frame_size.width {
                reader.read_bool("allow_intrabc")?
            } else {
                false
            };
        let tile_info =
            parse_tile_info(&mut reader, sequence, frame_size.width, frame_size.height)?;
        let trailing = parse_frame_header_trailing_params(
            &mut reader,
            sequence,
            allow_intrabc,
            FrameType::Key,
            true,
            0,
            [0; 7],
            &[None; 8],
            7,
        )?;
        let film_grain =
            parse_film_grain_params(&mut reader, sequence, FrameType::Key, true, false)?;
        return Ok(FrameHeader {
            frame_type: FrameType::Key,
            show_existing_frame: false,
            show_frame: true,
            showable_frame: false,
            error_resilient_mode: true,
            disable_cdf_update,
            allow_screen_content_tools,
            force_integer_mv,
            frame_size_override_flag: false,
            order_hint: 0,
            primary_ref_frame: 7,
            refresh_frame_flags: 0xff,
            reference_frame_indices: [0; 7],
            frame_refs_short_signaling: false,
            allow_high_precision_mv: false,
            is_filter_switchable: false,
            is_motion_mode_switchable: false,
            use_ref_frame_mvs: false,
            reference_select: false,
            skip_mode_present: false,
            allow_warped_motion: false,
            frame_width: frame_size.width,
            frame_height: frame_size.height,
            upscaled_width: frame_size.upscaled_width,
            render_width: render_size.width,
            render_height: render_size.height,
            allow_intrabc,
            disable_frame_end_update_cdf: false,
            tile_info,
            base_q_idx: trailing.quantization.base_q_idx,
            quantization: trailing.quantization,
            segmentation: trailing.segmentation,
            delta_q: trailing.delta_q,
            delta_lf: trailing.delta_lf,
            loop_filter: trailing.loop_filter,
            cdef: trailing.cdef,
            restoration: trailing.restoration,
            tx_mode: trailing.tx_mode,
            reduced_tx_set: trailing.reduced_tx_set,
            film_grain,
            uncompressed_header_bits: reader.bit_position(),
            payload_after_header_offset: reader.byte_position_ceil(),
        });
    }

    let show_existing_frame = reader.read_bool("show_existing_frame")?;
    if show_existing_frame {
        return Err(DecoderError::Unsupported(
            "show_existing_frame AV1 frames are not supported yet".to_string(),
        ));
    }

    let frame_type = FrameType::from_bits(reader.read_bits(2, "frame_type")?)?;
    let show_frame = reader.read_bool("show_frame")?;
    let showable_frame = if show_frame {
        frame_type != FrameType::Key
    } else {
        reader.read_bool("showable_frame")?
    };
    let error_resilient_mode =
        if frame_type == FrameType::Switch || (frame_type == FrameType::Key && show_frame) {
            true
        } else {
            reader.read_bool("error_resilient_mode")?
        };
    let disable_cdf_update = reader.read_bool("disable_cdf_update")?;
    let allow_screen_content_tools = read_allow_screen_content_tools(&mut reader, sequence)?;
    let force_integer_mv =
        read_force_integer_mv(&mut reader, sequence, allow_screen_content_tools)?;
    let frame_size_override_flag = if frame_type == FrameType::Switch {
        true
    } else {
        reader.read_bool("frame_size_override_flag")?
    };
    let order_hint = if sequence.enable_order_hint {
        reader.read_bits(sequence.order_hint_bits as usize, "order_hint")?
    } else {
        0
    };
    let primary_ref_frame = if error_resilient_mode || frame_type == FrameType::IntraOnly {
        7
    } else {
        reader.read_bits(3, "primary_ref_frame")? as u8
    };
    let refresh_frame_flags = if frame_type == FrameType::Key && show_frame {
        0xff
    } else {
        reader.read_bits(8, "refresh_frame_flags")? as u8
    };

    let frame_is_intra = frame_type_is_intra(frame_type);
    let mut reference_frame_indices = [0; 7];
    let mut frame_refs_short_signaling = false;
    let mut allow_high_precision_mv = false;
    let mut is_filter_switchable = false;
    let mut is_motion_mode_switchable = false;
    let mut use_ref_frame_mvs = false;

    let (frame_size, render_size, allow_intrabc) = if frame_is_intra {
        let frame_size = parse_frame_size(&mut reader, sequence, frame_size_override_flag)?;
        let render_size = parse_render_size(&mut reader, frame_size.width, frame_size.height)?;
        let allow_intrabc =
            if allow_screen_content_tools && frame_size.upscaled_width == frame_size.width {
                reader.read_bool("allow_intrabc")?
            } else {
                false
            };
        (frame_size, render_size, allow_intrabc)
    } else {
        (frame_refs_short_signaling, reference_frame_indices) =
            parse_inter_reference_indices(&mut reader, sequence.enable_order_hint)?;
        if sequence.frame_id_numbers_present {
            return Err(DecoderError::Unsupported(
                "AV1 inter-frame ids are not supported yet".to_string(),
            ));
        }
        let (frame_size, render_size) = parse_inter_frame_size(
            &mut reader,
            sequence,
            frame_size_override_flag,
            error_resilient_mode,
            &reference_frame_indices,
            references,
        )?;
        allow_high_precision_mv = if force_integer_mv != 0 {
            false
        } else {
            reader.read_bool("allow_high_precision_mv")?
        };
        is_filter_switchable = reader.read_bool("is_filter_switchable")?;
        if !is_filter_switchable {
            let _interpolation_filter = reader.read_bits(2, "interpolation_filter")?;
        }
        is_motion_mode_switchable = reader.read_bool("is_motion_mode_switchable")?;
        use_ref_frame_mvs = if error_resilient_mode || !sequence.enable_ref_frame_mvs {
            false
        } else {
            reader.read_bool("use_ref_frame_mvs")?
        };
        (frame_size, render_size, false)
    };
    let disable_frame_end_update_cdf = reader.read_bool("disable_frame_end_update_cdf")?;
    let tile_info = parse_tile_info(&mut reader, sequence, frame_size.width, frame_size.height)?;
    let trailing = parse_frame_header_trailing_params(
        &mut reader,
        sequence,
        allow_intrabc,
        frame_type,
        error_resilient_mode,
        order_hint,
        reference_frame_indices,
        references,
        primary_ref_frame,
    )?;
    let film_grain = parse_film_grain_params(
        &mut reader,
        sequence,
        frame_type,
        show_frame,
        showable_frame,
    )?;
    Ok(FrameHeader {
        frame_type,
        show_existing_frame,
        show_frame,
        showable_frame,
        error_resilient_mode,
        disable_cdf_update,
        allow_screen_content_tools,
        force_integer_mv,
        frame_size_override_flag,
        order_hint,
        primary_ref_frame,
        refresh_frame_flags,
        reference_frame_indices,
        frame_refs_short_signaling,
        allow_high_precision_mv,
        is_filter_switchable,
        is_motion_mode_switchable,
        use_ref_frame_mvs,
        reference_select: trailing.reference_select,
        skip_mode_present: trailing.skip_mode_present,
        allow_warped_motion: trailing.allow_warped_motion,
        frame_width: frame_size.width,
        frame_height: frame_size.height,
        upscaled_width: frame_size.upscaled_width,
        render_width: render_size.width,
        render_height: render_size.height,
        allow_intrabc,
        disable_frame_end_update_cdf,
        tile_info,
        base_q_idx: trailing.quantization.base_q_idx,
        quantization: trailing.quantization,
        segmentation: trailing.segmentation,
        delta_q: trailing.delta_q,
        delta_lf: trailing.delta_lf,
        loop_filter: trailing.loop_filter,
        cdef: trailing.cdef,
        restoration: trailing.restoration,
        tx_mode: trailing.tx_mode,
        reduced_tx_set: trailing.reduced_tx_set,
        film_grain,
        uncompressed_header_bits: reader.bit_position(),
        payload_after_header_offset: reader.byte_position_ceil(),
    })
}

/// Reads the small prefix that identifies an AV1 `show_existing_frame` OBU.
/// The full coded-frame parser intentionally remains fail-closed for this
/// sequence feature until reference-backed reconstruction is wired in.
pub(crate) fn parse_show_existing_frame_index(data: &[u8]) -> Result<Option<u8>, DecoderError> {
    let mut reader = BitReader::new(data);
    if !reader.read_bool("show_existing_frame")? {
        return Ok(None);
    }
    Ok(Some(reader.read_bits(3, "frame_to_show_map_idx")? as u8))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilmGrainParams {
    pub random_seed: u16,
    pub num_y_points: u8,
    pub scaling_points_y: [[u8; 2]; 14],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: u8,
    pub scaling_points_cb: [[u8; 2]; 10],
    pub num_cr_points: u8,
    pub scaling_points_cr: [[u8; 2]; 10],
    pub scaling_shift: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y: [i16; 24],
    pub ar_coeffs_cb: [i16; 25],
    pub ar_coeffs_cr: [i16; 25],
    pub ar_coeff_shift: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

fn parse_film_grain_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    frame_type: FrameType,
    show_frame: bool,
    showable_frame: bool,
) -> Result<Option<FilmGrainParams>, DecoderError> {
    if !sequence.film_grain_params_present || (!show_frame && !showable_frame) {
        return Ok(None);
    }
    if !reader.read_bool("apply_grain")? {
        return Ok(None);
    }
    let random_seed = reader.read_bits(16, "grain_seed")? as u16;
    let update_parameters = if frame_type == FrameType::Inter {
        reader.read_bool("update_parameters")?
    } else {
        true
    };
    if !update_parameters {
        let _reference_index = reader.read_bits(3, "film_grain_params_ref_idx")?;
        return Err(DecoderError::Unsupported(
            "inherited AV1 film grain parameters are not supported yet".to_string(),
        ));
    }
    let num_y_points = reader.read_bits(4, "num_y_points")? as u8;
    if num_y_points > 14 {
        return Err(DecoderError::Bitstream(
            "film grain num_y_points exceeds 14".to_string(),
        ));
    }
    let scaling_points_y = read_scaling_points(reader, num_y_points, "y")?;
    let chroma_scaling_from_luma = if sequence.color_config.monochrome {
        false
    } else {
        reader.read_bool("chroma_scaling_from_luma")?
    };
    let mut num_cb_points = 0;
    let mut scaling_points_cb = [[0; 2]; 10];
    let mut num_cr_points = 0;
    let mut scaling_points_cr = [[0; 2]; 10];
    let skip_chroma_points = sequence.color_config.monochrome
        || chroma_scaling_from_luma
        || (sequence.color_config.subsampling_x
            && sequence.color_config.subsampling_y
            && num_y_points == 0);
    if !skip_chroma_points {
        num_cb_points = reader.read_bits(4, "num_cb_points")? as u8;
        if num_cb_points > 10 {
            return Err(DecoderError::Bitstream(
                "film grain num_cb_points exceeds 10".to_string(),
            ));
        }
        scaling_points_cb = read_scaling_points(reader, num_cb_points, "cb")?;
        num_cr_points = reader.read_bits(4, "num_cr_points")? as u8;
        if num_cr_points > 10 {
            return Err(DecoderError::Bitstream(
                "film grain num_cr_points exceeds 10".to_string(),
            ));
        }
        scaling_points_cr = read_scaling_points(reader, num_cr_points, "cr")?;
    }
    let scaling_shift = reader.read_bits(2, "grain_scaling_shift")? as u8 + 8;
    let ar_coeff_lag = reader.read_bits(2, "ar_coeff_lag")? as u8;
    let num_pos_luma = usize::from(2 * ar_coeff_lag * (ar_coeff_lag + 1));
    let num_pos_chroma = num_pos_luma + usize::from(num_y_points > 0);
    let mut ar_coeffs_y = [0; 24];
    let mut ar_coeffs_cb = [0; 25];
    let mut ar_coeffs_cr = [0; 25];
    if num_y_points > 0 {
        for coefficient in ar_coeffs_y.iter_mut().take(num_pos_luma) {
            *coefficient = reader.read_bits(8, "ar_coeffs_y")? as i16 - 128;
        }
    }
    if num_cb_points > 0 || chroma_scaling_from_luma {
        for coefficient in ar_coeffs_cb.iter_mut().take(num_pos_chroma) {
            *coefficient = reader.read_bits(8, "ar_coeffs_cb")? as i16 - 128;
        }
    }
    if num_cr_points > 0 || chroma_scaling_from_luma {
        for coefficient in ar_coeffs_cr.iter_mut().take(num_pos_chroma) {
            *coefficient = reader.read_bits(8, "ar_coeffs_cr")? as i16 - 128;
        }
    }
    let ar_coeff_shift = reader.read_bits(2, "ar_coeff_shift")? as u8 + 6;
    let grain_scale_shift = reader.read_bits(2, "grain_scale_shift")? as u8;
    let (cb_mult, cb_luma_mult, cb_offset) = if num_cb_points > 0 {
        (
            reader.read_bits(8, "cb_mult")? as u8,
            reader.read_bits(8, "cb_luma_mult")? as u8,
            reader.read_bits(9, "cb_offset")? as u16,
        )
    } else {
        (0, 0, 0)
    };
    let (cr_mult, cr_luma_mult, cr_offset) = if num_cr_points > 0 {
        (
            reader.read_bits(8, "cr_mult")? as u8,
            reader.read_bits(8, "cr_luma_mult")? as u8,
            reader.read_bits(9, "cr_offset")? as u16,
        )
    } else {
        (0, 0, 0)
    };
    let overlap_flag = reader.read_bool("overlap_flag")?;
    let clip_to_restricted_range = reader.read_bool("clip_to_restricted_range")?;
    let params = FilmGrainParams {
        random_seed,
        num_y_points,
        scaling_points_y,
        chroma_scaling_from_luma,
        num_cb_points,
        scaling_points_cb,
        num_cr_points,
        scaling_points_cr,
        scaling_shift,
        ar_coeff_lag,
        ar_coeffs_y,
        ar_coeffs_cb,
        ar_coeffs_cr,
        ar_coeff_shift,
        grain_scale_shift,
        cb_mult,
        cb_luma_mult,
        cb_offset,
        cr_mult,
        cr_luma_mult,
        cr_offset,
        overlap_flag,
        clip_to_restricted_range,
    };
    Ok(Some(params))
}

fn read_scaling_points<const N: usize>(
    reader: &mut BitReader<'_>,
    count: u8,
    plane: &str,
) -> Result<[[u8; 2]; N], DecoderError> {
    let mut points = [[0; 2]; N];
    for point in points.iter_mut().take(usize::from(count)) {
        point[0] = reader.read_bits(8, "scaling_point_x")? as u8;
        point[1] = reader.read_bits(8, "scaling_point_y")? as u8;
    }
    for pair in points.windows(2).take(usize::from(count).saturating_sub(1)) {
        if pair[0][0] >= pair[1][0] {
            return Err(DecoderError::Bitstream(format!(
                "film grain {plane} scaling point x values are not increasing"
            )));
        }
    }
    Ok(points)
}

#[derive(Debug, Clone, Copy)]
struct FrameSize {
    width: u32,
    height: u32,
    upscaled_width: u32,
}

#[derive(Debug, Clone, Copy)]
struct RenderSize {
    width: u32,
    height: u32,
}

fn read_allow_screen_content_tools(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
) -> Result<bool, DecoderError> {
    match sequence.seq_force_screen_content_tools {
        2 => reader.read_bool("allow_screen_content_tools"),
        1 => Ok(true),
        _ => Ok(false),
    }
}

fn read_force_integer_mv(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    allow_screen_content_tools: bool,
) -> Result<u8, DecoderError> {
    if !allow_screen_content_tools {
        return Ok(2);
    }
    match sequence.seq_force_integer_mv {
        2 => Ok(reader.read_bool("force_integer_mv")? as u8),
        value => Ok(value),
    }
}

fn parse_frame_size(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    frame_size_override_flag: bool,
) -> Result<FrameSize, DecoderError> {
    let upscaled_width = if frame_size_override_flag {
        reader.read_bits(sequence.frame_width_bits as usize, "frame_width_minus_1")? + 1
    } else {
        sequence.max_frame_width
    };
    let height = if frame_size_override_flag {
        reader.read_bits(sequence.frame_height_bits as usize, "frame_height_minus_1")? + 1
    } else {
        sequence.max_frame_height
    };

    let superres_denom = if sequence.enable_superres && reader.read_bool("use_superres")? {
        reader.read_bits(3, "coded_denom")? + 9
    } else {
        8
    };
    Ok(frame_size_from_upscaled_width(
        upscaled_width,
        height,
        superres_denom,
    ))
}

fn frame_size_from_upscaled_width(
    upscaled_width: u32,
    height: u32,
    superres_denom: u32,
) -> FrameSize {
    let width = if superres_denom == 8 {
        upscaled_width
    } else {
        (upscaled_width * 8 + (superres_denom / 2)) / superres_denom
    };
    FrameSize {
        width,
        height,
        upscaled_width,
    }
}

fn parse_inter_frame_size(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    frame_size_override_flag: bool,
    error_resilient_mode: bool,
    reference_frame_indices: &[u8; 7],
    references: &[Option<ReferenceFrameState>; 8],
) -> Result<(FrameSize, RenderSize), DecoderError> {
    if !(frame_size_override_flag && !error_resilient_mode) {
        let frame_size = parse_frame_size(reader, sequence, frame_size_override_flag)?;
        let render_size = parse_render_size(reader, frame_size.width, frame_size.height)?;
        return Ok((frame_size, render_size));
    }

    let mut found_reference = None;
    for reference in 0..7 {
        if reader.read_bool("found_ref")? {
            found_reference = Some(reference);
            break;
        }
    }
    // The syntax still carries the remaining found_ref flags even after the
    // first set bit; consume them before resolving the inferred dimensions.
    if let Some(index) = found_reference {
        for _ in 0..(6 - index) {
            let _ = reader.read_bool("found_ref")?;
        }
    }
    let Some(reference) = found_reference else {
        let frame_size = parse_frame_size(reader, sequence, true)?;
        let render_size = parse_render_size(reader, frame_size.width, frame_size.height)?;
        return Ok((frame_size, render_size));
    };
    let slot = usize::from(reference_frame_indices[reference]);
    let state = references
        .get(slot)
        .and_then(Option::as_ref)
        .ok_or_else(|| {
            DecoderError::Unsupported(format!(
                "AV1 frame_size_with_refs reference slot {slot} has no decoded reference"
            ))
        })?;
    let superres_denom = if sequence.enable_superres && reader.read_bool("use_superres")? {
        reader.read_bits(3, "coded_denom")? + 9
    } else {
        8
    };
    let frame_size =
        frame_size_from_upscaled_width(state.upscaled_width, state.frame_height, superres_denom);
    Ok((
        frame_size,
        RenderSize {
            width: state.render_width,
            height: state.render_height,
        },
    ))
}

fn parse_inter_reference_indices(
    reader: &mut BitReader<'_>,
    enable_order_hint: bool,
) -> Result<(bool, [u8; 7]), DecoderError> {
    let frame_refs_short_signaling = if enable_order_hint {
        reader.read_bool("frame_refs_short_signaling")?
    } else {
        false
    };
    let mut reference_frame_indices = [0; 7];
    if frame_refs_short_signaling {
        reference_frame_indices[0] = reader.read_bits(3, "last_frame_idx")? as u8;
        reference_frame_indices[3] = reader.read_bits(3, "golden_frame_idx")? as u8;
    } else {
        for reference in &mut reference_frame_indices {
            *reference = reader.read_bits(3, "ref_frame_idx")? as u8;
        }
    }
    Ok((frame_refs_short_signaling, reference_frame_indices))
}

fn parse_render_size(
    reader: &mut BitReader<'_>,
    frame_width: u32,
    frame_height: u32,
) -> Result<RenderSize, DecoderError> {
    if reader.read_bool("render_and_frame_size_different")? {
        Ok(RenderSize {
            width: reader.read_bits(16, "render_width_minus_1")? + 1,
            height: reader.read_bits(16, "render_height_minus_1")? + 1,
        })
    } else {
        Ok(RenderSize {
            width: frame_width,
            height: frame_height,
        })
    }
}

fn parse_frame_header_trailing_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    allow_intrabc: bool,
    frame_type: FrameType,
    error_resilient_mode: bool,
    order_hint: u32,
    reference_frame_indices: [u8; 7],
    references: &[Option<ReferenceFrameState>; 8],
    primary_ref_frame: u8,
) -> Result<FrameHeaderTrailingParams, DecoderError> {
    let quantization = parse_quantization_params(reader, sequence)?;
    let segmentation = parse_segmentation_params(reader, primary_ref_frame)?;
    let delta_q = parse_delta_q_params(reader, quantization.base_q_idx)?;
    let delta_lf = parse_delta_lf_params(reader, delta_q.present, allow_intrabc)?;
    let coded_lossless = quantization.coded_lossless()
        && (!segmentation.enabled
            || (segmentation.delta_q == 0
                && segmentation.segment_delta_q.iter().all(|delta| *delta == 0)));
    let loop_filter = parse_loop_filter_params(reader, sequence, coded_lossless, allow_intrabc)?;
    let cdef = parse_cdef_params(reader, sequence, coded_lossless, allow_intrabc)?;
    let restoration = parse_lr_params(reader, sequence, coded_lossless, allow_intrabc)?;
    let tx_mode = parse_tx_mode(reader, coded_lossless)?;
    let frame_is_intra = frame_type_is_intra(frame_type);
    let reference_select = if frame_is_intra || error_resilient_mode {
        false
    } else {
        reader.read_bool("reference_select")?
    };
    let skip_mode_present = if skip_mode_allowed(
        sequence,
        frame_type,
        error_resilient_mode,
        order_hint,
        reference_select,
        &reference_frame_indices,
        references,
    ) {
        reader.read_bool("skip_mode_present")?
    } else {
        false
    };
    let allow_warped_motion =
        if frame_is_intra || error_resilient_mode || !sequence.enable_warped_motion {
            false
        } else {
            reader.read_bool("allow_warped_motion")?
        };
    let reduced_tx_set = reader.read_bool("reduced_tx_set")?;
    // Global-motion parameters are intentionally left for the prediction
    // decoder.  AVIF frame OBUs in the supported sequence path place the
    // tile-group boundary immediately after this header subset; consuming
    // speculative global-motion syntax here would shift that boundary and
    // make the whole sample undecodable.
    Ok(FrameHeaderTrailingParams {
        quantization,
        segmentation,
        delta_q,
        delta_lf,
        loop_filter,
        cdef,
        restoration,
        tx_mode,
        reference_select,
        skip_mode_present,
        allow_warped_motion,
        reduced_tx_set,
    })
}

fn skip_mode_allowed(
    sequence: &SequenceHeader,
    frame_type: FrameType,
    error_resilient_mode: bool,
    order_hint: u32,
    reference_select: bool,
    reference_frame_indices: &[u8; 7],
    references: &[Option<ReferenceFrameState>; 8],
) -> bool {
    if frame_type_is_intra(frame_type)
        || error_resilient_mode
        || !reference_select
        || !sequence.enable_order_hint
    {
        return false;
    }
    let mut forward = 0usize;
    let mut backward = 0usize;
    for &slot in reference_frame_indices {
        let Some(reference) = references.get(usize::from(slot)).and_then(Option::as_ref) else {
            continue;
        };
        match relative_order_hint_distance(
            sequence.order_hint_bits,
            reference.order_hint,
            order_hint,
        ) {
            distance if distance < 0 => forward += 1,
            distance if distance > 0 => backward += 1,
            _ => {}
        }
    }
    forward > 0 && (backward > 0 || forward > 1)
}

fn relative_order_hint_distance(bits: u8, reference: u32, current: u32) -> i32 {
    if bits == 0 {
        return 0;
    }
    let modulo = 1i32 << bits;
    let mask = modulo - 1;
    let mut distance = ((reference as i32 - current as i32) & mask) as i32;
    if distance & (modulo >> 1) != 0 {
        distance -= modulo;
    }
    distance
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeaderTrailingParams {
    pub quantization: QuantizationParams,
    pub segmentation: SegmentationParams,
    pub delta_q: DeltaQParams,
    pub delta_lf: DeltaLfParams,
    pub loop_filter: LoopFilterParams,
    pub cdef: CdefParams,
    pub restoration: RestorationParams,
    pub tx_mode: TxMode,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationParams {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub using_qmatrix: bool,
    /// Quantizer-matrix levels signalled by the frame header. Level 15 is the
    /// identity matrix and therefore has no effect on dequantization.
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

impl QuantizationParams {
    pub(crate) fn has_unsupported_qmatrix(&self) -> bool {
        self.using_qmatrix
            && [self.qm_y, self.qm_u, self.qm_v]
                .iter()
                .any(|&level| level > 15)
    }

    pub(crate) fn qmatrix_level(&self, plane: usize) -> u8 {
        match plane {
            0 => self.qm_y,
            1 => self.qm_u,
            _ => self.qm_v,
        }
    }

    pub(crate) fn coded_lossless(&self) -> bool {
        self.base_q_idx == 0
            && self.delta_q_y_dc == 0
            && self.delta_q_u_dc == 0
            && self.delta_q_u_ac == 0
            && self.delta_q_v_dc == 0
            && self.delta_q_v_ac == 0
    }
}

fn parse_quantization_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
) -> Result<QuantizationParams, DecoderError> {
    let base_q_idx = reader.read_bits(8, "base_q_idx")? as u8;
    let delta_q_y_dc = read_delta_q(reader, "delta_q_y_dc")?;
    let mut delta_q_u_dc = 0;
    let mut delta_q_u_ac = 0;
    let mut delta_q_v_dc = 0;
    let mut delta_q_v_ac = 0;

    if !sequence.color_config.monochrome {
        let diff_uv_delta = if sequence.color_config.separate_uv_delta_q {
            reader.read_bool("diff_uv_delta")?
        } else {
            false
        };
        delta_q_u_dc = read_delta_q(reader, "delta_q_u_dc")?;
        delta_q_u_ac = read_delta_q(reader, "delta_q_u_ac")?;
        if diff_uv_delta {
            delta_q_v_dc = read_delta_q(reader, "delta_q_v_dc")?;
            delta_q_v_ac = read_delta_q(reader, "delta_q_v_ac")?;
        } else {
            delta_q_v_dc = delta_q_u_dc;
            delta_q_v_ac = delta_q_u_ac;
        }
    }

    let using_qmatrix = reader.read_bool("using_qmatrix")?;
    let mut qm_y = 15;
    let mut qm_u = 15;
    let mut qm_v = 15;
    if using_qmatrix {
        qm_y = reader.read_bits(4, "qm_y")? as u8;
        qm_u = reader.read_bits(4, "qm_u")? as u8;
        if sequence.color_config.separate_uv_delta_q {
            qm_v = reader.read_bits(4, "qm_v")? as u8;
        } else {
            qm_v = qm_u;
        }
    }

    Ok(QuantizationParams {
        base_q_idx,
        delta_q_y_dc,
        delta_q_u_dc,
        delta_q_u_ac,
        delta_q_v_dc,
        delta_q_v_ac,
        using_qmatrix,
        qm_y,
        qm_u,
        qm_v,
    })
}

fn read_delta_q(reader: &mut BitReader<'_>, name: &str) -> Result<i8, DecoderError> {
    if reader.read_bool(name)? {
        Ok(read_inv_signed_literal(reader, 6, name)? as i8)
    } else {
        Ok(0)
    }
}

/// Reads AV1's inverse-signed literal. The syntax's `bits` value excludes the
/// sign bit; the encoded value is a two's-complement literal of `bits + 1`
/// bits, rather than a magnitude followed by a separate sign flag.
fn read_inv_signed_literal(
    reader: &mut BitReader<'_>,
    bits: usize,
    name: &str,
) -> Result<i32, DecoderError> {
    let raw = reader.read_bits(bits + 1, name)? as i32;
    let sign_bit = 1i32 << bits;
    let range = sign_bit << 1;
    Ok(if raw & sign_bit != 0 {
        raw - range
    } else {
        raw
    })
}

fn parse_segmentation_params(
    reader: &mut BitReader<'_>,
    primary_ref_frame: u8,
) -> Result<SegmentationParams, DecoderError> {
    if !reader.read_bool("segmentation_enabled")? {
        return Ok(SegmentationParams::default());
    }

    // Still-image AV1 uses primary_ref_frame == PRIMARY_REF_NONE, so these
    // flags are explicitly coded for the current frame. We parse the full
    // feature table even when it is not used by reconstruction, keeping the
    // reader aligned for the following delta-q and loop-filter syntax.
    let primary_ref_none = primary_ref_frame == 7;
    let update_map = if primary_ref_none {
        true
    } else {
        reader.read_bool("segmentation_update_map")?
    };
    let temporal_update = if update_map && !primary_ref_none {
        reader.read_bool("segmentation_temporal_update")?
    } else {
        false
    };
    let update_data = if primary_ref_none {
        true
    } else {
        reader.read_bool("segmentation_update_data")?
    };
    if !update_data {
        return Ok(SegmentationParams {
            enabled: true,
            update_map,
            temporal_update,
            preskip: false,
            delta_q: 0,
            segment_delta_q: [0; 8],
            segment_delta_lf: [[0; 4]; 8],
            segment_skip: [false; 8],
            last_active_segment: 0,
        });
    }

    const FEATURE_BITS: [usize; 8] = [8, 6, 6, 6, 6, 3, 0, 0];
    const FEATURE_SIGNED: [bool; 8] = [true, true, true, true, true, false, false, false];
    let mut segment_delta_q = [0i16; 8];
    let mut segment_delta_lf = [[0i8; 4]; 8];
    let mut segment_skip = [false; 8];
    let mut preskip = false;
    let mut last_active_segment = 0u8;
    for segment in 0..8 {
        for feature in 0..8 {
            let enabled = reader.read_bool("segmentation_feature_enabled")?;
            if !enabled {
                continue;
            }
            let bits = FEATURE_BITS[feature];
            if bits != 0 {
                let value = if FEATURE_SIGNED[feature] {
                    read_inv_signed_literal(reader, bits, "segmentation_feature_value")?
                } else {
                    reader.read_bits(bits, "segmentation_feature_value")? as i32
                };
                match feature {
                    0 => segment_delta_q[segment] = value as i16,
                    1..=4 => segment_delta_lf[segment][feature - 1] = value as i8,
                    _ => {}
                }
            }
            match feature {
                0..=4 => last_active_segment = segment as u8,
                5 => {
                    preskip = true;
                    last_active_segment = segment as u8;
                }
                6 => {
                    segment_skip[segment] = true;
                    preskip = true;
                    last_active_segment = segment as u8;
                }
                7 => {
                    preskip = true;
                    last_active_segment = segment as u8;
                }
                _ => unreachable!("segmentation feature index is bounded to 0..8"),
            }
        }
    }
    Ok(SegmentationParams {
        enabled: true,
        update_map,
        temporal_update,
        preskip,
        delta_q: segment_delta_q[0],
        segment_delta_q,
        segment_delta_lf,
        segment_skip,
        last_active_segment,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeltaQParams {
    pub present: bool,
    pub res: u8,
}

fn parse_delta_q_params(
    reader: &mut BitReader<'_>,
    base_q_idx: u8,
) -> Result<DeltaQParams, DecoderError> {
    let present = base_q_idx > 0 && reader.read_bool("delta_q_present")?;
    let res = if present {
        reader.read_bits(2, "delta_q_res")? as u8
    } else {
        0
    };
    Ok(DeltaQParams { present, res })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeltaLfParams {
    pub present: bool,
    pub res: u8,
    pub multi: bool,
}

fn parse_delta_lf_params(
    reader: &mut BitReader<'_>,
    delta_q_present: bool,
    allow_intrabc: bool,
) -> Result<DeltaLfParams, DecoderError> {
    let present = delta_q_present && !allow_intrabc && reader.read_bool("delta_lf_present")?;
    if !present {
        return Ok(DeltaLfParams::default());
    }
    Ok(DeltaLfParams {
        present,
        res: reader.read_bits(2, "delta_lf_res")? as u8,
        multi: reader.read_bool("delta_lf_multi")?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopFilterParams {
    pub levels: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
    pub ref_deltas: [i8; 8],
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilterParams {
    fn default() -> Self {
        Self {
            levels: [0; 4],
            sharpness: 0,
            delta_enabled: false,
            delta_update: false,
            ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            mode_deltas: [0; 2],
        }
    }
}

fn parse_loop_filter_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<LoopFilterParams, DecoderError> {
    if coded_lossless || allow_intrabc {
        return Ok(LoopFilterParams::default());
    }
    let mut levels = [0u8; 4];
    levels[0] = reader.read_bits(6, "loop_filter_level[0]")? as u8;
    levels[1] = reader.read_bits(6, "loop_filter_level[1]")? as u8;
    if !sequence.color_config.monochrome && (levels[0] != 0 || levels[1] != 0) {
        levels[2] = reader.read_bits(6, "loop_filter_level[2]")? as u8;
        levels[3] = reader.read_bits(6, "loop_filter_level[3]")? as u8;
    }
    let sharpness = reader.read_bits(3, "loop_filter_sharpness")? as u8;
    let delta_enabled = reader.read_bool("loop_filter_delta_enabled")?;
    let delta_update = delta_enabled && reader.read_bool("loop_filter_delta_update")?;
    let mut ref_deltas = [1i8, 0, 0, 0, -1, 0, -1, -1];
    let mut mode_deltas = [0i8; 2];
    if delta_update {
        for (index, delta) in ref_deltas.iter_mut().enumerate() {
            if reader.read_bool(&format!("update_ref_delta[{index}]"))? {
                *delta = read_signed_delta(reader, &format!("loop_filter_ref_delta[{index}]"))?;
            }
        }
        for (index, delta) in mode_deltas.iter_mut().enumerate() {
            if reader.read_bool(&format!("update_mode_delta[{index}]"))? {
                *delta = read_signed_delta(reader, &format!("loop_filter_mode_delta[{index}]"))?;
            }
        }
    }
    Ok(LoopFilterParams {
        levels,
        sharpness,
        delta_enabled,
        delta_update,
        ref_deltas,
        mode_deltas,
    })
}

fn read_signed_delta(reader: &mut BitReader<'_>, name: &str) -> Result<i8, DecoderError> {
    let magnitude = reader.read_bits(6, name)? as i8;
    if magnitude == 0 {
        return Ok(0);
    }
    let negative = reader.read_bool(name)?;
    Ok(if negative { -magnitude } else { magnitude })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdefStrength {
    pub y_pri: u8,
    pub y_sec: u8,
    pub uv_pri: u8,
    pub uv_sec: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdefParams {
    pub enabled: bool,
    pub damping: u8,
    pub bits: u8,
    pub strengths: [CdefStrength; 8],
}

impl Default for CdefParams {
    fn default() -> Self {
        Self {
            enabled: false,
            damping: 0,
            bits: 0,
            strengths: [CdefStrength {
                y_pri: 0,
                y_sec: 0,
                uv_pri: 0,
                uv_sec: 0,
            }; 8],
        }
    }
}

fn parse_cdef_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<CdefParams, DecoderError> {
    if !sequence.enable_cdef || coded_lossless || allow_intrabc {
        return Ok(CdefParams::default());
    }
    let damping = reader.read_bits(2, "cdef_damping_minus_3")? as u8 + 3;
    let cdef_bits = reader.read_bits(2, "cdef_bits")?;
    let mut strengths = CdefParams::default().strengths;
    for strength in strengths.iter_mut().take(1usize << cdef_bits) {
        strength.y_pri = reader.read_bits(4, "cdef_y_pri_strength")? as u8;
        strength.y_sec = match reader.read_bits(2, "cdef_y_sec_strength")? as u8 {
            3 => 4,
            value => value,
        };
        if !sequence.color_config.monochrome {
            strength.uv_pri = reader.read_bits(4, "cdef_uv_pri_strength")? as u8;
            strength.uv_sec = match reader.read_bits(2, "cdef_uv_sec_strength")? as u8 {
                3 => 4,
                value => value,
            };
        }
    }
    Ok(CdefParams {
        enabled: true,
        damping,
        bits: cdef_bits as u8,
        strengths,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RestorationParams {
    pub uses_lr: bool,
    pub lr_type: [u8; 3],
    pub unit_shift: u8,
    /// Additional chroma unit-size shift signalled for subsampled planes.
    pub uv_unit_shift: u8,
}

fn parse_lr_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<RestorationParams, DecoderError> {
    if !sequence.enable_restoration || coded_lossless || allow_intrabc {
        return Ok(RestorationParams::default());
    }
    let planes = if sequence.color_config.monochrome {
        1
    } else {
        3
    };
    let mut uses_lr = false;
    let mut lr_type = [0u8; 3];
    for plane_lr_type in lr_type.iter_mut().take(planes) {
        *plane_lr_type = match reader.read_bits(2, "lr_type")? {
            0 => 0,
            1 => 3,
            2 => 1,
            3 => 2,
            _ => unreachable!("two-bit restoration type"),
        };
        uses_lr |= *plane_lr_type != 0;
    }
    let mut unit_shift = 0;
    if uses_lr {
        unit_shift = reader.read_bool("lr_unit_shift")? as u8;
        if !sequence.use_128x128_superblock && unit_shift == 1 {
            unit_shift += reader.read_bool("lr_unit_extra_shift")? as u8;
        }
    }
    let subsampling =
        usize::from(sequence.color_config.subsampling_x && sequence.color_config.subsampling_y);
    let chroma_uses_lr = planes > 1 && (lr_type[1] != 0 || lr_type[2] != 0);
    let uv_unit_shift = if subsampling != 0 && chroma_uses_lr {
        reader.read_bool("lr_uv_shift")? as u8
    } else {
        0
    };
    Ok(RestorationParams {
        uses_lr,
        lr_type,
        unit_shift,
        uv_unit_shift,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxMode {
    Only4x4,
    Largest,
    Select,
}

fn parse_tx_mode(reader: &mut BitReader<'_>, coded_lossless: bool) -> Result<TxMode, DecoderError> {
    if coded_lossless {
        return Ok(TxMode::Only4x4);
    }
    if reader.read_bool("tx_mode")? {
        Ok(TxMode::Select)
    } else {
        Ok(TxMode::Largest)
    }
}

fn frame_type_is_intra(frame_type: FrameType) -> bool {
    matches!(frame_type, FrameType::Key | FrameType::IntraOnly)
}

#[cfg(test)]
mod tests {
    use super::{
        LoopFilterParams, SegmentationParams, parse_inter_reference_indices,
        parse_segmentation_params, parse_show_existing_frame_index, read_inv_signed_literal,
        read_signed_delta,
    };
    use crate::av1::bitstream::BitReader;

    fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
        let mut data = vec![0u8; bits.len().div_ceil(8)];
        for (index, bit) in bits.iter().copied().enumerate() {
            if bit {
                data[index / 8] |= 1 << (7 - index % 8);
            }
        }
        data
    }

    #[test]
    fn parses_inter_reference_indices_for_both_signalling_modes() {
        let (short, indices) = parse_inter_reference_indices(&mut BitReader::new(&[0; 4]), false)
            .expect("explicit inter references should parse");
        assert!(!short);
        assert_eq!(indices, [0; 7]);

        let mut bits = Vec::new();
        bits.push(true);
        push_unsigned(&mut bits, 3, 3);
        push_unsigned(&mut bits, 5, 3);
        let (short, indices) =
            parse_inter_reference_indices(&mut BitReader::new(&bits_to_bytes(&bits)), true)
                .expect("short inter references should parse");
        assert!(short);
        assert_eq!(indices[0], 3);
        assert_eq!(indices[3], 5);
    }

    fn push_unsigned(bits: &mut Vec<bool>, value: u32, width: usize) {
        bits.extend((0..width).rev().map(|bit| value & (1 << bit) != 0));
    }

    fn push_inv_signed(bits: &mut Vec<bool>, value: i32, width_without_sign: usize) {
        let width = width_without_sign + 1;
        let encoded = if value < 0 {
            (1i32 << width) + value
        } else {
            value
        };
        push_unsigned(bits, encoded as u32, width);
    }

    #[test]
    fn loop_filter_defaults_use_the_intra_reference_delta() {
        let params = LoopFilterParams::default();
        assert_eq!(params.ref_deltas[0], 1);
        assert_eq!(params.ref_deltas, [1, 0, 0, 0, -1, 0, -1, -1]);
        assert_eq!(params.mode_deltas, [0; 2]);
    }

    #[test]
    fn signed_loop_filter_delta_reads_magnitude_and_sign() {
        let mut positive = BitReader::new(&[0b0001_0100]);
        assert_eq!(read_signed_delta(&mut positive, "delta").unwrap(), 5);

        let mut negative = BitReader::new(&[0b0001_0110]);
        assert_eq!(read_signed_delta(&mut negative, "delta").unwrap(), -5);
    }

    #[test]
    fn inverse_signed_literal_uses_twos_complement() {
        let mut positive = BitReader::new(&[0x0a]);
        assert_eq!(
            read_inv_signed_literal(&mut positive, 6, "delta").unwrap(),
            5
        );

        let mut negative = BitReader::new(&[0xf6]);
        assert_eq!(
            read_inv_signed_literal(&mut negative, 6, "delta").unwrap(),
            -5
        );
    }

    #[test]
    fn parses_noop_segmentation_params() {
        let mut reader = BitReader::new(&[0x80, 0, 0, 0, 0, 0, 0, 0, 0]);
        let params = parse_segmentation_params(&mut reader, 7).unwrap();
        assert_eq!(
            params,
            SegmentationParams {
                enabled: true,
                update_map: true,
                temporal_update: false,
                preskip: false,
                delta_q: 0,
                segment_delta_q: [0; 8],
                segment_delta_lf: [[0; 4]; 8],
                segment_skip: [false; 8],
                last_active_segment: 0,
            }
        );
    }

    #[test]
    fn parses_segment_zero_alt_q_segmentation_feature() {
        let mut bits = vec![true, true];
        push_inv_signed(&mut bits, 5, 8);
        bits.extend(std::iter::repeat_n(false, 63));
        let data = bits_to_bytes(&bits);
        let mut reader = BitReader::new(&data);
        let params = parse_segmentation_params(&mut reader, 7).unwrap();
        assert_eq!(params.delta_q, 5);
        assert_eq!(params.effective_qindex(100), 105);

        let mut bits = vec![true];
        bits.extend(std::iter::repeat_n(false, 64));
        let data = bits_to_bytes(&bits);
        let mut reader = BitReader::new(&data);
        let params = parse_segmentation_params(&mut reader, 7).unwrap();
        assert_eq!(params.delta_q, 0);

        let mut bits = vec![true, true];
        push_inv_signed(&mut bits, -5, 8);
        bits.extend(std::iter::repeat_n(false, 63));
        let data = bits_to_bytes(&bits);
        let mut reader = BitReader::new(&data);
        let params = parse_segmentation_params(&mut reader, 7).unwrap();
        assert_eq!(params.delta_q, -5);
        assert_eq!(params.effective_qindex(3), 0);
    }

    #[test]
    fn parses_segment_loop_filter_feature() {
        let mut bits = vec![true];
        for segment in 0..8 {
            for feature in 0..8 {
                let enabled = segment == 0 && (1..=4).contains(&feature);
                bits.push(enabled);
                if enabled {
                    let value = [0, 5, 3, 7, 9][feature];
                    let signed = if matches!(feature, 2 | 4) {
                        -(value as i32)
                    } else {
                        value as i32
                    };
                    push_inv_signed(&mut bits, signed, 6);
                }
            }
        }
        let data = bits_to_bytes(&bits);

        let mut reader = BitReader::new(&data);
        let params = parse_segmentation_params(&mut reader, 7).unwrap();
        assert_eq!(params.segment_delta_lf[0], [5, -3, 7, -9]);
        assert_eq!(params.last_active_segment, 0);
    }

    #[test]
    fn accepts_alt_q_on_later_segment() {
        let mut reader = BitReader::new(&[0x80, 0x40, 0, 0, 0, 0, 0, 0, 0, 0]);
        let params = parse_segmentation_params(&mut reader, 7).unwrap();
        assert_eq!(params.segment_delta_q, [0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(params.last_active_segment, 1);
    }

    #[test]
    fn parses_alt_q_deltas_for_multiple_segments() {
        let mut bits = vec![true];
        for segment in 0..8 {
            for feature in 0..8 {
                let enabled = feature == 0 && segment < 2;
                bits.push(enabled);
                if enabled {
                    let value = if segment == 0 { 5u8 } else { 3u8 };
                    let signed = if segment == 1 {
                        -(value as i32)
                    } else {
                        value as i32
                    };
                    push_inv_signed(&mut bits, signed, 8);
                }
            }
        }
        let data = bits_to_bytes(&bits);

        let mut reader = BitReader::new(&data);
        let params = parse_segmentation_params(&mut reader, 7).unwrap();
        assert_eq!(params.segment_delta_q, [5, -3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(params.last_active_segment, 1);
        assert_eq!(params.effective_qindex_for_segment(10, 1), 7);
    }

    #[test]
    fn accepts_segment_skip_as_a_pre_skip_feature() {
        let mut bits = vec![true];
        for segment in 0..8 {
            for feature in 0..8 {
                let enabled = segment == 1 && feature == 6;
                bits.push(enabled);
            }
        }
        let mut data = vec![0u8; bits.len().div_ceil(8)];
        for (index, bit) in bits.into_iter().enumerate() {
            if bit {
                data[index / 8] |= 1 << (7 - index % 8);
            }
        }

        let mut reader = BitReader::new(&data);
        let params = parse_segmentation_params(&mut reader, 7).unwrap();
        assert!(params.preskip);
        assert!(params.segment_skip[1]);
        assert_eq!(params.last_active_segment, 1);
    }

    #[test]
    fn accepts_reference_and_globalmv_segmentation_features_for_still_headers() {
        for feature in [5, 7] {
            let mut bits = vec![true];
            for segment in 0..8 {
                for current_feature in 0..8 {
                    bits.push(segment == 0 && current_feature == feature);
                    if segment == 0 && current_feature == feature && current_feature == 5 {
                        bits.extend([false, false, false]);
                    }
                }
            }
            let mut data = vec![0u8; bits.len().div_ceil(8)];
            for (index, bit) in bits.into_iter().enumerate() {
                if bit {
                    data[index / 8] |= 1 << (7 - index % 8);
                }
            }
            let mut reader = BitReader::new(&data);
            let params = parse_segmentation_params(&mut reader, 7).unwrap();
            assert!(params.preskip);
            assert_eq!(params.last_active_segment, 0);
        }
    }

    #[test]
    fn reads_show_existing_frame_slot_prefix() {
        assert_eq!(
            parse_show_existing_frame_index(&[0b1001_0000]).unwrap(),
            Some(1)
        );
        assert_eq!(
            parse_show_existing_frame_index(&[0b0100_0000]).unwrap(),
            None
        );
    }
}

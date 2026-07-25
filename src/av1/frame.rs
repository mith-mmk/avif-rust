use super::bitstream::BitReader;
use super::sequence::SequenceHeader;
use super::syntax::BlockSize;
use super::tile::{TileInfo, parse_tile_info};
use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Key,
    Inter,
    IntraOnly,
    Switch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationFilter {
    Regular,
    Smooth,
    Sharp,
    Bilinear,
    /// The filter is selected per inter block by the switchable filter CDF.
    Switchable,
}

impl InterpolationFilter {
    fn from_bits(value: u32) -> Result<Self, DecoderError> {
        match value {
            0 => Ok(Self::Regular),
            1 => Ok(Self::Smooth),
            2 => Ok(Self::Sharp),
            3 => Ok(Self::Bilinear),
            _ => Err(DecoderError::Bitstream(format!(
                "interpolation_filter {value} is reserved"
            ))),
        }
    }

    pub(crate) fn from_switchable_symbol(value: usize) -> Result<Self, DecoderError> {
        match value {
            0 => Ok(Self::Regular),
            1 => Ok(Self::Smooth),
            2 => Ok(Self::Sharp),
            _ => Err(DecoderError::Bitstream(format!(
                "switchable interpolation filter {value} is invalid"
            ))),
        }
    }
}

/// AV1 global-motion model signalled for each of the seven inter references.
/// The matrices use the codec's 16-bit warped-model precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalMotionType {
    Identity,
    Translation,
    RotZoom,
    Affine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalMotionParams {
    pub types: [GlobalMotionType; 7],
    pub matrices: [[i32; 6]; 7],
}

impl Default for GlobalMotionParams {
    fn default() -> Self {
        Self {
            types: [GlobalMotionType::Identity; 7],
            matrices: [[0, 0, 1 << 16, 0, 0, 1 << 16]; 7],
        }
    }
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
    /// Order hints for the seven reference types, retained for compound
    /// distance-weighted blending.
    pub reference_order_hints: [Option<u32>; 7],
    pub frame_refs_short_signaling: bool,
    /// Current frame identifier when sequence-level frame-id signalling is
    /// enabled. `None` means the sequence does not carry frame IDs.
    pub frame_id: Option<u16>,
    pub allow_high_precision_mv: bool,
    pub is_filter_switchable: bool,
    pub interpolation_filter: InterpolationFilter,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    pub allow_warped_motion: bool,
    pub global_motion: GlobalMotionParams,
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
    pub film_grain: Option<FilmGrainParams>,
    pub frame_id: Option<u16>,
    pub global_motion: GlobalMotionParams,
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
/// values are also consumed for still-image headers; the AVIS path supplies
/// decoded reference slots for supported inter/switch reconstruction, while
/// unsupported prediction syntax remains fail-closed. Inherited film-grain
/// parameters are resolved from the stored reference metadata.
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
        let film_grain = parse_film_grain_params(
            &mut reader,
            sequence,
            FrameType::Key,
            true,
            false,
            &[None; 8],
        )?;
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
            reference_order_hints: [None; 7],
            frame_refs_short_signaling: false,
            frame_id: None,
            allow_high_precision_mv: false,
            is_filter_switchable: false,
            interpolation_filter: InterpolationFilter::Regular,
            is_motion_mode_switchable: false,
            use_ref_frame_mvs: false,
            reference_select: false,
            skip_mode_present: false,
            allow_warped_motion: false,
            global_motion: GlobalMotionParams::default(),
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
    let refresh_frame_flags =
        if (frame_type == FrameType::Key && show_frame) || frame_type == FrameType::Switch {
            0xff
        } else {
            reader.read_bits(8, "refresh_frame_flags")? as u8
        };
    let frame_id = read_current_frame_id(&mut reader, sequence)?;

    let frame_is_intra = frame_type_is_intra(frame_type);
    let mut reference_frame_indices = [0; 7];
    let mut reference_order_hints = [None; 7];
    let frame_order_hints = if error_resilient_mode
        && sequence.enable_order_hint
        && (!frame_is_intra || refresh_frame_flags != 0xff)
    {
        let mut frame_order_hints = [0u32; 8];
        for order_hint in &mut frame_order_hints {
            *order_hint = reader.read_bits(sequence.order_hint_bits as usize, "ref_order_hint")?;
        }
        Some(frame_order_hints)
    } else {
        None
    };
    let mut frame_refs_short_signaling = false;
    let mut allow_high_precision_mv = false;
    let mut is_filter_switchable = false;
    let mut interpolation_filter = InterpolationFilter::Regular;
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
        reference_order_hints = if let Some(frame_order_hints) = frame_order_hints {
            std::array::from_fn(|index| {
                frame_order_hints
                    .get(usize::from(reference_frame_indices[index]))
                    .copied()
            })
        } else {
            std::array::from_fn(|index| {
                references
                    .get(usize::from(reference_frame_indices[index]))
                    .and_then(|reference| reference.as_ref())
                    .map(|reference| reference.order_hint)
            })
        };
        validate_reference_frame_ids(frame_id, sequence, &reference_frame_indices, references)?;
        let (frame_size, render_size) = parse_inter_frame_size(
            &mut reader,
            sequence,
            frame_size_override_flag,
            error_resilient_mode,
            &reference_frame_indices,
            references,
        )?;
        allow_high_precision_mv = if force_integer_mv == 1 {
            false
        } else {
            reader.read_bool("allow_high_precision_mv")?
        };
        is_filter_switchable = reader.read_bool("is_filter_switchable")?;
        interpolation_filter = if is_filter_switchable {
            InterpolationFilter::Switchable
        } else {
            InterpolationFilter::from_bits(reader.read_bits(2, "interpolation_filter")?)?
        };
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
    let global_motion = if !frame_is_intra {
        read_global_motion_params(
            &mut reader,
            allow_high_precision_mv,
            &reference_frame_indices,
            references,
        )?
    } else {
        GlobalMotionParams::default()
    };
    let film_grain = parse_film_grain_params(
        &mut reader,
        sequence,
        frame_type,
        show_frame,
        showable_frame,
        references,
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
        reference_order_hints,
        frame_refs_short_signaling,
        frame_id,
        allow_high_precision_mv,
        is_filter_switchable,
        interpolation_filter,
        is_motion_mode_switchable,
        use_ref_frame_mvs,
        reference_select: trailing.reference_select,
        skip_mode_present: trailing.skip_mode_present,
        allow_warped_motion: trailing.allow_warped_motion,
        global_motion,
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

fn read_current_frame_id(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
) -> Result<Option<u16>, DecoderError> {
    if sequence.frame_id_numbers_present {
        Ok(Some(
            reader.read_bits(sequence.frame_id_length as usize, "current_frame_id")? as u16,
        ))
    } else {
        Ok(None)
    }
}

fn validate_reference_frame_ids(
    current_frame_id: Option<u16>,
    sequence: &SequenceHeader,
    reference_frame_indices: &[u8; 7],
    references: &[Option<ReferenceFrameState>; 8],
) -> Result<(), DecoderError> {
    let Some(current_frame_id) = current_frame_id else {
        return Ok(());
    };
    let modulus = 1u32 << sequence.frame_id_length;
    let max_age = 1u32 << sequence.delta_frame_id_length;
    let current = u32::from(current_frame_id);
    for &slot in reference_frame_indices {
        let Some(reference) = references.get(usize::from(slot)).and_then(Option::as_ref) else {
            continue;
        };
        let Some(reference_frame_id) = reference.frame_id else {
            return Err(DecoderError::Bitstream(format!(
                "AV1 reference frame slot {slot} has no frame ID"
            )));
        };
        let reference = u32::from(reference_frame_id);
        let age = (current + modulus - reference) % modulus;
        if age > max_age {
            return Err(DecoderError::Bitstream(format!(
                "AV1 reference frame slot {slot} has stale frame ID {reference_frame_id}"
            )));
        }
    }
    Ok(())
}

fn read_global_motion_params(
    reader: &mut BitReader<'_>,
    allow_high_precision_mv: bool,
    reference_frame_indices: &[u8; 7],
    references: &[Option<ReferenceFrameState>; 8],
) -> Result<GlobalMotionParams, DecoderError> {
    const WARPEDMODEL_PREC_BITS: i32 = 16;
    const GM_ALPHA_MAX: i32 = 1 << 12;
    const GM_ALPHA_PREC_BITS: i32 = 15;
    const GM_ALPHA_PREC_DIFF: i32 = WARPEDMODEL_PREC_BITS - GM_ALPHA_PREC_BITS;
    const GM_ALPHA_DECODE_FACTOR: i32 = 1 << GM_ALPHA_PREC_DIFF;
    const GM_ABS_TRANS_ONLY_BITS: usize = 9;
    const GM_TRANS_ONLY_DECODE_FACTOR: i32 = 1 << 13;
    const GM_TRANS_ONLY_PREC_DIFF: i32 = 13;
    const GM_ABS_TRANS_BITS: usize = 12;
    const GM_TRANS_DECODE_FACTOR: i32 = 1 << 10;
    const GM_TRANS_PREC_DIFF: i32 = 10;
    const SUBEXP_K: usize = 3;

    let mut result = GlobalMotionParams::default();
    for reference in 0..7 {
        // The first bit is the identity/non-identity discriminator.  AOM's
        // decoder names this `type` rather than `is_global`.
        if !reader.read_bool("global_motion_type")? {
            continue;
        }
        let motion_type = if reader.read_bool("global_motion_is_rot_zoom")? {
            GlobalMotionType::RotZoom
        } else if reader.read_bool("global_motion_is_translation")? {
            GlobalMotionType::Translation
        } else {
            GlobalMotionType::Affine
        };
        let mut matrix = references
            .get(usize::from(
                *reference_frame_indices.get(reference).ok_or_else(|| {
                    DecoderError::Bitstream(
                        "AV1 global motion reference index is invalid".to_string(),
                    )
                })?,
            ))
            .and_then(Option::as_ref)
            .map(|state| state.global_motion.matrices[reference])
            .unwrap_or(result.matrices[reference]);
        match motion_type {
            GlobalMotionType::Identity => {}
            GlobalMotionType::RotZoom | GlobalMotionType::Affine => {
                let alpha_n = (GM_ALPHA_MAX + 1) as usize;
                let alpha_ref = (matrix[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS);
                matrix[2] = read_signed_primitive_refsubexpfin(
                    reader,
                    alpha_n,
                    SUBEXP_K,
                    alpha_ref,
                    "global_motion_alpha",
                )? * GM_ALPHA_DECODE_FACTOR
                    + (1 << WARPEDMODEL_PREC_BITS);
                let alpha_ref = matrix[3] >> GM_ALPHA_PREC_DIFF;
                matrix[3] = read_signed_primitive_refsubexpfin(
                    reader,
                    alpha_n,
                    SUBEXP_K,
                    alpha_ref,
                    "global_motion_alpha",
                )? * GM_ALPHA_DECODE_FACTOR;
            }
            GlobalMotionType::Translation => {}
        }
        if motion_type == GlobalMotionType::Affine {
            let alpha_n = (GM_ALPHA_MAX + 1) as usize;
            let alpha_ref = matrix[4] >> GM_ALPHA_PREC_DIFF;
            matrix[4] = read_signed_primitive_refsubexpfin(
                reader,
                alpha_n,
                SUBEXP_K,
                alpha_ref,
                "global_motion_alpha",
            )? * GM_ALPHA_DECODE_FACTOR;
            let alpha_ref = (matrix[5] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS);
            matrix[5] = read_signed_primitive_refsubexpfin(
                reader,
                alpha_n,
                SUBEXP_K,
                alpha_ref,
                "global_motion_alpha",
            )? * GM_ALPHA_DECODE_FACTOR
                + (1 << WARPEDMODEL_PREC_BITS);
        } else if motion_type == GlobalMotionType::RotZoom {
            matrix[4] = -matrix[3];
            matrix[5] = matrix[2];
        }
        if matches!(
            motion_type,
            GlobalMotionType::Translation | GlobalMotionType::RotZoom | GlobalMotionType::Affine
        ) {
            let (trans_bits, trans_factor, trans_prec_diff) =
                if motion_type == GlobalMotionType::Translation {
                    (
                        GM_ABS_TRANS_ONLY_BITS - usize::from(!allow_high_precision_mv),
                        GM_TRANS_ONLY_DECODE_FACTOR * (1 << usize::from(!allow_high_precision_mv)),
                        GM_TRANS_ONLY_PREC_DIFF + i32::from(!allow_high_precision_mv),
                    )
                } else {
                    (
                        GM_ABS_TRANS_BITS,
                        GM_TRANS_DECODE_FACTOR,
                        GM_TRANS_PREC_DIFF,
                    )
                };
            let trans_n = (1usize << trans_bits) + 1;
            for index in 0..2 {
                let trans_ref = matrix[index] >> trans_prec_diff;
                matrix[index] = read_signed_primitive_refsubexpfin(
                    reader,
                    trans_n,
                    SUBEXP_K,
                    trans_ref,
                    "global_motion_translation",
                )? * trans_factor;
            }
        }
        result.types[reference] = motion_type;
        result.matrices[reference] = matrix;
    }
    Ok(result)
}

fn read_signed_primitive_refsubexpfin(
    reader: &mut BitReader<'_>,
    n: usize,
    k: usize,
    reference: i32,
    name: &str,
) -> Result<i32, DecoderError> {
    let reference = reference
        .checked_add(
            i32::try_from(n - 1)
                .map_err(|_| DecoderError::InvalidParam(format!("{name} range is too large")))?,
        )
        .ok_or_else(|| DecoderError::Bitstream(format!("{name} reference overflows")))?;
    let scaled_n = n
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| DecoderError::InvalidParam(format!("{name} range is too large")))?;
    let value = read_primitive_refsubexpfin(reader, scaled_n, k, reference as usize, name)?;
    Ok(i32::try_from(value)
        .map_err(|_| DecoderError::Bitstream(format!("{name} value overflows")))?
        - i32::try_from(n - 1)
            .map_err(|_| DecoderError::InvalidParam(format!("{name} range is too large")))?)
}

fn read_primitive_refsubexpfin(
    reader: &mut BitReader<'_>,
    n: usize,
    k: usize,
    reference: usize,
    name: &str,
) -> Result<usize, DecoderError> {
    let value = read_primitive_subexpfin(reader, n, k, name)?;
    Ok(inv_recenter_finite_nonneg(n, reference.min(n - 1), value))
}

fn read_primitive_subexpfin(
    reader: &mut BitReader<'_>,
    n: usize,
    k: usize,
    name: &str,
) -> Result<usize, DecoderError> {
    let mut index = 0usize;
    let mut mk = 0usize;
    loop {
        let bits = if index == 0 { k } else { k + index - 1 };
        let step = 1usize << bits;
        if n <= mk + 3 * step {
            return read_primitive_quniform(reader, n - mk, name).map(|value| value + mk);
        }
        if !reader.read_bool(name)? {
            return Ok(reader.read_bits(bits, name)? as usize + mk);
        }
        index += 1;
        mk += step;
    }
}

fn read_primitive_quniform(
    reader: &mut BitReader<'_>,
    n: usize,
    name: &str,
) -> Result<usize, DecoderError> {
    if n <= 1 {
        return Ok(0);
    }
    let bits = usize::BITS as usize - n.leading_zeros() as usize;
    let threshold = (1usize << bits) - n;
    let value = reader.read_bits(bits - 1, name)? as usize;
    if value < threshold {
        Ok(value)
    } else {
        Ok((value << 1) - threshold + reader.read_bits(1, name)? as usize)
    }
}

fn inv_recenter_finite_nonneg(n: usize, reference: usize, value: usize) -> usize {
    let inv_recenter = |r: usize, v: usize| {
        if v > (r << 1) {
            v
        } else if v & 1 == 0 {
            (v >> 1) + r
        } else {
            r - ((v + 1) >> 1)
        }
    };
    if (reference << 1) <= n {
        inv_recenter(reference, value)
    } else {
        n - 1 - inv_recenter(n - 1 - reference, value)
    }
}

impl GlobalMotionParams {
    pub(crate) fn motion_vector(
        &self,
        reference: usize,
        block_size: BlockSize,
        x: usize,
        y: usize,
        allow_high_precision_mv: bool,
        force_integer_mv: bool,
    ) -> Result<(i32, i32), DecoderError> {
        let motion_type = *self.types.get(reference).ok_or_else(|| {
            DecoderError::Bitstream("AV1 global motion reference is invalid".to_string())
        })?;
        let matrix = *self.matrices.get(reference).ok_or_else(|| {
            DecoderError::Bitstream("AV1 global motion matrix is invalid".to_string())
        })?;
        if motion_type == GlobalMotionType::Identity {
            return Ok((0, 0));
        }
        if motion_type == GlobalMotionType::Translation {
            let row = matrix[0] >> 13;
            let col = matrix[1] >> 13;
            return Ok(if force_integer_mv {
                (round_mv_to_integer(row), round_mv_to_integer(col))
            } else {
                (row, col)
            });
        }

        let center_x = i64::try_from(x)
            .map_err(|_| DecoderError::InvalidParam("AV1 global motion x overflows".to_string()))?
            + i64::try_from(block_size.width() / 2 - 1)
                .map_err(|_| DecoderError::InvalidParam("AV1 block width overflows".to_string()))?;
        let center_y = i64::try_from(y)
            .map_err(|_| DecoderError::InvalidParam("AV1 global motion y overflows".to_string()))?
            + i64::try_from(block_size.height() / 2 - 1).map_err(|_| {
                DecoderError::InvalidParam("AV1 block height overflows".to_string())
            })?;
        let xc = (i64::from(matrix[2]) - (1_i64 << 16)) * center_x
            + i64::from(matrix[3]) * center_y
            + i64::from(matrix[0]);
        let yc = i64::from(matrix[4]) * center_x
            + (i64::from(matrix[5]) - (1_i64 << 16)) * center_y
            + i64::from(matrix[1]);
        let precision = if allow_high_precision_mv { 13 } else { 14 };
        let mut row = round_power_of_two_signed(yc, precision);
        let mut col = round_power_of_two_signed(xc, precision);
        if !allow_high_precision_mv {
            row *= 2;
            col *= 2;
        }
        if force_integer_mv {
            row = round_mv_to_integer(row);
            col = round_mv_to_integer(col);
        }
        Ok((row, col))
    }
}

fn round_power_of_two_signed(value: i64, bits: u32) -> i32 {
    let offset = 1_i64 << bits.saturating_sub(1);
    let rounded = if value < 0 {
        -((-value + offset) >> bits)
    } else {
        (value + offset) >> bits
    };
    rounded.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn round_mv_to_integer(value: i32) -> i32 {
    let remainder = value % 8;
    if remainder == 0 {
        value
    } else if remainder.abs() > 4 {
        value - remainder + if remainder > 0 { 8 } else { -8 }
    } else {
        value - remainder
    }
}

/// Reads the small prefix that identifies an AV1 `show_existing_frame` OBU.
/// The full coded-frame parser intentionally remains fail-closed for this
/// sequence-only prefix; the AVIS sequence dispatcher resolves the referenced
/// frame from its slot before decoding the complete sample.
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
    references: &[Option<ReferenceFrameState>; 8],
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
        let reference_index = reader.read_bits(3, "film_grain_params_ref_idx")? as usize;
        let reference = references
            .get(reference_index)
            .and_then(Option::as_ref)
            .and_then(|reference| reference.film_grain)
            .ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 inherited film grain reference slot {reference_index} is unavailable"
                ))
            })?;
        let mut inherited = reference;
        inherited.random_seed = random_seed;
        return Ok(Some(inherited));
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CdefStrength {
    pub y_pri: u8,
    pub y_sec: u8,
    pub uv_pri: u8,
    pub uv_sec: u8,
    /// CDEF skip flags are retained in the private model; legacy AVIF
    /// reduced-header writers omit these syntax bits and therefore default
    /// them to false during parsing.
    pub y_filter_skip: bool,
    pub uv_filter_skip: bool,
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
                y_filter_skip: false,
                uv_filter_skip: false,
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
        strength.y_sec = cdef_secondary_strength(reader.read_bits(2, "cdef_y_sec_strength")? as u8);
        if !sequence.color_config.monochrome {
            strength.uv_pri = reader.read_bits(4, "cdef_uv_pri_strength")? as u8;
            strength.uv_sec =
                cdef_secondary_strength(reader.read_bits(2, "cdef_uv_sec_strength")? as u8);
        }
    }
    Ok(CdefParams {
        enabled: true,
        damping,
        bits: cdef_bits as u8,
        strengths,
    })
}

#[inline]
fn cdef_secondary_strength(value: u8) -> u8 {
    match value {
        3 => 4,
        value => value,
    }
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
        FilmGrainParams, FrameType, GlobalMotionParams, GlobalMotionType, InterpolationFilter,
        LoopFilterParams, ReferenceFrameState, SegmentationParams, parse_film_grain_params,
        parse_inter_reference_indices, parse_segmentation_params, parse_show_existing_frame_index,
        read_current_frame_id, read_global_motion_params, read_inv_signed_literal,
        read_signed_delta, validate_reference_frame_ids,
    };
    use crate::DecoderError;
    use crate::av1::bitstream::BitReader;

    #[test]
    fn interpolation_filter_symbols_follow_av1_order() {
        assert_eq!(
            InterpolationFilter::from_bits(0).unwrap(),
            InterpolationFilter::Regular
        );
        assert_eq!(
            InterpolationFilter::from_bits(1).unwrap(),
            InterpolationFilter::Smooth
        );
        assert_eq!(
            InterpolationFilter::from_bits(2).unwrap(),
            InterpolationFilter::Sharp
        );
        assert_eq!(
            InterpolationFilter::from_bits(3).unwrap(),
            InterpolationFilter::Bilinear
        );
        assert!(InterpolationFilter::from_bits(4).is_err());
        assert_eq!(
            InterpolationFilter::from_switchable_symbol(2).unwrap(),
            InterpolationFilter::Sharp
        );
        assert!(InterpolationFilter::from_switchable_symbol(3).is_err());
    }
    use crate::av1::syntax::BlockSize;

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
    fn cdef_secondary_strength_three_maps_to_four() {
        assert_eq!(super::cdef_secondary_strength(0), 0);
        assert_eq!(super::cdef_secondary_strength(2), 2);
        assert_eq!(super::cdef_secondary_strength(3), 4);
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
    fn inherited_film_grain_reuses_reference_parameters_with_new_seed() {
        let data = include_bytes!("../../test_data/images/WML2Viewer.avif");
        let info = crate::container::parse_avif(data).unwrap();
        let sequence_payload = crate::obu::find_obu_payload(
            &info.primary_item_payload,
            crate::obu::ObuType::SequenceHeader,
        )
        .unwrap()
        .unwrap();
        let mut sequence = super::super::sequence::parse_sequence_header(sequence_payload).unwrap();
        sequence.film_grain_params_present = true;

        let reference_grain = FilmGrainParams {
            random_seed: 0x1111,
            num_y_points: 1,
            scaling_points_y: [[12, 34]; 14],
            chroma_scaling_from_luma: true,
            num_cb_points: 0,
            scaling_points_cb: [[0, 0]; 10],
            num_cr_points: 0,
            scaling_points_cr: [[0, 0]; 10],
            scaling_shift: 8,
            ar_coeff_lag: 0,
            ar_coeffs_y: [0; 24],
            ar_coeffs_cb: [0; 25],
            ar_coeffs_cr: [0; 25],
            ar_coeff_shift: 6,
            grain_scale_shift: 0,
            cb_mult: 0,
            cb_luma_mult: 0,
            cb_offset: 0,
            cr_mult: 0,
            cr_luma_mult: 0,
            cr_offset: 0,
            overlap_flag: false,
            clip_to_restricted_range: false,
        };
        let mut references = [None; 8];
        references[3] = Some(ReferenceFrameState {
            frame_width: 1,
            frame_height: 1,
            upscaled_width: 1,
            render_width: 1,
            render_height: 1,
            order_hint: 0,
            film_grain: Some(reference_grain),
            frame_id: None,
            global_motion: GlobalMotionParams::default(),
        });
        let mut bits = vec![true];
        push_unsigned(&mut bits, 0x2345, 16);
        bits.push(false);
        push_unsigned(&mut bits, 3, 3);
        let parsed = parse_film_grain_params(
            &mut BitReader::new(&bits_to_bytes(&bits)),
            &sequence,
            FrameType::Inter,
            true,
            false,
            &references,
        )
        .unwrap()
        .unwrap();
        assert_eq!(parsed.random_seed, 0x2345);
        let mut expected = reference_grain;
        expected.random_seed = 0x2345;
        assert_eq!(parsed, expected);
    }

    #[test]
    fn reads_current_frame_id_only_when_sequence_signals_ids() {
        let data = include_bytes!("../../test_data/images/WML2Viewer.avif");
        let info = crate::container::parse_avif(data).unwrap();
        let sequence_payload = crate::obu::find_obu_payload(
            &info.primary_item_payload,
            crate::obu::ObuType::SequenceHeader,
        )
        .unwrap()
        .unwrap();
        let mut sequence = super::super::sequence::parse_sequence_header(sequence_payload).unwrap();
        let mut reader = BitReader::new(&[0b1010_0000]);
        assert_eq!(read_current_frame_id(&mut reader, &sequence).unwrap(), None);
        sequence.frame_id_numbers_present = true;
        sequence.frame_id_length = 4;
        let mut reader = BitReader::new(&[0b1010_0000]);
        assert_eq!(
            read_current_frame_id(&mut reader, &sequence).unwrap(),
            Some(0b1010)
        );
    }

    #[test]
    fn rejects_reference_frame_ids_outside_the_allowed_age_window() {
        let data = include_bytes!("../../test_data/images/WML2Viewer.avif");
        let info = crate::container::parse_avif(data).unwrap();
        let sequence_payload = crate::obu::find_obu_payload(
            &info.primary_item_payload,
            crate::obu::ObuType::SequenceHeader,
        )
        .unwrap()
        .unwrap();
        let mut sequence = super::super::sequence::parse_sequence_header(sequence_payload).unwrap();
        sequence.frame_id_numbers_present = true;
        sequence.frame_id_length = 4;
        sequence.delta_frame_id_length = 2;
        let mut references = [None; 8];
        references[0] = Some(ReferenceFrameState {
            frame_width: 1,
            frame_height: 1,
            upscaled_width: 1,
            render_width: 1,
            render_height: 1,
            order_hint: 0,
            film_grain: None,
            frame_id: Some(1),
            global_motion: GlobalMotionParams::default(),
        });
        let indices = [0; 7];
        assert!(validate_reference_frame_ids(Some(3), &sequence, &indices, &references).is_ok());
        references[0].as_mut().unwrap().frame_id = Some(9);
        let error =
            validate_reference_frame_ids(Some(3), &sequence, &indices, &references).unwrap_err();
        assert!(
            matches!(error, crate::DecoderError::Bitstream(message) if message.contains("stale frame ID"))
        );
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

    #[test]
    fn global_motion_identity_vector_consumes_all_reference_types() {
        let mut reader = BitReader::new(&[0; 1]);
        let params = read_global_motion_params(&mut reader, false, &[0; 7], &[None; 8]).unwrap();
        assert_eq!(params, GlobalMotionParams::default());
        assert_eq!(reader.bit_position(), 7);
    }

    #[test]
    fn global_motion_parser_rejects_truncated_non_identity_model() {
        let mut reader = BitReader::new(&[0b1000_0000]);
        let error = read_global_motion_params(&mut reader, false, &[0; 7], &[None; 8]).unwrap_err();
        assert!(matches!(error, DecoderError::NotEnoughData(_)));
    }

    #[test]
    fn global_motion_vectors_follow_translation_and_block_center_models() {
        let mut params = GlobalMotionParams::default();
        params.types[0] = GlobalMotionType::Translation;
        params.matrices[0][0] = 8_192;
        params.matrices[0][1] = -16_384;
        assert_eq!(
            params
                .motion_vector(0, BlockSize::Block8x8, 0, 0, false, false)
                .unwrap(),
            (1, -2)
        );

        params.types[0] = GlobalMotionType::Affine;
        params.matrices[0] = [8_192, -8_192, 65_536, 0, 0, 65_536];
        assert_eq!(
            params
                .motion_vector(0, BlockSize::Block8x8, 0, 0, true, false)
                .unwrap(),
            (-1, 1)
        );
    }

    #[test]
    fn global_motion_parser_decodes_zero_translation_model() {
        let mut bits = vec![true, false, true];
        bits.extend([false; 8]);
        bits.extend([false; 6]);
        let data = bits_to_bytes(&bits);
        let mut reader = BitReader::new(&data);
        let params = read_global_motion_params(&mut reader, false, &[0; 7], &[None; 8]).unwrap();
        assert_eq!(params.types[0], GlobalMotionType::Translation);
        assert_eq!(params.matrices[0], [0, 0, 1 << 16, 0, 0, 1 << 16]);
        assert_eq!(reader.bit_position(), 17);
    }
}

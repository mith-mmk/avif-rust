use super::*;
use crate::DecoderError;
use crate::av1::{CdfContext, ReferenceFrameState};
use crate::container::AvifInfo;
use crate::obu::{ObuType, parse_obu_stream};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Av1SequenceSampleUnit {
    pub(super) payload: Vec<u8>,
    has_sequence_header: bool,
}

/// Splits an AVIS sample into the coded frame units it contains.
///
/// An AV1 OBU_FRAME is already self-contained.  For the separate
/// OBU_FRAME_HEADER form, the following tile-group OBUs belong to that frame
/// until the next frame boundary.  AVIS samples may contain hidden reference
/// frames followed by a displayed frame, so decoding only the first coded OBU
/// loses the reference state needed by show-existing-frame.
pub(super) fn split_av1_sequence_sample(
    sample: &[u8],
) -> Result<Vec<Av1SequenceSampleUnit>, DecoderError> {
    let obus = parse_obu_stream(sample)?;
    let sequence_prefix = obus
        .iter()
        .find(|obu| obu.obu_type == ObuType::SequenceHeader)
        .map(|obu| encode_obu(obu.obu_type, obu.payload))
        .transpose()?;
    let has_sequence_header = sequence_prefix.is_some();
    let mut units = Vec::new();
    let mut current = Vec::new();
    let mut current_active = false;

    let finish_current = |units: &mut Vec<Av1SequenceSampleUnit>, current: &mut Vec<u8>| {
        if !current.is_empty() {
            units.push(Av1SequenceSampleUnit {
                payload: std::mem::take(current),
                has_sequence_header,
            });
        }
    };

    for obu in obus {
        match obu.obu_type {
            ObuType::SequenceHeader => {}
            ObuType::TemporalDelimiter if current_active => {
                finish_current(&mut units, &mut current);
                current_active = false;
            }
            ObuType::Frame => {
                if current_active {
                    finish_current(&mut units, &mut current);
                }
                if let Some(prefix) = sequence_prefix.as_deref() {
                    current.extend_from_slice(prefix);
                }
                current.extend_from_slice(&encode_obu(obu.obu_type, obu.payload)?);
                finish_current(&mut units, &mut current);
                current_active = false;
            }
            ObuType::FrameHeader => {
                if current_active {
                    finish_current(&mut units, &mut current);
                }
                if let Some(prefix) = sequence_prefix.as_deref() {
                    current.extend_from_slice(prefix);
                }
                current.extend_from_slice(&encode_obu(obu.obu_type, obu.payload)?);
                current_active = true;
            }
            ObuType::TileGroup if current_active => {
                current.extend_from_slice(&encode_obu(obu.obu_type, obu.payload)?);
            }
            _ => {}
        }
    }
    if current_active {
        finish_current(&mut units, &mut current);
    }
    Ok(units)
}

#[cfg(test)]
pub(super) fn decode_sequence_frame_from_info(
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

#[cfg(test)]
pub(super) fn decode_sequence_frames_from_info(
    info: &AvifInfo,
) -> Result<Vec<DecodedFrame>, DecoderError> {
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

#[cfg(test)]
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
    let mut state = SequenceDecodeState::new(info)?;
    let frame_capacity = if collect_all { stop_index + 1 } else { 1 };
    let mut frames = Vec::with_capacity(frame_capacity);
    while state.next_sample_index <= stop_index {
        let index = state.next_sample_index;
        let frame = state.next_sample(info, samples)?.ok_or_else(|| {
            DecoderError::Bitstream(format!("AVIS sequence ended before sample {stop_index}"))
        })?;
        if collect_all {
            frames.push(frame);
        } else if index == stop_index {
            return Ok(vec![frame]);
        }
    }
    Ok(frames)
}

/// Mutable AV1 track state shared by the public incremental decoder and the
/// indexed/batch compatibility helpers.
///
/// Cloning this value is intentionally cheap: frame planes, CDFs, motion
/// fields, and the shared sequence prefix are reference counted. A sample is
/// decoded against a clone and committed only after the whole sample succeeds,
/// so an error never advances or partially mutates the public decoder cursor.
#[derive(Debug, Clone, Default)]
pub(super) struct SequenceDecodeState {
    sequence_prefix: Arc<[u8]>,
    references: FrameReferenceSlots,
    cdf_states: Option<Arc<Vec<CdfContext>>>,
    next_sample_index: usize,
    #[cfg(test)]
    decoded_sample_count: usize,
}

impl SequenceDecodeState {
    pub(super) fn new(info: &AvifInfo) -> Result<Self, DecoderError> {
        let primary_obus = parse_obu_stream(&info.primary_item_payload)?;
        let sequence_obu = primary_obus
            .iter()
            .find(|obu| obu.obu_type == ObuType::SequenceHeader)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 sequence header OBU is missing".to_string())
            })?;
        Ok(Self {
            sequence_prefix: Arc::from(
                encode_obu(sequence_obu.obu_type, sequence_obu.payload)?.into_boxed_slice(),
            ),
            ..Self::default()
        })
    }

    pub(super) fn next_sample(
        &mut self,
        info: &AvifInfo,
        samples: &[Vec<u8>],
    ) -> Result<Option<DecodedFrame>, DecoderError> {
        let Some(sample) = samples.get(self.next_sample_index) else {
            return Ok(None);
        };
        let mut next = self.clone();
        let index = next.next_sample_index;
        let frame = next.decode_sample(info, sample, index)?;
        next.next_sample_index += 1;
        #[cfg(test)]
        {
            next.decoded_sample_count += 1;
        }
        *self = next;
        Ok(Some(frame))
    }

    #[cfg(test)]
    pub(super) fn decoded_sample_count(&self) -> usize {
        self.decoded_sample_count
    }

    fn decode_sample(
        &mut self,
        info: &AvifInfo,
        sample: &[u8],
        index: usize,
    ) -> Result<DecodedFrame, DecoderError> {
        let units = split_av1_sequence_sample(sample)?;
        if units.is_empty() {
            return Err(DecoderError::Bitstream(
                "AVIS sample has no coded frame OBU".to_string(),
            ));
        }
        let mut last_visible_frame = None;
        for (unit_index, unit) in units.iter().enumerate() {
            let sample_info = crate::container::inspect_av1_sequence_sample(&unit.payload)?;
            let kind = sample_info.kind.ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AVIS sample {index} unit {unit_index} has no coded frame OBU"
                ))
            })?;
            match kind {
                crate::container::AvifSequenceSampleKind::Key
                | crate::container::AvifSequenceSampleKind::IntraOnly => {
                    // A per-sample Sequence Header is authoritative when an AVIS
                    // track changes its sample description.  For the common case
                    // where it is omitted, inspect the shared prefix and sample
                    // as separate OBU streams to avoid rebuilding a concatenated
                    // temporary payload.
                    let av1_config = (!unit.has_sequence_header)
                        .then_some(info.av1_config.as_deref())
                        .flatten();
                    let headers = parse_av1_sequence_sample_headers(
                        info,
                        &self.sequence_prefix,
                        &unit.payload,
                        unit.has_sequence_header,
                        av1_config,
                    )?;
                    let initial_cdfs = if headers.frame.primary_ref_frame != 7 {
                        self.references
                            .cdf_for_header(&headers.frame)
                            .or_else(|| self.cdf_states.as_ref().map(|cdf| cdf.as_slice()))
                    } else {
                        None
                    };
                    let (decoded_state, next_cdf_states) =
                        decode_still_frame_with_filter_policy_and_state_and_references_and_cdf(
                            &headers,
                            Some(info),
                            true,
                            std::array::from_fn(|_| None),
                            initial_cdfs,
                            true,
                            true,
                        )?;
                    let (decoded, motion_field) =
                        finish_decoded_still_frame(&headers, decoded_state, true)?;
                    let mut reference_cdfs = if headers.frame.disable_frame_end_update_cdf {
                        initial_cdfs.map(ToOwned::to_owned).unwrap_or_else(|| {
                            vec![CdfContext::new(headers.frame.base_q_idx); next_cdf_states.len()]
                        })
                    } else {
                        next_cdf_states.clone()
                    };
                    if !headers.frame.disable_frame_end_update_cdf {
                        for cdf in &mut reference_cdfs {
                            cdf.reset_symbol_counters();
                        }
                    }
                    self.references.refresh_with_cdf_and_motion(
                        headers.frame.refresh_frame_flags,
                        &decoded,
                        &headers.frame,
                        &reference_cdfs,
                        &motion_field,
                    );
                    self.references.set_previous_motion_field(motion_field);
                    if !headers.frame.disable_frame_end_update_cdf {
                        self.cdf_states = Some(Arc::new(next_cdf_states));
                    }
                    if headers.frame.show_frame {
                        last_visible_frame = Some(decoded);
                    }
                }
                crate::container::AvifSequenceSampleKind::ShowExisting {
                    frame_to_show_map_idx,
                } => {
                    let decoded = self.references.frame_to_show(frame_to_show_map_idx)?;
                    last_visible_frame = Some(decoded);
                }
                crate::container::AvifSequenceSampleKind::Inter
                | crate::container::AvifSequenceSampleKind::Switch => {
                    let av1_config = (!unit.has_sequence_header)
                        .then_some(info.av1_config.as_deref())
                        .flatten();
                    let reference_states = self.references.states();
                    let headers = parse_av1_sequence_sample_headers_with_references(
                        info,
                        &self.sequence_prefix,
                        &unit.payload,
                        unit.has_sequence_header,
                        av1_config,
                        &reference_states,
                    )?;
                    #[cfg(test)]
                    if std::env::var_os("AVIF_ENTROPY_TRACE").is_some() {
                        let start = headers.tile_group.group.data_start_offset;
                        let end = start
                            .saturating_add(8)
                            .min(headers.tile_group.tile_data.len());
                        eprintln!(
                            "entropy-trace headers sample={index} unit={unit_index} kind={kind:?} show={} order_hint={} primary_ref={} refresh={:#04x} disable_cdf={} disable_frame_end_cdf={} refs={:?} reference_select={} skip_mode={}/{:?} base_q={} delta_q={} segmentation={:?} global={:?} header_bits={} tile_start={start} tile_bytes={:02x?}",
                            headers.frame.show_frame,
                            headers.frame.order_hint,
                            headers.frame.primary_ref_frame,
                            headers.frame.refresh_frame_flags,
                            headers.frame.disable_cdf_update,
                            headers.frame.disable_frame_end_update_cdf,
                            headers.frame.reference_frame_indices,
                            headers.frame.reference_select,
                            headers.frame.skip_mode_present,
                            headers.frame.skip_mode_frame,
                            headers.frame.base_q_idx,
                            headers.frame.delta_q.present,
                            headers.frame.segmentation,
                            headers.frame.global_motion.types,
                            headers.frame.uncompressed_header_bits,
                            &headers.tile_group.tile_data[start..end]
                        );
                    }
                    let temporal_motion_field = self
                        .references
                        .temporal_motion_field(&headers.frame, headers.sequence.order_hint_bits);
                    let initial_cdfs = if headers.frame.primary_ref_frame != 7 {
                        self.references
                            .cdf_for_header(&headers.frame)
                            .or_else(|| self.cdf_states.as_ref().map(|cdf| cdf.as_slice()))
                    } else {
                        None
                    };
                    let (decoded_state, next_cdf_states) =
                        match decode_still_frame_with_filter_policy_and_state_and_references_and_cdf_and_motion(
                            &headers,
                            Some(info),
                            true,
                            self.references.buffers(),

                            initial_cdfs,
                            temporal_motion_field,
                            true,
                            true,
                        ) {
                            Ok(decoded) => decoded,
                            Err(DecoderError::Unsupported(message)) => {
                                return Err(DecoderError::Unsupported(format!(
                                    "AVIS sample {index} unit {unit_index} uses unsupported {kind:?} frame prediction: {message}"
                                )));
                            }
                            Err(DecoderError::Bitstream(message)) => {
                                return Err(DecoderError::Bitstream(format!(
                                    "AVIS sample {index} unit {unit_index} {kind:?} frame: {message}"
                                )));
                            }
                            Err(err) => return Err(err),
                        };
                    let (decoded, motion_field) =
                        finish_decoded_still_frame(&headers, decoded_state, true)?;
                    let mut reference_cdfs = if headers.frame.disable_frame_end_update_cdf {
                        initial_cdfs.map(ToOwned::to_owned).unwrap_or_else(|| {
                            vec![CdfContext::new(headers.frame.base_q_idx); next_cdf_states.len()]
                        })
                    } else {
                        next_cdf_states.clone()
                    };
                    if !headers.frame.disable_frame_end_update_cdf {
                        for cdf in &mut reference_cdfs {
                            cdf.reset_symbol_counters();
                        }
                    }
                    self.references.refresh_with_cdf_and_motion(
                        headers.frame.refresh_frame_flags,
                        &decoded,
                        &headers.frame,
                        &reference_cdfs,
                        &motion_field,
                    );
                    self.references.set_previous_motion_field(motion_field);
                    if !headers.frame.disable_frame_end_update_cdf {
                        self.cdf_states = Some(Arc::new(next_cdf_states));
                    }
                    if headers.frame.show_frame {
                        last_visible_frame = Some(decoded);
                    }
                }
            }
        }
        last_visible_frame.ok_or_else(|| {
            DecoderError::Bitstream(format!("AVIS sample {index} has no displayed frame"))
        })
    }
}

#[cfg(test)]
pub(super) fn avis_parallel_work_is_large_enough(info: &AvifInfo, sample_count: usize) -> bool {
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
    previous_motion_field: Option<Arc<MotionField>>,
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
    motion_field: Arc<MotionField>,
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
        let motion_field = MotionField::empty(0, 0);
        self.refresh_with_cdf_and_motion(
            refresh_frame_flags,
            frame,
            header,
            cdf_states,
            &motion_field,
        );
    }

    fn refresh_with_cdf_and_motion(
        &mut self,
        refresh_frame_flags: u8,
        frame: &DecodedFrame,
        header: &FrameHeader,
        cdf_states: &[CdfContext],
        motion_field: &MotionField,
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
                    motion_field: Arc::new(motion_field.clone()),
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

    fn set_previous_motion_field(&mut self, motion_field: MotionField) {
        self.previous_motion_field = Some(Arc::new(motion_field));
    }

    fn temporal_motion_field(
        &self,
        header: &FrameHeader,
        order_hint_bits: u8,
    ) -> Option<Arc<MotionField>> {
        if !header.use_ref_frame_mvs {
            return None;
        }
        let mi_cols = usize::try_from(header.frame_width).ok()?.div_ceil(4);
        let mi_rows = usize::try_from(header.frame_height).ok()?.div_ceil(4);
        let mut field = MotionField::empty(mi_cols, mi_rows);
        field.order_hint_bits = order_hint_bits;
        field.order_hint = header.order_hint;
        field.reference_order_hints = header.reference_order_hints;
        field.reference_frame_indices = header.reference_frame_indices;
        field.projected = true;

        let mut project = |reference_type: usize, direction: i32| -> bool {
            let Some(&slot) = header.reference_frame_indices.get(reference_type) else {
                return false;
            };
            let Ok(start) = self.get(slot) else {
                return false;
            };
            if matches!(start.frame_type, FrameType::Key | FrameType::IntraOnly) {
                return false;
            }
            let source = &start.motion_field;
            if source.mi_cols != mi_cols || source.mi_rows != mi_rows {
                return false;
            }
            let start_to_current =
                relative_order_hint_distance(order_hint_bits, start.order_hint, header.order_hint);
            let start_to_current = if direction == 2 {
                -start_to_current
            } else {
                start_to_current
            };
            for source_row in (0..mi_rows).step_by(2) {
                for source_col in (0..mi_cols).step_by(2) {
                    let source_index = source_row * mi_cols + source_col;
                    let Some(motion_vector) = source.motion_vectors[source_index] else {
                        continue;
                    };
                    let Some(source_reference_type) = source.reference_frames[source_index] else {
                        continue;
                    };
                    let Some(source_reference_hint) = source
                        .reference_order_hints
                        .get(usize::from(source_reference_type))
                        .copied()
                        .flatten()
                    else {
                        continue;
                    };
                    let reference_offset = relative_order_hint_distance(
                        order_hint_bits,
                        start.order_hint,
                        source_reference_hint,
                    );
                    if reference_offset <= 0
                        || reference_offset.abs() > 31
                        || start_to_current.abs() > 31
                    {
                        continue;
                    }
                    let projected = project_temporal_motion_vector(
                        motion_vector,
                        start_to_current,
                        reference_offset,
                    );
                    let source_row = source_row / 2;
                    let source_col = source_col / 2;
                    let row_offset = motion_offset(projected.0);
                    let col_offset = motion_offset(projected.1);
                    let target_row = if direction == 2 {
                        source_row as i32 - row_offset
                    } else {
                        source_row as i32 + row_offset
                    };
                    let target_col = if direction == 2 {
                        source_col as i32 - col_offset
                    } else {
                        source_col as i32 + col_offset
                    };
                    let base_row = (source_row >> 3) << 3;
                    let base_col = (source_col >> 3) << 3;
                    if target_row < base_row as i32
                        || target_row >= (base_row + 8) as i32
                        || target_col < (base_col as i32 - 8)
                        || target_col >= (base_col + 16) as i32
                        || target_row < 0
                        || target_col < 0
                        || target_row as usize >= mi_rows.div_ceil(2)
                        || target_col as usize >= mi_cols.div_ceil(2)
                    {
                        continue;
                    }
                    let target_index = target_row as usize * 2 * mi_cols + target_col as usize * 2;
                    field.motion_vectors[target_index] = Some(motion_vector);
                    field.reference_offsets[target_index] = Some(reference_offset);
                }
            }
            true
        };

        // `av1_setup_motion_field` spends the LAST stamp whenever the slot
        // exists, even if projection fails because LAST is a key frame.
        let mut ref_stamp = 2;
        if header
            .reference_frame_indices
            .first()
            .is_some_and(|slot| self.get(*slot).is_ok())
        {
            let is_last_overlay = header
                .reference_frame_indices
                .first()
                .and_then(|slot| self.get(*slot).ok())
                .and_then(|start| start.motion_field.reference_order_hints[6])
                .zip(header.reference_order_hints[3])
                .is_some_and(|(altref_hint, golden_hint)| altref_hint == golden_hint);
            if !is_last_overlay {
                let _ = project(0, 2);
            }
            ref_stamp -= 1;
        }
        for reference_type in [4usize, 5] {
            if header.reference_order_hints[reference_type]
                .map(|hint| {
                    relative_order_hint_distance(order_hint_bits, hint, header.order_hint) > 0
                })
                .unwrap_or(false)
                && project(reference_type, 0)
            {
                ref_stamp -= 1;
            }
        }
        if ref_stamp >= 0
            && header.reference_order_hints[6]
                .map(|hint| {
                    relative_order_hint_distance(order_hint_bits, hint, header.order_hint) > 0
                })
                .unwrap_or(false)
            && project(6, 0)
        {
            ref_stamp -= 1;
        }
        if ref_stamp >= 0 {
            let _ = project(1, 2);
        }
        Some(Arc::new(field))
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

fn relative_order_hint_distance(bits: u8, reference: u32, current: u32) -> i32 {
    let modulo = 1i32 << bits;
    let mask = modulo - 1;
    let mut distance = (reference as i32 - current as i32) & mask;
    if distance & (modulo >> 1) != 0 {
        distance -= modulo;
    }
    distance
}

fn motion_offset(value: i32) -> i32 {
    if value >= 0 {
        value >> 6
    } else {
        -((-value) >> 6)
    }
}

fn project_temporal_motion_vector(
    motion_vector: (i32, i32),
    numerator: i32,
    denominator: i32,
) -> (i32, i32) {
    const DIV_MULT: [i32; 32] = [
        0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170,
        1092, 1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
    ];
    fn scale(value: i32, numerator: i32, denominator: i32, div_mult: &[i32; 32]) -> i32 {
        let product = i64::from(value)
            * i64::from(numerator.clamp(-31, 31))
            * i64::from(div_mult[denominator.clamp(0, 31) as usize]);
        let rounded = if product < 0 {
            -((-product + (1 << 13)) >> 14)
        } else {
            (product + (1 << 13)) >> 14
        };
        rounded.clamp(-32767, 32767) as i32
    }
    (
        scale(motion_vector.0, numerator, denominator, &DIV_MULT),
        scale(motion_vector.1, numerator, denominator, &DIV_MULT),
    )
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
        let Some(header) = reference_header() else {
            return;
        };
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
        let Some(mut header) = reference_header() else {
            return;
        };
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
        let Some(mut header) = reference_header() else {
            return;
        };
        header.primary_ref_frame = 0;
        header.reference_frame_indices[0] = 3;
        let cdf = CdfContext::new(header.base_q_idx);

        slots.refresh_with_cdf(1 << 3, &frame(3), &header, std::slice::from_ref(&cdf));

        let restored = slots.cdf_for_header(&header).unwrap();
        assert_eq!(restored, [cdf].as_slice());
    }

    fn wml2viewer_data() -> Option<Vec<u8>> {
        crate::test_support::wml2viewer_avif()
    }

    fn reference_header() -> Option<FrameHeader> {
        let data = wml2viewer_data()?;
        let info = parse_avif(&data).ok()?;
        Some(parse_av1_headers(&info).ok()?.frame)
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
        let Some(data) = wml2viewer_data() else {
            return;
        };
        let mut info = parse_avif(&data).unwrap();
        info.major_brand = *b"avis";
        info.sequence_sample_payloads = vec![
            info.primary_item_payload.clone(),
            encode_obu(ObuType::FrameHeader, &[0x80]).unwrap(),
        ];
        let expected = decode_frame_bytes(&data).unwrap();
        let shown = decode_sequence_frame_from_info(&info, 1).unwrap();
        assert_eq!(shown, expected);
    }

    #[test]
    fn batch_show_existing_sample_reuses_decoded_key_frame() {
        let Some(data) = wml2viewer_data() else {
            return;
        };
        let mut info = parse_avif(&data).unwrap();
        info.major_brand = *b"avis";
        info.sequence_sample_payloads = vec![
            info.primary_item_payload.clone(),
            encode_obu(ObuType::FrameHeader, &[0x80]).unwrap(),
        ];
        let expected = decode_frame_bytes(&data).unwrap();
        let frames = decode_sequence_frames_from_info(&info).unwrap();
        assert_eq!(frames, vec![expected.clone(), expected]);
    }

    #[test]
    fn incremental_state_decodes_each_sample_once_and_stops_idempotently() {
        let Some(data) = wml2viewer_data() else {
            return;
        };
        let mut info = parse_avif(&data).unwrap();
        info.major_brand = *b"avis";
        let samples = vec![
            info.primary_item_payload.clone(),
            encode_obu(ObuType::FrameHeader, &[0x80]).unwrap(),
        ];
        let expected = decode_frame_bytes(&data).unwrap();
        let mut state = SequenceDecodeState::new(&info).unwrap();

        let first = state.next_sample(&info, &samples).unwrap().unwrap();
        let shown = state.next_sample(&info, &samples).unwrap().unwrap();
        assert_eq!(first, expected);
        assert_eq!(shown, expected);
        assert_eq!(state.decoded_sample_count(), samples.len());
        assert!(state.next_sample(&info, &samples).unwrap().is_none());
        assert!(state.next_sample(&info, &samples).unwrap().is_none());
        assert_eq!(state.decoded_sample_count(), samples.len());
    }

    #[test]
    fn incremental_state_does_not_advance_after_sample_error() {
        let Some(data) = wml2viewer_data() else {
            return;
        };
        let mut info = parse_avif(&data).unwrap();
        info.major_brand = *b"avis";
        let first = info.primary_item_payload.clone();
        let valid_second = encode_obu(ObuType::FrameHeader, &[0x80]).unwrap();
        let mut state = SequenceDecodeState::new(&info).unwrap();

        state
            .next_sample(&info, std::slice::from_ref(&first))
            .unwrap()
            .unwrap();
        let invalid_samples = vec![first.clone(), vec![0xff]];
        assert!(state.next_sample(&info, &invalid_samples).is_err());
        assert_eq!(state.decoded_sample_count(), 1);

        let valid_samples = vec![first, valid_second];
        assert!(state.next_sample(&info, &valid_samples).unwrap().is_some());
        assert_eq!(state.decoded_sample_count(), 2);
    }

    #[test]
    fn small_avis_sequences_avoid_thread_pool_overhead() {
        let Some(data) = wml2viewer_data() else {
            return;
        };
        let mut info = parse_avif(&data).unwrap();
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
        let Some(data) = wml2viewer_data() else {
            return;
        };
        let mut info = parse_avif(&data).unwrap();
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
    #[ignore = "requires AVIF_ENTROPY_TRACE_FIXTURE for test-only AOM accounting comparison"]
    fn trace_avis_entropy_fixture() {
        let path = std::env::var_os("AVIF_ENTROPY_TRACE_FIXTURE")
            .expect("AVIF_ENTROPY_TRACE_FIXTURE must point to an AVIS fixture");
        let index = std::env::var("AVIF_ENTROPY_TRACE_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        let data = std::fs::read(path).expect("trace AVIS fixture should be readable");
        let info = parse_avif(&data).expect("trace AVIS fixture should parse");
        if let Some(output) = std::env::var_os("AVIF_ENTROPY_TRACE_DUMP_OBU") {
            let stream = info
                .sequence_sample_payloads
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            std::fs::write(output, stream).expect("trace AV1 stream should be writable");
        }
        if let Some(output) = std::env::var_os("AVIF_ENTROPY_TRACE_DUMP_IVF") {
            let width = u16::try_from(info.width.expect("trace AVIS width is required"))
                .expect("trace AVIS width must fit IVF");
            let height = u16::try_from(info.height.expect("trace AVIS height is required"))
                .expect("trace AVIS height must fit IVF");
            let frame_count = u32::try_from(info.sequence_sample_payloads.len())
                .expect("trace AVIS frame count must fit IVF");
            let mut ivf = Vec::new();
            ivf.extend_from_slice(b"DKIF");
            ivf.extend_from_slice(&0_u16.to_le_bytes());
            ivf.extend_from_slice(&32_u16.to_le_bytes());
            ivf.extend_from_slice(b"AV01");
            ivf.extend_from_slice(&width.to_le_bytes());
            ivf.extend_from_slice(&height.to_le_bytes());
            ivf.extend_from_slice(&1_u32.to_le_bytes());
            ivf.extend_from_slice(&1_u32.to_le_bytes());
            ivf.extend_from_slice(&frame_count.to_le_bytes());
            ivf.extend_from_slice(&0_u32.to_le_bytes());
            for (timestamp, sample) in info.sequence_sample_payloads.iter().enumerate() {
                let size = u32::try_from(sample.len()).expect("trace sample size must fit IVF");
                ivf.extend_from_slice(&size.to_le_bytes());
                ivf.extend_from_slice(&(timestamp as u64).to_le_bytes());
                ivf.extend_from_slice(sample);
            }
            std::fs::write(output, ivf).expect("trace IVF stream should be writable");
        }
        eprintln!(
            "entropy-trace samples={} selected={index}",
            info.sequence_sample_payloads.len()
        );
        for (sample_index, sample) in info
            .sequence_sample_payloads
            .iter()
            .take(index + 1)
            .enumerate()
        {
            let units = split_av1_sequence_sample(sample).expect("trace sample should split");
            for (unit_index, unit) in units.iter().enumerate() {
                let inspected = crate::container::inspect_av1_sequence_sample(&unit.payload)
                    .expect("trace unit should inspect");
                eprintln!(
                    "entropy-trace sample={sample_index} unit={unit_index} len={} kind={:?}",
                    unit.payload.len(),
                    inspected.kind
                );
            }
        }
        let frame = decode_sequence_frame_bytes(&data, index)
            .expect("diagnostic fixture must pass strict entropy validation");
        eprintln!(
            "entropy-trace result=ok dimensions={}x{} bit_depth={}",
            frame.width, frame.height, frame.bit_depth
        );
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
pub(super) fn decode_hidden_key_frame_show_existing(
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

#[cfg(test)]
use crate::Rgba16ImageBuffer;
use crate::allocation::{AllocationClass, AllocationTicket, admit_fresh_with, fresh_replacement};
use crate::av1::ColorRange;
use crate::av1::PostFilterState;
#[cfg(test)]
use crate::av1::alloc_frame_buffers;
#[cfg(test)]
use crate::av1::decode_luma_root_block_prefix_with_post_filter_state_and_entropy;
use crate::av1::{
    Av1CodecConfiguration, BlockFilterState, BlockModeProbe, CdfContext, ChromaSamplePosition,
    ColorConfig, FilmGrainParams, FrameBuffers, FrameDecodePlan, FrameHeader, FrameType,
    GlobalMotionParams,
    LoopFilterParams, MotionField, NativeDecodePlanAllocation, NativeFrameAllocation,
    NativeTileGroupAllocation, PartitionProbe, PlaneBuffer, PlaneLayout, QuantState,
    ReferenceFrameState, ResidualProbe, SegmentationParams, SequenceHeader, TileEntropyState,
    TileGroup, alloc_coded_frame_buffers, alloc_coded_frame_buffers_with_budget, apply_film_grain,
    apply_superres_horizontal, build_still_decode_plan, build_still_decode_plan_with_budget,
    cdef_adjust_primary_strength, cdef_chroma_direction,
    cdef_filter_block_region_with_edge_mode_into_bit_depth_visible_scaled,
    cdef_find_direction_with_variance_visible, crop_frame_buffers_to_plan,
    crop_native_frame_buffers_to_plan, deblock_filter_edge_with_visible_bounds,
    decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion,
    decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget,
    parse_av1_config, parse_frame_header, parse_frame_header_with_references_and_metadata,
    parse_sequence_header, parse_sequence_header_with_metadata, parse_show_existing_frame_index,
    parse_tile_group, parse_tile_group_with_budget, plan_transform_blocks_with_tx_size,
    prepare_tile_entropy, probe_first_block_residuals, probe_tile_block_modes,
    probe_tile_partitions, sgrproj_filter_unit_into_with_fixed_scratch_bit_depth_visible,
    sgrproj_filter_unit_into_with_scratch_bit_depth_visible,
    wiener_filter_unit_into_with_fixed_scratch_bit_depth_visible,
    wiener_filter_unit_into_with_scratch_bit_depth_visible,
};
use crate::compat::{DataMap, DecodeOptions};
use crate::container::{
    AuxiliaryImage, AvifAnimation, AvifInfo, ColorInformation, GridCell, ImageMirror,
    ImageRotation, PixelInformation, SampleTransformInput, SampleTransformToken,
    grid_cell_alpha_item_id, grid_has_cell_alpha, parse_avif, parse_sample_transform,
};
use crate::obu::{ObuType, find_obu_payloads_in_parts, parse_obu_stream};
use crate::{DecoderError, ImageBuffer};
use bin_rs::reader::BinaryReader;
use std::io::SeekFrom;
use std::sync::Arc;

type Error = Box<dyn std::error::Error>;

mod callback;
mod composition;
#[cfg(test)]
mod deblock_budget_tests;
mod frame;
#[cfg(test)]
mod native_hookup_tests;
#[allow(clippy::items_after_test_module)]
mod sequence;
#[allow(clippy::items_after_test_module)]
mod still;
#[cfg(test)]
mod strict_alpha_tests;
#[cfg(test)]
mod strict_grid_tests;

pub use frame::{
    AvifSequenceDecoder, DecodedFrame, DecodedGainMapFrame, DecodedSequenceFrame,
    PreparedSequenceCommit, PreparedSequenceFrame, StrictAvifSequenceDecoder,
    decode_frame_bytes, decode_frame_bytes_strict, decode_frame_bytes_strict_with_limits,
    decode_gain_map_frame_bytes, decode_sequence_frame_bytes, decode_sequence_frames_bytes,
};
#[cfg(test)]
use frame::{resample_gain_map, unpremultiply_rgba8, unpremultiply_rgba16};
use sequence::decode_hidden_key_frame_show_existing;
use still::{
    append_alpha_plane, append_alpha_plane_buffer, decode_alpha_auxiliary_frame,
    decode_alpha_grid_plane, decode_grid_frame, decode_grid_frame_raw, decode_grid_image,
    decode_still_image,
};
#[cfg(test)]
use still::{apply_alpha_rows, grid_composition_tests};

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
    if let Some(frame) = decode_sample_transform_frame(&data, &info, None)? {
        let mut image = frame.to_rgba8()?;
        composition::apply_image_transforms(
            &mut image,
            info.clean_aperture,
            info.mirror,
            info.rotation,
        )?;
        emit_metadata(&info, None, None, option)?;
        callback::emit_single(option.drawer, &image)?;
        return Ok(());
    }
    if info.primary_grid.is_some() {
        let image = decode_grid_image(&info)?;
        emit_metadata(&info, None, None, option)?;
        callback::emit_single(option.drawer, &image)?;
        return Ok(());
    }
    if info.sequence_sample_payloads.len() > 1 {
        let mut sequence_decoder = AvifSequenceDecoder::new(&data)?;
        let mut headers = parse_av1_headers(&info)?;
        // The probes replay entropy traversal and are only consumed by the
        // diagnostic metadata below. Keep the normal draw path to one decode;
        // callers that request debug output explicitly still receive the probes.
        if option.debug_flag > 0 {
            populate_diagnostic_probes(&mut headers);
        }
        emit_metadata(
            &info,
            Some(&headers),
            Some(sequence_decoder.animation()),
            option,
        )?;
        callback::emit_animation(
            option.drawer,
            &mut sequence_decoder,
            info.clean_aperture,
            info.mirror,
            info.rotation,
        )?;
        return Ok(());
    }
    let mut headers = parse_av1_headers(&info)?;
    // The probes replay entropy traversal and are only consumed by the
    // diagnostic metadata below. Keep the normal draw path to one decode;
    // callers that request debug output explicitly still receive the probes.
    if option.debug_flag > 0 {
        populate_diagnostic_probes(&mut headers);
    }
    emit_metadata(&info, Some(&headers), None, option)?;
    let image = decode_still_image(&headers, Some(&info))?;

    callback::emit_single(option.drawer, &image)?;
    Ok(())
}

pub fn decode_bytes(data: &[u8]) -> Result<ImageBuffer, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, true)?;
    if let Some(frame) = decode_sample_transform_frame(data, &info, None)? {
        let mut image = frame.to_rgba8()?;
        composition::apply_image_transforms(
            &mut image,
            info.clean_aperture,
            info.mirror,
            info.rotation,
        )?;
        return Ok(image);
    }
    if info.primary_grid.is_some() {
        return decode_grid_image(&info);
    }
    if let Some(frame) = decode_hidden_key_frame_show_existing(&info)? {
        let mut image = frame.to_rgba8()?;
        composition::apply_image_transforms(
            &mut image,
            info.clean_aperture,
            info.mirror,
            info.rotation,
        )?;
        return Ok(image);
    }
    let headers = parse_av1_headers(&info)?;
    decode_still_image(&headers, Some(&info))
}

pub(super) struct Av1Headers {
    config: Option<Av1CodecConfiguration>,
    sequence: SequenceHeader,
    frame: FrameHeader,
    tile_group: ParsedTileGroup,
    decode_plan: FrameDecodePlan,
    quant_state: QuantState,
    /// Strict-native header/materialization owners.  Legacy callers leave
    /// this empty and retain their historical allocation behaviour.
    native_allocation: Option<NativeHeaderAllocation>,
}

/// Tickets for the temporary AV1 objects that are needed between completing a
/// frame header and publishing decoded planes.  They are deliberately kept
/// out of the public AV1 structs: strict-native decoding releases them before
/// it starts materializing the selected alpha header, while the legacy parser
/// continues to expose the same owned vectors as before.
struct NativeHeaderAllocation {
    tile_info: Option<crate::av1::NativeTileInfoAllocation>,
    tile_group: Option<NativeTileGroupAllocation>,
    decode_plan: Option<NativeDecodePlanAllocation>,
    tile_data: AllocationTicket,
}

impl NativeHeaderAllocation {
    fn empty() -> Self {
        Self {
            tile_info: None,
            tile_group: None,
            decode_plan: None,
            tile_data: AllocationTicket::new(AllocationClass::Frame),
        }
    }

    fn release(&mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        budget.release_token(&mut self.tile_data)?;
        if let Some(allocation) = self.decode_plan.as_mut() {
            allocation.release(budget)?;
        }
        if let Some(allocation) = self.tile_group.as_mut() {
            allocation.release(budget)?;
        }
        if let Some(allocation) = self.tile_info.as_mut() {
            allocation.release(budget)?;
        }
        Ok(())
    }
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

/// Borrowed metadata needed while parsing one AV1 item.  Native auxiliary
/// items use this source directly instead of fabricating a cloned `AvifInfo`;
/// the legacy parser still constructs the same view from the public
/// container projection.
#[derive(Clone, Copy)]
pub(super) struct ItemHeaderSource<'a> {
    pub(super) payload: &'a [u8],
    pub(super) pixel_information: Option<&'a PixelInformation>,
    pub(super) color_information: Option<&'a ColorInformation>,
    pub(super) width: Option<u32>,
    pub(super) height: Option<u32>,
    pub(super) av1_config: Option<&'a [u8]>,
}

#[derive(Clone, Copy)]
pub(super) struct ItemHeaderParts<'a> {
    pub(super) sequence_payload: &'a [u8],
    pub(super) frame_payload: Option<&'a [u8]>,
    pub(super) frame_header_payload: Option<&'a [u8]>,
}

impl<'a> ItemHeaderSource<'a> {
    fn from_info(info: &'a AvifInfo) -> Self {
        Self {
            payload: &info.primary_item_payload,
            pixel_information: info.pixel_information.as_ref(),
            color_information: info.color_information.as_ref(),
            width: info.width,
            height: info.height,
            av1_config: info.av1_config.as_deref(),
        }
    }

    pub(super) fn from_auxiliary(auxiliary: &'a AuxiliaryImage) -> Self {
        Self {
            payload: &auxiliary.payload,
            pixel_information: None,
            color_information: None,
            width: None,
            height: None,
            av1_config: None,
        }
    }
}

fn parse_av1_headers(info: &AvifInfo) -> Result<Av1Headers, DecoderError> {
    parse_av1_headers_from_parts_with_references(
        info,
        &[&info.primary_item_payload],
        info.av1_config.as_deref(),
        &[None; 8],
    )
}

fn parse_av1_headers_with_frame_prefix(
    info: &AvifInfo,
    frame_prefix: crate::av1::FramePrefix<'_, '_>,
) -> Result<Av1Headers, DecoderError> {
    parse_av1_headers_from_parts_with_prefix(
        info,
        &[&info.primary_item_payload],
        info.av1_config.as_deref(),
        &[None; 8],
        Some(frame_prefix),
    )
}

fn parse_av1_headers_from_parts_with_references(
    info: &AvifInfo,
    payloads: &[&[u8]],
    av1_config: Option<&[u8]>,
    references: &[Option<ReferenceFrameState>; 8],
) -> Result<Av1Headers, DecoderError> {
    let source = ItemHeaderSource::from_info(info);
    parse_av1_headers_from_source_with_prefix(
        &source, payloads, av1_config, references, None, None, None,
    )
}

fn parse_av1_headers_from_parts_with_prefix(
    info: &AvifInfo,
    payloads: &[&[u8]],
    av1_config: Option<&[u8]>,
    references: &[Option<ReferenceFrameState>; 8],
    frame_prefix: Option<crate::av1::FramePrefix<'_, '_>>,
) -> Result<Av1Headers, DecoderError> {
    let source = ItemHeaderSource::from_info(info);
    parse_av1_headers_from_source_with_prefix(
        &source,
        payloads,
        av1_config,
        references,
        frame_prefix,
        None,
        None,
    )
}

pub(super) fn parse_av1_headers_for_item_with_prefix(
    source: &ItemHeaderSource<'_>,
    parts: ItemHeaderParts<'_>,
    frame_prefix: crate::av1::FramePrefix<'_, '_>,
) -> Result<Av1Headers, DecoderError> {
    parse_av1_headers_from_source_with_prefix(
        source,
        &[source.payload],
        source.av1_config,
        &[None; 8],
        Some(frame_prefix),
        Some(parts),
        None,
    )
}

fn parse_av1_headers_from_source_with_prefix(
    source: &ItemHeaderSource<'_>,
    payloads: &[&[u8]],
    av1_config: Option<&[u8]>,
    references: &[Option<ReferenceFrameState>; 8],
    mut frame_prefix: Option<crate::av1::FramePrefix<'_, '_>>,
    header_parts: Option<ItemHeaderParts<'_>>,
    native_budget: Option<&mut crate::container::DecodeBudget>,
) -> Result<Av1Headers, DecoderError> {
    let [
        sequence_payload,
        frame_payload,
        frame_header_payload,
        _tile_group_payload,
    ] = if let Some(parts) = header_parts {
        [
            Some(parts.sequence_payload),
            parts.frame_payload,
            parts.frame_header_payload,
            None,
        ]
    } else {
        find_obu_payloads_in_parts(
            payloads,
            [
                ObuType::SequenceHeader,
                ObuType::Frame,
                ObuType::FrameHeader,
                ObuType::TileGroup,
            ],
        )?
    };
    let sequence_payload = sequence_payload
        .ok_or_else(|| DecoderError::Bitstream("AV1 sequence header OBU is missing".to_string()))?;
    let (sequence, sequence_metadata) = parse_sequence_header_with_metadata(sequence_payload)?;
    validate_extended_pixi(source.pixel_information, &sequence.color_config)?;
    validate_color_metadata(source.color_information, &sequence.color_config)?;
    let config = av1_config.map(parse_av1_config).transpose()?;
    if let Some(config) = config {
        validate_av1_config(&config, &sequence)?;
    }
    let mut native_tile_info_allocation = None;
    let (frame, tile_group_payload) = if let Some(frame_payload) = frame_payload {
        let frame = if let Some(prefix) = frame_prefix.take() {
            if let Some(budget) = native_budget {
                let (frame, allocation) =
                    crate::av1::finish_frame_header_with_budget(prefix, budget)?;
                native_tile_info_allocation = Some(allocation);
                frame
            } else {
                crate::av1::finish_frame_header(prefix)?
            }
        } else {
            parse_frame_header_with_references_and_metadata(
                frame_payload,
                &sequence,
                &sequence_metadata,
                references,
            )?
        };
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
        let frame = if let Some(prefix) = frame_prefix.take() {
            if let Some(budget) = native_budget {
                let (frame, allocation) =
                    crate::av1::finish_frame_header_with_budget(prefix, budget)?;
                native_tile_info_allocation = Some(allocation);
                frame
            } else {
                crate::av1::finish_frame_header(prefix)?
            }
        } else {
            parse_frame_header_with_references_and_metadata(
                frame_header_payload,
                &sequence,
                &sequence_metadata,
                references,
            )?
        };
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
    if let Some(width) = source.width
        && width != frame.upscaled_width
    {
        return Err(DecoderError::Bitstream(format!(
            "AVIF ispe width {width} does not match AV1 upscaled width {}",
            frame.upscaled_width
        )));
    }
    if let Some(height) = source.height
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
    let native_allocation = native_tile_info_allocation.map(|tile_info| NativeHeaderAllocation {
        tile_info: Some(tile_info),
        tile_group: None,
        decode_plan: None,
        tile_data: AllocationTicket::new(AllocationClass::Frame),
    });
    Ok(Av1Headers {
        config,
        sequence,
        frame,
        tile_group: tile_group_payload,
        decode_plan,
        quant_state,
        native_allocation,
    })
}

/// Strict-native counterpart of the historical prefix parser.  It shares the
/// syntax and validation path, then transfers every dynamic header/tile/plan
/// owner into the carried DecodeBudget.  Public and legacy callers continue
/// to use `parse_av1_headers_with_frame_prefix` unchanged.
fn parse_av1_headers_with_frame_prefix_and_budget(
    info: &AvifInfo,
    frame_prefix: crate::av1::FramePrefix<'_, '_>,
    budget: &mut crate::container::DecodeBudget,
) -> Result<Av1Headers, DecoderError> {
    let source = ItemHeaderSource::from_info(info);
    match select_strict_native_frame_route(source.payload)? {
        StrictNativeFrameRoute::Normal { frame_payload } => {
            parse_native_normal_frame_obu_headers_with_budget(
                info,
                frame_payload,
                frame_prefix,
                budget,
            )
        }
        StrictNativeFrameRoute::Split { .. } => {
            parse_strict_native_item_headers_with_prefix_and_budget(&source, frame_prefix, budget)
        }
    }
}

/// The only coded-frame layouts accepted by strict native still decoding.
/// This classifier is deliberately allocation-free and fail-closed: a second
/// coded frame, a tile group before its frame header, or a non-tile OBU inside
/// a split frame is never sent through the historical, unbounded merger.
#[derive(Clone, Copy)]
pub(super) enum StrictNativeFrameRoute<'a> {
    Normal { frame_payload: &'a [u8] },
    Split { frame_header_payload: &'a [u8] },
}

pub(super) fn select_strict_native_frame_route(
    payload: &[u8],
) -> Result<StrictNativeFrameRoute<'_>, DecoderError> {
    let mut normal = None;
    let mut split_header = None;
    let mut split_tile_groups = 0usize;
    let mut split_started = false;

    for obu in crate::obu::ObuIter::new(payload) {
        let obu = obu?;
        match obu.obu_type {
            ObuType::Frame => {
                if normal.replace(obu.payload).is_some() || split_header.is_some() {
                    return Err(DecoderError::Unsupported(
                        "bounded native decode requires exactly one AV1 coded frame".to_string(),
                    ));
                }
            }
            ObuType::FrameHeader => {
                if normal.is_some() || split_header.replace(obu.payload).is_some() {
                    return Err(DecoderError::Unsupported(
                        "bounded native decode does not accept multiple AV1 frame headers"
                            .to_string(),
                    ));
                }
                split_started = true;
            }
            ObuType::TileGroup => {
                if normal.is_some() || !split_started {
                    return Err(DecoderError::Unsupported(
                        "bounded native decode requires tile groups to follow one AV1 frame header"
                            .to_string(),
                    ));
                }
                split_tile_groups = split_tile_groups.checked_add(1).ok_or_else(|| {
                    DecoderError::InvalidParam("AV1 tile group count overflows".to_string())
                })?;
            }
            _ if normal.is_some() => {
                return Err(DecoderError::Unsupported(
                    "bounded native decode does not accept OBUs after its AV1 OBU_FRAME"
                        .to_string(),
                ));
            }
            _ if split_started => {
                return Err(DecoderError::Unsupported(
                    "bounded native split frame contains an unsupported OBU between header and tiles"
                        .to_string(),
                ));
            }
            _ => {}
        }
    }

    match (normal, split_header, split_tile_groups) {
        (Some(frame_payload), None, 0) => Ok(StrictNativeFrameRoute::Normal { frame_payload }),
        (None, Some(frame_header_payload), groups) if groups != 0 => {
            Ok(StrictNativeFrameRoute::Split {
                frame_header_payload,
            })
        }
        (None, Some(_), 0) => Err(DecoderError::Bitstream(
            "AV1 tile group OBU is missing".to_string(),
        )),
        _ => Err(DecoderError::Unsupported(
            "bounded native decode accepts one AV1 OBU_FRAME or one split frame-header/tile-group sequence"
                .to_string(),
        )),
    }
}

pub(super) fn select_strict_native_sequence_header(payload: &[u8]) -> Result<&[u8], DecoderError> {
    let mut sequence = None;
    for obu in crate::obu::ObuIter::new(payload) {
        let obu = obu?;
        if obu.obu_type == ObuType::SequenceHeader {
            if sequence.replace(obu.payload).is_some() {
                return Err(DecoderError::Unsupported(
                    "bounded native decode does not accept multiple AV1 sequence headers"
                        .to_string(),
                ));
            }
        }
    }
    sequence
        .ok_or_else(|| DecoderError::Bitstream("AV1 sequence header OBU is missing".to_string()))
}

#[cfg(test)]
pub(super) fn select_native_normal_frame_obu_for_test(
    payload: &[u8],
) -> Result<Option<&[u8]>, DecoderError> {
    // This test-only compatibility probe predates the fail-closed strict
    // router. Keep its historical `None` result for ambiguous input; the
    // production strict path above turns that same input into an error.
    let mut frame = None;
    for obu in crate::obu::ObuIter::new(payload) {
        let obu = obu?;
        match obu.obu_type {
            ObuType::Frame if frame.replace(obu.payload).is_some() => return Ok(None),
            ObuType::FrameHeader | ObuType::TileGroup => return Ok(None),
            _ => {}
        }
    }
    Ok(frame)
}

/// Bounded materializer for the normal single-`OBU_FRAME` still shape.  It
/// never rehomes legacy vectors: tile entries, decode-plan entries, and the
/// sole copied payload each receive their final owner ticket before they are
/// populated.  Entropy diagnostics are deliberately absent from this strict
/// route; decoding initializes entropy directly from the final payload.
fn parse_native_normal_frame_obu_headers_with_budget(
    info: &AvifInfo,
    frame_payload: &[u8],
    frame_prefix: crate::av1::FramePrefix<'_, '_>,
    budget: &mut crate::container::DecodeBudget,
) -> Result<Av1Headers, DecoderError> {
    let source = ItemHeaderSource::from_info(info);
    parse_native_normal_frame_obu_headers_from_source_with_budget(
        &source,
        select_strict_native_sequence_header(source.payload)?,
        frame_payload,
        frame_prefix,
        budget,
    )
}

fn parse_native_normal_frame_obu_headers_from_source_with_budget(
    source: &ItemHeaderSource<'_>,
    sequence_payload: &[u8],
    frame_payload: &[u8],
    frame_prefix: crate::av1::FramePrefix<'_, '_>,
    budget: &mut crate::container::DecodeBudget,
) -> Result<Av1Headers, DecoderError> {
    let (sequence, _) = parse_sequence_header_with_metadata(sequence_payload)?;
    validate_extended_pixi(source.pixel_information, &sequence.color_config)?;
    validate_color_metadata(source.color_information, &sequence.color_config)?;
    let config = source.av1_config.map(parse_av1_config).transpose()?;
    if let Some(config) = config {
        validate_av1_config(&config, &sequence)?;
    }

    let mut allocation = NativeHeaderAllocation::empty();
    let result = (|| {
        let (frame, tile_info_allocation) =
            crate::av1::finish_frame_header_with_budget(frame_prefix, budget)?;
        allocation.tile_info = Some(tile_info_allocation);
        if frame.payload_after_header_offset > frame_payload.len() {
            return Err(DecoderError::Bitstream(
                "AV1 frame payload offset points outside OBU_FRAME".to_string(),
            ));
        }
        let (group, tile_group_allocation) = parse_tile_group_with_budget(
            frame_payload,
            frame.payload_after_header_offset * 8,
            &frame.tile_info,
            budget,
        )?;
        allocation.tile_group = Some(tile_group_allocation);
        let (decode_plan, decode_plan_allocation) =
            build_still_decode_plan_with_budget(&sequence, &frame, &group, budget)?;
        allocation.decode_plan = Some(decode_plan_allocation);
        let quant_state =
            QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;

        // This is the only owned copy of the `OBU_FRAME` payload on the
        // normal strict path. `TilePayload` offsets remain valid because the
        // final owner keeps the original payload layout intact.
        let (mut tile_data, tile_data_ticket) = admit_fresh_with(
            budget,
            frame_payload.len(),
            AllocationClass::Frame,
            "native AV1 frame tile data",
            fresh_replacement::<u8>,
        )?;
        allocation.tile_data = tile_data_ticket;
        tile_data.values_mut().extend_from_slice(frame_payload);

        Ok(Av1Headers {
            config,
            sequence,
            frame,
            tile_group: ParsedTileGroup {
                from_frame_obu: true,
                payload_len: frame_payload
                    .len()
                    .checked_sub(group.data_start_offset)
                    .ok_or_else(|| {
                        DecoderError::Bitstream(
                            "AV1 tile group data offset points outside OBU_FRAME".to_string(),
                        )
                    })?,
                tile_data: tile_data.into_vec(),
                group,
                entropy_states: Vec::new(),
                partition_probes: Vec::new(),
                block_mode_probes: Vec::new(),
                residual_probes: Vec::new(),
            },
            decode_plan,
            quant_state,
            native_allocation: None,
        })
    })();
    match result {
        Ok(mut headers) => {
            headers.native_allocation = Some(allocation);
            Ok(headers)
        }
        Err(error) => {
            allocation.release(budget)?;
            Err(error)
        }
    }
}

/// Runs one AVIS unit through the strict-native header materializer.
///
/// Sequence samples cannot use [`parse_av1_headers_from_source_with_prefix`]:
/// that historical route creates a payload list, prepares diagnostic entropy
/// state, and merges split tile groups before a native budget can reject the
/// allocation.  This entry point keeps the unit payload borrowed, parses its
/// frame prefix with the caller's reference slots, and then selects the same
/// final-owner normal/split materializers used by bounded still decoding.
///
/// The callback is executed while the header owners are live.  Regardless of
/// callback success, the header/tile/plan owners are dropped and their tickets
/// are released before this function returns.
pub(super) fn with_strict_sequence_headers<T, F>(
    info: &AvifInfo,
    sequence_prefix: &[u8],
    sample: &[u8],
    sample_has_sequence_header: bool,
    av1_config: Option<&[u8]>,
    references: &[Option<ReferenceFrameState>; 8],
    limits: &crate::limits::NativeDecodeLimits,
    budget: &mut crate::container::DecodeBudget,
    callback: F,
) -> Result<T, DecoderError>
where
    F: FnOnce(&Av1Headers) -> Result<T, DecoderError>,
{
    with_strict_sequence_headers_and_budget(
        info,
        sequence_prefix,
        sample,
        sample_has_sequence_header,
        av1_config,
        references,
        limits,
        budget,
        |headers, _budget| callback(headers),
    )
}

/// Budget-aware callback variant of [`with_strict_sequence_headers`].  The
/// callback runs while every native header owner is live and receives the same
/// ledger used by the header materializer, so a decoded unit can admit its
/// frame and entropy owners without creating a second accounting domain.
pub(super) fn with_strict_sequence_headers_and_budget<T, F>(
    info: &AvifInfo,
    sequence_prefix: &[u8],
    sample: &[u8],
    sample_has_sequence_header: bool,
    av1_config: Option<&[u8]>,
    references: &[Option<ReferenceFrameState>; 8],
    limits: &crate::limits::NativeDecodeLimits,
    budget: &mut crate::container::DecodeBudget,
    callback: F,
) -> Result<T, DecoderError>
where
    F: FnOnce(&Av1Headers, &mut crate::container::DecodeBudget) -> Result<T, DecoderError>,
{
    let headers = parse_av1_sequence_sample_headers_with_budget(
        info,
        sequence_prefix,
        sample,
        sample_has_sequence_header,
        av1_config,
        references,
        limits,
        budget,
    )?;
    let result = callback(&headers, budget);
    let release = drop_native_headers_and_release(headers, budget);
    match (result, release) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    }
}

/// Parses one AVIS unit's frame header without entering the legacy header
/// parser.  The route and geometry checks are borrowed/allocation-free; only
/// the final native normal/split owners are materialized.
fn parse_av1_sequence_sample_headers_with_budget(
    info: &AvifInfo,
    sequence_prefix: &[u8],
    sample: &[u8],
    sample_has_sequence_header: bool,
    av1_config: Option<&[u8]>,
    references: &[Option<ReferenceFrameState>; 8],
    limits: &crate::limits::NativeDecodeLimits,
    budget: &mut crate::container::DecodeBudget,
) -> Result<Av1Headers, DecoderError> {
    let sequence_payload = if sample_has_sequence_header {
        select_strict_native_sequence_header(sample)?
    } else {
        select_strict_native_sequence_header(sequence_prefix)?
    };
    let (sequence, sequence_metadata) = parse_sequence_header_with_metadata(sequence_payload)?;
    let source = ItemHeaderSource {
        payload: sample,
        pixel_information: info.pixel_information.as_ref(),
        color_information: info.color_information.as_ref(),
        width: info.width,
        height: info.height,
        av1_config,
    };
    validate_extended_pixi(source.pixel_information, &sequence.color_config)?;
    validate_color_metadata(source.color_information, &sequence.color_config)?;
    let config = source.av1_config.map(parse_av1_config).transpose()?;
    if let Some(config) = config.as_ref() {
        validate_av1_config(config, &sequence)?;
    }

    let route = select_strict_native_frame_route(sample)?;
    let (frame_payload, split) = match route {
        StrictNativeFrameRoute::Normal { frame_payload } => (frame_payload, false),
        StrictNativeFrameRoute::Split {
            frame_header_payload,
        } => (frame_header_payload, true),
    };
    let prefix = crate::av1::parse_frame_prefix(
        frame_payload,
        &sequence,
        &sequence_metadata,
        references,
    )?;
    frame::validate_native_frame_prefix_geometry_limits(&sequence, &prefix, limits)?;

    if split {
        parse_native_split_frame_headers_from_source_with_budget(
            &source,
            sequence_payload,
            frame_payload,
            prefix,
            budget,
        )
    } else {
        parse_native_normal_frame_obu_headers_from_source_with_budget(
            &source,
            sequence_payload,
            frame_payload,
            prefix,
            budget,
        )
    }
}

/// Computes the allocation-free upper bound for one strict AVIS unit's
/// header owners.  The bound covers the two tile-boundary arrays, the final
/// tile table, the decode-plan plane/tile tables, and the one final tile-data
/// copy.  It deliberately does not include entropy state or decoded planes:
/// those belong to the sequence-state/frame allocation stages.
pub(super) fn strict_sequence_header_peak_bytes(
    info: &AvifInfo,
    sequence_prefix: &[u8],
    sample: &[u8],
    sample_has_sequence_header: bool,
    av1_config: Option<&[u8]>,
    references: &[Option<ReferenceFrameState>; 8],
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<usize, DecoderError> {
    let sequence_payload = if sample_has_sequence_header {
        select_strict_native_sequence_header(sample)?
    } else {
        select_strict_native_sequence_header(sequence_prefix)?
    };
    let (sequence, sequence_metadata) = parse_sequence_header_with_metadata(sequence_payload)?;
    let source = ItemHeaderSource {
        payload: sample,
        pixel_information: info.pixel_information.as_ref(),
        color_information: info.color_information.as_ref(),
        width: info.width,
        height: info.height,
        av1_config,
    };
    validate_extended_pixi(source.pixel_information, &sequence.color_config)?;
    validate_color_metadata(source.color_information, &sequence.color_config)?;
    if let Some(config) = source.av1_config.map(parse_av1_config).transpose()? {
        validate_av1_config(&config, &sequence)?;
    }
    let route = select_strict_native_frame_route(sample)?;
    let frame_payload = match route {
        StrictNativeFrameRoute::Normal { frame_payload }
        | StrictNativeFrameRoute::Split {
            frame_header_payload: frame_payload,
        } => frame_payload,
    };
    let prefix = crate::av1::parse_frame_prefix(
        frame_payload,
        &sequence,
        &sequence_metadata,
        references,
    )?;
    frame::validate_native_frame_prefix_geometry_limits(&sequence, &prefix, limits)?;
    let (width, height, _, _, _) = prefix.geometry();
    let width = usize::try_from(width)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
    let height = usize::try_from(height)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
    let sb_size = if sequence.use_128x128_superblock {
        128usize
    } else {
        64usize
    };
    let mi_cols = width
        .checked_add(3)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS tile geometry overflows".to_string()))?
        / 4;
    let mi_rows = height
        .checked_add(3)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS tile geometry overflows".to_string()))?
        / 4;
    let sb_cols = mi_cols.div_ceil(sb_size / 4);
    let sb_rows = mi_rows.div_ceil(sb_size / 4);
    let tile_cols_starts = sb_cols
        .checked_add(1)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS tile columns overflow".to_string()))?;
    let tile_rows_starts = sb_rows
        .checked_add(1)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS tile rows overflow".to_string()))?;
    let tile_count = sb_cols
        .checked_mul(sb_rows)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS tile count overflows".to_string()))?;
    let plane_count: usize = if sequence.color_config.monochrome {
        1
    } else {
        3
    };
    let mut bytes = tile_cols_starts
        .checked_mul(std::mem::size_of::<u32>())
        .and_then(|value| {
            tile_rows_starts
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|rows| value.checked_add(rows))
        })
        .and_then(|value| {
            tile_count
                .checked_mul(std::mem::size_of::<crate::av1::TilePayload>())
                .and_then(|tiles| value.checked_add(tiles))
        })
        .and_then(|value| {
            plane_count
                .checked_mul(std::mem::size_of::<crate::av1::PlaneLayout>())
                .and_then(|planes| value.checked_add(planes))
        })
        .and_then(|value| {
            tile_count
                .checked_mul(std::mem::size_of::<crate::av1::TileDecodePlan>())
                .and_then(|tiles| value.checked_add(tiles))
        })
        .and_then(|value| value.checked_add(sample.len()))
        .ok_or_else(|| DecoderError::InvalidParam("AVIS strict header peak overflows".to_string()))?;
    // A split unit's final tile-data owner is no larger than the complete
    // sample.  A normal unit's copied OBU_FRAME is covered by the same bound.
    bytes = bytes.max(sample.len());
    Ok(bytes)
}

/// Strict-native header dispatcher shared by the master and the selected
/// alpha.  It deliberately does not delegate split streams to the historical
/// `parse_av1_headers_from_source_with_prefix` route, which owns per-group
/// payload vectors before it can enforce the native allocation contract.
fn parse_strict_native_item_headers_with_prefix_and_budget(
    source: &ItemHeaderSource<'_>,
    frame_prefix: crate::av1::FramePrefix<'_, '_>,
    budget: &mut crate::container::DecodeBudget,
) -> Result<Av1Headers, DecoderError> {
    let sequence_payload = select_strict_native_sequence_header(source.payload)?;
    match select_strict_native_frame_route(source.payload)? {
        StrictNativeFrameRoute::Normal { frame_payload } => {
            parse_native_normal_frame_obu_headers_from_source_with_budget(
                source,
                sequence_payload,
                frame_payload,
                frame_prefix,
                budget,
            )
        }
        StrictNativeFrameRoute::Split {
            frame_header_payload,
        } => parse_native_split_frame_headers_from_source_with_budget(
            source,
            sequence_payload,
            frame_header_payload,
            frame_prefix,
            budget,
        ),
    }
}

#[derive(Clone, Copy)]
struct StrictSplitTilePlan {
    tile_count: usize,
    tile_data_len: usize,
}

/// Walks only the tile groups belonging to the single strict split frame.
/// The classifier has already ruled out another coded frame and all
/// interstitial OBUs, but this routine repeats that structural check so its
/// two independent passes cannot drift apart.
fn walk_strict_split_tile_groups<F>(
    payload: &[u8],
    tile_info: &crate::av1::TileInfo,
    mut visit: F,
) -> Result<StrictSplitTilePlan, DecoderError>
where
    F: FnMut(&[u8], crate::av1::TilePayload) -> Result<(), DecoderError>,
{
    let expected_count = tile_info
        .tile_cols
        .checked_mul(tile_info.tile_rows)
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile count overflows".to_string()))?;
    if expected_count == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 tile count is zero".to_string(),
        ));
    }
    let mut saw_header = false;
    let mut next_tile = 0u32;
    let mut tile_count = 0usize;
    let mut tile_data_len = 0usize;
    for obu in crate::obu::ObuIter::new(payload) {
        let obu = obu?;
        match obu.obu_type {
            ObuType::FrameHeader if !saw_header => saw_header = true,
            ObuType::TileGroup if saw_header => {
                let syntax = crate::av1::walk_tile_group(obu.payload, 0, tile_info, |tile| {
                    if tile.tile_id != next_tile {
                        return Err(DecoderError::Unsupported(
                            "bounded native split frame requires complete, in-order tile groups"
                                .to_string(),
                        ));
                    }
                    let bytes = obu
                        .payload
                        .get(
                            tile.offset..tile.offset.checked_add(tile.len).ok_or_else(|| {
                                DecoderError::Bitstream(
                                    "AV1 tile payload end overflows".to_string(),
                                )
                            })?,
                        )
                        .ok_or_else(|| {
                            DecoderError::Bitstream("AV1 tile payload range is invalid".to_string())
                        })?;
                    visit(bytes, tile)?;
                    next_tile = next_tile.checked_add(1).ok_or_else(|| {
                        DecoderError::InvalidParam("AV1 tile ID overflows".to_string())
                    })?;
                    tile_count = tile_count.checked_add(1).ok_or_else(|| {
                        DecoderError::InvalidParam("AV1 tile count overflows".to_string())
                    })?;
                    tile_data_len = tile_data_len.checked_add(tile.len).ok_or_else(|| {
                        DecoderError::InvalidParam("AV1 merged tile data overflows".to_string())
                    })?;
                    Ok(())
                })?;
                if syntax.start_tile > syntax.end_tile {
                    return Err(DecoderError::Bitstream(
                        "AV1 tile group range is invalid".to_string(),
                    ));
                }
            }
            ObuType::FrameHeader | ObuType::Frame | ObuType::TileGroup => {
                return Err(DecoderError::Unsupported(
                    "bounded native split frame has an ambiguous coded OBU sequence".to_string(),
                ));
            }
            _ if saw_header => {
                return Err(DecoderError::Unsupported(
                    "bounded native split frame contains an unsupported OBU between header and tiles"
                        .to_string(),
                ));
            }
            _ => {}
        }
    }
    if !saw_header || tile_count == 0 || next_tile != expected_count {
        return Err(DecoderError::Unsupported(
            "bounded native split frame requires all AV1 tiles exactly once".to_string(),
        ));
    }
    Ok(StrictSplitTilePlan {
        tile_count,
        tile_data_len,
    })
}

/// Bounded materializer for one FRAME_HEADER plus complete in-order
/// TILE_GROUP sequence. The first pass is borrowed syntax only. The second
/// creates exactly one descriptor owner and one final merged tile-data owner;
/// no payload vectors or per-tile copies are constructed along the way.
fn parse_native_split_frame_headers_from_source_with_budget(
    source: &ItemHeaderSource<'_>,
    sequence_payload: &[u8],
    _frame_header_payload: &[u8],
    frame_prefix: crate::av1::FramePrefix<'_, '_>,
    budget: &mut crate::container::DecodeBudget,
) -> Result<Av1Headers, DecoderError> {
    let (sequence, _) = parse_sequence_header_with_metadata(sequence_payload)?;
    validate_extended_pixi(source.pixel_information, &sequence.color_config)?;
    validate_color_metadata(source.color_information, &sequence.color_config)?;
    let config = source.av1_config.map(parse_av1_config).transpose()?;
    if let Some(config) = config {
        validate_av1_config(&config, &sequence)?;
    }

    let mut allocation = NativeHeaderAllocation::empty();
    let result = (|| {
        let (frame, tile_info_allocation) =
            crate::av1::finish_frame_header_with_budget(frame_prefix, budget)?;
        allocation.tile_info = Some(tile_info_allocation);

        let split_plan =
            walk_strict_split_tile_groups(source.payload, &frame.tile_info, |_, _| Ok(()))?;
        let (mut tiles, tiles_ticket) = admit_fresh_with(
            budget,
            split_plan.tile_count,
            AllocationClass::Frame,
            "native AV1 split tile group entries",
            fresh_replacement::<crate::av1::TilePayload>,
        )?;
        allocation.tile_group = Some(crate::av1::NativeTileGroupAllocation::from_ticket(
            tiles_ticket,
        ));
        let (mut tile_data, tile_data_ticket) = admit_fresh_with(
            budget,
            split_plan.tile_data_len,
            AllocationClass::Frame,
            "native AV1 split frame tile data",
            fresh_replacement::<u8>,
        )?;
        allocation.tile_data = tile_data_ticket;

        walk_strict_split_tile_groups(source.payload, &frame.tile_info, |bytes, tile| {
            let offset = tile_data.values_mut().len();
            tile_data.values_mut().extend_from_slice(bytes);
            tiles.values_mut().push(crate::av1::TilePayload {
                tile_id: tile.tile_id,
                offset,
                len: tile.len,
            });
            Ok(())
        })?;
        if tiles.values_mut().len() != split_plan.tile_count
            || tile_data.values_mut().len() != split_plan.tile_data_len
        {
            return Err(DecoderError::Bitstream(
                "AV1 split tile merge changed between validation and materialization".to_string(),
            ));
        }
        let group = TileGroup {
            start_tile: 0,
            end_tile: frame
                .tile_info
                .tile_cols
                .checked_mul(frame.tile_info.tile_rows)
                .ok_or_else(|| DecoderError::Bitstream("AV1 tile count overflows".to_string()))?
                .checked_sub(1)
                .ok_or_else(|| DecoderError::Bitstream("AV1 tile count is zero".to_string()))?,
            data_start_offset: 0,
            tiles: tiles.into_vec(),
        };
        let (decode_plan, decode_plan_allocation) =
            build_still_decode_plan_with_budget(&sequence, &frame, &group, budget)?;
        allocation.decode_plan = Some(decode_plan_allocation);
        let quant_state =
            QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
        Ok(Av1Headers {
            config,
            sequence,
            frame,
            tile_group: ParsedTileGroup {
                from_frame_obu: false,
                payload_len: split_plan.tile_data_len,
                tile_data: tile_data.into_vec(),
                group,
                entropy_states: Vec::new(),
                partition_probes: Vec::new(),
                block_mode_probes: Vec::new(),
                residual_probes: Vec::new(),
            },
            decode_plan,
            quant_state,
            native_allocation: None,
        })
    })();
    match result {
        Ok(mut headers) => {
            headers.native_allocation = Some(allocation);
            Ok(headers)
        }
        Err(error) => {
            allocation.release(budget)?;
            Err(error)
        }
    }
}

pub(super) fn parse_av1_headers_for_item_with_prefix_and_budget(
    source: &ItemHeaderSource<'_>,
    parts: ItemHeaderParts<'_>,
    frame_prefix: crate::av1::FramePrefix<'_, '_>,
    budget: &mut crate::container::DecodeBudget,
) -> Result<Av1Headers, DecoderError> {
    let _ = parts;
    parse_strict_native_item_headers_with_prefix_and_budget(source, frame_prefix, budget)
}

/// Retires a strict-native AV1 header only after every vector stored in the
/// header has been dropped.  Releasing its tickets while `FrameHeader`,
/// `TileGroup`, or `FrameDecodePlan` still owns storage would make the live
/// DecodeBudget lie during the alpha-header phase.
pub(super) fn drop_native_headers_and_release(
    mut headers: Av1Headers,
    budget: &mut crate::container::DecodeBudget,
) -> Result<(), DecoderError> {
    let mut allocation = headers.native_allocation.take();
    drop(headers);
    if let Some(allocation) = allocation.as_mut() {
        allocation.release(budget)?;
    }
    Ok(())
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

fn decode_still_frame(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
) -> Result<DecodedFrame, DecoderError> {
    decode_still_frame_with_filter_policy(headers, info, true)
}

/// Strict-native still decode with its moved container budget kept live through
/// coded allocation, post-filter crop, and the optional auxiliary alpha.
/// Legacy and sequence paths deliberately continue to use `decode_still_frame`.
pub(super) fn decode_still_frame_with_native_budget(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
    budget: &mut crate::container::DecodeBudget,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(DecodedFrame, NativeFrameAllocation), DecoderError> {
    let (frame, _, _, allocation, mut entropy_allocation) =
        decode_still_frame_with_native_budget_and_state(
            headers,
            info,
            std::array::from_fn(|_| None),
            None,
            false,
            false,
            budget,
            limits,
        )?;
    entropy_allocation.release(budget)?;
    Ok((frame, allocation))
}

/// Private result of one strict AVIS key unit.  The returned sidecar owns the
/// budget tickets for the vectors returned alongside the frame.  Callers must
/// drop the frame, CDF vector, and motion field before releasing the sidecar.
#[allow(dead_code)]
pub(super) struct NativeSequenceUnit {
    pub(super) frame: DecodedFrame,
    pub(super) cdf_states: Vec<CdfContext>,
    pub(super) motion_field: MotionField,
    pub(super) frame_allocation: NativeFrameAllocation,
    pub(super) allocation: NativeSequenceUnitAllocation,
}

impl NativeSequenceUnit {
    /// Drops all returned owners before releasing their tickets.  A future
    /// sequence-state handoff may transfer the tickets instead; keeping this
    /// explicit method here prevents callers from releasing while the vectors
    /// are still alive.
    #[allow(dead_code)]
    pub(super) fn release(
        mut self,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<(), DecoderError> {
        let mut frame_allocation = self.frame_allocation;
        drop(self.frame);
        drop(self.cdf_states);
        drop(self.motion_field);
        self.allocation.release(budget)?;
        frame_allocation.release(budget)
    }
}

/// Budget owners for one decoded sequence unit.  This remains private because
/// the public `DecodedFrame` API has no owner token; sequence hand-off will
/// move these tickets into the state ledger in a later slice.
pub(super) struct NativeSequenceUnitAllocation {
    color_information: crate::allocation::AllocationTicket,
    cdf_states: crate::allocation::AllocationTicket,
    motion_field: crate::allocation::AllocationTicket,
}

impl NativeSequenceUnitAllocation {
    pub(super) fn release(
        &mut self,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<(), DecoderError> {
        budget.release_token(&mut self.color_information)?;
        budget.release_token(&mut self.cdf_states)?;
        budget.release_token(&mut self.motion_field)?;
        Ok(())
    }
}

fn strict_motion_field_bytes(headers: &Av1Headers) -> Result<usize, DecoderError> {
    let mi_cols = usize::try_from(
        headers
            .frame
            .frame_width
            .div_ceil(8)
            .checked_mul(2)
            .ok_or_else(|| DecoderError::InvalidParam("AVIS motion columns overflow".to_string()))?,
    )
    .map_err(|_| DecoderError::InvalidParam("AVIS motion columns overflow".to_string()))?;
    let mi_rows = usize::try_from(
        headers
            .frame
            .frame_height
            .div_ceil(8)
            .checked_mul(2)
            .ok_or_else(|| DecoderError::InvalidParam("AVIS motion rows overflow".to_string()))?,
    )
    .map_err(|_| DecoderError::InvalidParam("AVIS motion rows overflow".to_string()))?;
    let count = mi_cols
        .checked_mul(mi_rows)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS motion count overflows".to_string()))?;
    count
        .checked_mul(std::mem::size_of::<Option<u8>>())
        .and_then(|bytes| {
            count
                .checked_mul(std::mem::size_of::<Option<(i32, i32)>>())
                .and_then(|vectors| bytes.checked_add(vectors))
        })
        .and_then(|bytes| {
            count
                .checked_mul(std::mem::size_of::<Option<i32>>())
                .and_then(|offsets| bytes.checked_add(offsets))
        })
        .ok_or_else(|| DecoderError::InvalidParam("AVIS motion allocation overflows".to_string()))
}

/// Decodes one normal `OBU_FRAME` key unit through the strict native path.
///
/// The sequence header and frame header owners remain live while the callback
/// is not involved: this primitive is intentionally a direct unit boundary
/// for the sequence decoder.  It rejects inter/intra-only/show-existing and
/// split shapes so those stateful paths cannot accidentally bypass the strict
/// ownership contract.  CDF and motion vectors are collected for the caller's
/// state refresh and are covered by the returned sidecar.
#[allow(dead_code)]
pub(super) fn decode_sequence_unit_with_native_budget(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
    limits: &crate::limits::NativeDecodeLimits,
    budget: &mut crate::container::DecodeBudget,
) -> Result<NativeSequenceUnit, DecoderError> {
    if !headers.tile_group.from_frame_obu {
        return Err(DecoderError::Unsupported(
            "strict sequence unit requires a normal OBU_FRAME".to_string(),
        ));
    }
    if headers.frame.frame_type != FrameType::Key {
        return Err(DecoderError::Unsupported(
            "strict sequence unit primitive accepts key frames only".to_string(),
        ));
    }
    if headers.frame.show_existing_frame {
        return Err(DecoderError::Unsupported(
            "strict sequence unit primitive rejects show-existing frames".to_string(),
        ));
    }
    let (frame, cdf_states, motion_field, frame_allocation, entropy_allocation) =
        decode_still_frame_with_native_budget_and_state(
            headers,
            info,
            std::array::from_fn(|_| None),
            None,
            true,
            true,
            budget,
            limits,
        )?;
    let motion_field = match motion_field {
        Some(motion_field) => motion_field,
        None => {
            let mut frame_allocation = frame_allocation;
            frame_allocation.release(budget)?;
            let mut entropy_allocation = entropy_allocation;
            entropy_allocation.release(budget)?;
            return Err(DecoderError::InvalidParam(
                "strict sequence key unit did not produce a motion field".to_string(),
            ));
        }
    };
    let result = NativeSequenceUnit {
        frame,
        cdf_states,
        motion_field,
        frame_allocation,
        allocation: entropy_allocation,
    };
    // The frame allocation is returned as part of the sidecar.  The entropy
    // tickets have already been reconciled by the helper and remain attached
    // to `result` until its caller explicitly releases them.
    Ok(result)
}

fn decode_still_frame_with_native_budget_and_state(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    initial_cdfs: Option<&[CdfContext]>,
    collect_cdf: bool,
    collect_motion: bool,
    budget: &mut crate::container::DecodeBudget,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(
    DecodedFrame,
    Vec<CdfContext>,
    Option<MotionField>,
    NativeFrameAllocation,
    NativeSequenceUnitAllocation,
), DecoderError> {
    validate_public_decode_tools(headers)?;
    let planned_cdf_bytes = if collect_cdf {
        headers
            .tile_group
            .group
            .tiles
            .len()
            .checked_mul(std::mem::size_of::<CdfContext>())
            .ok_or_else(|| DecoderError::InvalidParam("AVIS CDF allocation overflows".to_string()))?
    } else {
        0
    };
    let planned_motion_bytes = if collect_motion {
        strict_motion_field_bytes(headers)?
    } else {
        0
    };
    let (mut buffers, mut allocation) =
        alloc_coded_frame_buffers_with_budget(&headers.decode_plan, budget, limits)?;
    let mut entropy_allocation = NativeSequenceUnitAllocation {
        color_information: crate::allocation::AllocationTicket::new(
            crate::allocation::AllocationClass::Frame,
        ),
        cdf_states: crate::allocation::AllocationTicket::new(
            crate::allocation::AllocationClass::Frame,
        ),
        motion_field: crate::allocation::AllocationTicket::new(
            crate::allocation::AllocationClass::Frame,
        ),
    };
    // Reserve the upper bound before the decoder creates the final CDF and
    // motion vectors.  The returned capacities are reconciled below, so a
    // platform allocator with a larger capacity cannot escape the budget.
    if planned_cdf_bytes != 0 {
        entropy_allocation.cdf_states = match budget.reserve_existing_bytes(
            crate::allocation::AllocationClass::Frame,
            planned_cdf_bytes,
            "native AVIS CDF states",
        ) {
            Ok(ticket) => ticket,
            Err(error) => {
                allocation.release(budget)?;
                return Err(error);
            }
        };
    }
    if let Some(color_information) = info.and_then(|value| value.color_information.as_ref()) {
        let requested_bytes = color_information.payload.len();
        entropy_allocation.color_information = match budget.reserve_existing_bytes(
            crate::allocation::AllocationClass::Frame,
            requested_bytes,
            "native AVIS decoded colour metadata",
        ) {
            Ok(ticket) => ticket,
            Err(error) => {
                allocation.release(budget)?;
                entropy_allocation.release(budget)?;
                return Err(error);
            }
        };
    }
    if planned_motion_bytes != 0 {
        entropy_allocation.motion_field = match budget.reserve_existing_bytes(
            crate::allocation::AllocationClass::Frame,
            planned_motion_bytes,
            "native AVIS motion field",
        ) {
            Ok(ticket) => ticket,
            Err(error) => {
                allocation.release(budget)?;
                entropy_allocation.release(budget)?;
                return Err(error);
            }
        };
    }
    let decoded = decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
        &headers.tile_group.tile_data,
        &headers.tile_group.group,
        &headers.sequence,
        &headers.frame,
        &headers.decode_plan,
        &mut buffers,
        usize::MAX,
        true,
        false,
        reference_buffers,
        initial_cdfs,
        collect_cdf,
        collect_motion,
        None,
        budget,
    );
    let (prefix, post_filter_state, final_cdfs, motion_field) = match decoded {
        Ok(decoded) => decoded,
        Err(error) => {
            allocation.release(budget)?;
            entropy_allocation.release(budget)?;
            return Err(error);
        }
    };
    if let Some(error) = prefix.next_unsupported {
        post_filter_state.release(budget)?;
        allocation.release(budget)?;
        entropy_allocation.release(budget)?;
        return Err(error);
    }
    let cdf_bytes = match final_cdfs
        .capacity()
        .checked_mul(std::mem::size_of::<CdfContext>())
    {
        Some(bytes) => bytes,
        None => {
            post_filter_state.release(budget)?;
            allocation.release(budget)?;
            entropy_allocation.release(budget)?;
            return Err(DecoderError::InvalidParam(
                "AVIS CDF capacity overflows".to_string(),
            ));
        }
    };
    if cdf_bytes > entropy_allocation.cdf_states.charged_capacity_bytes {
        post_filter_state.release(budget)?;
        allocation.release(budget)?;
        entropy_allocation.release(budget)?;
        return Err(DecoderError::InvalidParam(
            "AVIS CDF allocation exceeded its strict preflight".to_string(),
        ));
    }
    let cdf_delta = entropy_allocation.cdf_states.charged_capacity_bytes - cdf_bytes;
    budget.release_class_bytes(crate::allocation::AllocationClass::Frame, cdf_delta)?;
    entropy_allocation.cdf_states.charged_capacity_bytes = cdf_bytes;
    let motion_bytes = match motion_field.as_ref().map_or(Ok(0), |motion| {
        motion
            .reference_frames
            .capacity()
            .checked_mul(std::mem::size_of::<Option<u8>>())
            .and_then(|bytes| {
                motion
                    .motion_vectors
                    .capacity()
                    .checked_mul(std::mem::size_of::<Option<(i32, i32)>>())
                    .and_then(|vectors| bytes.checked_add(vectors))
            })
            .and_then(|bytes| {
                motion
                    .reference_offsets
                    .capacity()
                    .checked_mul(std::mem::size_of::<Option<i32>>())
                    .and_then(|offsets| bytes.checked_add(offsets))
            })
            .ok_or_else(|| DecoderError::InvalidParam("AVIS motion capacity overflows".to_string()))
    }) {
        Ok(bytes) => bytes,
        Err(error) => {
            post_filter_state.release(budget)?;
            allocation.release(budget)?;
            entropy_allocation.release(budget)?;
            return Err(error);
        }
    };
    if motion_bytes > entropy_allocation.motion_field.charged_capacity_bytes {
        post_filter_state.release(budget)?;
        allocation.release(budget)?;
        entropy_allocation.release(budget)?;
        return Err(DecoderError::InvalidParam(
            "AVIS motion allocation exceeded its strict preflight".to_string(),
        ));
    }
    let motion_delta = entropy_allocation.motion_field.charged_capacity_bytes - motion_bytes;
    budget.release_class_bytes(crate::allocation::AllocationClass::Frame, motion_delta)?;
    entropy_allocation.motion_field.charged_capacity_bytes = motion_bytes;
    let mut frame = DecodedFrame {
        width: headers.decode_plan.width,
        height: headers.decode_plan.height,
        render_width: headers.decode_plan.render_width,
        render_height: headers.decode_plan.render_height,
        bit_depth: headers.decode_plan.bit_depth,
        color_config: headers.sequence.color_config,
        color_information: info.and_then(|info| info.color_information.clone()),
        alpha_premultiplied: info.is_some_and(|info| info.alpha_premultiplied),
        buffers,
    };
    let actual_color_bytes = frame
        .color_information
        .as_ref()
        .map_or(0, |color| color.payload.capacity());
    if actual_color_bytes > entropy_allocation.color_information.charged_capacity_bytes {
        post_filter_state.release(budget)?;
        allocation.release(budget)?;
        entropy_allocation.release(budget)?;
        return Err(DecoderError::InvalidParam(
            "AVIS decoded colour metadata exceeded its strict preflight".to_string(),
        ));
    }
    let color_delta = entropy_allocation
        .color_information
        .charged_capacity_bytes
        - actual_color_bytes;
    budget.release_class_bytes(crate::allocation::AllocationClass::Frame, color_delta)?;
    entropy_allocation.color_information.charged_capacity_bytes = actual_color_bytes;
    if let Err(error) =
        apply_deblock_stage_with_budget(&mut frame, &headers.frame, &post_filter_state, budget)
    {
        drop(frame);
        post_filter_state.release(budget)?;
        drop(final_cdfs);
        drop(motion_field);
        entropy_allocation.release(budget)?;
        allocation.release(budget)?;
        return Err(error);
    }
    let mut strict_restoration_owner = if headers.frame.restoration.uses_lr {
        let plan = match StrictRestorationBoundaryPlan::for_frame(&frame) {
            Ok(plan) => plan,
            Err(error) => {
                drop(frame);
                post_filter_state.release(budget)?;
                drop(final_cdfs);
                drop(motion_field);
                entropy_allocation.release(budget)?;
                allocation.release(budget)?;
                return Err(error);
            }
        };
        match StrictRestorationBoundaryOwner::admit(plan, &frame, budget) {
            Ok(owner) => Some(owner),
            Err(error) => {
                drop(frame);
                post_filter_state.release(budget)?;
                drop(final_cdfs);
                drop(motion_field);
                entropy_allocation.release(budget)?;
                allocation.release(budget)?;
                return Err(error);
            }
        }
    } else {
        None
    };
    if let Err(error) =
        apply_cdef_stage_with_budget(&mut frame, &headers.frame, &post_filter_state, budget)
    {
        drop(frame);
        if let Some(owner) = strict_restoration_owner.take() {
            owner.release(budget)?;
        }
        post_filter_state.release(budget)?;
        drop(final_cdfs);
        drop(motion_field);
        entropy_allocation.release(budget)?;
        allocation.release(budget)?;
        return Err(error);
    }
    let mut strict_restoration_workspace = if let Some(boundary) = strict_restoration_owner.take() {
        let snapshot_plan = match StrictRestorationSnapshotPlan::for_frame(
            &frame,
            &post_filter_state,
            &[1, 2],
        ) {
            Ok(plan) => plan,
            Err(error) => {
                drop(frame);
                boundary.release(budget)?;
                post_filter_state.release(budget)?;
                drop(final_cdfs);
                drop(motion_field);
                entropy_allocation.release(budget)?;
                allocation.release(budget)?;
                return Err(error);
            }
        };
        match StrictRestorationWorkspaceOwner::admit(boundary, snapshot_plan, &frame, budget) {
            Ok(owner) => Some(owner),
            Err(error) => {
                drop(frame);
                post_filter_state.release(budget)?;
                drop(final_cdfs);
                drop(motion_field);
                entropy_allocation.release(budget)?;
                allocation.release(budget)?;
                return Err(error);
            }
        }
    } else {
        None
    };
    let restoration_unit_size = headers.decode_plan.superblock_size.checked_shl(u32::from(
        headers.frame.restoration.unit_shift,
    ));
    let mut strict_restoration_execution =
        if let Some(workspace) = strict_restoration_workspace.take() {
            let restoration_unit_size = match restoration_unit_size {
                    Some(size) => size,
                    None => {
                        let error = release_strict_restoration_workspace_on_error(
                            workspace,
                            budget,
                            DecoderError::InvalidParam(
                                "native AV1 restoration unit size overflows".to_string(),
                            ),
                        );
                        drop(frame);
                        post_filter_state.release(budget)?;
                        drop(final_cdfs);
                        drop(motion_field);
                        entropy_allocation.release(budget)?;
                        allocation.release(budget)?;
                        return Err(error);
                    }
                };
            let execution_plan =
                match StrictRestorationExecutionScratchPlan::for_frame(
                    &frame,
                    &post_filter_state,
                    &[1, 2],
                    restoration_unit_size,
                ) {
                    Ok(plan) => plan,
                    Err(error) => {
                        let error = release_strict_restoration_workspace_on_error(
                            workspace, budget, error,
                        );
                        drop(frame);
                        post_filter_state.release(budget)?;
                        drop(final_cdfs);
                        drop(motion_field);
                        entropy_allocation.release(budget)?;
                        allocation.release(budget)?;
                        return Err(error);
                    }
                };
            match StrictRestorationExecutionScratchOwner::admit(
                workspace,
                execution_plan,
                &frame,
                &post_filter_state,
                &[1, 2],
                restoration_unit_size,
                budget,
            ) {
                Ok(owner) => Some(owner),
                Err(error) => {
                    drop(frame);
                    post_filter_state.release(budget)?;
                    drop(final_cdfs);
                    drop(motion_field);
                    entropy_allocation.release(budget)?;
                    allocation.release(budget)?;
                    return Err(error);
                }
            }
        } else {
            None
        };
    if let Some(owner) = strict_restoration_execution.as_mut() {
        let restoration_unit_size = restoration_unit_size
            .expect("strict restoration owner requires a valid unit size");
        apply_loop_restoration_stage_with_strict_workspace(
            &mut frame,
            &post_filter_state,
            restoration_unit_size,
            &[1, 2],
            owner,
        );
    }
    if let Some(owner) = strict_restoration_execution.take() {
        owner.release(budget)?;
    }
    if let Err(error) = crop_native_frame_buffers_to_plan(
        &mut frame.buffers,
        &mut allocation,
        &headers.decode_plan,
        budget,
        limits,
    ) {
        post_filter_state.release(budget)?;
        allocation.release(budget)?;
        entropy_allocation.release(budget)?;
        return Err(error);
    }
    // Native limits reject super-resolution before this path. Keeping the
    // explicit guard makes a future parser relaxation fail closed instead of
    // allocating an unbudgeted resize buffer here.
    if headers.decode_plan.upscaled_width != headers.decode_plan.width {
        post_filter_state.release(budget)?;
        allocation.release(budget)?;
        entropy_allocation.release(budget)?;
        return Err(DecoderError::Unsupported(
            "bounded native decode does not yet budget AV1 super-resolution".to_string(),
        ));
    }
    if let Some(film_grain) = headers.frame.film_grain {
        apply_film_grain(&mut frame.buffers, &frame.color_config, &film_grain);
    }
    post_filter_state.release(budget)?;
    Ok((
        frame,
        final_cdfs,
        motion_field,
        allocation,
        entropy_allocation,
    ))
}

fn decode_sample_transform_frame(
    data: &[u8],
    info: &AvifInfo,
    limits: Option<&crate::limits::NativeDecodeLimits>,
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
        frames.push(decode_sample_transform_input(input, limits)?);
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
    limits: Option<&crate::limits::NativeDecodeLimits>,
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
        if limits.is_some() {
            return Err(DecoderError::Unsupported(
                "bounded native decode does not yet support grid inputs to Sample Transform"
                    .to_string(),
            ));
        }
        input_info.primary_grid = Some(grid.clone());
        decode_grid_frame(&input_info)
    } else if let Some(limits) = limits {
        frame::decode_frame_bytes_strict_from_info_for_derived(&input_info, limits)
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
    finish_decoded_still_frame(headers, decoded, validate_filters).map(|(frame, _)| frame)
}

fn finish_decoded_still_frame(
    headers: &Av1Headers,
    decoded: DecodedStillFrame,
    validate_filters: bool,
) -> Result<(DecodedFrame, MotionField), DecoderError> {
    let DecodedStillFrame {
        mut frame,
        post_filter_state,
        film_grain,
        motion_field,
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
    Ok((frame, motion_field))
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

const STRICT_DEBLOCK_FILTER_GRID_LABEL: &str = "native AV1 deblock filter grid";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictDeblockFilterGridPlan {
    width: usize,
    height: usize,
    len: usize,
    requested_bytes: usize,
}

impl StrictDeblockFilterGridPlan {
    fn for_frame(width: usize, height: usize) -> Result<Self, DecoderError> {
        if width == 0 || height == 0 {
            return Err(DecoderError::InvalidParam(
                "native AV1 deblock filter grid dimensions are empty".to_string(),
            ));
        }
        const STEP: usize = 8;
        let width = width
            .checked_add(STEP - 1)
            .map(|value| value / STEP)
            .ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 deblock filter grid width overflows".to_string(),
                )
            })?;
        let height = height
            .checked_add(STEP - 1)
            .map(|value| value / STEP)
            .ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 deblock filter grid height overflows".to_string(),
                )
            })?;
        let len = width.checked_mul(height).ok_or_else(|| {
            DecoderError::InvalidParam(
                "native AV1 deblock filter grid length overflows".to_string(),
            )
        })?;
        let requested_bytes = crate::allocation::capacity_bytes::<Option<BlockFilterState>>(
            len,
            STRICT_DEBLOCK_FILTER_GRID_LABEL,
        )?;
        Ok(Self {
            width,
            height,
            len,
            requested_bytes,
        })
    }
}

struct StrictDeblockFilterGridOwner {
    grid: Vec<Option<BlockFilterState>>,
    ticket: AllocationTicket,
}

impl StrictDeblockFilterGridOwner {
    fn admit(
        plan: StrictDeblockFilterGridPlan,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        let (candidate, ticket) = admit_fresh_with(
            budget,
            plan.len,
            AllocationClass::Frame,
            STRICT_DEBLOCK_FILTER_GRID_LABEL,
            fresh_replacement::<Option<BlockFilterState>>,
        )?;
        let mut grid = candidate.into_vec();
        // `admit_fresh_with` has already checked the actual capacity.  Filling
        // the admitted storage changes only the logical length and cannot
        // allocate, which keeps the plane pass allocation-free for this grid.
        grid.resize(plan.len, None);
        Ok(Self { grid, ticket })
    }

    fn release(mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        drop(self.grid);
        budget.release_token(&mut self.ticket)
    }
}

const STRICT_DEBLOCK_VERTICAL_ORDER_LABEL: &str = "native AV1 deblock vertical boundary order";
const STRICT_DEBLOCK_HORIZONTAL_ORDER_LABEL: &str = "native AV1 deblock horizontal boundary order";
const STRICT_DEBLOCK_VERTICAL_PREVIOUS_LABEL: &str =
    "native AV1 deblock vertical previous-boundary index";
const STRICT_DEBLOCK_HORIZONTAL_PREVIOUS_LABEL: &str =
    "native AV1 deblock horizontal previous-boundary index";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictDeblockBoundaryOrderPlan {
    count: usize,
    requested_bytes: usize,
}

impl StrictDeblockBoundaryOrderPlan {
    fn for_count(count: usize) -> Result<Self, DecoderError> {
        let per_order =
            crate::allocation::capacity_bytes::<usize>(count, STRICT_DEBLOCK_VERTICAL_ORDER_LABEL)?;
        let requested_bytes = per_order.checked_mul(4).ok_or_else(|| {
            DecoderError::InvalidParam(
                "native AV1 deblock boundary order bytes overflow".to_string(),
            )
        })?;
        Ok(Self {
            count,
            requested_bytes,
        })
    }
}

struct StrictDeblockBoundaryOrderOwner {
    vertical: Vec<usize>,
    horizontal: Vec<usize>,
    previous_vertical: Vec<usize>,
    previous_horizontal: Vec<usize>,
    tickets: [AllocationTicket; 4],
}

impl StrictDeblockBoundaryOrderOwner {
    fn admit(
        plan: StrictDeblockBoundaryOrderPlan,
        state: &PostFilterState,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        if state.transform_boundaries.len() != plan.count {
            return Err(DecoderError::InvalidParam(
                "native AV1 deblock boundary order plan is stale".to_string(),
            ));
        }
        validate_strict_deblock_previous_edges(state)?;
        let labels = [
            STRICT_DEBLOCK_VERTICAL_ORDER_LABEL,
            STRICT_DEBLOCK_HORIZONTAL_ORDER_LABEL,
            STRICT_DEBLOCK_VERTICAL_PREVIOUS_LABEL,
            STRICT_DEBLOCK_HORIZONTAL_PREVIOUS_LABEL,
        ];
        let mut admitted = std::array::from_fn::<_, 4, _>(|_| None);
        for (slot, label) in labels.into_iter().enumerate() {
            match admit_fresh_with(
                budget,
                plan.count,
                AllocationClass::Frame,
                label,
                fresh_replacement::<usize>,
            ) {
                Ok(result) => admitted[slot] = Some(result),
                Err(error) => {
                    let mut first_release_error = None;
                    for entry in admitted.iter_mut().rev() {
                        if let Some((candidate, mut ticket)) = entry.take() {
                            drop(candidate);
                            if let Err(release_error) = budget.release_token(&mut ticket) {
                                first_release_error.get_or_insert(release_error);
                            }
                        }
                    }
                    return Err(first_release_error.unwrap_or(error));
                }
            }
        }
        let [
            Some((vertical_candidate, vertical_ticket)),
            Some((horizontal_candidate, horizontal_ticket)),
            Some((previous_vertical_candidate, previous_vertical_ticket)),
            Some((previous_horizontal_candidate, previous_horizontal_ticket)),
        ] = admitted
        else {
            unreachable!("all strict deblock order candidates must be admitted")
        };
        let mut vertical = vertical_candidate.into_vec();
        let mut horizontal = horizontal_candidate.into_vec();
        let mut previous_vertical = previous_vertical_candidate.into_vec();
        let mut previous_horizontal = previous_horizontal_candidate.into_vec();
        vertical.resize(plan.count, 0);
        horizontal.resize(plan.count, 0);
        previous_vertical.resize(plan.count, 0);
        previous_horizontal.resize(plan.count, 0);
        for (index, slot) in vertical.iter_mut().enumerate() {
            *slot = index;
        }
        for (index, slot) in horizontal.iter_mut().enumerate() {
            *slot = index;
        }
        for (index, slot) in previous_vertical.iter_mut().enumerate() {
            *slot = index;
        }
        for (index, slot) in previous_horizontal.iter_mut().enumerate() {
            *slot = index;
        }
        sort_strict_deblock_vertical_order(&mut vertical, &state.transform_boundaries);
        sort_strict_deblock_horizontal_order(&mut horizontal, &state.transform_boundaries);
        sort_strict_deblock_previous_vertical_order(
            &mut previous_vertical,
            &state.transform_boundaries,
        );
        sort_strict_deblock_previous_horizontal_order(
            &mut previous_horizontal,
            &state.transform_boundaries,
        );
        Ok(Self {
            vertical,
            horizontal,
            previous_vertical,
            previous_horizontal,
            tickets: [
                vertical_ticket,
                horizontal_ticket,
                previous_vertical_ticket,
                previous_horizontal_ticket,
            ],
        })
    }

    fn release(mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        drop(self.vertical);
        drop(self.horizontal);
        drop(self.previous_vertical);
        drop(self.previous_horizontal);
        let mut first_error = None;
        for ticket in &mut self.tickets {
            if let Err(error) = budget.release_token(ticket) {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Sorts an already-admitted index buffer without requesting sorting scratch.
/// The original index is the final key so `sort_unstable_by_key` has the same
/// deterministic tie order as the legacy stable sort while remaining bounded
/// by the caller's pre-admitted vector only.
fn sort_strict_deblock_vertical_order(
    indices: &mut [usize],
    boundaries: &[crate::av1::TransformBoundary],
) {
    indices.sort_unstable_by_key(|&index| {
        let boundary = &boundaries[index];
        (
            boundary.block.plane,
            boundary.block.y,
            boundary.block.x,
            index,
        )
    });
}

fn sort_strict_deblock_horizontal_order(
    indices: &mut [usize],
    boundaries: &[crate::av1::TransformBoundary],
) {
    indices.sort_unstable_by_key(|&index| {
        let boundary = &boundaries[index];
        (
            boundary.block.plane,
            boundary.block.x,
            boundary.block.y,
            index,
        )
    });
}

fn sort_strict_deblock_previous_vertical_order(
    indices: &mut [usize],
    boundaries: &[crate::av1::TransformBoundary],
) {
    indices.sort_unstable_by_key(|&index| {
        let boundary = &boundaries[index];
        (
            boundary.block.plane,
            boundary
                .block
                .x
                .checked_add(boundary.block.tx_size.width())
                .expect("validated vertical deblock edge"),
            index,
        )
    });
}

fn sort_strict_deblock_previous_horizontal_order(
    indices: &mut [usize],
    boundaries: &[crate::av1::TransformBoundary],
) {
    indices.sort_unstable_by_key(|&index| {
        let boundary = &boundaries[index];
        (
            boundary.block.plane,
            boundary
                .block
                .y
                .checked_add(boundary.block.tx_size.height())
                .expect("validated horizontal deblock edge"),
            index,
        )
    });
}

fn validate_strict_deblock_previous_edges(state: &PostFilterState) -> Result<(), DecoderError> {
    for boundary in &state.transform_boundaries {
        boundary
            .block
            .x
            .checked_add(boundary.block.tx_size.width())
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 vertical deblock edge overflows".to_string())
            })?;
        boundary
            .block
            .y
            .checked_add(boundary.block.tx_size.height())
            .ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 horizontal deblock edge overflows".to_string(),
                )
            })?;
    }
    Ok(())
}

fn first_strict_deblock_previous_vertical<'a>(
    indices: &[usize],
    boundaries: &'a [crate::av1::TransformBoundary],
    plane: usize,
    edge_x: usize,
    edge_y: usize,
) -> Option<&'a crate::av1::TransformBoundary> {
    let key = (plane, edge_x);
    let start = indices.partition_point(|&index| {
        let boundary = &boundaries[index];
        (
            boundary.block.plane,
            boundary.block.x + boundary.block.tx_size.width(),
        ) < key
    });
    let end = start
        + indices[start..].partition_point(|&index| {
            let boundary = &boundaries[index];
            (
                boundary.block.plane,
                boundary.block.x + boundary.block.tx_size.width(),
            ) <= key
        });
    indices[start..end].iter().find_map(|&index| {
        let boundary = &boundaries[index];
        (edge_y >= boundary.block.y && edge_y < boundary.block.y + boundary.block.tx_size.height())
            .then_some(boundary)
    })
}

fn first_strict_deblock_previous_horizontal<'a>(
    indices: &[usize],
    boundaries: &'a [crate::av1::TransformBoundary],
    plane: usize,
    edge_y: usize,
    edge_x: usize,
) -> Option<&'a crate::av1::TransformBoundary> {
    let key = (plane, edge_y);
    let start = indices.partition_point(|&index| {
        let boundary = &boundaries[index];
        (
            boundary.block.plane,
            boundary.block.y + boundary.block.tx_size.height(),
        ) < key
    });
    let end = start
        + indices[start..].partition_point(|&index| {
            let boundary = &boundaries[index];
            (
                boundary.block.plane,
                boundary.block.y + boundary.block.tx_size.height(),
            ) <= key
        });
    indices[start..end].iter().find_map(|&index| {
        let boundary = &boundaries[index];
        (edge_x >= boundary.block.x && edge_x < boundary.block.x + boundary.block.tx_size.width())
            .then_some(boundary)
    })
}

struct StrictDeblockWorkspaceOwner {
    grid: StrictDeblockFilterGridOwner,
    orders: StrictDeblockBoundaryOrderOwner,
}

impl StrictDeblockWorkspaceOwner {
    fn admit(
        grid_plan: StrictDeblockFilterGridPlan,
        order_plan: StrictDeblockBoundaryOrderPlan,
        state: &PostFilterState,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        let requested_bytes = grid_plan
            .requested_bytes
            .checked_add(order_plan.requested_bytes)
            .ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 deblock workspace bytes overflow".to_string(),
                )
            })?;
        budget.check_additional_frame(requested_bytes)?;
        let grid = StrictDeblockFilterGridOwner::admit(grid_plan, budget)?;
        let orders = match StrictDeblockBoundaryOrderOwner::admit(order_plan, state, budget) {
            Ok(orders) => orders,
            Err(error) => {
                grid.release(budget)?;
                return Err(error);
            }
        };
        Ok(Self { grid, orders })
    }

    fn release(self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        let orders_result = self.orders.release(budget);
        let grid_result = self.grid.release(budget);
        orders_result.and(grid_result)
    }
}

enum DeblockBoundaryOrder<'a> {
    References(std::slice::Iter<'a, &'a crate::av1::TransformBoundary>),
    Indices {
        indices: std::slice::Iter<'a, usize>,
        boundaries: &'a [crate::av1::TransformBoundary],
    },
}

impl<'a> Iterator for DeblockBoundaryOrder<'a> {
    type Item = &'a crate::av1::TransformBoundary;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::References(iter) => iter.next().copied(),
            Self::Indices {
                indices,
                boundaries,
            } => indices.next().and_then(|&index| boundaries.get(index)),
        }
    }
}

fn populate_deblock_filter_grid(
    filter_grid: &mut [Option<BlockFilterState>],
    filter_grid_width: usize,
    filter_grid_height: usize,
    state: &PostFilterState,
) {
    const FILTER_GRID_STEP: usize = 8;
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
}

fn apply_deblock_stage_with_budget(
    frame: &mut DecodedFrame,
    frame_header: &FrameHeader,
    state: &PostFilterState,
    budget: &mut crate::container::DecodeBudget,
) -> Result<(), DecoderError> {
    if state.block_filter_states.is_empty() || state.transform_boundaries.is_empty() {
        return Ok(());
    }
    if !deblock_has_active_strengths(&frame_header.loop_filter, &frame_header.segmentation, state) {
        return Ok(());
    }
    let grid_plan = StrictDeblockFilterGridPlan::for_frame(frame.width, frame.height)?;
    let order_plan = StrictDeblockBoundaryOrderPlan::for_count(state.transform_boundaries.len())?;
    let mut workspace = StrictDeblockWorkspaceOwner::admit(grid_plan, order_plan, state, budget)?;
    populate_deblock_filter_grid(
        &mut workspace.grid.grid,
        grid_plan.width,
        grid_plan.height,
        state,
    );
    let frame_width = frame.width;
    let frame_height = frame.height;
    let bit_depth = frame.bit_depth;
    let subsampling_x = frame.color_config.subsampling_x;
    let subsampling_y = frame.color_config.subsampling_y;
    #[cfg(not(target_family = "wasm"))]
    if post_filter_parallel_work_is_large_enough(frame) {
        std::thread::scope(|scope| {
            for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
                let filter_grid = &workspace.grid.grid;
                let boundary_orders = &workspace.orders;
                scope.spawn(move || {
                    apply_deblock_plane_with_filter_grid(
                        plane,
                        plane_index,
                        frame_width,
                        frame_height,
                        bit_depth,
                        subsampling_x,
                        subsampling_y,
                        frame_header,
                        state,
                        Some(filter_grid),
                        Some(boundary_orders),
                    );
                });
            }
        });
    } else {
        for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
            apply_deblock_plane_with_filter_grid(
                plane,
                plane_index,
                frame_width,
                frame_height,
                bit_depth,
                subsampling_x,
                subsampling_y,
                frame_header,
                state,
                Some(&workspace.grid.grid),
                Some(&workspace.orders),
            );
        }
    }
    #[cfg(target_family = "wasm")]
    for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
        apply_deblock_plane_with_filter_grid(
            plane,
            plane_index,
            frame_width,
            frame_height,
            bit_depth,
            subsampling_x,
            subsampling_y,
            frame_header,
            state,
            Some(&workspace.grid.grid),
            Some(&workspace.orders),
        );
    }
    workspace.release(budget)
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
    apply_deblock_plane_with_filter_grid(
        plane,
        target_plane_index,
        frame_width,
        frame_height,
        bit_depth,
        subsampling_x,
        subsampling_y,
        frame_header,
        state,
        None,
        None,
    );
}

fn apply_deblock_plane_with_filter_grid(
    plane: &mut PlaneBuffer,
    target_plane_index: usize,
    frame_width: usize,
    frame_height: usize,
    bit_depth: u8,
    subsampling_x: bool,
    subsampling_y: bool,
    frame_header: &FrameHeader,
    state: &PostFilterState,
    shared_filter_grid: Option<&[Option<crate::av1::BlockFilterState>]>,
    shared_boundary_orders: Option<&StrictDeblockBoundaryOrderOwner>,
) {
    const FILTER_GRID_STEP: usize = 8;
    let filter_grid_width = frame_width.div_ceil(FILTER_GRID_STEP);
    let filter_grid_height = frame_height.div_ceil(FILTER_GRID_STEP);
    let filter_grid = match shared_filter_grid {
        Some(filter_grid) => std::borrow::Cow::Borrowed(filter_grid),
        None => {
            let mut filter_grid = vec![None; filter_grid_width * filter_grid_height];
            populate_deblock_filter_grid(
                &mut filter_grid,
                filter_grid_width,
                filter_grid_height,
                state,
            );
            std::borrow::Cow::Owned(filter_grid)
        }
    };
    let block_filter_state_at = |x: usize, y: usize| {
        filter_grid
            .get((y / FILTER_GRID_STEP) * filter_grid_width + x / FILTER_GRID_STEP)
            .copied()
            .flatten()
    };
    let mut previous_vertical = shared_boundary_orders.is_none().then(|| {
        std::collections::HashMap::<(usize, usize), Vec<(usize, usize, usize, bool, bool)>>::new()
    });
    let mut previous_horizontal = shared_boundary_orders.is_none().then(|| {
        std::collections::HashMap::<(usize, usize), Vec<(usize, usize, usize, bool, bool)>>::new()
    });
    if let (Some(previous_vertical), Some(previous_horizontal)) =
        (previous_vertical.as_mut(), previous_horizontal.as_mut())
    {
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
    }
    let mut legacy_boundaries = shared_boundary_orders
        .is_none()
        .then(|| state.transform_boundaries.iter().collect::<Vec<_>>());
    for vertical in [true, false] {
        let mut previous_edge = None;
        let boundary_order = if let Some(orders) = shared_boundary_orders {
            if vertical {
                DeblockBoundaryOrder::Indices {
                    indices: orders.vertical.iter(),
                    boundaries: &state.transform_boundaries,
                }
            } else {
                DeblockBoundaryOrder::Indices {
                    indices: orders.horizontal.iter(),
                    boundaries: &state.transform_boundaries,
                }
            }
        } else {
            let boundaries = legacy_boundaries
                .as_mut()
                .expect("legacy deblock boundaries must be present");
            if vertical {
                boundaries.sort_by_key(|boundary| {
                    (boundary.block.plane, boundary.block.y, boundary.block.x)
                });
            } else {
                boundaries.sort_by_key(|boundary| {
                    (boundary.block.plane, boundary.block.x, boundary.block.y)
                });
            }
            DeblockBoundaryOrder::References(boundaries.iter())
        };
        for boundary in boundary_order {
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
                } else if let Some(orders) = shared_boundary_orders {
                    let previous = if vertical {
                        first_strict_deblock_previous_vertical(
                            &orders.previous_vertical,
                            &state.transform_boundaries,
                            plane_index,
                            edge_x,
                            edge_y,
                        )
                    } else {
                        first_strict_deblock_previous_horizontal(
                            &orders.previous_horizontal,
                            &state.transform_boundaries,
                            plane_index,
                            edge_y,
                            edge_x,
                        )
                    };
                    previous
                        .map(|previous| {
                            if vertical {
                                previous.block.tx_size.width()
                            } else {
                                previous.block.tx_size.height()
                            }
                        })
                        .unwrap_or(dimension)
                } else if vertical {
                    previous_vertical
                        .as_ref()
                        .expect("legacy vertical deblock map must be present")
                        .get(&(plane_index, edge_x))
                        .into_iter()
                        .flat_map(|entries| entries.iter())
                        .find(|(y, height, _, _, _)| edge_y >= *y && edge_y < *y + *height)
                        .map(|(_, _, width, _, _)| *width)
                        .unwrap_or(dimension)
                } else {
                    previous_horizontal
                        .as_ref()
                        .expect("legacy horizontal deblock map must be present")
                        .get(&(plane_index, edge_y))
                        .into_iter()
                        .flat_map(|entries| entries.iter())
                        .find(|(x, width, _, _, _)| edge_x >= *x && edge_x < *x + *width)
                        .map(|(_, _, height, _, _)| *height)
                        .unwrap_or(dimension)
                };
                let previous_skipped = if plane_index != 0 {
                    previous_block.is_some_and(|previous| previous.skip && previous.is_inter)
                } else if let Some(orders) = shared_boundary_orders {
                    let previous = if vertical {
                        first_strict_deblock_previous_vertical(
                            &orders.previous_vertical,
                            &state.transform_boundaries,
                            plane_index,
                            edge_x,
                            edge_y,
                        )
                    } else {
                        first_strict_deblock_previous_horizontal(
                            &orders.previous_horizontal,
                            &state.transform_boundaries,
                            plane_index,
                            edge_y,
                            edge_x,
                        )
                    };
                    previous.is_some_and(|previous| previous.skip && previous.is_inter)
                } else if vertical {
                    previous_vertical
                        .as_ref()
                        .expect("legacy vertical deblock map must be present")
                        .get(&(plane_index, edge_x))
                        .into_iter()
                        .flat_map(|entries| entries.iter())
                        .find(|(y, height, _, _, _)| edge_y >= *y && edge_y < *y + *height)
                        .is_some_and(|(_, _, _, skip, is_inter)| *skip && *is_inter)
                } else {
                    previous_horizontal
                        .as_ref()
                        .expect("legacy horizontal deblock map must be present")
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
    motion_field: MotionField,
}

const STRICT_CDEF_INDEX_LABEL: &str = "native AV1 CDEF index table";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictCdefIndexPlan {
    units_width: usize,
    units_height: usize,
    len: usize,
    requested_bytes: usize,
}

impl StrictCdefIndexPlan {
    fn for_visible_dimensions(width: usize, height: usize) -> Result<Self, DecoderError> {
        if width == 0 || height == 0 {
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF visible dimensions must be nonzero".to_string(),
            ));
        }
        let units_width = width
            .checked_add(63)
            .map(|value| value / 64)
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF unit width overflows".to_string())
            })?;
        let units_height = height
            .checked_add(63)
            .map(|value| value / 64)
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF unit height overflows".to_string())
            })?;
        let len = units_width.checked_mul(units_height).ok_or_else(|| {
            DecoderError::InvalidParam("native AV1 CDEF index count overflows".to_string())
        })?;
        let requested_bytes = crate::allocation::capacity_bytes::<u8>(len, STRICT_CDEF_INDEX_LABEL)?;
        Ok(Self {
            units_width,
            units_height,
            len,
            requested_bytes,
        })
    }
}

struct StrictCdefIndexOwner {
    indices: Vec<u8>,
    ticket: AllocationTicket,
}

impl StrictCdefIndexOwner {
    fn admit(
        plan: StrictCdefIndexPlan,
        bits: u8,
        state: &PostFilterState,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        if bits > 8 {
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF bits exceed index width".to_string(),
            ));
        }
        let unit_mask = (1usize.checked_shl(u32::from(bits)).ok_or_else(|| {
            DecoderError::InvalidParam("native AV1 CDEF strength mask overflows".to_string())
        })?)
        .checked_sub(1)
        .ok_or_else(|| DecoderError::InvalidParam("native AV1 CDEF strength mask underflows".to_string()))?;
        let (candidate, ticket) = admit_fresh_with(
            budget,
            plan.len,
            AllocationClass::Frame,
            STRICT_CDEF_INDEX_LABEL,
            fresh_replacement::<u8>,
        )?;
        let mut indices = candidate.into_vec();
        indices.resize(plan.len, u8::MAX);
        if let Err(error) = populate_strict_cdef_indices(
            &mut indices,
            plan.units_width,
            unit_mask,
            state,
        ) {
            drop(indices);
            let mut ticket = ticket;
            if let Err(release_error) = budget.release_token(&mut ticket) {
                return Err(release_error);
            }
            return Err(error);
        }
        Ok(Self { indices, ticket })
    }

    fn release(mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        drop(self.indices);
        budget.release_token(&mut self.ticket)
    }
}

const STRICT_CDEF_MASK_LABEL: &str = "native AV1 CDEF filtered-block mask";
const STRICT_CDEF_ORIGINS_LABEL: &str = "native AV1 CDEF block origins";
type StrictCdefOrigin = (usize, usize, usize);
type StrictCdefDirectionBlock = (usize, usize, usize, usize, i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictCdefScratchPlan {
    blocks_width: usize,
    blocks_height: usize,
    total_blocks: usize,
    mask_words: usize,
    mask_bytes: usize,
    origins_bytes: usize,
    requested_bytes: usize,
}

impl StrictCdefScratchPlan {
    fn for_visible_dimensions(width: usize, height: usize) -> Result<Self, DecoderError> {
        if width == 0 || height == 0 {
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF scratch dimensions must be nonzero".to_string(),
            ));
        }
        let blocks_width = width
            .checked_add(7)
            .map(|value| value / 8)
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF block width overflows".to_string())
            })?;
        let blocks_height = height
            .checked_add(7)
            .map(|value| value / 8)
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF block height overflows".to_string())
            })?;
        let total_blocks = blocks_width.checked_mul(blocks_height).ok_or_else(|| {
            DecoderError::InvalidParam("native AV1 CDEF block count overflows".to_string())
        })?;
        let mask_words = total_blocks
            .checked_add(63)
            .map(|value| value / 64)
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF mask count overflows".to_string())
            })?;
        let mask_bytes = crate::allocation::capacity_bytes::<u64>(
            mask_words,
            STRICT_CDEF_MASK_LABEL,
        )?;
        let origins_bytes = crate::allocation::capacity_bytes::<StrictCdefOrigin>(
            total_blocks,
            STRICT_CDEF_ORIGINS_LABEL,
        )?;
        let requested_bytes = mask_bytes.checked_add(origins_bytes).ok_or_else(|| {
            DecoderError::InvalidParam("native AV1 CDEF scratch bytes overflow".to_string())
        })?;
        Ok(Self {
            blocks_width,
            blocks_height,
            total_blocks,
            mask_words,
            mask_bytes,
            origins_bytes,
            requested_bytes,
        })
    }
}

struct StrictCdefScratchOwner {
    origins: Vec<StrictCdefOrigin>,
    ticket: AllocationTicket,
}

impl StrictCdefScratchOwner {
    fn admit(
        plan: StrictCdefScratchPlan,
        index_owner: StrictCdefIndexOwner,
        units_width: usize,
        state: &PostFilterState,
        visible_width: usize,
        visible_height: usize,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        if plan.blocks_width != visible_width.div_ceil(8)
            || plan.blocks_height != visible_height.div_ceil(8)
        {
            let index_owner = index_owner;
            index_owner.release(budget)?;
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF scratch plan is stale".to_string(),
            ));
        }
        if let Err(error) = validate_strict_cdef_filter_endpoints(state, visible_width, visible_height)
        {
            let index_owner = index_owner;
            index_owner.release(budget)?;
            return Err(error);
        }
        if let Err(error) = budget.check_additional_frame(plan.requested_bytes) {
            let index_owner = index_owner;
            index_owner.release(budget)?;
            return Err(error);
        }

        let (mask_candidate, mut mask_ticket) = match admit_fresh_with(
            budget,
            plan.mask_words,
            AllocationClass::Frame,
            STRICT_CDEF_MASK_LABEL,
            fresh_replacement::<u64>,
        ) {
            Ok(result) => result,
            Err(error) => {
                let index_owner = index_owner;
                return match index_owner.release(budget) {
                    Ok(()) => Err(error),
                    Err(release_error) => Err(release_error),
                };
            }
        };
        let (origins_candidate, mut origins_ticket) = match admit_fresh_with(
            budget,
            plan.total_blocks,
            AllocationClass::Frame,
            STRICT_CDEF_ORIGINS_LABEL,
            fresh_replacement::<StrictCdefOrigin>,
        ) {
            Ok(result) => result,
            Err(error) => {
                drop(mask_candidate);
                let mut ticket = mask_ticket;
                let release_result = budget.release_token(&mut ticket);
                let index_owner = index_owner;
                let index_release_result = index_owner.release(budget);
                return match (release_result, index_release_result) {
                    (Err(release_error), _) | (_, Err(release_error)) => Err(release_error),
                    (Ok(()), Ok(())) => Err(error),
                };
            }
        };
        let mut mask = mask_candidate.into_vec();
        let mut origins = origins_candidate.into_vec();
        mask.resize(plan.mask_words, 0);
        if let Err(error) = populate_strict_cdef_scratch(
            &mut mask,
            &mut origins,
            plan,
            &index_owner.indices,
            units_width,
            state,
            visible_width,
            visible_height,
        ) {
            drop(origins);
            drop(mask);
            let mut origins_ticket = origins_ticket;
            let mut mask_ticket = mask_ticket;
            let mut first_error = Some(error);
            if let Err(release_error) = budget.release_token(&mut origins_ticket) {
                first_error.get_or_insert(release_error);
            }
            if let Err(release_error) = budget.release_token(&mut mask_ticket) {
                first_error.get_or_insert(release_error);
            }
            let index_owner = index_owner;
            if let Err(release_error) = index_owner.release(budget) {
                first_error.get_or_insert(release_error);
            }
            return Err(first_error.expect("strict CDEF scratch error is present"));
        }
        drop(mask);
        let mut first_error = budget.release_token(&mut mask_ticket).err();
        let index_owner = index_owner;
        if let Err(error) = index_owner.release(budget) {
            first_error.get_or_insert(error);
        }
        if let Some(error) = first_error {
            drop(origins);
            if let Err(release_error) = budget.release_token(&mut origins_ticket) {
                return Err(release_error);
            }
            return Err(error);
        }
        Ok(Self {
            origins,
            ticket: origins_ticket,
        })
    }

    fn release(mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        drop(self.origins);
        budget.release_token(&mut self.ticket)
    }
}

const STRICT_CDEF_DIRECTION_BLOCK_LABEL: &str = "native AV1 CDEF direction blocks";

#[derive(Clone, Copy)]
struct StrictCdefDirectionContext<'a> {
    luma_samples: &'a [u16],
    luma_width: usize,
    luma_height: usize,
    visible_width: usize,
    visible_height: usize,
    coeff_shift: u8,
}

struct StrictCdefDirectionBlockOwner {
    blocks: Vec<StrictCdefDirectionBlock>,
    ticket: AllocationTicket,
}

impl StrictCdefDirectionBlockOwner {
    fn admit(
        origins_owner: StrictCdefScratchOwner,
        context: StrictCdefDirectionContext<'_>,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        let count = origins_owner.origins.len();
        let requested_bytes = crate::allocation::capacity_bytes::<StrictCdefDirectionBlock>(
            count,
            STRICT_CDEF_DIRECTION_BLOCK_LABEL,
        )?;
        if let Err(error) = budget.check_additional_frame(requested_bytes) {
            origins_owner.release(budget)?;
            return Err(error);
        }
        let (candidate, mut ticket) = match admit_fresh_with(
            budget,
            count,
            AllocationClass::Frame,
            STRICT_CDEF_DIRECTION_BLOCK_LABEL,
            fresh_replacement::<StrictCdefDirectionBlock>,
        ) {
            Ok(result) => result,
            Err(error) => {
                origins_owner.release(budget)?;
                return Err(error);
            }
        };
        let mut blocks = candidate.into_vec();
        blocks.resize(count, (0, 0, 0, 0, 0));
        fill_strict_cdef_direction_blocks(&origins_owner.origins, &mut blocks, context);
        if let Err(error) = origins_owner.release(budget) {
            drop(blocks);
            return match budget.release_token(&mut ticket) {
                Ok(()) => Err(error),
                Err(release_error) => Err(release_error),
            };
        }
        Ok(Self { blocks, ticket })
    }

    fn release(mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        drop(self.blocks);
        budget.release_token(&mut self.ticket)
    }
}

const STRICT_CDEF_SNAPSHOT_LABELS: [&str; 4] = [
    "native AV1 CDEF coded samples snapshot Y",
    "native AV1 CDEF coded samples snapshot U",
    "native AV1 CDEF coded samples snapshot V",
    "native AV1 CDEF coded samples snapshot alpha",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictCdefSnapshotPlan {
    plane_count: usize,
    active_planes: [bool; 4],
    sample_lengths: [usize; 4],
    sample_bytes: [usize; 4],
    requested_bytes: usize,
}

impl StrictCdefSnapshotPlan {
    fn for_frame(
        frame: &DecodedFrame,
        active_planes: [bool; 4],
    ) -> Result<Self, DecoderError> {
        if frame.buffers.planes.len() > 4 {
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF snapshot plane count exceeds owner width".to_string(),
            ));
        }
        let mut lengths = [0; 4];
        for (plane_index, plane) in frame.buffers.planes.iter().enumerate() {
            lengths[plane_index] = plane.samples.len();
        }
        Self::for_frame_lengths(
            &lengths[..frame.buffers.planes.len()],
            active_planes,
            Some(frame),
        )
    }

    #[cfg(test)]
    fn for_sample_lengths(lengths: &[usize]) -> Result<Self, DecoderError> {
        if lengths.len() > 4 {
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF snapshot plane count exceeds owner width".to_string(),
            ));
        }
        let mut active_planes = [false; 4];
        active_planes[..lengths.len()].fill(true);
        Self::for_frame_lengths(lengths, active_planes, None)
    }

    fn for_frame_lengths(
        lengths: &[usize],
        active_planes: [bool; 4],
        frame: Option<&DecodedFrame>,
    ) -> Result<Self, DecoderError> {
        if lengths.len() > 4 {
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF snapshot plane count exceeds owner width".to_string(),
            ));
        }
        if active_planes[lengths.len()..].iter().any(|active| *active) {
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF snapshot active plane is missing".to_string(),
            ));
        }
        let mut sample_lengths = [0; 4];
        let mut sample_bytes = [0; 4];
        let mut requested_bytes: usize = 0;
        for (plane_index, &sample_length) in lengths.iter().enumerate() {
            if !active_planes[plane_index] {
                continue;
            }
            if let Some(frame) = frame {
                let plane = frame.buffers.planes.get(plane_index).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 CDEF snapshot plane is missing".to_string(),
                    )
                })?;
                let layout_samples = plane
                    .layout
                    .width
                    .checked_mul(plane.layout.height)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam(
                            "native AV1 CDEF snapshot plane dimensions overflow".to_string(),
                        )
                    })?;
                if layout_samples != plane.layout.sample_count || layout_samples != sample_length {
                    return Err(DecoderError::InvalidParam(
                        "native AV1 CDEF snapshot plane sample count mismatch".to_string(),
                    ));
                }
            }
            let bytes = crate::allocation::capacity_bytes::<u16>(
                sample_length,
                STRICT_CDEF_SNAPSHOT_LABELS[plane_index],
            )?;
            requested_bytes = requested_bytes.checked_add(bytes).ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF snapshot bytes overflow".to_string())
            })?;
            sample_lengths[plane_index] = sample_length;
            sample_bytes[plane_index] = bytes;
        }
        Ok(Self {
            plane_count: lengths.len(),
            active_planes,
            sample_lengths,
            sample_bytes,
            requested_bytes,
        })
    }
}

struct StrictCdefSnapshotOwner {
    snapshots: [Option<Vec<u16>>; 4],
    tickets: [Option<AllocationTicket>; 4],
    direction_owner: StrictCdefDirectionBlockOwner,
}

impl StrictCdefSnapshotOwner {
    fn admit(
        direction_owner: StrictCdefDirectionBlockOwner,
        plan: StrictCdefSnapshotPlan,
        frame: &DecodedFrame,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        if plan.plane_count != frame.buffers.planes.len()
            || frame
                .buffers
                .planes
                .iter()
                .enumerate()
                .any(|(index, plane)| {
                    plan.active_planes[index] && plane.samples.len() != plan.sample_lengths[index]
                })
        {
            return match direction_owner.release(budget) {
                Ok(()) => Err(DecoderError::InvalidParam(
                    "native AV1 CDEF snapshot plan is stale".to_string(),
                )),
                Err(error) => Err(error),
            };
        }
        // This check must happen before the first candidate is allocated.  The
        // actual-capacity checks below still protect the transaction when an
        // allocator returns a larger capacity than requested.
        if let Err(error) = budget.check_additional_frame(plan.requested_bytes) {
            direction_owner.release(budget)?;
            return Err(error);
        }

        let mut candidates: [Option<crate::allocation::FreshReplacement<u16>>; 4] =
            std::array::from_fn(|_| None);
        let mut tickets: [Option<AllocationTicket>; 4] = std::array::from_fn(|_| None);
        for plane_index in 0..plan.plane_count {
            if !plan.active_planes[plane_index] {
                continue;
            }
            let (candidate, ticket) = match admit_fresh_with(
                budget,
                plan.sample_lengths[plane_index],
                AllocationClass::Frame,
                STRICT_CDEF_SNAPSHOT_LABELS[plane_index],
                fresh_replacement::<u16>,
            ) {
                Ok(result) => result,
                Err(error) => {
                    let cleanup = release_strict_cdef_snapshot_candidates(
                        &mut candidates,
                        &mut tickets,
                        budget,
                    );
                    let block_cleanup = direction_owner.release(budget);
                    return match (cleanup, block_cleanup) {
                        (Err(cleanup_error), _) | (_, Err(cleanup_error)) => Err(cleanup_error),
                        (Ok(()), Ok(())) => Err(error),
                    };
                }
            };
            candidates[plane_index] = Some(candidate);
            tickets[plane_index] = Some(ticket);
        }

        // Keep all source planes untouched until every destination candidate
        // has passed admission.  The copies themselves cannot allocate because
        // each candidate was admitted with the exact source length.
        for plane_index in 0..plan.plane_count {
            if !plan.active_planes[plane_index] {
                continue;
            }
            let candidate = candidates[plane_index]
                .as_mut()
                .expect("CDEF snapshot candidate must be present");
            let source = &frame.buffers.planes[plane_index].samples;
            if candidate.values_mut().capacity() < source.len() {
                let cleanup = release_strict_cdef_snapshot_candidates(
                    &mut candidates,
                    &mut tickets,
                    budget,
                );
                let block_cleanup = direction_owner.release(budget);
                let error = DecoderError::InvalidParam(
                    "native AV1 CDEF snapshot candidate is too small".to_string(),
                );
                return match (cleanup, block_cleanup) {
                    (Err(cleanup_error), _) | (_, Err(cleanup_error)) => Err(cleanup_error),
                    (Ok(()), Ok(())) => Err(error),
                };
            }
            candidate.values_mut().extend_from_slice(source);
        }

        let mut snapshots = std::array::from_fn(|_| None);
        for plane_index in 0..plan.plane_count {
            if !plan.active_planes[plane_index] {
                continue;
            }
            snapshots[plane_index] = Some(
                candidates[plane_index]
                    .take()
                    .expect("CDEF snapshot candidate must be present")
                    .into_vec(),
            );
        }
        Ok(Self {
            snapshots,
            tickets,
            direction_owner,
        })
    }

    fn blocks(&self) -> &[StrictCdefDirectionBlock] {
        &self.direction_owner.blocks
    }

    fn release(mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        let mut first_error = None;
        for plane_index in (0..self.snapshots.len()).rev() {
            drop(self.snapshots[plane_index].take());
            if let Some(ticket) = self.tickets[plane_index].as_mut()
                && let Err(error) = budget.release_token(ticket)
            {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => {
                if let Err(release_error) = self.direction_owner.release(budget) {
                    return Err(release_error);
                }
                Err(error)
            }
            None => self.direction_owner.release(budget),
        }
    }
}

fn release_strict_cdef_snapshot_candidates(
    candidates: &mut [Option<crate::allocation::FreshReplacement<u16>>; 4],
    tickets: &mut [Option<AllocationTicket>; 4],
    budget: &mut crate::container::DecodeBudget,
) -> Result<(), DecoderError> {
    let mut first_error = None;
    for plane_index in (0..candidates.len()).rev() {
        drop(candidates[plane_index].take());
        if let Some(ticket) = tickets[plane_index].as_mut()
            && let Err(error) = budget.release_token(ticket)
        {
            first_error.get_or_insert(error);
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn fill_strict_cdef_direction_blocks(
    origins: &[StrictCdefOrigin],
    blocks: &mut [StrictCdefDirectionBlock],
    context: StrictCdefDirectionContext<'_>,
) {
    #[cfg(not(target_family = "wasm"))]
    if origins.len() >= PARALLEL_CDEF_DIRECTION_MIN_BLOCKS
        && context.luma_width.saturating_mul(context.luma_height) >= 128 * 1024
    {
        fill_strict_cdef_direction_blocks_parallel(origins, blocks, context, 4);
        return;
    }
    fill_strict_cdef_direction_blocks_serial(origins, blocks, context);
}

fn fill_strict_cdef_direction_blocks_serial(
    origins: &[StrictCdefOrigin],
    blocks: &mut [StrictCdefDirectionBlock],
    context: StrictCdefDirectionContext<'_>,
) {
    for (slot, &(x, y, index)) in origins.iter().enumerate() {
        blocks[slot] = cdef_direction_block(
            context.luma_samples,
            context.luma_width,
            context.luma_height,
            context.visible_width,
            context.visible_height,
            context.coeff_shift,
            x,
            y,
            index,
        );
    }
}

#[cfg(not(target_family = "wasm"))]
fn fill_strict_cdef_direction_blocks_parallel(
    origins: &[StrictCdefOrigin],
    blocks: &mut [StrictCdefDirectionBlock],
    context: StrictCdefDirectionContext<'_>,
    worker_budget: usize,
) {
    if worker_budget <= 1 || origins.len() < PARALLEL_CDEF_DIRECTION_MIN_BLOCKS {
        fill_strict_cdef_direction_blocks_serial(origins, blocks, context);
        return;
    }
    let midpoint = origins.len() / 2;
    let (left_origins, right_origins) = origins.split_at(midpoint);
    let (left_blocks, right_blocks) = blocks.split_at_mut(midpoint);
    std::thread::scope(|scope| {
        let left = scope.spawn(|| {
            fill_strict_cdef_direction_blocks_parallel(
                left_origins,
                left_blocks,
                context,
                worker_budget / 2,
            )
        });
        fill_strict_cdef_direction_blocks_parallel(
            right_origins,
            right_blocks,
            context,
            worker_budget - worker_budget / 2,
        );
        left.join().expect("CDEF direction worker panicked");
    });
}

fn validate_strict_cdef_filter_endpoints(
    state: &PostFilterState,
    visible_width: usize,
    visible_height: usize,
) -> Result<(), DecoderError> {
    for block in &state.block_filter_states {
        let end_x = block.x.checked_add(block.block_size.width()).ok_or_else(|| {
            DecoderError::InvalidParam("native AV1 CDEF block x endpoint overflows".to_string())
        })?;
        let end_y = block.y.checked_add(block.block_size.height()).ok_or_else(|| {
            DecoderError::InvalidParam("native AV1 CDEF block y endpoint overflows".to_string())
        })?;
        if block.x > visible_width || block.y > visible_height {
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF block origin exceeds visible frame".to_string(),
            ));
        }
        let _ = (end_x, end_y);
    }
    Ok(())
}

fn populate_strict_cdef_scratch(
    mask: &mut [u64],
    origins: &mut Vec<StrictCdefOrigin>,
    plan: StrictCdefScratchPlan,
    indices: &[u8],
    units_width: usize,
    state: &PostFilterState,
    visible_width: usize,
    visible_height: usize,
) -> Result<(), DecoderError> {
    let filtered_blocks_width = plan.blocks_width;
    for block in &state.block_filter_states {
        if block.skip {
            continue;
        }
        let start_x = (block.x / 8).min(plan.blocks_width);
        let start_y = (block.y / 8).min(plan.blocks_height);
        let end_x = block
            .x
            .checked_add(block.block_size.width())
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF block x endpoint overflows".to_string())
            })?
            .div_ceil(8)
            .min(plan.blocks_width);
        let end_y = block
            .y
            .checked_add(block.block_size.height())
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF block y endpoint overflows".to_string())
            })?
            .div_ceil(8)
            .min(plan.blocks_height);
        for y in start_y..end_y {
            for x in start_x..end_x {
                let bit = y
                    .checked_mul(filtered_blocks_width)
                    .and_then(|row| row.checked_add(x))
                    .ok_or_else(|| {
                        DecoderError::InvalidParam(
                            "native AV1 CDEF mask index overflows".to_string(),
                        )
                    })?;
                let word = bit / 64;
                let offset = bit % 64;
                let slot = mask.get_mut(word).ok_or_else(|| {
                    DecoderError::InvalidParam("native AV1 CDEF mask index is out of range".to_string())
                })?;
                *slot |= 1u64 << offset;
            }
        }
    }
    for y in (0..visible_height).step_by(8) {
        for x in (0..visible_width).step_by(8) {
            let bit = (y / 8)
                .checked_mul(filtered_blocks_width)
                .and_then(|row| row.checked_add(x / 8))
                .ok_or_else(|| {
                    DecoderError::InvalidParam("native AV1 CDEF origin index overflows".to_string())
                })?;
            if (mask[bit / 64] & (1u64 << (bit % 64))) == 0 {
                continue;
            }
            let index = (y / 64)
                .checked_mul(units_width)
                .and_then(|row| row.checked_add(x / 64))
                .ok_or_else(|| {
                    DecoderError::InvalidParam("native AV1 CDEF unit index overflows".to_string())
                })?;
            let value = indices.get(index).copied().ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF unit index is out of range".to_string())
            })?;
            origins.push((x, y, if value == u8::MAX { 0 } else { usize::from(value) }));
        }
    }
    Ok(())
}

fn populate_strict_cdef_indices(
    indices: &mut [u8],
    units_width: usize,
    unit_mask: usize,
    state: &PostFilterState,
) -> Result<(), DecoderError> {
    for (x, y, value) in state
        .cdef_units
        .iter()
        .map(|unit| (unit.x, unit.y, unit.index))
        .chain(
            state
                .cdef_blocks
                .iter()
                .map(|block| (block.x, block.y, block.index)),
        )
    {
        let row = y / 64;
        let column = x / 64;
        let index = row
            .checked_mul(units_width)
            .and_then(|offset| offset.checked_add(column))
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 CDEF index offset overflows".to_string())
            })?;
        if let Some(slot) = indices.get_mut(index) {
            *slot = (value as usize & unit_mask) as u8;
        }
    }
    Ok(())
}

fn apply_cdef_stage(frame: &mut DecodedFrame, frame_header: &FrameHeader, state: &PostFilterState) {
    apply_cdef_stage_with_optional_indices(frame, frame_header, state, None, None, None);
}

fn apply_cdef_stage_with_budget(
    frame: &mut DecodedFrame,
    frame_header: &FrameHeader,
    state: &PostFilterState,
    budget: &mut crate::container::DecodeBudget,
) -> Result<(), DecoderError> {
    if !frame_header.cdef.enabled
        || (state.cdef_units.is_empty() && state.cdef_blocks.is_empty())
        || !cdef_has_active_strengths(&frame_header.cdef)
    {
        return Ok(());
    }
    let luma = frame.buffers.planes.first().ok_or_else(|| {
        DecoderError::InvalidParam("native AV1 CDEF requires a luma plane".to_string())
    })?;
    let visible_luma_width = frame.width.min(luma.layout.width);
    let visible_luma_height = frame.height.min(luma.layout.height);
    let plan = StrictCdefIndexPlan::for_visible_dimensions(visible_luma_width, visible_luma_height)?;
    let index_owner = StrictCdefIndexOwner::admit(plan, frame_header.cdef.bits, state, budget)?;
    if !cdef_indices_have_active_strengths(&frame_header.cdef, &index_owner.indices) {
        return index_owner.release(budget);
    }
    let scratch_plan = match StrictCdefScratchPlan::for_visible_dimensions(
        visible_luma_width,
        visible_luma_height,
    ) {
        Ok(plan) => plan,
        Err(error) => {
            let index_owner = index_owner;
            index_owner.release(budget)?;
            return Err(error);
        }
    };
    let scratch_owner = StrictCdefScratchOwner::admit(
        scratch_plan,
        index_owner,
        plan.units_width,
        state,
        visible_luma_width,
        visible_luma_height,
        budget,
    )?;
    if scratch_owner.origins.is_empty() {
        return scratch_owner.release(budget);
    }
    let luma_width = luma.layout.width;
    let luma_height = luma.layout.height;
    let direction_owner = StrictCdefDirectionBlockOwner::admit(
        scratch_owner,
        StrictCdefDirectionContext {
            luma_samples: &luma.samples,
            luma_width,
            luma_height,
            visible_width: visible_luma_width,
            visible_height: visible_luma_height,
            coeff_shift: frame.bit_depth.saturating_sub(8),
        },
        budget,
    )?;
    let active_planes = match strict_cdef_active_planes(
        &direction_owner.blocks,
        &frame_header.cdef,
        frame.buffers.planes.len(),
    ) {
        Ok(active_planes) => active_planes,
        Err(error) => {
            direction_owner.release(budget)?;
            return Err(error);
        }
    };
    let snapshot_plan = match StrictCdefSnapshotPlan::for_frame(frame, active_planes) {
        Ok(plan) => plan,
        Err(error) => {
            direction_owner.release(budget)?;
            return Err(error);
        }
    };
    let snapshot_owner = match StrictCdefSnapshotOwner::admit(
        direction_owner,
        snapshot_plan,
        frame,
        budget,
    ) {
        Ok(owner) => owner,
        Err(error) => return Err(error),
    };
    apply_cdef_stage_with_optional_indices(
        frame,
        frame_header,
        state,
        None,
        Some(snapshot_owner.blocks()),
        Some(&snapshot_owner.snapshots),
    );
    snapshot_owner.release(budget)
}

fn apply_cdef_stage_with_optional_indices(
    frame: &mut DecodedFrame,
    frame_header: &FrameHeader,
    state: &PostFilterState,
    strict_indices: Option<&[u8]>,
    strict_blocks: Option<&[StrictCdefDirectionBlock]>,
    strict_snapshots: Option<&[Option<Vec<u16>>; 4]>,
) {
    if !frame_header.cdef.enabled
        || (if strict_indices.is_some() || strict_blocks.is_some() {
            state.cdef_units.is_empty() && state.cdef_blocks.is_empty()
        } else {
            state.cdef_units.is_empty()
        })
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
    let owned_cdef_indices = (strict_indices.is_none() && strict_blocks.is_none()).then(|| {
        let mut indices = vec![u8::MAX; cdef_units_width * cdef_units_height];
        for unit in &state.cdef_units {
            let index = (unit.y / 64) * cdef_units_width + unit.x / 64;
            if let Some(slot) = indices.get_mut(index) {
                *slot = (unit.index as usize & unit_mask) as u8;
            }
        }
        for block in &state.cdef_blocks {
            let index = (block.y / 64) * cdef_units_width + block.x / 64;
            if let Some(slot) = indices.get_mut(index) {
                *slot = (block.index as usize & unit_mask) as u8;
            }
        }
        indices
    });
    let cdef_indices = strict_indices.or(owned_cdef_indices.as_deref());
    if strict_blocks.is_none() {
        let cdef_indices = cdef_indices.expect("CDEF index table must be available");
        if !cdef_indices_have_active_strengths(&frame_header.cdef, cdef_indices) {
            return;
        }
    }
    let cdef = frame_header.cdef;
    let filtered_blocks_width = visible_luma_width.div_ceil(8);
    let owned_cdef_origins = strict_blocks.is_none().then(|| {
        let cdef_indices = cdef_indices.expect("CDEF index table must be available");
        let filtered_blocks = cdef_filtered_block_mask(
            &state.block_filter_states,
            visible_luma_width,
            visible_luma_height,
        );
        let mut origins = Vec::new();
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
                origins.push((x, y, index));
            }
        }
        origins
    });
    let cdef_block_origins: &[StrictCdefOrigin] = owned_cdef_origins
        .as_deref()
        .unwrap_or(&[]);
    let owned_cdef_blocks = strict_blocks.is_none().then(|| {
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
        for &(x, y, index) in cdef_block_origins {
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
    #[cfg(target_family = "wasm")]
    for &(x, y, index) in cdef_block_origins {
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
        cdef_blocks
    });
    let cdef_blocks: &[StrictCdefDirectionBlock] = strict_blocks
        .or(owned_cdef_blocks.as_deref())
        .expect("CDEF direction blocks must be available");
    let subsampling_x = frame.color_config.subsampling_x;
    let subsampling_y = frame.color_config.subsampling_y;
    // Entropy reconstruction is complete and each plane owns an independent
    // source/output buffer, so the expensive directional filtering can run
    // concurrently without sharing mutable state.
    #[cfg(not(target_family = "wasm"))]
    if post_filter_parallel_work_is_large_enough(frame) {
        std::thread::scope(|scope| {
            for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
                scope.spawn(move || {
                    apply_cdef_plane_for_mode(
                        plane,
                        plane_index,
                        subsampling_x,
                        subsampling_y,
                        cdef_coeff_shift,
                        cdef,
                        ceil_shift(visible_luma_width, usize::from(subsampling_x)),
                        ceil_shift(visible_luma_height, usize::from(subsampling_y)),
                        cdef_blocks,
                        strict_snapshots,
                    );
                });
            }
        });
        return;
    }
    for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
        apply_cdef_plane_for_mode(
            plane,
            plane_index,
            subsampling_x,
            subsampling_y,
            cdef_coeff_shift,
            cdef,
            ceil_shift(visible_luma_width, usize::from(subsampling_x)),
            ceil_shift(visible_luma_height, usize::from(subsampling_y)),
            cdef_blocks,
            strict_snapshots,
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

fn cdef_plane_has_configured_strength(
    plane_index: usize,
    cdef: crate::av1::CdefParams,
    cdef_blocks: &[(usize, usize, usize, usize, i32)],
) -> bool {
    cdef_blocks.iter().any(|&(_, _, index, _, _)| {
        let Some(strength) = cdef.strengths.get(index) else {
            return false;
        };
        if plane_index == 0 {
            strength.y_pri != 0 || strength.y_sec != 0
        } else {
            strength.uv_pri != 0 || strength.uv_sec != 0
        }
    })
}

fn apply_cdef_plane_for_mode(
    plane: &mut PlaneBuffer,
    plane_index: usize,
    subsampling_x: bool,
    subsampling_y: bool,
    cdef_coeff_shift: u8,
    cdef: crate::av1::CdefParams,
    visible_width: usize,
    visible_height: usize,
    cdef_blocks: &[(usize, usize, usize, usize, i32)],
    strict_snapshots: Option<&[Option<Vec<u16>>; 4]>,
) {
    if let Some(snapshots) = strict_snapshots {
        if let Some(source) = snapshots.get(plane_index).and_then(Option::as_deref) {
            apply_cdef_plane_from_source(
                source,
                &mut plane.samples,
                plane.layout.width,
                plane.layout.height,
                plane_index,
                subsampling_x,
                subsampling_y,
                cdef_coeff_shift,
                cdef,
                visible_width,
                visible_height,
                cdef_blocks,
            );
        }
    } else {
        apply_cdef_plane(
            plane,
            plane_index,
            subsampling_x,
            subsampling_y,
            cdef_coeff_shift,
            cdef,
            visible_width,
            visible_height,
            cdef_blocks,
        );
    }
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
    if !cdef_plane_has_configured_strength(plane_index, cdef, cdef_blocks) {
        plane.samples = source;
        return;
    }
    // Cloning preserves the source snapshot while avoiding a zero-fill pass
    // before the filtered blocks overwrite their regions.
    let mut output = source.clone();
    apply_cdef_plane_from_source(
        &source,
        &mut output,
        plane.layout.width,
        plane.layout.height,
        plane_index,
        subsampling_x,
        subsampling_y,
        cdef_coeff_shift,
        cdef,
        visible_width,
        visible_height,
        cdef_blocks,
    );
    plane.samples = output;
}

fn apply_cdef_plane_from_source(
    source: &[u16],
    output: &mut [u16],
    width: usize,
    height: usize,
    plane_index: usize,
    subsampling_x: bool,
    subsampling_y: bool,
    cdef_coeff_shift: u8,
    cdef: crate::av1::CdefParams,
    visible_width: usize,
    visible_height: usize,
    cdef_blocks: &[(usize, usize, usize, usize, i32)],
) {
    if source.len() != output.len()
        || !cdef_plane_has_configured_strength(plane_index, cdef, cdef_blocks)
    {
        return;
    }
    let visible_width = visible_width.min(width);
    let visible_height = visible_height.min(height);
    let plane_subsampling_x = usize::from(plane_index > 0 && subsampling_x);
    let plane_subsampling_y = usize::from(plane_index > 0 && subsampling_y);
    let scale_x = 1usize << plane_subsampling_x;
    let scale_y = 1usize << plane_subsampling_y;
    let mut filtered = [0u16; 64];
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

fn strict_cdef_active_planes(
    blocks: &[StrictCdefDirectionBlock],
    cdef: &crate::av1::CdefParams,
    plane_count: usize,
) -> Result<[bool; 4], DecoderError> {
    if plane_count > 4 {
        return Err(DecoderError::InvalidParam(
            "native AV1 CDEF plane count exceeds owner width".to_string(),
        ));
    }
    let strength_count = 1usize.checked_shl(u32::from(cdef.bits)).ok_or_else(|| {
        DecoderError::InvalidParam("native AV1 CDEF strength count overflows".to_string())
    })?;
    let strength_count = strength_count.min(cdef.strengths.len());
    let mut active = [false; 4];
    for &(_, _, index, _, _) in blocks {
        if index >= strength_count {
            return Err(DecoderError::InvalidParam(
                "native AV1 CDEF strength index is out of range".to_string(),
            ));
        }
        let strength = cdef.strengths.get(index).ok_or_else(|| {
            DecoderError::InvalidParam("native AV1 CDEF strength index is out of range".to_string())
        })?;
        if strength.y_pri != 0 || strength.y_sec != 0 {
            active[0] = true;
        }
        if strength.uv_pri != 0 || strength.uv_sec != 0 {
            for plane in active.iter_mut().take(plane_count).skip(1) {
                *plane = true;
            }
        }
    }
    Ok(active)
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
    apply_loop_restoration_stage_with_source(
        frame,
        state,
        unit_size,
        enabled_types,
        boundaries.map(RestorationBoundaryCollection::Legacy),
    );
}

#[allow(dead_code)]
fn apply_loop_restoration_stage_with_flat_boundaries(
    frame: &mut DecodedFrame,
    state: &PostFilterState,
    unit_size: usize,
    enabled_types: &[u8],
    owner: &StrictRestorationBoundaryOwner,
) {
    apply_loop_restoration_stage_with_source(
        frame,
        state,
        unit_size,
        enabled_types,
        Some(RestorationBoundaryCollection::Flat(owner)),
    );
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum RestorationBoundaryCollection<'a> {
    Legacy(&'a [RestorationBoundaryRows]),
    Flat(&'a StrictRestorationBoundaryOwner),
}

impl<'a> RestorationBoundaryCollection<'a> {
    fn for_plane(self, plane_index: usize) -> Option<RestorationBoundaryInput<'a>> {
        match self {
            Self::Legacy(boundaries) => boundaries
                .get(plane_index)
                .map(RestorationBoundaryInput::Legacy),
            Self::Flat(owner) => owner
                .metadata
                .get(plane_index)
                .and_then(Option::as_deref)
                .zip(owner.samples.get(plane_index).and_then(Option::as_deref))
                .map(|(metadata, samples)| RestorationBoundaryInput::Flat { metadata, samples }),
        }
    }
}

#[derive(Clone, Copy)]
enum RestorationBoundaryInput<'a> {
    Legacy(&'a RestorationBoundaryRows),
    Flat {
        metadata: &'a [StrictRestorationBoundaryMeta],
        samples: &'a [u16],
    },
}

fn apply_loop_restoration_stage_with_source(
    frame: &mut DecodedFrame,
    state: &PostFilterState,
    unit_size: usize,
    enabled_types: &[u8],
    boundaries: Option<RestorationBoundaryCollection<'_>>,
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
                let boundaries = boundaries.and_then(|boundaries| boundaries.for_plane(plane_index));
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
        let boundaries = boundaries.and_then(|boundaries| boundaries.for_plane(plane_index));
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

#[derive(Debug, Clone, Default)]
struct RestorationBoundaryRows {
    rows: Vec<(usize, Vec<u16>)>,
}

const STRICT_RESTORATION_BOUNDARY_META_LABEL: &str =
    "native AV1 restoration boundary row metadata Y";
const STRICT_RESTORATION_BOUNDARY_SAMPLES_LABEL: &str =
    "native AV1 restoration boundary row samples Y";
const STRICT_RESTORATION_BOUNDARY_META_LABELS: [&str; 4] = [
    STRICT_RESTORATION_BOUNDARY_META_LABEL,
    "native AV1 restoration boundary row metadata U",
    "native AV1 restoration boundary row metadata V",
    "native AV1 restoration boundary row metadata alpha",
];
const STRICT_RESTORATION_BOUNDARY_SAMPLES_LABELS: [&str; 4] = [
    STRICT_RESTORATION_BOUNDARY_SAMPLES_LABEL,
    "native AV1 restoration boundary row samples U",
    "native AV1 restoration boundary row samples V",
    "native AV1 restoration boundary row samples alpha",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictRestorationBoundaryMeta {
    row: usize,
    offset: usize,
    len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictRestorationPatchMeta {
    row: usize,
    start_x: usize,
    offset: usize,
    len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictRestorationBoundaryPlan {
    plane_count: usize,
    metadata_lengths: [usize; 4],
    sample_lengths: [usize; 4],
    metadata_bytes: [usize; 4],
    sample_bytes: [usize; 4],
    requested_bytes: usize,
}

impl StrictRestorationBoundaryPlan {
    fn for_frame(frame: &DecodedFrame) -> Result<Self, DecoderError> {
        if frame.buffers.planes.len() > 4 {
            return Err(DecoderError::InvalidParam(
                "native AV1 restoration boundary plane count exceeds owner width".to_string(),
            ));
        }
        let mut metadata_lengths = [0; 4];
        let mut sample_lengths = [0; 4];
        let mut metadata_bytes = [0; 4];
        let mut sample_bytes = [0; 4];
        let mut requested_bytes = 0usize;
        for (plane_index, plane) in frame.buffers.planes.iter().enumerate() {
            let (metadata_len, sample_len) =
                strict_restoration_boundary_lengths(frame, plane)?;
            let metadata = crate::allocation::capacity_bytes::<StrictRestorationBoundaryMeta>(
                metadata_len,
                STRICT_RESTORATION_BOUNDARY_META_LABEL,
            )?;
            let samples = crate::allocation::capacity_bytes::<u16>(
                sample_len,
                STRICT_RESTORATION_BOUNDARY_SAMPLES_LABEL,
            )?;
            metadata_lengths[plane_index] = metadata_len;
            sample_lengths[plane_index] = sample_len;
            metadata_bytes[plane_index] = metadata;
            sample_bytes[plane_index] = samples;
            requested_bytes = requested_bytes
                .checked_add(metadata)
                .and_then(|total| total.checked_add(samples))
                .ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration boundary bytes overflow".to_string(),
                    )
                })?;
        }
        Ok(Self {
            plane_count: frame.buffers.planes.len(),
            metadata_lengths,
            sample_lengths,
            metadata_bytes,
            sample_bytes,
            requested_bytes,
        })
    }
}

fn strict_restoration_boundary_lengths(
    frame: &DecodedFrame,
    plane: &PlaneBuffer,
) -> Result<(usize, usize), DecoderError> {
    let shift_y = usize::from(plane.layout.subsampling_y);
    if shift_y >= usize::BITS as usize {
        return Err(DecoderError::InvalidParam(
            "native AV1 restoration boundary subsampling overflows".to_string(),
        ));
    }
    let visible_height = frame.height.div_ceil(1usize << shift_y);
    let stripe_height = 64usize.checked_shr(shift_y as u32).ok_or_else(|| {
        DecoderError::InvalidParam(
            "native AV1 restoration boundary stripe height overflows".to_string(),
        )
    })?;
    let stripe_offset = 8usize.checked_shr(shift_y as u32).ok_or_else(|| {
        DecoderError::InvalidParam(
            "native AV1 restoration boundary stripe offset overflows".to_string(),
        )
    })?;
    if stripe_height == 0 || plane.layout.width == 0 {
        return Ok((0, 0));
    }
    if plane.layout.width.checked_mul(plane.layout.height) != Some(plane.samples.len()) {
        return Err(DecoderError::InvalidParam(
            "native AV1 restoration boundary plane layout is inconsistent".to_string(),
        ));
    }
    let mut metadata_len = 0usize;
    let mut sample_len = 0usize;
    let mut stripe = 0usize;
    loop {
        let stripe_start = stripe
            .checked_mul(stripe_height)
            .ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration boundary stripe start overflows".to_string(),
                )
            })?
            .saturating_sub(stripe_offset);
        if stripe_start >= visible_height {
            break;
        }
        let stripe_end = stripe
            .checked_add(1)
            .and_then(|value| value.checked_mul(stripe_height))
            .and_then(|value| value.checked_sub(stripe_offset))
            .ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration boundary stripe endpoint overflows".to_string(),
                )
            })?
            .min(visible_height);
        let mut count_rows = |start: usize, end: usize| -> Result<(), DecoderError> {
            for row in start..end {
                let offset = row.checked_mul(plane.layout.width).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration boundary row offset overflows".to_string(),
                    )
                })?;
                let end = offset.checked_add(plane.layout.width).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration boundary row endpoint overflows".to_string(),
                    )
                })?;
                if end > plane.samples.len() {
                    return Err(DecoderError::InvalidParam(
                        "native AV1 restoration boundary row exceeds coded plane".to_string(),
                    ));
                }
                metadata_len = metadata_len.checked_add(1).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration boundary metadata count overflows".to_string(),
                    )
                })?;
                sample_len = sample_len.checked_add(plane.layout.width).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration boundary sample count overflows".to_string(),
                    )
                })?;
            }
            Ok(())
        };
        if stripe > 0 {
            count_rows(stripe_start.saturating_sub(2), stripe_start)?;
        }
        if stripe_end < visible_height {
            let end = stripe_end.checked_add(2).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration boundary row range overflows".to_string(),
                )
            })?;
            count_rows(stripe_end, end.min(visible_height))?;
        }
        stripe = stripe.checked_add(1).ok_or_else(|| {
            DecoderError::InvalidParam(
                "native AV1 restoration boundary stripe count overflows".to_string(),
            )
        })?;
    }
    Ok((metadata_len, sample_len))
}

struct StrictRestorationBoundaryOwner {
    metadata: [Option<Vec<StrictRestorationBoundaryMeta>>; 4],
    samples: [Option<Vec<u16>>; 4],
    tickets: [Option<AllocationTicket>; 8],
}

const STRICT_RESTORATION_SNAPSHOT_LABELS: [&str; 4] = [
    "native AV1 restoration coded samples snapshot Y",
    "native AV1 restoration coded samples snapshot U",
    "native AV1 restoration coded samples snapshot V",
    "native AV1 restoration coded samples snapshot alpha",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictRestorationSnapshotPlan {
    plane_count: usize,
    active_planes: [bool; 4],
    sample_lengths: [usize; 4],
    requested_bytes: usize,
}

impl StrictRestorationSnapshotPlan {
    fn for_frame(
        frame: &DecodedFrame,
        state: &PostFilterState,
        enabled_types: &[u8],
    ) -> Result<Self, DecoderError> {
        if frame.buffers.planes.len() > 4 {
            return Err(DecoderError::InvalidParam(
                "native AV1 restoration snapshot plane count exceeds owner width".to_string(),
            ));
        }
        let mut active_planes = [false; 4];
        for unit in &state.restoration_units {
            if !enabled_types.contains(&unit.restoration_type) {
                continue;
            }
            if unit.plane >= frame.buffers.planes.len() {
                return Err(DecoderError::InvalidParam(
                    "native AV1 restoration unit plane is out of range".to_string(),
                ));
            }
            let plane = &frame.buffers.planes[unit.plane];
            let shift_y = u32::from(plane.layout.subsampling_y);
            let divisor = 1usize.checked_shl(shift_y).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration unit subsampling overflows".to_string(),
                )
            })?;
            let visible_height = frame.height.div_ceil(divisor);
            if unit.x >= plane.layout.width || unit.y >= visible_height {
                return Err(DecoderError::InvalidParam(
                    "native AV1 restoration unit origin is outside the coded plane".to_string(),
                ));
            }
            match unit.restoration_type {
                1 if unit.wiener.is_none() => {
                    return Err(DecoderError::InvalidParam(
                        "native AV1 restoration Wiener parameters are missing".to_string(),
                    ));
                }
                2 if unit.sgrproj.is_none() || unit.sgrproj_index.is_none() => {
                    return Err(DecoderError::InvalidParam(
                        "native AV1 restoration SGR parameters are missing".to_string(),
                    ));
                }
                _ => {}
            }
            active_planes[unit.plane] = true;
        }
        let mut sample_lengths = [0; 4];
        let mut requested_bytes = 0usize;
        for (plane_index, plane) in frame.buffers.planes.iter().enumerate() {
            if !active_planes[plane_index] {
                continue;
            }
            let layout_samples = plane
                .layout
                .width
                .checked_mul(plane.layout.height)
                .ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration snapshot plane dimensions overflow".to_string(),
                    )
                })?;
            if layout_samples != plane.layout.sample_count || layout_samples != plane.samples.len()
            {
                return Err(DecoderError::InvalidParam(
                    "native AV1 restoration snapshot plane sample count mismatch".to_string(),
                ));
            }
            let bytes = crate::allocation::capacity_bytes::<u16>(
                plane.samples.len(),
                STRICT_RESTORATION_SNAPSHOT_LABELS[plane_index],
            )?;
            sample_lengths[plane_index] = plane.samples.len();
            requested_bytes = requested_bytes.checked_add(bytes).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration snapshot bytes overflow".to_string(),
                )
            })?;
        }
        Ok(Self {
            plane_count: frame.buffers.planes.len(),
            active_planes,
            sample_lengths,
            requested_bytes,
        })
    }
}

struct StrictRestorationSnapshotOwner {
    snapshots: [Option<Vec<u16>>; 4],
    tickets: [Option<AllocationTicket>; 4],
    active_planes: [bool; 4],
    sample_lengths: [usize; 4],
}

struct StrictRestorationWorkspaceOwner {
    boundary: StrictRestorationBoundaryOwner,
    snapshots: StrictRestorationSnapshotOwner,
    snapshot_plan: StrictRestorationSnapshotPlan,
}

impl StrictRestorationSnapshotOwner {
    fn admit(
        plan: StrictRestorationSnapshotPlan,
        frame: &DecodedFrame,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        if plan.plane_count != frame.buffers.planes.len() {
            return Err(DecoderError::InvalidParam(
                "native AV1 restoration snapshot plan is stale".to_string(),
            ));
        }
        budget.check_additional_frame(plan.requested_bytes)?;
        let mut candidates: [Option<crate::allocation::FreshReplacement<u16>>; 4] =
            std::array::from_fn(|_| None);
        let mut tickets: [Option<AllocationTicket>; 4] = std::array::from_fn(|_| None);
        for plane_index in 0..plan.plane_count {
            if !plan.active_planes[plane_index] {
                continue;
            }
            let (candidate, ticket) = match admit_fresh_with(
                budget,
                plan.sample_lengths[plane_index],
                AllocationClass::Frame,
                STRICT_RESTORATION_SNAPSHOT_LABELS[plane_index],
                fresh_replacement::<u16>,
            ) {
                Ok(value) => value,
                Err(error) => {
                    return Err(release_strict_restoration_snapshot_candidates(
                        &mut candidates,
                        &mut tickets,
                        budget,
                        error,
                    ));
                }
            };
            candidates[plane_index] = Some(candidate);
            tickets[plane_index] = Some(ticket);
        }
        for plane_index in 0..plan.plane_count {
            if !plan.active_planes[plane_index] {
                continue;
            }
            let candidate = candidates[plane_index]
                .as_mut()
                .expect("restoration snapshot candidate must exist");
            let source = &frame.buffers.planes[plane_index].samples;
            if candidate.values_mut().capacity() < source.len() {
                let error = DecoderError::InvalidParam(
                    "native AV1 restoration snapshot candidate is too small".to_string(),
                );
                return Err(release_strict_restoration_snapshot_candidates(
                    &mut candidates,
                    &mut tickets,
                    budget,
                    error,
                ));
            }
            candidate.values_mut().extend_from_slice(source);
        }
        let mut snapshots = std::array::from_fn(|_| None);
        for plane_index in 0..plan.plane_count {
            if plan.active_planes[plane_index] {
                snapshots[plane_index] = Some(
                    candidates[plane_index]
                        .take()
                        .expect("restoration snapshot candidate must exist")
                        .into_vec(),
                );
            }
        }
        Ok(Self {
            snapshots,
            tickets,
            active_planes: plan.active_planes,
            sample_lengths: plan.sample_lengths,
        })
    }

    fn release(mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        for plane_index in (0..self.snapshots.len()).rev() {
            drop(self.snapshots[plane_index].take());
        }
        let mut first_error = None;
        for ticket in self.tickets.iter_mut().rev().flatten() {
            if let Err(error) = budget.release_token(ticket) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl StrictRestorationWorkspaceOwner {
    fn admit(
        boundary: StrictRestorationBoundaryOwner,
        plan: StrictRestorationSnapshotPlan,
        frame: &DecodedFrame,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        match StrictRestorationSnapshotOwner::admit(plan, frame, budget) {
            Ok(snapshots) => Ok(Self {
                boundary,
                snapshots,
                snapshot_plan: plan,
            }),
            Err(error) => match boundary.release(budget) {
                Ok(()) => Err(error),
                Err(release_error) => Err(release_error),
            },
        }
    }

    fn release(self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        let snapshots_result = self.snapshots.release(budget);
        let boundary_result = self.boundary.release(budget);
        snapshots_result.and(boundary_result)
    }
}

#[allow(dead_code)]
const STRICT_RESTORATION_SCRATCH_LABELS: [[&str; 7]; 4] = [
    [
        "native AV1 restoration Wiener scratch Y",
        "native AV1 restoration SGR scratch 0 Y",
        "native AV1 restoration SGR scratch 1 Y",
        "native AV1 restoration SGR scratch 2 Y",
        "native AV1 restoration SGR scratch 3 Y",
        "native AV1 restoration patch metadata Y",
        "native AV1 restoration patch samples Y",
    ],
    [
        "native AV1 restoration Wiener scratch U",
        "native AV1 restoration SGR scratch 0 U",
        "native AV1 restoration SGR scratch 1 U",
        "native AV1 restoration SGR scratch 2 U",
        "native AV1 restoration SGR scratch 3 U",
        "native AV1 restoration patch metadata U",
        "native AV1 restoration patch samples U",
    ],
    [
        "native AV1 restoration Wiener scratch V",
        "native AV1 restoration SGR scratch 0 V",
        "native AV1 restoration SGR scratch 1 V",
        "native AV1 restoration SGR scratch 2 V",
        "native AV1 restoration SGR scratch 3 V",
        "native AV1 restoration patch metadata V",
        "native AV1 restoration patch samples V",
    ],
    [
        "native AV1 restoration Wiener scratch alpha",
        "native AV1 restoration SGR scratch 0 alpha",
        "native AV1 restoration SGR scratch 1 alpha",
        "native AV1 restoration SGR scratch 2 alpha",
        "native AV1 restoration SGR scratch 3 alpha",
        "native AV1 restoration patch metadata alpha",
        "native AV1 restoration patch samples alpha",
    ],
];

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StrictRestorationExecutionScratchPlan {
    plane_count: usize,
    wiener_lengths: [usize; 4],
    sgr_lengths: [[usize; 4]; 4],
    patch_metadata_lengths: [usize; 4],
    patch_sample_lengths: [usize; 4],
    requested_bytes: usize,
    fingerprint: u64,
}

fn strict_restoration_scratch_fingerprint(
    frame: &DecodedFrame,
    state: &PostFilterState,
    enabled_types: &[u8],
    unit_size: usize,
) -> u64 {
    fn mix(hash: &mut u64, value: u64) {
        *hash ^= value;
        *hash = hash.wrapping_mul(1_099_511_628_211);
    }
    let mut fingerprint = 14_695_981_039_346_656_037;
    mix(&mut fingerprint, frame.width as u64);
    mix(&mut fingerprint, frame.height as u64);
    mix(&mut fingerprint, unit_size as u64);
    mix(&mut fingerprint, enabled_types.len() as u64);
    for &restoration_type in enabled_types {
        mix(&mut fingerprint, u64::from(restoration_type));
    }
    mix(&mut fingerprint, frame.buffers.planes.len() as u64);
    for plane in &frame.buffers.planes {
        mix(&mut fingerprint, plane.layout.width as u64);
        mix(&mut fingerprint, plane.layout.height as u64);
        mix(&mut fingerprint, u64::from(plane.layout.subsampling_x));
        mix(&mut fingerprint, u64::from(plane.layout.subsampling_y));
        mix(&mut fingerprint, plane.layout.sample_count as u64);
        mix(&mut fingerprint, plane.samples.len() as u64);
    }
    mix(&mut fingerprint, state.restoration_units.len() as u64);
    for unit in &state.restoration_units {
        if !enabled_types.contains(&unit.restoration_type) {
            continue;
        }
        mix(&mut fingerprint, unit.plane as u64);
        mix(&mut fingerprint, unit.x as u64);
        mix(&mut fingerprint, unit.y as u64);
        mix(&mut fingerprint, u64::from(unit.restoration_type));
        match unit.wiener {
            Some(filters) => {
                mix(&mut fingerprint, 1);
                for axis in filters {
                    for tap in axis {
                        mix(&mut fingerprint, tap as i64 as u64);
                    }
                }
            }
            None => mix(&mut fingerprint, 0),
        }
        match unit.sgrproj {
            Some(xqd) => {
                mix(&mut fingerprint, 1);
                for value in xqd {
                    mix(&mut fingerprint, value as i64 as u64);
                }
            }
            None => mix(&mut fingerprint, 0),
        }
        mix(
            &mut fingerprint,
            unit.sgrproj_index.map_or(0, |index| u64::from(index) + 1),
        );
    }
    fingerprint
}

#[allow(dead_code)]
impl StrictRestorationExecutionScratchPlan {
    fn for_frame(
        frame: &DecodedFrame,
        state: &PostFilterState,
        enabled_types: &[u8],
        unit_size: usize,
    ) -> Result<Self, DecoderError> {
        if unit_size == 0 || frame.buffers.planes.len() > 4 {
            return Err(DecoderError::InvalidParam(
                "native AV1 restoration scratch dimensions are invalid".to_string(),
            ));
        }
        let unit_margin = unit_size.checked_add(unit_size / 2).ok_or_else(|| {
            DecoderError::InvalidParam(
                "native AV1 restoration scratch unit dimensions overflow".to_string(),
            )
        })?;
        let mut wiener_lengths = [0; 4];
        let mut sgr_lengths = [[0; 4]; 4];
        let mut patch_metadata_lengths = [0; 4];
        let mut patch_sample_lengths = [0; 4];
        for unit in &state.restoration_units {
            if !enabled_types.contains(&unit.restoration_type) {
                continue;
            }
            let plane = frame.buffers.planes.get(unit.plane).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration scratch unit plane is out of range".to_string(),
                )
            })?;
            let divisor = 1usize
                .checked_shl(u32::from(plane.layout.subsampling_y))
                .ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration scratch subsampling overflows".to_string(),
                    )
                })?;
            let visible_width = frame.width.div_ceil(
                1usize
                    .checked_shl(u32::from(plane.layout.subsampling_x))
                    .ok_or_else(|| {
                        DecoderError::InvalidParam(
                            "native AV1 restoration scratch subsampling overflows".to_string(),
                        )
                    })?,
            );
            let visible_height = frame.height.div_ceil(divisor);
            let layout_samples = plane
                .layout
                .width
                .checked_mul(plane.layout.height)
                .ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration scratch plane dimensions overflow".to_string(),
                    )
                })?;
            if layout_samples != plane.layout.sample_count || layout_samples != plane.samples.len()
            {
                return Err(DecoderError::InvalidParam(
                    "native AV1 restoration scratch plane sample count mismatch".to_string(),
                ));
            }
            if unit.x >= plane.layout.width || unit.y >= visible_height {
                return Err(DecoderError::InvalidParam(
                    "native AV1 restoration scratch unit origin is outside the plane".to_string(),
                ));
            }
            if matches!(unit.restoration_type, 1) && unit.wiener.is_none() {
                return Err(DecoderError::InvalidParam(
                    "native AV1 restoration Wiener parameters are missing".to_string(),
                ));
            }
            let sgr_index = if unit.restoration_type == 2 {
                if unit.sgrproj.is_none() {
                    return Err(DecoderError::InvalidParam(
                        "native AV1 restoration SGR parameters are missing".to_string(),
                    ));
                }
                Some(unit.sgrproj_index.ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration SGR index is missing".to_string(),
                    )
                })?)
            } else {
                None
            };
            if !matches!(unit.restoration_type, 1 | 2) {
                continue;
            }
            let remaining_width = visible_width.saturating_sub(unit.x);
            let unit_width = if remaining_width < unit_margin {
                remaining_width
            } else {
                unit_size
            };
            let origin_y = unit.y.saturating_sub(8);
            let remaining_height = visible_height.saturating_sub(unit.y);
            let unit_extent = if remaining_height < unit_margin {
                remaining_height
            } else {
                unit_size
            };
            let unit_end = unit.y.checked_add(unit_extent).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration scratch unit endpoint overflows".to_string(),
                )
            })?;
            let end_y = if unit_end < visible_height {
                unit_end.saturating_sub(8)
            } else {
                visible_height
            };
            let mut stripe_y = origin_y;
            while stripe_y < end_y {
                let frame_stripe = stripe_y.checked_add(8).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration scratch stripe overflows".to_string(),
                    )
                })? / 64;
                let stripe_height =
                    (64usize - usize::from(frame_stripe == 0) * 8).min(end_y - stripe_y);
                if stripe_height == 0 {
                    break;
                }
                let patch_start_x = unit.x.saturating_sub(3);
                let patch_end_x = unit
                    .x
                    .checked_add(unit_width)
                    .and_then(|value| value.checked_add(3))
                    .ok_or_else(|| {
                        DecoderError::InvalidParam(
                            "native AV1 restoration patch endpoint overflows".to_string(),
                        )
                    })?
                    .min(visible_width)
                    .min(plane.layout.width);
                let patch_span = patch_end_x.saturating_sub(patch_start_x);
                let stripe_end = stripe_y.checked_add(stripe_height).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration patch stripe endpoint overflows".to_string(),
                    )
                })?;
                let patch_rows = usize::from(frame_stripe > 0) * 3
                    + usize::from(stripe_end < visible_height) * 3;
                patch_metadata_lengths[unit.plane] =
                    patch_metadata_lengths[unit.plane].max(patch_rows);
                let patch_samples = patch_rows.checked_mul(patch_span).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration patch sample count overflows".to_string(),
                    )
                })?;
                patch_sample_lengths[unit.plane] =
                    patch_sample_lengths[unit.plane].max(patch_samples);
                let mut chunk_x = 0;
                while chunk_x < unit_width {
                    let chunk_width = 64usize.min(unit_width - chunk_x);
                    if unit.restoration_type == 1 {
                        let length = chunk_width
                            .checked_mul(stripe_height.checked_add(6).ok_or_else(|| {
                                DecoderError::InvalidParam(
                                    "native AV1 Wiener scratch dimensions overflow".to_string(),
                                )
                            })?)
                            .ok_or_else(|| {
                                DecoderError::InvalidParam(
                                    "native AV1 Wiener scratch dimensions overflow".to_string(),
                                )
                            })?;
                        wiener_lengths[unit.plane] = wiener_lengths[unit.plane].max(length);
                    } else if let Some(index) = sgr_index {
                        let length = chunk_width
                            .checked_add(4)
                            .and_then(|width| {
                                stripe_height
                                    .checked_add(4)
                                    .and_then(|height| width.checked_mul(height))
                            })
                            .ok_or_else(|| {
                                DecoderError::InvalidParam(
                                    "native AV1 SGR scratch dimensions overflow".to_string(),
                                )
                            })?;
                        let index = usize::from(index);
                        if index >= 16 {
                            return Err(DecoderError::InvalidParam(
                                "native AV1 restoration SGR index is out of range".to_string(),
                            ));
                        }
                        let first_radius = index < 10 || index >= 14;
                        let second_radius = index < 14;
                        if first_radius {
                            sgr_lengths[unit.plane][0] = sgr_lengths[unit.plane][0].max(length);
                            sgr_lengths[unit.plane][1] = sgr_lengths[unit.plane][1].max(length);
                        }
                        if second_radius {
                            sgr_lengths[unit.plane][2] = sgr_lengths[unit.plane][2].max(length);
                            sgr_lengths[unit.plane][3] = sgr_lengths[unit.plane][3].max(length);
                        }
                    }
                    chunk_x = chunk_x.checked_add(chunk_width).ok_or_else(|| {
                        DecoderError::InvalidParam(
                            "native AV1 restoration scratch chunk count overflows".to_string(),
                        )
                    })?;
                }
                stripe_y = stripe_y.checked_add(stripe_height).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration scratch stripe count overflows".to_string(),
                    )
                })?;
            }
        }
        let mut requested_bytes = 0usize;
        for plane_index in 0..frame.buffers.planes.len() {
            requested_bytes = requested_bytes
                .checked_add(crate::allocation::capacity_bytes::<i32>(
                    wiener_lengths[plane_index],
                    STRICT_RESTORATION_SCRATCH_LABELS[plane_index][0],
                )?)
                .ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration scratch bytes overflow".to_string(),
                    )
                })?;
            for scratch_index in 0..4 {
                requested_bytes = requested_bytes
                    .checked_add(crate::allocation::capacity_bytes::<i32>(
                        sgr_lengths[plane_index][scratch_index],
                        STRICT_RESTORATION_SCRATCH_LABELS[plane_index][scratch_index + 1],
                    )?)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam(
                            "native AV1 restoration scratch bytes overflow".to_string(),
                        )
                    })?;
            }
            let patch_metadata_bytes = crate::allocation::capacity_bytes::<
                StrictRestorationPatchMeta,
            >(
                patch_metadata_lengths[plane_index],
                STRICT_RESTORATION_SCRATCH_LABELS[plane_index][5],
            )?;
            let patch_sample_bytes = crate::allocation::capacity_bytes::<u16>(
                patch_sample_lengths[plane_index],
                STRICT_RESTORATION_SCRATCH_LABELS[plane_index][6],
            )?;
            requested_bytes = requested_bytes
                .checked_add(patch_metadata_bytes)
                .and_then(|bytes| bytes.checked_add(patch_sample_bytes))
                .ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration patch bytes overflow".to_string(),
                    )
                })?;
        }
        Ok(Self {
            plane_count: frame.buffers.planes.len(),
            wiener_lengths,
            sgr_lengths,
            patch_metadata_lengths,
            patch_sample_lengths,
            requested_bytes,
            fingerprint: strict_restoration_scratch_fingerprint(
                frame,
                state,
                enabled_types,
                unit_size,
            ),
        })
    }
}

#[allow(dead_code)]
struct StrictRestorationScratchPlaneOwner {
    wiener: Option<Vec<i32>>,
    sgr: [Option<Vec<i32>>; 4],
    patch_metadata: Option<Vec<StrictRestorationPatchMeta>>,
    patch_samples: Option<Vec<u16>>,
    tickets: [Option<AllocationTicket>; 7],
}

#[allow(dead_code)]
struct StrictRestorationExecutionScratchOwner {
    workspace: StrictRestorationWorkspaceOwner,
    planes: [Option<StrictRestorationScratchPlaneOwner>; 4],
}

#[allow(dead_code)]
impl StrictRestorationExecutionScratchOwner {
    fn admit(
        workspace: StrictRestorationWorkspaceOwner,
        plan: StrictRestorationExecutionScratchPlan,
        frame: &DecodedFrame,
        state: &PostFilterState,
        enabled_types: &[u8],
        unit_size: usize,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        let compatible_plan = match StrictRestorationExecutionScratchPlan::for_frame(
            frame,
            state,
            enabled_types,
            unit_size,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                return Err(release_strict_restoration_workspace_on_error(
                    workspace, budget, error,
                ));
            }
        };
        let compatible_snapshot =
            match StrictRestorationSnapshotPlan::for_frame(frame, state, enabled_types) {
                Ok(plan) => plan,
                Err(error) => {
                    return Err(release_strict_restoration_workspace_on_error(
                        workspace, budget, error,
                    ));
                }
            };
        if plan != compatible_plan
            || plan.plane_count != frame.buffers.planes.len()
            || workspace.snapshot_plan != compatible_snapshot
            || workspace.snapshots.active_planes != compatible_snapshot.active_planes
            || workspace.snapshots.sample_lengths != compatible_snapshot.sample_lengths
        {
            return Err(release_strict_restoration_workspace_on_error(
                workspace,
                budget,
                DecoderError::InvalidParam(
                    "native AV1 restoration scratch plan is stale".to_string(),
                ),
            ));
        }
        for plane_index in 0..plan.plane_count {
            let snapshot = workspace.snapshots.snapshots[plane_index].as_ref();
            let valid = if compatible_snapshot.active_planes[plane_index] {
                snapshot.is_some_and(|values| {
                    values.len() == compatible_snapshot.sample_lengths[plane_index]
                })
            } else {
                snapshot.is_none()
            };
            if !valid {
                return Err(release_strict_restoration_workspace_on_error(
                    workspace,
                    budget,
                    DecoderError::InvalidParam(
                        "native AV1 restoration snapshot owner is inconsistent".to_string(),
                    ),
                ));
            }
        }
        let mut candidates: [[Option<crate::allocation::FreshReplacement<i32>>; 5]; 4] =
            std::array::from_fn(|_| std::array::from_fn(|_| None));
        let mut patch_metadata_candidates: [Option<
            crate::allocation::FreshReplacement<StrictRestorationPatchMeta>,
        >; 4] = std::array::from_fn(|_| None);
        let mut patch_samples_candidates: [Option<crate::allocation::FreshReplacement<u16>>; 4] =
            std::array::from_fn(|_| None);
        let mut tickets: [[Option<AllocationTicket>; 7]; 4] =
            std::array::from_fn(|_| std::array::from_fn(|_| None));
        if let Err(error) = budget.check_additional_frame(plan.requested_bytes) {
            return Err(release_strict_restoration_workspace_on_error(
                workspace, budget, error,
            ));
        }
        for plane_index in 0..plan.plane_count {
            let lengths = [
                plan.wiener_lengths[plane_index],
                plan.sgr_lengths[plane_index][0],
                plan.sgr_lengths[plane_index][1],
                plan.sgr_lengths[plane_index][2],
                plan.sgr_lengths[plane_index][3],
            ];
            for scratch_index in 0..5 {
                if lengths[scratch_index] == 0 {
                    continue;
                }
                let (candidate, ticket) = match admit_fresh_with(
                    budget,
                    lengths[scratch_index],
                    AllocationClass::Frame,
                    STRICT_RESTORATION_SCRATCH_LABELS[plane_index][scratch_index],
                    fresh_replacement::<i32>,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        let error = release_strict_restoration_scratch_candidates(
                            &mut candidates,
                            &mut tickets,
                            &mut patch_metadata_candidates,
                            &mut patch_samples_candidates,
                            budget,
                            error,
                        );
                        return Err(release_strict_restoration_workspace_on_error(
                            workspace, budget, error,
                        ));
                    }
                };
                candidates[plane_index][scratch_index] = Some(candidate);
                tickets[plane_index][scratch_index] = Some(ticket);
            }
            if plan.patch_metadata_lengths[plane_index] > 0 {
                let (candidate, ticket) = match admit_fresh_with(
                    budget,
                    plan.patch_metadata_lengths[plane_index],
                    AllocationClass::Frame,
                    STRICT_RESTORATION_SCRATCH_LABELS[plane_index][5],
                    fresh_replacement::<StrictRestorationPatchMeta>,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        let error = release_strict_restoration_scratch_candidates(
                            &mut candidates,
                            &mut tickets,
                            &mut patch_metadata_candidates,
                            &mut patch_samples_candidates,
                            budget,
                            error,
                        );
                        return Err(release_strict_restoration_workspace_on_error(
                            workspace, budget, error,
                        ));
                    }
                };
                patch_metadata_candidates[plane_index] = Some(candidate);
                tickets[plane_index][5] = Some(ticket);
            }
            if plan.patch_sample_lengths[plane_index] > 0 {
                let (candidate, ticket) = match admit_fresh_with(
                    budget,
                    plan.patch_sample_lengths[plane_index],
                    AllocationClass::Frame,
                    STRICT_RESTORATION_SCRATCH_LABELS[plane_index][6],
                    fresh_replacement::<u16>,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        let error = release_strict_restoration_scratch_candidates(
                            &mut candidates,
                            &mut tickets,
                            &mut patch_metadata_candidates,
                            &mut patch_samples_candidates,
                            budget,
                            error,
                        );
                        return Err(release_strict_restoration_workspace_on_error(
                            workspace, budget, error,
                        ));
                    }
                };
                patch_samples_candidates[plane_index] = Some(candidate);
                tickets[plane_index][6] = Some(ticket);
            }
        }
        for plane_index in 0..plan.plane_count {
            let lengths = [
                plan.wiener_lengths[plane_index],
                plan.sgr_lengths[plane_index][0],
                plan.sgr_lengths[plane_index][1],
                plan.sgr_lengths[plane_index][2],
                plan.sgr_lengths[plane_index][3],
            ];
            for scratch_index in 0..5 {
                if lengths[scratch_index] == 0 {
                    continue;
                }
                let candidate = candidates[plane_index][scratch_index]
                    .as_mut()
                    .expect("restoration scratch candidate must exist");
                if candidate.values_mut().capacity() < lengths[scratch_index] {
                    let error = DecoderError::InvalidParam(
                        "native AV1 restoration scratch candidate is too small".to_string(),
                    );
                    let error = release_strict_restoration_scratch_candidates(
                        &mut candidates,
                        &mut tickets,
                        &mut patch_metadata_candidates,
                        &mut patch_samples_candidates,
                        budget,
                        error,
                    );
                        return Err(release_strict_restoration_workspace_on_error(
                            workspace, budget, error,
                        ));
                }
                candidate.values_mut().resize(lengths[scratch_index], 0);
            }
            let patch_lengths = [
                plan.patch_metadata_lengths[plane_index],
                plan.patch_sample_lengths[plane_index],
            ];
            if patch_lengths[0] > 0 {
                let candidate = patch_metadata_candidates[plane_index]
                    .as_mut()
                    .expect("restoration patch metadata candidate must exist");
                if candidate.values_mut().capacity() < patch_lengths[0] {
                    let error = DecoderError::InvalidParam(
                        "native AV1 restoration patch metadata candidate is too small".to_string(),
                    );
                    let error = release_strict_restoration_scratch_candidates(
                        &mut candidates,
                        &mut tickets,
                        &mut patch_metadata_candidates,
                        &mut patch_samples_candidates,
                        budget,
                        error,
                    );
                    return Err(release_strict_restoration_workspace_on_error(
                        workspace, budget, error,
                    ));
                }
                candidate.values_mut().resize(patch_lengths[0], StrictRestorationPatchMeta {
                    row: 0,
                    start_x: 0,
                    offset: 0,
                    len: 0,
                });
            }
            if patch_lengths[1] > 0 {
                let candidate = patch_samples_candidates[plane_index]
                    .as_mut()
                    .expect("restoration patch samples candidate must exist");
                if candidate.values_mut().capacity() < patch_lengths[1] {
                    let error = DecoderError::InvalidParam(
                        "native AV1 restoration patch samples candidate is too small".to_string(),
                    );
                    let error = release_strict_restoration_scratch_candidates(
                        &mut candidates,
                        &mut tickets,
                        &mut patch_metadata_candidates,
                        &mut patch_samples_candidates,
                        budget,
                        error,
                    );
                    return Err(release_strict_restoration_workspace_on_error(
                        workspace, budget, error,
                    ));
                }
                candidate.values_mut().resize(patch_lengths[1], 0);
            }
        }
        let mut planes = std::array::from_fn(|_| None);
        for plane_index in 0..plan.plane_count {
            let mut plane_tickets = std::array::from_fn(|_| None);
            let wiener = candidates[plane_index][0]
                .take()
                .map(|candidate| candidate.into_vec());
            plane_tickets[0] = tickets[plane_index][0].take();
            let mut sgr = std::array::from_fn(|_| None);
            for scratch_index in 0..4 {
                sgr[scratch_index] = candidates[plane_index][scratch_index + 1]
                    .take()
                    .map(|candidate| candidate.into_vec());
                plane_tickets[scratch_index + 1] = tickets[plane_index][scratch_index + 1].take();
            }
            let patch_metadata = patch_metadata_candidates[plane_index]
                .take()
                .map(|candidate| candidate.into_vec());
            plane_tickets[5] = tickets[plane_index][5].take();
            let patch_samples = patch_samples_candidates[plane_index]
                .take()
                .map(|candidate| candidate.into_vec());
            plane_tickets[6] = tickets[plane_index][6].take();
            if wiener.is_some()
                || sgr.iter().any(Option::is_some)
                || patch_metadata.is_some()
                || patch_samples.is_some()
            {
                planes[plane_index] = Some(StrictRestorationScratchPlaneOwner {
                    wiener,
                    sgr,
                    patch_metadata,
                    patch_samples,
                    tickets: plane_tickets,
                });
            }
        }
        Ok(Self { workspace, planes })
    }

    fn release(mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        for plane in self.planes.iter_mut().rev().flatten() {
            drop(plane.wiener.take());
            for scratch in plane.sgr.iter_mut().rev() {
                drop(scratch.take());
            }
            drop(plane.patch_metadata.take());
            drop(plane.patch_samples.take());
        }
        let mut first_error = None;
        for plane in self.planes.iter_mut().rev().flatten() {
            for ticket in plane.tickets.iter_mut().rev().flatten() {
                if let Err(error) = budget.release_token(ticket) {
                    first_error.get_or_insert(error);
                }
            }
        }
        if let Err(error) = self.workspace.release(budget) {
            first_error.get_or_insert(error);
        }
        first_error.map_or(Ok(()), Err)
    }
}

#[allow(dead_code)]
fn release_strict_restoration_scratch_candidates(
    candidates: &mut [[Option<crate::allocation::FreshReplacement<i32>>; 5]; 4],
    tickets: &mut [[Option<AllocationTicket>; 7]; 4],
    patch_metadata_candidates: &mut [
        Option<crate::allocation::FreshReplacement<StrictRestorationPatchMeta>>;
        4
    ],
    patch_samples_candidates: &mut [Option<crate::allocation::FreshReplacement<u16>>; 4],
    budget: &mut crate::container::DecodeBudget,
    error: DecoderError,
) -> DecoderError {
    for plane in candidates.iter_mut().rev() {
        for candidate in plane.iter_mut().rev() {
            drop(candidate.take());
        }
    }
    for candidate in patch_samples_candidates.iter_mut().rev() {
        drop(candidate.take());
    }
    for candidate in patch_metadata_candidates.iter_mut().rev() {
        drop(candidate.take());
    }
    let mut first_error = None;
    for plane in tickets.iter_mut().rev() {
        for ticket in plane.iter_mut().rev().flatten() {
            if let Err(release_error) = budget.release_token(ticket) {
                first_error.get_or_insert(release_error);
            }
        }
    }
    first_error.unwrap_or(error)
}

fn release_strict_restoration_workspace_on_error(
    workspace: StrictRestorationWorkspaceOwner,
    budget: &mut crate::container::DecodeBudget,
    error: DecoderError,
) -> DecoderError {
    match workspace.release(budget) {
        Ok(()) => error,
        Err(release_error) => release_error,
    }
}

fn release_strict_restoration_snapshot_candidates(
    candidates: &mut [Option<crate::allocation::FreshReplacement<u16>>; 4],
    tickets: &mut [Option<AllocationTicket>; 4],
    budget: &mut crate::container::DecodeBudget,
    error: DecoderError,
) -> DecoderError {
    for candidate in candidates.iter_mut().rev() {
        drop(candidate.take());
    }
    let mut first_error = None;
    for ticket in tickets.iter_mut().rev().flatten() {
        if let Err(release_error) = budget.release_token(ticket) {
            first_error.get_or_insert(release_error);
        }
    }
    first_error.unwrap_or(error)
}

impl StrictRestorationBoundaryOwner {
    fn admit(
        plan: StrictRestorationBoundaryPlan,
        frame: &DecodedFrame,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Self, DecoderError> {
        if plan.plane_count != frame.buffers.planes.len() {
            return Err(DecoderError::InvalidParam(
                "native AV1 restoration boundary plan is stale".to_string(),
            ));
        }
        budget.check_additional_frame(plan.requested_bytes)?;
        let mut metadata_candidates: [Option<crate::allocation::FreshReplacement<
            StrictRestorationBoundaryMeta,
        >>; 4] = std::array::from_fn(|_| None);
        let mut sample_candidates: [Option<crate::allocation::FreshReplacement<u16>>; 4] =
            std::array::from_fn(|_| None);
        let mut metadata_tickets: [Option<AllocationTicket>; 4] =
            std::array::from_fn(|_| None);
        let mut sample_tickets: [Option<AllocationTicket>; 4] = std::array::from_fn(|_| None);
        for plane_index in 0..plan.plane_count {
            let metadata = match admit_fresh_with(
                budget,
                plan.metadata_lengths[plane_index],
                AllocationClass::Frame,
                STRICT_RESTORATION_BOUNDARY_META_LABELS[plane_index],
                fresh_replacement::<StrictRestorationBoundaryMeta>,
            ) {
                Ok(value) => value,
                Err(error) => {
                    return Err(release_strict_restoration_candidates(
                        &mut metadata_candidates,
                        &mut sample_candidates,
                        &mut metadata_tickets,
                        &mut sample_tickets,
                        budget,
                        error,
                    ));
                }
            };
            metadata_candidates[plane_index] = Some(metadata.0);
            metadata_tickets[plane_index] = Some(metadata.1);
            let samples = match admit_fresh_with(
                budget,
                plan.sample_lengths[plane_index],
                AllocationClass::Frame,
                STRICT_RESTORATION_BOUNDARY_SAMPLES_LABELS[plane_index],
                fresh_replacement::<u16>,
            ) {
                Ok(value) => value,
                Err(error) => {
                    return Err(release_strict_restoration_candidates(
                        &mut metadata_candidates,
                        &mut sample_candidates,
                        &mut metadata_tickets,
                        &mut sample_tickets,
                        budget,
                        error,
                    ));
                }
            };
            sample_candidates[plane_index] = Some(samples.0);
            sample_tickets[plane_index] = Some(samples.1);
        }
        let mut metadata = std::array::from_fn(|_| None);
        let mut samples = std::array::from_fn(|_| None);
        for plane_index in 0..plan.plane_count {
            let plane = &frame.buffers.planes[plane_index];
            let metadata_candidate = metadata_candidates[plane_index]
                .as_mut()
                .expect("restoration metadata candidate must exist");
            let sample_candidate = sample_candidates[plane_index]
                .as_mut()
                .expect("restoration sample candidate must exist");
            if let Err(error) = populate_strict_restoration_boundary_values(
                frame,
                plane,
                metadata_candidate.values_mut(),
                sample_candidate.values_mut(),
                plan.metadata_lengths[plane_index],
                plan.sample_lengths[plane_index],
            ) {
                return Err(release_strict_restoration_candidates(
                    &mut metadata_candidates,
                    &mut sample_candidates,
                    &mut metadata_tickets,
                    &mut sample_tickets,
                    budget,
                    error,
                ));
            }
            metadata[plane_index] = Some(
                metadata_candidates[plane_index]
                    .take()
                    .expect("restoration metadata candidate must exist")
                    .into_vec(),
            );
            samples[plane_index] = Some(
                sample_candidates[plane_index]
                    .take()
                    .expect("restoration sample candidate must exist")
                    .into_vec(),
            );
        }
        let mut tickets: [Option<AllocationTicket>; 8] = std::array::from_fn(|_| None);
        for plane_index in 0..plan.plane_count {
            tickets[plane_index] = metadata_tickets[plane_index].take();
            tickets[4 + plane_index] = sample_tickets[plane_index].take();
        }
        Ok(Self {
            metadata,
            samples,
            tickets,
        })
    }

    fn release(mut self, budget: &mut crate::container::DecodeBudget) -> Result<(), DecoderError> {
        for plane_index in (0..4).rev() {
            drop(self.metadata[plane_index].take());
            drop(self.samples[plane_index].take());
        }
        let mut first_error = None;
        for ticket in self.tickets.iter_mut().rev().flatten() {
            if let Err(error) = budget.release_token(ticket) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

fn release_strict_restoration_candidates(
    metadata: &mut [Option<crate::allocation::FreshReplacement<StrictRestorationBoundaryMeta>>; 4],
    samples: &mut [Option<crate::allocation::FreshReplacement<u16>>; 4],
    metadata_tickets: &mut [Option<AllocationTicket>; 4],
    sample_tickets: &mut [Option<AllocationTicket>; 4],
    budget: &mut crate::container::DecodeBudget,
    error: DecoderError,
) -> DecoderError {
    for plane_index in (0..4).rev() {
        drop(samples[plane_index].take());
        drop(metadata[plane_index].take());
    }
    let mut first_error = None;
    for ticket in sample_tickets.iter_mut().rev().chain(metadata_tickets.iter_mut().rev()).flatten() {
        if let Err(release_error) = budget.release_token(ticket) {
            first_error.get_or_insert(release_error);
        }
    }
    first_error.unwrap_or(error)
}

fn populate_strict_restoration_boundary_values(
    frame: &DecodedFrame,
    plane: &PlaneBuffer,
    metadata: &mut Vec<StrictRestorationBoundaryMeta>,
    samples: &mut Vec<u16>,
    expected_metadata: usize,
    expected_samples: usize,
) -> Result<(), DecoderError> {
    let shift_y = usize::from(plane.layout.subsampling_y);
    let shift = u32::try_from(shift_y).map_err(|_| {
        DecoderError::InvalidParam(
            "native AV1 restoration boundary subsampling overflows".to_string(),
        )
    })?;
    let divisor = 1usize.checked_shl(shift).ok_or_else(|| {
        DecoderError::InvalidParam(
            "native AV1 restoration boundary subsampling overflows".to_string(),
        )
    })?;
    let visible_height = frame.height.div_ceil(divisor);
    let stripe_height = 64usize.checked_shr(shift).ok_or_else(|| {
        DecoderError::InvalidParam(
            "native AV1 restoration boundary stripe height overflows".to_string(),
        )
    })?;
    let stripe_offset = 8usize.checked_shr(shift).ok_or_else(|| {
        DecoderError::InvalidParam(
            "native AV1 restoration boundary stripe offset overflows".to_string(),
        )
    })?;
    if metadata.capacity() < expected_metadata || samples.capacity() < expected_samples {
        return Err(DecoderError::InvalidParam(
            "native AV1 restoration boundary candidate is too small".to_string(),
        ));
    }
    let mut stripe = 0usize;
    loop {
        let stripe_start = stripe
            .checked_mul(stripe_height)
            .ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration boundary stripe start overflows".to_string(),
                )
            })?
            .saturating_sub(stripe_offset);
        if stripe_start >= visible_height {
            break;
        }
        let stripe_end = stripe
            .checked_add(1)
            .and_then(|value| value.checked_mul(stripe_height))
            .and_then(|value| value.checked_sub(stripe_offset))
            .ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration boundary stripe endpoint overflows".to_string(),
                )
            })?
            .min(visible_height);
        let mut copy_rows = |start: usize, end: usize| -> Result<(), DecoderError> {
            for row in start..end {
                let offset = row.checked_mul(plane.layout.width).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration boundary row offset overflows".to_string(),
                    )
                })?;
                let end = offset.checked_add(plane.layout.width).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration boundary row endpoint overflows".to_string(),
                    )
                })?;
                let source = plane.samples.get(offset..end).ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 restoration boundary row exceeds coded plane".to_string(),
                    )
                })?;
                metadata.push(StrictRestorationBoundaryMeta {
                    row,
                    offset: samples.len(),
                    len: plane.layout.width,
                });
                samples.extend_from_slice(source);
            }
            Ok(())
        };
        if stripe > 0 {
            copy_rows(stripe_start.saturating_sub(2), stripe_start)?;
        }
        if stripe_end < visible_height {
            let end = stripe_end.checked_add(2).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 restoration boundary row range overflows".to_string(),
                )
            })?;
            copy_rows(stripe_end, end.min(visible_height))?;
        }
        stripe = stripe.checked_add(1).ok_or_else(|| {
            DecoderError::InvalidParam(
                "native AV1 restoration boundary stripe count overflows".to_string(),
            )
        })?;
    }
    if metadata.len() != expected_metadata || samples.len() != expected_samples {
        return Err(DecoderError::InvalidParam(
            "native AV1 restoration boundary plan is stale".to_string(),
        ));
    }
    Ok(())
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

#[cfg(test)]
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
    patch_restoration_stripe_boundaries_source(
        source,
        width,
        height,
        visible_width,
        visible_height,
        origin_x,
        unit_width,
        stripe_y,
        stripe_height,
        frame_stripe,
        RestorationBoundaryInput::Legacy(boundaries),
        saved,
    );
}

fn patch_restoration_stripe_boundaries_source(
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
    boundaries: RestorationBoundaryInput<'_>,
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
        let Some(samples) = restoration_boundary_samples(boundaries, boundary_row)
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

fn restoration_boundary_samples<'a>(
    boundaries: RestorationBoundaryInput<'a>,
    row: usize,
) -> Option<&'a [u16]> {
    match boundaries {
        RestorationBoundaryInput::Legacy(boundaries) => boundaries
            .rows
            .iter()
            .find_map(|&(candidate, ref samples)| (candidate == row).then_some(samples.as_slice())),
        RestorationBoundaryInput::Flat { metadata, samples } => metadata
            .iter()
            .find(|entry| entry.row == row)
            .and_then(|entry| samples.get(entry.offset..entry.offset + entry.len)),
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

fn apply_loop_restoration_stage_with_strict_workspace(
    frame: &mut DecodedFrame,
    state: &PostFilterState,
    unit_size: usize,
    enabled_types: &[u8],
    owner: &mut StrictRestorationExecutionScratchOwner,
) {
    let bit_depth = frame.bit_depth;
    let (workspace, scratch_planes) = (&mut owner.workspace, &mut owner.planes);
    #[cfg(not(target_family = "wasm"))]
    if post_filter_parallel_work_is_large_enough(frame) {
        let width = frame.width;
        let height = frame.height;
        let plane_count = frame.buffers.planes.len();
        let boundary = &workspace.boundary;
        let mut planes = frame.buffers.planes.as_mut_slice();
        let mut snapshots = workspace.snapshots.snapshots.as_mut_slice();
        let mut scratch_planes = scratch_planes.as_mut_slice();
        std::thread::scope(|scope| {
            for plane_index in 0..plane_count {
                let (plane_head, plane_tail) = planes.split_at_mut(1);
                planes = plane_tail;
                let (snapshot_head, snapshot_tail) = snapshots.split_at_mut(1);
                snapshots = snapshot_tail;
                let (scratch_head, scratch_tail) = scratch_planes.split_at_mut(1);
                scratch_planes = scratch_tail;
                let plane = &mut plane_head[0];
                let source = snapshot_head[0].as_mut();
                let scratch = scratch_head[0].as_mut();
                let boundaries = boundary
                    .metadata
                    .get(plane_index)
                    .and_then(Option::as_deref)
                    .zip(
                        boundary
                            .samples
                            .get(plane_index)
                            .and_then(Option::as_deref),
                    )
                    .map(|(metadata, samples)| {
                        RestorationBoundaryInput::Flat { metadata, samples }
                    });
                let visible_width =
                    width.div_ceil(1usize << usize::from(plane.layout.subsampling_x));
                let visible_height =
                    height.div_ceil(1usize << usize::from(plane.layout.subsampling_y));
                scope.spawn(move || {
                    if let (Some(source), Some(scratch)) = (source, scratch) {
                        apply_loop_restoration_plane_strict(
                            plane,
                            plane_index,
                            source,
                            state,
                            unit_size,
                            enabled_types,
                            bit_depth,
                            visible_width,
                            visible_height,
                            boundaries,
                            scratch,
                        );
                    }
                });
            }
        });
        return;
    }
    for (plane_index, plane) in frame.buffers.planes.iter_mut().enumerate() {
        let Some(scratch) = scratch_planes[plane_index].as_mut() else {
            continue;
        };
        let Some(source) = workspace.snapshots.snapshots[plane_index].as_mut() else {
            continue;
        };
        let boundaries = workspace
            .boundary
            .metadata
            .get(plane_index)
            .and_then(Option::as_deref)
            .zip(
                workspace
                    .boundary
                    .samples
                    .get(plane_index)
                    .and_then(Option::as_deref),
            )
            .map(|(metadata, samples)| RestorationBoundaryInput::Flat { metadata, samples });
        let visible_width = frame.width.div_ceil(1usize << usize::from(plane.layout.subsampling_x));
        let visible_height = frame.height.div_ceil(1usize << usize::from(plane.layout.subsampling_y));
        apply_loop_restoration_plane_strict(
            plane,
            plane_index,
            source,
            state,
            unit_size,
            enabled_types,
            bit_depth,
            visible_width,
            visible_height,
            boundaries,
            scratch,
        );
    }
}

fn apply_loop_restoration_plane_strict(
    plane: &mut PlaneBuffer,
    plane_index: usize,
    source: &mut [u16],
    state: &PostFilterState,
    unit_size: usize,
    enabled_types: &[u8],
    bit_depth: u8,
    visible_width: usize,
    visible_height: usize,
    boundaries: Option<RestorationBoundaryInput<'_>>,
    scratch: &mut StrictRestorationScratchPlaneOwner,
) {
    const RESTORATION_UNIT_OFFSET: usize = 8;
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
        let mut stripe_y = origin_y;
        while stripe_y < end_y {
            let frame_stripe = (stripe_y + RESTORATION_UNIT_OFFSET) / 64;
            let nominal_height = 64 - usize::from(frame_stripe == 0) * RESTORATION_UNIT_OFFSET;
            let stripe_height = nominal_height.min(end_y - stripe_y);
            let patch_count = if let (Some(boundaries), Some(metadata), Some(samples)) = (
                boundaries,
                scratch.patch_metadata.as_mut(),
                scratch.patch_samples.as_mut(),
            ) {
                patch_restoration_stripe_boundaries_strict(
                    source,
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
                    metadata,
                    samples,
                )
            } else {
                0
            };
            let mut chunk_x = 0;
            while chunk_x < unit_width {
                let x = unit.x + chunk_x;
                let chunk_width = 64.min(unit_width - chunk_x);
                match unit.restoration_type {
                    1 => {
                        let Some(mut filters) = unit.wiener else { break };
                        if plane_index > 0 {
                            filters[0][0] = 0;
                            filters[1][0] = 0;
                        }
                        let wiener = scratch.wiener.as_mut().expect("strict Wiener scratch");
                        wiener_filter_unit_into_with_fixed_scratch_bit_depth_visible(
                            source,
                            &mut plane.samples,
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
                            wiener,
                        );
                    }
                    2 => {
                        let (Some(index), Some(xqd)) = (unit.sgrproj_index, unit.sgrproj) else {
                            break;
                        };
                        let (s0, rest) = scratch.sgr.split_at_mut(1);
                        let (s1, rest) = rest.split_at_mut(1);
                        let (s2, s3) = rest.split_at_mut(1);
                        let mut unused0 = [];
                        let mut unused1 = [];
                        let mut unused2 = [];
                        let mut unused3 = [];
                        let mut sgr = [
                            s0[0]
                                .as_mut()
                                .map_or(unused0.as_mut_slice(), Vec::as_mut_slice),
                            s1[0]
                                .as_mut()
                                .map_or(unused1.as_mut_slice(), Vec::as_mut_slice),
                            s2[0]
                                .as_mut()
                                .map_or(unused2.as_mut_slice(), Vec::as_mut_slice),
                            s3[0]
                                .as_mut()
                                .map_or(unused3.as_mut_slice(), Vec::as_mut_slice),
                        ];
                        sgrproj_filter_unit_into_with_fixed_scratch_bit_depth_visible(
                            source,
                            &mut plane.samples,
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
                            &mut sgr,
                        );
                    }
                    _ => break,
                }
                chunk_x += chunk_width;
            }
            if patch_count > 0 {
                restore_restoration_stripe_boundaries_strict(
                    source,
                    plane.layout.width,
                    scratch.patch_metadata.as_ref().expect("strict patch metadata"),
                    scratch.patch_samples.as_ref().expect("strict patch samples"),
                    patch_count,
                );
            }
            stripe_y += stripe_height;
        }
    }
}

fn patch_restoration_stripe_boundaries_strict(
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
    boundaries: RestorationBoundaryInput<'_>,
    metadata: &mut [StrictRestorationPatchMeta],
    samples: &mut [u16],
) -> usize {
    let start_x = origin_x.saturating_sub(3);
    let end_x = origin_x
        .saturating_add(unit_width)
        .saturating_add(3)
        .min(visible_width)
        .min(width);
    let mut count = 0;
    let mut sample_offset = 0;
    let mut patch_row = |target_row: usize, boundary_row: usize| {
        if target_row >= height || count >= metadata.len() {
            return;
        }
        let Some(boundary_samples) = restoration_boundary_samples(boundaries, boundary_row) else {
            return;
        };
        let start = target_row * width + start_x;
        let end = target_row * width + end_x;
        let len = end - start;
        let Some(saved) = samples.get_mut(sample_offset..sample_offset + len) else {
            return;
        };
        let Some(replacement) = boundary_samples.get(start_x..end_x) else {
            return;
        };
        saved.copy_from_slice(&source[start..end]);
        metadata[count] = StrictRestorationPatchMeta {
            row: target_row,
            start_x,
            offset: sample_offset,
            len,
        };
        source[start..end].copy_from_slice(replacement);
        count += 1;
        sample_offset += len;
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
    count
}

fn restore_restoration_stripe_boundaries_strict(
    source: &mut [u16],
    width: usize,
    metadata: &[StrictRestorationPatchMeta],
    samples: &[u16],
    count: usize,
) {
    for entry in metadata.iter().take(count) {
        let start = entry.row * width + entry.start_x;
        if let Some(destination) = source.get_mut(start..start + entry.len) {
            if let Some(saved) = samples.get(entry.offset..entry.offset + entry.len) {
                destination.copy_from_slice(saved);
            }
        }
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
    boundaries: Option<RestorationBoundaryInput<'_>>,
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
                patch_restoration_stripe_boundaries_source(
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
    decode_still_frame_with_filter_policy_and_state_and_references_and_cdf_and_motion(
        headers,
        info,
        validate_filters,
        reference_buffers,
        initial_cdfs,
        None,
        validate_entropy,
        collect_cdf,
    )
}

fn decode_still_frame_with_filter_policy_and_state_and_references_and_cdf_and_motion(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
    validate_filters: bool,
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    initial_cdfs: Option<&[CdfContext]>,
    temporal_motion_field: Option<Arc<MotionField>>,
    validate_entropy: bool,
    collect_cdf: bool,
) -> Result<(DecodedStillFrame, Vec<CdfContext>), DecoderError> {
    if validate_filters {
        validate_public_decode_tools(headers)?;
    }
    let mut buffers = alloc_coded_frame_buffers(&headers.decode_plan)?;
    let (prefix, post_filter_state, final_cdfs, motion_field) =
        decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion(
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
            true,
            temporal_motion_field,
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
            motion_field: motion_field.unwrap_or_else(|| {
                MotionField::empty(
                    usize::try_from(headers.frame.frame_width)
                        .unwrap_or_default()
                        .div_ceil(4),
                    usize::try_from(headers.frame.frame_height)
                        .unwrap_or_default()
                        .div_ceil(4),
                )
            }),
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
    if let Some(config) = config
        && !matches!(config.bit_depth(), 8 | 10 | 12)
    {
        return Err(DecoderError::Unsupported(format!(
            "AV1 {}-bit quantization is not supported by public decode yet",
            config.bit_depth()
        )));
    }
    // Monochrome and sub-sampled YUV layouts are decoded by the AV1
    // plane/reconstruction path.
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
    if let Some(sequence_position) = sequence.color_config.chroma_sample_position
        && config.chroma_sample_position != sequence_position
    {
        return Err(DecoderError::Bitstream(format!(
            "av1C chroma sample position {:?} does not match sequence header {:?}",
            config.chroma_sample_position, sequence_position
        )));
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
    let nclx_color_unspecified = nclx.color_primaries == 2
        && nclx.transfer_characteristics == 2
        && nclx.matrix_coefficients == 2;
    if nclx.full_range_flag != av1_full_range && !nclx_color_unspecified {
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
    animation: Option<&AvifAnimation>,
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
    if let Some(animation) = animation {
        let loop_count = match animation.repetition_count {
            crate::container::AvifRepetitionCount::Finite(value) => value,
            crate::container::AvifRepetitionCount::Infinite => 0,
            crate::container::AvifRepetitionCount::Unknown => 1,
        };
        let duration_ms = animation
            .color_timing
            .iter()
            .map(|timing| timing.duration_ms)
            .try_fold(0_u64, u64::checked_add)
            .ok_or_else(|| "AVIS millisecond duration overflows u64".to_string())?;
        option.drawer.set_metadata(
            "Animation frames",
            DataMap::UInt(animation.color_timing.len() as u64),
        )?;
        option
            .drawer
            .set_metadata("Animation loop count", DataMap::UInt(loop_count as u64))?;
        option.drawer.set_metadata(
            "Animation timescale",
            DataMap::UInt(animation.color_timescale),
        )?;
        option
            .drawer
            .set_metadata("Animation duration", DataMap::UInt(duration_ms))?;
        option.drawer.set_metadata(
            "Animation frame PTS",
            DataMap::UIntAllay(
                animation
                    .color_timing
                    .iter()
                    .map(|timing| timing.pts_in_timescales)
                    .collect(),
            ),
        )?;
        option.drawer.set_metadata(
            "Animation frame durations",
            DataMap::UIntAllay(
                animation
                    .color_timing
                    .iter()
                    .map(|timing| timing.duration_in_timescales)
                    .collect(),
            ),
        )?;
        option.drawer.set_metadata(
            "Animation frame durations ms",
            DataMap::UIntAllay(
                animation
                    .color_timing
                    .iter()
                    .map(|timing| timing.duration_ms)
                    .collect(),
            ),
        )?;
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
        let Ok(data) = std::fs::read(&avif_path) else {
            return;
        };
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
        let Ok(data) = std::fs::read(root.join("test_data/images/WML2Viewer.avif")) else {
            return;
        };
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
            if header & 0x02 != 0 {
                loop {
                    let byte = info.primary_item_payload[cursor];
                    cursor += 1;
                    payload_len |= usize::from(byte & 0x7f) << shift;
                    if byte & 0x80 == 0 {
                        break;
                    }
                    shift += 7;
                }
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

#[cfg(test)]
mod incremental_sequence_state_tests {
    use super::*;

    #[test]
    fn external_alpha_sequence_decodes_each_track_sample_once() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("test/images/external/avif/unsupported/colors-animated-8bpc-alpha-exif-xmp.avif");
        if !path.is_file() {
            eprintln!("external alpha AVIS sample is unavailable; skipping linear-state audit");
            return;
        }
        let data = std::fs::read(path).unwrap();
        let expected = decode_sequence_frames_bytes(&data).unwrap();
        let indexed = (0..expected.len())
            .map(|index| decode_sequence_frame_bytes(&data, index).unwrap())
            .collect::<Vec<_>>();
        let mut decoder = AvifSequenceDecoder::new(&data).unwrap();
        let mut incremental = Vec::new();
        while let Some(decoded) = decoder.next_frame().unwrap() {
            incremental.push(decoded.frame);
        }

        assert_eq!(incremental, expected);
        assert_eq!(indexed, expected);
        assert_eq!(
            decoder.decoded_track_sample_counts(),
            (expected.len(), Some(expected.len()))
        );
        assert!(decoder.next_frame().unwrap().is_none());
        assert_eq!(
            decoder.decoded_track_sample_counts(),
            (expected.len(), Some(expected.len()))
        );
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

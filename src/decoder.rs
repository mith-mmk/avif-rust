use crate::av1::{
    Av1CodecConfiguration, BlockModeProbe, FrameDecodePlan, FrameHeader, PartitionProbe,
    QuantState, ResidualProbe, SequenceHeader, TileEntropyState, TileGroup, alloc_frame_buffers,
    build_still_decode_plan, decode_luma_root_block_prefix, frame_buffers_to_rgba_8,
    parse_av1_config, parse_frame_header, parse_sequence_header, parse_tile_group,
    plan_transform_blocks, prepare_tile_entropy, probe_first_block_residuals,
    probe_tile_block_modes, probe_tile_partitions,
};
use crate::compat::{DataMap, DecodeOptions, InitOptions};
use crate::container::{AvifInfo, parse_avif};
use crate::obu::{ObuType, find_obu_payloads};
use crate::{DecoderError, ImageBuffer};
use bin_rs::reader::BinaryReader;
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
    let data = read_to_end(reader)?;
    let info = parse_avif(&data)?;
    let headers = parse_av1_headers(&info)?;
    emit_metadata(&info, Some(&headers), option)?;
    let image = decode_still_image(&headers)?;

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
    let headers = parse_av1_headers(&info)?;
    decode_still_image(&headers)
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

fn parse_av1_headers(info: &AvifInfo) -> Result<Av1Headers, DecoderError> {
    let [
        sequence_payload,
        frame_payload,
        frame_header_payload,
        tile_group_payload,
    ] = find_obu_payloads(
        &info.primary_item_payload,
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
    let config = info
        .av1_config
        .as_deref()
        .map(parse_av1_config)
        .transpose()?;
    if let Some(config) = config {
        validate_av1_config(&config, &sequence)?;
    }
    let (frame, tile_group_payload, tile_data) = if let Some(frame_payload) = frame_payload {
        let frame = parse_frame_header(frame_payload, &sequence)?;
        if frame.payload_after_header_offset > frame_payload.len() {
            return Err(DecoderError::Bitstream(
                "AV1 frame payload offset points outside OBU_FRAME".to_string(),
            ));
        }
        let tile_group = parse_tile_group(
            frame_payload,
            frame.uncompressed_header_bits,
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
            frame_payload,
        )
    } else {
        let frame_header_payload = frame_header_payload.ok_or_else(|| {
            DecoderError::Bitstream("AV1 frame header OBU is missing".to_string())
        })?;
        let frame = parse_frame_header(frame_header_payload, &sequence)?;
        let tile_group_payload = tile_group_payload
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile group OBU is missing".to_string()))?;
        let tile_group = parse_tile_group(tile_group_payload, 0, &frame.tile_info)?;
        let entropy_states = prepare_tile_entropy(tile_group_payload, &tile_group, &frame)?;
        (
            frame,
            ParsedTileGroup {
                from_frame_obu: false,
                payload_len: tile_group_payload.len(),
                tile_data: tile_group_payload.to_vec(),
                group: tile_group,
                entropy_states,
                partition_probes: Vec::new(),
                block_mode_probes: Vec::new(),
                residual_probes: Vec::new(),
            },
            tile_group_payload,
        )
    };
    let decode_plan = build_still_decode_plan(&sequence, &frame, &tile_group_payload.group)?;
    let quant_state =
        QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
    let mut tile_group_payload = tile_group_payload;
    tile_group_payload.partition_probes =
        probe_tile_partitions(tile_data, &tile_group_payload.group, &frame, &decode_plan)?;
    tile_group_payload.block_mode_probes = probe_tile_block_modes(
        tile_data,
        &tile_group_payload.group,
        &sequence,
        &frame,
        &decode_plan,
    )?;
    tile_group_payload.residual_probes = probe_first_block_residuals(
        tile_data,
        &tile_group_payload.group,
        &sequence,
        &frame,
        &decode_plan,
    )?;

    Ok(Av1Headers {
        config,
        sequence,
        frame,
        tile_group: tile_group_payload,
        decode_plan,
        quant_state,
    })
}

fn decode_still_image(headers: &Av1Headers) -> Result<ImageBuffer, DecoderError> {
    let mut buffers = alloc_frame_buffers(&headers.decode_plan)?;
    let prefix = decode_luma_root_block_prefix(
        &headers.tile_group.tile_data,
        &headers.tile_group.group,
        &headers.sequence,
        &headers.frame,
        &headers.decode_plan,
        &mut buffers,
        usize::MAX,
    )?;
    if let Some(err) = prefix.next_unsupported {
        return Err(err);
    }
    frame_buffers_to_rgba_8(&buffers, &headers.sequence.color_config)
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
    Ok(())
}

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
            let first_block_transforms = plan_transform_blocks(
                0,
                0,
                0,
                block_mode.block_size,
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

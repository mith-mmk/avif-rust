use crate::av1::ColorRange;
use crate::av1::PostFilterState;
use crate::av1::{
    Av1CodecConfiguration, BlockModeProbe, ColorConfig, FrameBuffers, FrameDecodePlan, FrameHeader,
    PartitionProbe, QuantState, ResidualProbe, SequenceHeader, TileEntropyState, TileGroup,
    alloc_frame_buffers, build_still_decode_plan,
    decode_luma_root_block_prefix_with_post_filter_state_and_entropy, frame_buffers_to_rgba_8,
    frame_buffers_to_rgba_16, parse_av1_config, parse_frame_header, parse_sequence_header,
    parse_tile_group, plan_transform_blocks_with_tx_size, prepare_tile_entropy,
    probe_first_block_residuals, probe_tile_block_modes, probe_tile_partitions,
};
use crate::compat::{DataMap, DecodeOptions, InitOptions};
use crate::container::{AvifInfo, ColorInformation, parse_avif};
use crate::obu::{ObuType, count_obus, find_obu_payloads};
use crate::{DecoderError, ImageBuffer, Rgba16ImageBuffer};
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
    let headers = parse_av1_headers(&info)?;
    decode_still_image(&headers, Some(&info))
}

/// Decoded still-frame planes before colour conversion.
///
/// Samples are stored as native AV1 source planes in raster order. The current
/// decoder only supports a subset of still-image tools, but this type is the
/// conformance-test boundary for exact Y/U/V/alpha plane comparisons.
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

impl DecodedFrame {
    pub fn to_rgba8(&self) -> Result<ImageBuffer, DecoderError> {
        self.validate_color_management_for_rgba()?;
        frame_buffers_to_rgba_8(&self.buffers, &self.color_config)
    }

    pub fn to_rgba16(&self) -> Result<Rgba16ImageBuffer, DecoderError> {
        self.validate_color_management_for_rgba()?;
        frame_buffers_to_rgba_16(&self.buffers, &self.color_config)
    }

    fn validate_color_management_for_rgba(&self) -> Result<(), DecoderError> {
        if self
            .color_information
            .as_ref()
            .is_some_and(|color| color.icc_profile().is_some())
        {
            return Err(DecoderError::Unsupported(
                "AVIF ICC colour management for RGBA conversion is not supported yet".to_string(),
            ));
        }
        Ok(())
    }
}

/// Decodes a still AVIF image from memory into high-precision source planes.
pub fn decode_frame_bytes(data: &[u8]) -> Result<DecodedFrame, DecoderError> {
    let info = parse_avif(data)?;
    let headers = parse_av1_headers(&info)?;
    decode_still_frame(&headers, Some(&info))
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
    let tile_group_obu_count = count_obus(&info.primary_item_payload, ObuType::TileGroup)?;
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
    validate_color_metadata(info.color_information.as_ref(), &sequence.color_config)?;
    let config = info
        .av1_config
        .as_deref()
        .map(parse_av1_config)
        .transpose()?;
    if let Some(config) = config {
        validate_av1_config(&config, &sequence)?;
    }
    if frame_payload.is_none() && tile_group_obu_count > 1 {
        return Err(DecoderError::Unsupported(
            "AVIF multiple tile-group OBUs for one frame are not supported yet".to_string(),
        ));
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
    if let Some(width) = info.width {
        if width != frame.frame_width {
            return Err(DecoderError::Bitstream(format!(
                "AVIF ispe width {width} does not match AV1 frame width {}",
                frame.frame_width
            )));
        }
    }
    if let Some(height) = info.height {
        if height != frame.frame_height {
            return Err(DecoderError::Bitstream(format!(
                "AVIF ispe height {height} does not match AV1 frame height {}",
                frame.frame_height
            )));
        }
    }
    let decode_plan = build_still_decode_plan(&sequence, &frame, &tile_group_payload.group)?;
    let quant_state =
        QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
    let mut tile_group_payload = tile_group_payload;
    tile_group_payload.partition_probes = probe_tile_partitions(
        tile_data,
        &tile_group_payload.group,
        &sequence,
        &frame,
        &decode_plan,
    )?;
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

fn decode_still_image(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
) -> Result<ImageBuffer, DecoderError> {
    let frame = decode_still_frame(headers, info)?;
    frame.to_rgba8()
}

fn decode_still_frame(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
) -> Result<DecodedFrame, DecoderError> {
    decode_still_frame_with_filter_policy(headers, info, true)
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
    Ok(apply_post_filter_stage(decoded))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecodedStillFrame {
    frame: DecodedFrame,
    post_filter_state: PostFilterState,
}

fn apply_post_filter_stage(decoded: DecodedStillFrame) -> DecodedFrame {
    let DecodedStillFrame {
        frame,
        post_filter_state: _post_filter_state,
    } = decoded;
    frame
}

fn decode_still_frame_with_filter_policy_and_state(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
    validate_filters: bool,
) -> Result<DecodedStillFrame, DecoderError> {
    if info.is_some_and(|info| {
        info.clean_aperture.is_some() || info.rotation.is_some() || info.mirror.is_some()
    }) {
        return Err(DecoderError::Unsupported(
            "AVIF clap/irot/imir composition is not supported yet".to_string(),
        ));
    }
    if info.is_some_and(|info| info.primary_grid.is_some()) {
        return Err(DecoderError::Unsupported(
            "AVIF image grid composition is not supported yet".to_string(),
        ));
    }
    if info.is_some_and(|info| !info.alpha_auxiliary_items.is_empty()) {
        return Err(DecoderError::Unsupported(
            "AVIF alpha auxiliary item composition is not supported yet".to_string(),
        ));
    }
    if validate_filters {
        validate_public_decode_tools(headers)?;
    }
    let mut buffers = alloc_frame_buffers(&headers.decode_plan)?;
    let (prefix, post_filter_state) =
        decode_luma_root_block_prefix_with_post_filter_state_and_entropy(
            &headers.tile_group.tile_data,
            &headers.tile_group.group,
            &headers.sequence,
            &headers.frame,
            &headers.decode_plan,
            &mut buffers,
            usize::MAX,
            validate_filters,
        )?;
    if let Some(err) = prefix.next_unsupported {
        return Err(err);
    }
    Ok(DecodedStillFrame {
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
    })
}

fn validate_public_decode_tools(headers: &Av1Headers) -> Result<(), DecoderError> {
    if headers.decode_plan.uses_cdef {
        return Err(DecoderError::Unsupported(
            "AV1 CDEF filtering is not supported by public decode yet".to_string(),
        ));
    }
    if headers.decode_plan.uses_restoration {
        return Err(DecoderError::Unsupported(
            "AV1 loop restoration is not supported by public decode yet".to_string(),
        ));
    }
    if headers
        .frame
        .loop_filter
        .levels
        .iter()
        .any(|level| *level != 0)
    {
        return Err(DecoderError::Unsupported(
            "AV1 deblocking filter is not supported by public decode yet".to_string(),
        ));
    }
    if headers.sequence.film_grain_params_present {
        return Err(DecoderError::Unsupported(
            "AV1 film grain is not supported by public decode yet".to_string(),
        ));
    }
    if headers.frame.quantization.using_qmatrix {
        return Err(DecoderError::Unsupported(
            "AV1 quantization matrices are not supported by public decode yet".to_string(),
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
    fn color_metadata_rejects_explicit_nclx_mismatch() {
        let error = validate_color_metadata(Some(&nclx(9, 13, 0, true)), &config()).unwrap_err();
        assert!(
            matches!(error, DecoderError::Bitstream(message) if message.contains("does not match"))
        );
    }
}

#[cfg(test)]
mod prefilter_diagnostic_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    #[ignore = "diagnostic comparison against the local WML2Viewer pre-filter oracle"]
    fn reports_wml2viewer_prefilter_mismatches() {
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
            "diagnostic frame base_q_idx={} quant={:?}",
            headers.frame.base_q_idx, headers.frame.quantization
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
        for block in prefix
            .blocks
            .iter()
            .filter(|block| block.y == 0 && (128..160).contains(&block.x))
        {
            eprintln!(
                "diagnostic block ({}, {}) size={:?} transforms={:?}",
                block.x,
                block.y,
                block.block_size,
                block
                    .transforms
                    .iter()
                    .map(|transform| {
                        (
                            transform.transform.x,
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
            eprintln!("prefilter plane {plane_index}: first={first:?}, mismatches={mismatches}");
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
                    "prefilter plane {plane_index} window {start}..{end}: actual={:?}, expected={:?}",
                    &actual[start..end],
                    &expected[start..end]
                );
            }
        }
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

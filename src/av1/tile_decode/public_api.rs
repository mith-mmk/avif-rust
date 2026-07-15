use super::decode_flow::{decode_luma_block_tree, decode_luma_root_block};
use super::post_filter_state::PostFilterState;
use super::{
    BlockModeProbe, DecodedBlockPrefix, DecodedLumaBlock, DecodedTransform, PartitionProbe,
    ResidualProbe, TileDecoder, TileEntropyState,
};
use crate::DecoderError;
use crate::av1::decode::{FrameBuffers, FrameDecodePlan};
use crate::av1::entropy::EntropyDecoder;
use crate::av1::frame::FrameHeader;
use crate::av1::predict::{IntraEdges, predict_intra};
use crate::av1::quant::QuantState;
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::Partition;
use crate::av1::tile_decode::partition_syntax::root_block_size;
use crate::av1::tile_group::{TileGroup, TilePayload};
use crate::av1::transform::{
    QuantizedTransform, plan_transform_blocks_with_tx_size, reconstruct_lossless_transform_block,
    reconstruct_transform_block,
};

pub fn prepare_tile_entropy(
    data: &[u8],
    tile_group: &TileGroup,
    frame: &FrameHeader,
) -> Result<Vec<TileEntropyState>, DecoderError> {
    if tile_group.tiles.is_empty() {
        return Err(DecoderError::Bitstream(
            "AV1 tile group has no tile payloads".to_string(),
        ));
    }

    let mut states = Vec::with_capacity(tile_group.tiles.len());
    for tile in &tile_group.tiles {
        let payload = tile_payload_bytes(data, tile)?;
        let decoder = EntropyDecoder::new(payload, frame.disable_cdf_update)?;
        states.push(TileEntropyState {
            tile_id: tile.tile_id,
            payload_offset: tile.offset,
            payload_len: tile.len,
            entropy_start_bits: decoder.bit_position(),
        });
    }
    Ok(states)
}

pub fn probe_tile_partitions(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<PartitionProbe>, DecoderError> {
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        probes.push(decoder.read_root_partition(tile_plan, sequence)?);
    }
    Ok(probes)
}

pub fn probe_tile_block_modes(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<BlockModeProbe>, DecoderError> {
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        let partition = decoder.read_first_leaf_partition(tile_plan, sequence)?;
        if partition.partition == Partition::None {
            probes.push(decoder.read_intra_frame_block_mode(
                sequence,
                frame,
                tile_plan,
                partition.block_size,
                tile_plan.pixel_x,
                tile_plan.pixel_y,
            )?);
        }
    }
    Ok(probes)
}

pub fn probe_first_block_residuals(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<ResidualProbe>, DecoderError> {
    let quant_state =
        QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        let partition = decoder.read_first_leaf_partition(tile_plan, sequence)?;
        if partition.partition == Partition::None {
            let block_mode = decoder.read_intra_frame_block_mode(
                sequence,
                frame,
                tile_plan,
                partition.block_size,
                tile_plan.pixel_x,
                tile_plan.pixel_y,
            )?;
            let transforms = plan_transform_blocks_with_tx_size(
                0,
                0,
                0,
                block_mode.block_size,
                block_mode.tx_size,
                plan.width,
                plan.height,
            );
            probes.push(decoder.read_first_transform_residual(
                tile_plan.tile_id,
                frame,
                &block_mode,
                &transforms,
                quant_state,
                sequence.color_config.bit_depth,
            )?);
        }
    }
    Ok(probes)
}

pub fn decode_first_luma_transform(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
) -> Result<ResidualProbe, DecoderError> {
    let quant_state =
        QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
    let tile_payload = tile_group.tiles.first().ok_or_else(|| {
        DecoderError::Bitstream("AV1 tile group has no tile payloads".to_string())
    })?;
    let payload = tile_payload_bytes(data, tile_payload)?;
    let tile_plan = plan
        .tiles
        .first()
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile decode plan is missing".to_string()))?;
    let mut decoder = TileDecoder::new(payload, frame)?;
    let partition = decoder.read_first_leaf_partition(tile_plan, sequence)?;

    let block_mode = decoder.read_intra_frame_block_mode(
        sequence,
        frame,
        tile_plan,
        partition.block_size,
        tile_plan.pixel_x,
        tile_plan.pixel_y,
    )?;
    let transforms = plan_transform_blocks_with_tx_size(
        0,
        tile_plan.pixel_x,
        tile_plan.pixel_y,
        block_mode.block_size,
        block_mode.tx_size,
        plan.width,
        plan.height,
    );
    let residual = decoder.read_first_transform_residual(
        tile_plan.tile_id,
        frame,
        &block_mode,
        &transforms,
        quant_state,
        sequence.color_config.bit_depth,
    )?;

    if residual.skipped || residual.first_non_zero_transform.is_none() {
        return Ok(residual);
    }
    let transform = residual
        .first_non_zero_transform
        .expect("checked first_non_zero_transform");
    let tx_type = residual
        .tx_type
        .ok_or_else(|| DecoderError::Bitstream("AV1 residual tx_type is missing".to_string()))?;
    let coefficients = residual
        .first_quantized_coefficients
        .as_ref()
        .ok_or_else(|| {
            DecoderError::Bitstream("AV1 residual quantized coefficients are missing".to_string())
        })?;
    let mid = 1u16 << (sequence.color_config.bit_depth - 1);
    let above = vec![mid; transform.tx_size.width()];
    let left = vec![mid; transform.tx_size.height()];
    let prediction = predict_intra(
        block_mode.y_mode,
        block_mode.angle_delta_y,
        transform.tx_size.width(),
        transform.tx_size.height(),
        IntraEdges {
            above: Some(&above),
            left: Some(&left),
            above_left: Some(mid),
            bit_depth: sequence.color_config.bit_depth,
        },
    )?;
    let quantized = QuantizedTransform {
        block: transform,
        tx_type,
        coefficients: coefficients.clone(),
    };
    let luma = buffers
        .planes
        .get_mut(0)
        .ok_or_else(|| DecoderError::Bitstream("AV1 luma plane is missing".to_string()))?;
    if frame.quantization.coded_lossless() {
        reconstruct_lossless_transform_block(
            luma,
            &quantized,
            quant_state.plane(transform.plane),
            &prediction,
            sequence.color_config.bit_depth,
        )?;
    } else {
        reconstruct_transform_block(
            luma,
            &quantized,
            quant_state.plane(transform.plane),
            &prediction,
            sequence.color_config.bit_depth,
        )?;
    }

    Ok(residual)
}

pub fn decode_first_luma_block(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
) -> Result<Vec<DecodedTransform>, DecoderError> {
    let tile_payload = tile_group.tiles.first().ok_or_else(|| {
        DecoderError::Bitstream("AV1 tile group has no tile payloads".to_string())
    })?;
    let payload = tile_payload_bytes(data, tile_payload)?;
    let tile_plan = plan
        .tiles
        .first()
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile decode plan is missing".to_string()))?;
    let mut decoder = TileDecoder::new(payload, frame)?;
    let block = decode_luma_root_block(
        &mut decoder,
        sequence,
        frame,
        tile_plan,
        plan,
        buffers,
        tile_plan.pixel_x,
        tile_plan.pixel_y,
    )?;
    Ok(block.transforms)
}

pub fn decode_luma_root_blocks(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
) -> Result<Vec<DecodedLumaBlock>, DecoderError> {
    Ok(
        decode_luma_root_block_prefix(
            data, tile_group, sequence, frame, plan, buffers, max_blocks,
        )?
        .blocks,
    )
}

pub fn decode_luma_root_block_prefix(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
) -> Result<DecodedBlockPrefix, DecoderError> {
    // Keep the diagnostic prefix path's entropy traversal independent from
    // post-filter state aggregation. The public/test prefix oracle relies on
    // returning at the same block boundary as the historical decoder.
    if tile_group.tiles.is_empty() {
        return Err(DecoderError::Bitstream(
            "AV1 tile group has no tile payloads".to_string(),
        ));
    }
    let mut blocks = Vec::new();
    let mut block_budget = max_blocks;
    for (tile_index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(tile_index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        for sb_row in tile_plan.sb_row_start..tile_plan.sb_row_end {
            decoder.reset_left_superblock_contexts();
            for sb_col in tile_plan.sb_col_start..tile_plan.sb_col_end {
                if block_budget == 0 {
                    return Ok(DecodedBlockPrefix {
                        blocks,
                        next_unsupported: None,
                    });
                }
                let x = (sb_col as usize * plan.superblock_size).min(plan.width);
                let y = (sb_row as usize * plan.superblock_size).min(plan.height);
                decoder.read_restoration_units(sequence, x, y)?;
                let decoded = match decode_luma_block_tree(
                    &mut decoder,
                    sequence,
                    frame,
                    tile_plan,
                    plan,
                    buffers,
                    root_block_size(sequence),
                    x,
                    y,
                    &mut block_budget,
                ) {
                    Ok(blocks) => blocks,
                    Err(err @ DecoderError::Unsupported(_)) if !blocks.is_empty() => {
                        return Ok(DecodedBlockPrefix {
                            blocks,
                            next_unsupported: Some(err),
                        });
                    }
                    Err(err) => return Err(err),
                };
                blocks.extend(decoded);
            }
        }
    }
    Ok(DecodedBlockPrefix {
        blocks,
        next_unsupported: None,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal prefix decode exposes each independently testable pipeline input"
)]
pub(crate) fn decode_luma_root_block_prefix_with_post_filter_state_and_entropy(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
    validate_entropy: bool,
) -> Result<(DecodedBlockPrefix, PostFilterState), DecoderError> {
    if tile_group.tiles.is_empty() {
        return Err(DecoderError::Bitstream(
            "AV1 tile group has no tile payloads".to_string(),
        ));
    }
    let mut blocks = Vec::new();
    let mut block_budget = max_blocks;
    let mut post_filter_state = PostFilterState::default();

    for (tile_index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(tile_index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        for sb_row in tile_plan.sb_row_start..tile_plan.sb_row_end {
            decoder.reset_left_superblock_contexts();
            for sb_col in tile_plan.sb_col_start..tile_plan.sb_col_end {
                if block_budget == 0 {
                    post_filter_state.merge(decoder.take_post_filter_state());
                    return Ok((
                        DecodedBlockPrefix {
                            blocks,
                            next_unsupported: None,
                        },
                        post_filter_state,
                    ));
                }
                let x = (sb_col as usize * plan.superblock_size).min(plan.width);
                let y = (sb_row as usize * plan.superblock_size).min(plan.height);
                decoder.read_restoration_units(sequence, x, y)?;
                let decoded = match decode_luma_block_tree(
                    &mut decoder,
                    sequence,
                    frame,
                    tile_plan,
                    plan,
                    buffers,
                    root_block_size(sequence),
                    x,
                    y,
                    &mut block_budget,
                ) {
                    Ok(blocks) => blocks,
                    Err(err @ DecoderError::Unsupported(_)) if !blocks.is_empty() => {
                        post_filter_state.merge(decoder.take_post_filter_state());
                        return Ok((
                            DecodedBlockPrefix {
                                blocks,
                                next_unsupported: Some(err),
                            },
                            post_filter_state,
                        ));
                    }
                    Err(err) => return Err(err),
                };
                post_filter_state.record_luma_blocks(&decoded);
                blocks.extend(decoded);
            }
        }
        if validate_entropy {
            decoder.finish_entropy()?;
        }
        post_filter_state.merge(decoder.take_post_filter_state());
    }

    Ok((
        DecodedBlockPrefix {
            blocks,
            next_unsupported: None,
        },
        post_filter_state,
    ))
}

fn tile_payload_bytes<'a>(
    data: &'a [u8],
    tile_payload: &TilePayload,
) -> Result<&'a [u8], DecoderError> {
    let end = tile_payload
        .offset
        .checked_add(tile_payload.len)
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
    data.get(tile_payload.offset..end).ok_or_else(|| {
        DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::{parse_frame_header, parse_sequence_header, parse_tile_group};
    use crate::container::parse_avif;
    use crate::obu::{ObuType, find_obu_payload};

    #[test]
    fn tile_payload_bytes_checks_bounds_for_each_tile() {
        let data = b"0123456789";
        let first = TilePayload {
            tile_id: 0,
            offset: 2,
            len: 3,
        };
        assert_eq!(tile_payload_bytes(data, &first).unwrap(), b"234");

        let truncated = TilePayload {
            tile_id: 1,
            offset: 8,
            len: 4,
        };
        assert!(matches!(
            tile_payload_bytes(data, &truncated),
            Err(DecoderError::NotEnoughData(_))
        ));
    }

    #[test]
    fn prepares_sample_tile_entropy_state() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("samples")
                .join("WML2Viewer.avif"),
        )
        .expect("sample AVIF should exist");
        let info = parse_avif(&data).unwrap();
        let sequence_payload =
            find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
                .unwrap()
                .expect("sequence header OBU should exist");
        let sequence = parse_sequence_header(sequence_payload).unwrap();
        let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
            .unwrap()
            .expect("frame OBU should exist");
        let frame = parse_frame_header(frame_payload, &sequence).unwrap();
        let tile_group = parse_tile_group(
            frame_payload,
            frame.uncompressed_header_bits,
            &frame.tile_info,
        )
        .unwrap();

        let states = prepare_tile_entropy(frame_payload, &tile_group, &frame).unwrap();

        assert_eq!(states.len(), 1);
        assert_eq!(states[0].tile_id, 0);
        assert_eq!(states[0].entropy_start_bits, 15);
        assert!(states[0].payload_len > 0);
    }
}

use super::{DecodedLumaBlock, TileDecoder};
use crate::DecoderError;
use crate::av1::decode::{FrameBuffers, FrameDecodePlan, TileDecodePlan};
use crate::av1::frame::FrameHeader;
use crate::av1::quant::QuantState;
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::{BlockSize, Partition, PredictionMode, UvPredictionMode};
use crate::av1::tile_decode::reconstruction::decode_plane_block_unit;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResidualUnit {
    plane_index: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

pub(super) fn decode_luma_root_block(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    x: usize,
    y: usize,
) -> Result<DecodedLumaBlock, DecoderError> {
    let partition = decoder.read_first_leaf_partition(tile_plan, sequence)?;

    decode_luma_leaf_block(
        decoder,
        sequence,
        frame,
        tile_plan,
        plan,
        buffers,
        partition.block_size,
        x,
        y,
    )
}

pub(super) fn decode_luma_leaf_block(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    block_size: BlockSize,
    x: usize,
    y: usize,
) -> Result<DecodedLumaBlock, DecoderError> {
    let block_mode =
        decoder.read_intra_frame_block_mode(sequence, frame, tile_plan, block_size, x, y)?;
    decoder.record_block_filter_state(
        x,
        y,
        block_mode.block_size,
        block_mode.skip,
        block_mode.y_mode,
        block_mode.uv_mode,
    );
    decoder.record_cdef_index(frame, x, y, block_mode.cdef_idx);
    if std::env::var_os("AVIF_TRACE_WML2_MODES").is_some() && (64..96).contains(&x) && y < 32 {
        eprintln!(
            "Rust mode x={x} size={:?} skip={} y={:?} uv={:?} tx={:?} state={:?}",
            block_size,
            block_mode.skip,
            block_mode.y_mode,
            block_mode.uv_mode,
            block_mode.tx_size,
            decoder.reader.trace_state()
        );
    }
    let quant_state =
        QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
    let chroma = if !sequence.color_config.monochrome {
        let uv_mode = block_mode.uv_mode.ok_or_else(|| {
            DecoderError::Bitstream("AV1 chroma block mode is missing".to_string())
        })?;
        let (chroma_mode, cfl) = match uv_mode {
            UvPredictionMode::Intra(mode) => (mode, None),
            UvPredictionMode::Cfl => (PredictionMode::Dc, decoder.current_cfl),
        };
        Some((chroma_mode, cfl))
    } else {
        None
    };

    let plane_count = if chroma.is_some() { 3 } else { 1 };
    let mut decoded = Vec::new();
    for unit in residual_unit_order(block_mode.block_size, x, y, plane_count) {
        let (prediction_mode, angle_delta, filter_intra_mode, smooth_neighbour, cfl_alpha_q3) =
            match unit.plane_index {
                0 => (
                    block_mode.y_mode,
                    block_mode.angle_delta_y,
                    block_mode.filter_intra_mode,
                    block_mode.y_smooth_neighbour,
                    None,
                ),
                1 => {
                    let (mode, cfl) = chroma.expect("chroma planes require chroma mode state");
                    (
                        mode,
                        block_mode.angle_delta_uv,
                        None,
                        block_mode.uv_smooth_neighbour,
                        cfl.map(|params| params.alpha_u_q3),
                    )
                }
                2 => {
                    let (mode, cfl) = chroma.expect("chroma planes require chroma mode state");
                    (
                        mode,
                        block_mode.angle_delta_uv,
                        None,
                        block_mode.uv_smooth_neighbour,
                        cfl.map(|params| params.alpha_v_q3),
                    )
                }
                _ => unreachable!("AV1 has at most three image planes"),
            };
        let unit_decoded = decode_plane_block_unit(
            decoder,
            sequence,
            frame,
            plan,
            buffers,
            &block_mode,
            unit.plane_index,
            prediction_mode,
            angle_delta,
            filter_intra_mode,
            smooth_neighbour,
            cfl_alpha_q3,
            x,
            y,
            unit.x,
            unit.y,
            unit.width,
            unit.height,
            quant_state,
        )?;
        if unit.plane_index == 0 {
            decoded.extend(unit_decoded);
        }
    }

    Ok(DecodedLumaBlock {
        x,
        y,
        block_size: block_mode.block_size,
        palette: block_mode.palette,
        transforms: decoded,
    })
}

fn residual_unit_order(
    block_size: BlockSize,
    x: usize,
    y: usize,
    plane_count: usize,
) -> Vec<ResidualUnit> {
    const MAX_UNIT_SIZE: usize = 64;
    let mut units = Vec::new();
    for unit_y in (y..y + block_size.height()).step_by(MAX_UNIT_SIZE) {
        for unit_x in (x..x + block_size.width()).step_by(MAX_UNIT_SIZE) {
            let width = MAX_UNIT_SIZE.min(x + block_size.width() - unit_x);
            let height = MAX_UNIT_SIZE.min(y + block_size.height() - unit_y);
            for plane_index in 0..plane_count {
                units.push(ResidualUnit {
                    plane_index,
                    x: unit_x,
                    y: unit_y,
                    width,
                    height,
                });
            }
        }
    }
    units
}

#[cfg(test)]
mod tests {
    use super::{ResidualUnit, residual_unit_order};
    use crate::av1::syntax::BlockSize;

    #[test]
    fn residuals_interleave_planes_within_64x64_units() {
        let units = residual_unit_order(BlockSize::Block128x128, 768, 0, 3);
        assert_eq!(units.len(), 12);
        assert_eq!(
            &units[..6],
            &[
                ResidualUnit {
                    plane_index: 0,
                    x: 768,
                    y: 0,
                    width: 64,
                    height: 64
                },
                ResidualUnit {
                    plane_index: 1,
                    x: 768,
                    y: 0,
                    width: 64,
                    height: 64
                },
                ResidualUnit {
                    plane_index: 2,
                    x: 768,
                    y: 0,
                    width: 64,
                    height: 64
                },
                ResidualUnit {
                    plane_index: 0,
                    x: 832,
                    y: 0,
                    width: 64,
                    height: 64
                },
                ResidualUnit {
                    plane_index: 1,
                    x: 832,
                    y: 0,
                    width: 64,
                    height: 64
                },
                ResidualUnit {
                    plane_index: 2,
                    x: 832,
                    y: 0,
                    width: 64,
                    height: 64
                },
            ]
        );
    }
}

pub(super) fn decode_luma_block_tree(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    block_size: BlockSize,
    x: usize,
    y: usize,
    block_budget: &mut usize,
) -> Result<Vec<DecodedLumaBlock>, DecoderError> {
    if *block_budget == 0 || x >= plan.width || y >= plan.height {
        return Ok(Vec::new());
    }
    // AV1 does not signal a partition for blocks smaller than 8x8; they are
    // implicit leaves even when one dimension is 4 pixels.
    if block_size.width() < 8 || block_size.height() < 8 {
        let block = decode_luma_leaf_block(
            decoder, sequence, frame, tile_plan, plan, buffers, block_size, x, y,
        )?;
        *block_budget -= 1;
        return Ok(vec![block]);
    }
    let partition = decoder
        .read_partition(tile_plan, block_size, x, y)?
        .partition;
    let decoded = match partition {
        Partition::None => {
            let block = decode_luma_leaf_block(
                decoder, sequence, frame, tile_plan, plan, buffers, block_size, x, y,
            )?;
            *block_budget -= 1;
            Ok(vec![block])
        }
        Partition::Horizontal => {
            let subsize = block_size.horizontal_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[(subsize, x, y), (subsize, x, y + subsize.height())],
                block_budget,
            )
        }
        Partition::Vertical => {
            let subsize = block_size.vertical_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[(subsize, x, y), (subsize, x + subsize.width(), y)],
                block_budget,
            )
        }
        Partition::Split => {
            let subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 split partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_children(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                subsize,
                &[
                    (x, y),
                    (x + subsize.width(), y),
                    (x, y + subsize.height()),
                    (x + subsize.width(), y + subsize.height()),
                ],
                block_budget,
            )
        }
        Partition::HorizontalA => {
            let split_subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal-a partition for {block_size:?} is not supported yet"
                ))
            })?;
            let horizontal_subsize = block_size.horizontal_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal-a partition tail for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (split_subsize, x, y),
                    (split_subsize, x + split_subsize.width(), y),
                    (horizontal_subsize, x, y + split_subsize.height()),
                ],
                block_budget,
            )
        }
        Partition::HorizontalB => {
            let horizontal_subsize = block_size.horizontal_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal-b partition head for {block_size:?} is not supported yet"
                ))
            })?;
            let split_subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal-b partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (horizontal_subsize, x, y),
                    (split_subsize, x, y + horizontal_subsize.height()),
                    (
                        split_subsize,
                        x + split_subsize.width(),
                        y + horizontal_subsize.height(),
                    ),
                ],
                block_budget,
            )
        }
        Partition::VerticalA => {
            let split_subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical-a partition head for {block_size:?} is not supported yet"
                ))
            })?;
            let vertical_subsize = block_size.vertical_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical-a partition tail for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (split_subsize, x, y),
                    (split_subsize, x, y + split_subsize.height()),
                    (vertical_subsize, x + split_subsize.width(), y),
                ],
                block_budget,
            )
        }
        Partition::VerticalB => {
            let vertical_subsize = block_size.vertical_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical-b partition head for {block_size:?} is not supported yet"
                ))
            })?;
            let split_subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical-b partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (vertical_subsize, x, y),
                    (split_subsize, x + vertical_subsize.width(), y),
                    (
                        split_subsize,
                        x + vertical_subsize.width(),
                        y + split_subsize.height(),
                    ),
                ],
                block_budget,
            )
        }
        Partition::Horizontal4 => {
            let subsize = block_size.horizontal_4_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal4 partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (subsize, x, y),
                    (subsize, x, y + subsize.height()),
                    (subsize, x, y + subsize.height() * 2),
                    (subsize, x, y + subsize.height() * 3),
                ],
                block_budget,
            )
        }
        Partition::Vertical4 => {
            let subsize = block_size.vertical_4_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical4 partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (subsize, x, y),
                    (subsize, x + subsize.width(), y),
                    (subsize, x + subsize.width() * 2, y),
                    (subsize, x + subsize.width() * 3, y),
                ],
                block_budget,
            )
        }
    }?;
    decoder.update_ext_partition_context(x, y, block_size, partition)?;
    Ok(decoded)
}

fn decode_luma_partition_children(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    subsize: BlockSize,
    children: &[(usize, usize)],
    block_budget: &mut usize,
) -> Result<Vec<DecodedLumaBlock>, DecoderError> {
    let mut blocks = Vec::new();
    for &(sub_x, sub_y) in children {
        if *block_budget == 0 {
            return Ok(blocks);
        }
        let decoded = decode_luma_block_tree(
            decoder,
            sequence,
            frame,
            tile_plan,
            plan,
            buffers,
            subsize,
            sub_x,
            sub_y,
            block_budget,
        )?;
        blocks.extend(decoded);
    }
    Ok(blocks)
}

fn decode_luma_partition_runs(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    children: &[(BlockSize, usize, usize)],
    block_budget: &mut usize,
) -> Result<Vec<DecodedLumaBlock>, DecoderError> {
    let mut blocks = Vec::new();
    for &(subsize, sub_x, sub_y) in children {
        if *block_budget == 0 {
            return Ok(blocks);
        }
        if sub_x >= plan.width || sub_y >= plan.height {
            continue;
        }
        let decoded = decode_luma_leaf_block(
            decoder, sequence, frame, tile_plan, plan, buffers, subsize, sub_x, sub_y,
        )?;
        *block_budget -= 1;
        blocks.push(decoded);
    }
    Ok(blocks)
}

use super::{DecodedLumaBlock, TileDecoder, is_chroma_reference};
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

#[expect(
    clippy::too_many_arguments,
    reason = "explicit decoder and frame state avoids aliasing mutable reconstruction buffers"
)]
pub(super) fn decode_luma_root_block(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    x: usize,
    y: usize,
    collect_diagnostics: bool,
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
        collect_diagnostics,
    )?
    .ok_or_else(|| DecoderError::Bitstream("diagnostic luma block was not collected".to_string()))
}

#[expect(
    clippy::too_many_arguments,
    reason = "explicit decoder and frame state avoids aliasing mutable reconstruction buffers"
)]
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
    collect_diagnostics: bool,
) -> Result<Option<DecodedLumaBlock>, DecoderError> {
    let chroma_reference = is_chroma_reference(sequence, block_size, x, y);
    let block_mode = decoder.read_intra_frame_block_mode_with_chroma_reference(
        sequence,
        frame,
        tile_plan,
        block_size,
        x,
        y,
        chroma_reference,
    )?;
    decoder.record_block_filter_state(
        x,
        y,
        block_mode.block_size,
        block_mode.segment_id,
        block_mode.skip,
        block_mode.is_inter,
        block_mode.reference_frame,
        block_mode
            .motion_vector
            .is_some_and(|(x, y)| x != 0 || y != 0),
        block_mode.y_mode,
        block_mode.uv_mode,
        block_mode.delta_lf,
    );
    decoder.record_cdef_index(frame, x, y, block_mode.cdef_idx);
    if let Some(mv) = block_mode.intra_block_copy_mv {
        decoder.set_intra_bc_mv(x, y, block_mode.block_size, mv);
    }
    let quant_state = QuantState::from_qindex(
        &frame.quantization,
        block_mode.qindex,
        sequence.color_config.bit_depth,
    )?;
    let chroma = if chroma_reference {
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
    let transform_columns = block_mode
        .block_size
        .width()
        .div_ceil(block_mode.tx_size.width());
    let transform_rows = block_mode
        .block_size
        .height()
        .div_ceil(block_mode.tx_size.height());
    let mut decoded =
        collect_diagnostics.then(|| Vec::with_capacity(transform_columns * transform_rows));
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
        decode_plane_block_unit(
            decoder,
            sequence,
            frame,
            tile_plan,
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
            decoded.as_mut(),
        )?;
    }

    Ok(decoded.map(|transforms| DecodedLumaBlock {
        x,
        y,
        block_size: block_mode.block_size,
        palette: block_mode.palette,
        transforms,
    }))
}

fn residual_unit_order(
    block_size: BlockSize,
    x: usize,
    y: usize,
    plane_count: usize,
) -> ResidualUnitIter {
    ResidualUnitIter {
        x,
        y,
        width: block_size.width(),
        height: block_size.height(),
        plane_count,
        local_x: 0,
        local_y: 0,
        plane_index: 0,
    }
}

struct ResidualUnitIter {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    plane_count: usize,
    local_x: usize,
    local_y: usize,
    plane_index: usize,
}

impl Iterator for ResidualUnitIter {
    type Item = ResidualUnit;

    fn next(&mut self) -> Option<Self::Item> {
        const MAX_UNIT_SIZE: usize = 64;
        if self.local_y >= self.height || self.plane_count == 0 {
            return None;
        }
        let unit = ResidualUnit {
            plane_index: self.plane_index,
            x: self.x + self.local_x,
            y: self.y + self.local_y,
            width: MAX_UNIT_SIZE.min(self.width - self.local_x),
            height: MAX_UNIT_SIZE.min(self.height - self.local_y),
        };
        self.plane_index += 1;
        if self.plane_index == self.plane_count {
            self.plane_index = 0;
            self.local_x += MAX_UNIT_SIZE;
            if self.local_x >= self.width {
                self.local_x = 0;
                self.local_y += MAX_UNIT_SIZE;
            }
        }
        Some(unit)
    }
}

#[cfg(test)]
#[expect(
    clippy::items_after_test_module,
    reason = "the focused residual-order test stays next to its helper"
)]
mod tests {
    use super::{ResidualUnit, residual_unit_order};
    use crate::av1::syntax::BlockSize;

    #[test]
    fn residuals_interleave_planes_within_64x64_units() {
        let units: Vec<_> = residual_unit_order(BlockSize::Block128x128, 768, 0, 3).collect();
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

#[expect(
    clippy::too_many_arguments,
    reason = "recursive AV1 partition traversal keeps shared mutable state explicit"
)]
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
    blocks: &mut Vec<DecodedLumaBlock>,
    decoded_block_count: &mut usize,
    collect_diagnostics: bool,
) -> Result<(), DecoderError> {
    let coded_width = decoder.mi_cols << 2;
    let coded_height = decoder.mi_rows << 2;
    if *block_budget == 0 || x >= coded_width || y >= coded_height {
        return Ok(());
    }
    // AV1 does not signal a partition for blocks smaller than 8x8; they are
    // implicit leaves even when one dimension is 4 pixels.
    if block_size.width() < 8 || block_size.height() < 8 {
        let block = decode_luma_leaf_block(
            decoder,
            sequence,
            frame,
            tile_plan,
            plan,
            buffers,
            block_size,
            x,
            y,
            collect_diagnostics,
        )?;
        *block_budget -= 1;
        *decoded_block_count += 1;
        if let Some(block) = block {
            blocks.push(block);
        }
        return Ok(());
    }
    let partition = decoder
        .read_partition(tile_plan, block_size, x, y)?
        .partition;
    match partition {
        Partition::None => {
            let block = decode_luma_leaf_block(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                block_size,
                x,
                y,
                collect_diagnostics,
            )?;
            *block_budget -= 1;
            *decoded_block_count += 1;
            if let Some(block) = block {
                blocks.push(block);
            }
            Ok(())
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
                blocks,
                decoded_block_count,
                collect_diagnostics,
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
                blocks,
                decoded_block_count,
                collect_diagnostics,
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
                blocks,
                decoded_block_count,
                collect_diagnostics,
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
                blocks,
                decoded_block_count,
                collect_diagnostics,
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
                blocks,
                decoded_block_count,
                collect_diagnostics,
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
                blocks,
                decoded_block_count,
                collect_diagnostics,
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
                blocks,
                decoded_block_count,
                collect_diagnostics,
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
                blocks,
                decoded_block_count,
                collect_diagnostics,
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
                blocks,
                decoded_block_count,
                collect_diagnostics,
            )
        }
    }?;
    decoder.update_ext_partition_context(x, y, block_size, partition)?;
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "recursive AV1 partition traversal keeps shared mutable state explicit"
)]
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
    blocks: &mut Vec<DecodedLumaBlock>,
    decoded_block_count: &mut usize,
    collect_diagnostics: bool,
) -> Result<(), DecoderError> {
    for &(sub_x, sub_y) in children {
        if *block_budget == 0 {
            return Ok(());
        }
        decode_luma_block_tree(
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
            blocks,
            decoded_block_count,
            collect_diagnostics,
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "recursive AV1 partition traversal keeps shared mutable state explicit"
)]
fn decode_luma_partition_runs(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    children: &[(BlockSize, usize, usize)],
    block_budget: &mut usize,
    blocks: &mut Vec<DecodedLumaBlock>,
    decoded_block_count: &mut usize,
    collect_diagnostics: bool,
) -> Result<(), DecoderError> {
    for &(subsize, sub_x, sub_y) in children {
        if *block_budget == 0 {
            return Ok(());
        }
        if sub_x >= decoder.mi_cols << 2 || sub_y >= decoder.mi_rows << 2 {
            continue;
        }
        let decoded = decode_luma_leaf_block(
            decoder,
            sequence,
            frame,
            tile_plan,
            plan,
            buffers,
            subsize,
            sub_x,
            sub_y,
            collect_diagnostics,
        )?;
        *block_budget -= 1;
        *decoded_block_count += 1;
        if let Some(decoded) = decoded {
            blocks.push(decoded);
        }
    }
    Ok(())
}

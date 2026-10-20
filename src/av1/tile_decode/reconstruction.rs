#[path = "compound_mask.rs"]
mod compound_mask;
use super::diagnostic::ObmcNeighbors;
#[path = "interintra.rs"]
mod interintra;
#[path = "obmc_reference.rs"]
mod obmc_reference;
#[path = "sub8_chroma.rs"]
mod sub8_chroma;
use super::tx_type_syntax::inter_chroma_tx_type;
use super::{
    BlockModeProbe, CompoundMask, DecodedTransform, InterIntraMode, LocalWarpSample, MotionMode,
    PalettePlaneInfo, TileDecoder,
};
use crate::DecoderError;
use crate::av1::decode::{FrameBuffers, FrameDecodePlan, PlaneBuffer};
use crate::av1::frame::{FrameHeader, GlobalMotionType, InterpolationFilter};
use crate::av1::predict::{
    IntraEdges, predict_filter_intra_into, predict_intra_with_edge_filter_into,
};
use crate::av1::quant::QuantState;
use crate::av1::reconstruct::{read_intra_edges_into_with_bounds, write_plane_block};
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::{PredictionMode, TxSize, TxType};
use crate::av1::tile_decode::palette::PALETTE_MAX_SIZE;
use crate::av1::tile_decode::warped_filter::AV1_WARPED_FILTERS;
use crate::av1::transform::{
    TransformBlock, iter_transform_blocks_with_tx_size,
    reconstruct_lossless_transform_block_parts_into, reconstruct_transform_block_parts_into,
};

#[derive(Clone, Copy)]
struct MaskGeometry {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[expect(
    clippy::too_many_arguments,
    reason = "AV1 plane reconstruction keeps syntax, geometry, and mutable buffers explicit"
)]
pub(super) fn decode_plane_block_unit(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &crate::av1::TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    block_mode: &BlockModeProbe,
    plane_index: usize,
    prediction_mode: PredictionMode,
    angle_delta: Option<i8>,
    filter_intra_mode: Option<usize>,
    smooth_neighbour: bool,
    cfl_alpha_q3: Option<i8>,
    x: usize,
    y: usize,
    unit_x: usize,
    unit_y: usize,
    unit_width: usize,
    unit_height: usize,
    quant_state: QuantState,
    mut decoded_luma: Option<&mut Vec<DecodedTransform>>,
) -> Result<(), DecoderError> {
    let layout = plan.planes.get(plane_index).ok_or_else(|| {
        DecoderError::Bitstream(format!("AV1 plane {plane_index} decode plan is missing"))
    })?;
    let subsampling_x = usize::from(layout.subsampling_x);
    let subsampling_y = usize::from(layout.subsampling_y);
    let coded_width = decoder.mi_cols << 2;
    let coded_height = decoder.mi_rows << 2;
    let plane_width = ceil_shift(coded_width, subsampling_x);
    let plane_height = ceil_shift(coded_height, subsampling_y);
    // The luma plane layout is naturally full-resolution even when the
    // sequence is subsampled.  Use the sequence configuration for luma so
    // that YUV intra references cannot cross a tile boundary.  Identity RGB
    // content is kept full-frame because its planes are independent GBR
    // channels rather than YUV luma/chroma.
    let identity_rgb = !sequence.color_config.monochrome
        && sequence
            .color_config
            .color_description
            .is_some_and(|description| matches!(description.matrix_coefficients, 0 | 3));
    let use_tile_bounds = !identity_rgb && plan.tiles.len() > 1;
    let (tile_left, tile_top, tile_right, tile_bottom) = if !use_tile_bounds {
        (0, 0, plane_width, plane_height)
    } else {
        (
            tile_plan.pixel_x >> subsampling_x,
            tile_plan.pixel_y >> subsampling_y,
            ceil_shift(
                tile_plan.pixel_x.saturating_add(tile_plan.pixel_width),
                subsampling_x,
            ),
            ceil_shift(
                tile_plan.pixel_y.saturating_add(tile_plan.pixel_height),
                subsampling_y,
            ),
        )
    };
    let block_x = plane_block_origin(x, block_mode.block_size.width(), subsampling_x);
    let block_y = plane_block_origin(y, block_mode.block_size.height(), subsampling_y);
    let plane_block_size = if plane_index == 0 {
        block_mode.block_size
    } else {
        plane_block_size(block_mode.block_size, subsampling_x, subsampling_y)
    };
    let has_nonzero_mv = block_mode
        .motion_vector
        .is_some_and(|(x, y)| x != 0 || y != 0);
    let (luma_plane, plane) = if plane_index == 0 {
        let plane = buffers.planes.get_mut(0).ok_or_else(|| {
            DecoderError::Bitstream("AV1 luma plane buffer is missing".to_string())
        })?;
        (None, plane)
    } else {
        let (preceding, target) = buffers.planes.split_at_mut(plane_index);
        let luma = preceding.first().ok_or_else(|| {
            DecoderError::Bitstream("AV1 luma plane buffer is missing".to_string())
        })?;
        let plane = target.first_mut().ok_or_else(|| {
            DecoderError::Bitstream(format!("AV1 plane {plane_index} buffer is missing"))
        })?;
        (Some(luma), plane)
    };
    let reference_buffer = block_mode
        .reference_frame
        .map(|slot| decoder.reference_buffer(slot))
        .transpose()?;
    let reference_plane = reference_buffer
        .as_ref()
        .and_then(|buffers| buffers.planes.get(plane_index));
    let secondary_reference_buffer = block_mode
        .reference_frame_secondary
        .map(|slot| decoder.reference_buffer(slot))
        .transpose()?;
    let secondary_reference_plane = secondary_reference_buffer
        .as_ref()
        .and_then(|buffers| buffers.planes.get(plane_index));
    let compound_mask_references = reference_buffer
        .as_ref()
        .and_then(|a| a.planes.first())
        .zip(
            secondary_reference_buffer
                .as_ref()
                .and_then(|b| b.planes.first()),
        )
        .map(|(primary, secondary)| compound_mask::References { primary, secondary });
    if block_mode.is_inter && reference_plane.is_none() {
        return Err(DecoderError::Unsupported(format!(
            "AV1 inter reference plane {plane_index} is unavailable"
        )));
    }
    let sub8_chroma = if plane_index > 0 && block_mode.is_inter {
        sub8_chroma::plan(
            decoder,
            x,
            y,
            block_mode.block_size,
            subsampling_x,
            subsampling_y,
        )?
    } else {
        None
    };
    let obmc_neighbors = if block_mode.motion_mode == MotionMode::Obmc {
        decoder.obmc_neighbors(x, y, block_mode.block_size)
    } else {
        ObmcNeighbors::default()
    };
    let obmc_references = if block_mode.motion_mode == MotionMode::Obmc {
        Some(obmc_reference::Plan::new(decoder, &obmc_neighbors)?)
    } else {
        None
    };
    let chroma_tx_size = || {
        if frame.coded_lossless() {
            TxSize::Tx4x4
        } else {
            match plane_block_size.largest_supported_rect_tx_size() {
                TxSize::Tx64x64 | TxSize::Tx64x32 | TxSize::Tx32x64 => TxSize::Tx32x32,
                TxSize::Tx64x16 => TxSize::Tx32x16,
                TxSize::Tx16x64 => TxSize::Tx16x32,
                tx_size => tx_size,
            }
        }
    };
    let transforms = if plane_index > 0 {
        // Chroma uses max_uv_txsize for the prediction block. Variable luma
        // transform partitions do not subdivide the U/V coefficient blocks.
        iter_transform_blocks_with_tx_size(
            plane_index,
            block_x,
            block_y,
            plane_block_size,
            chroma_tx_size(),
            plane_width,
            plane_height,
        )
        .collect()
    } else if block_mode.transform_blocks.is_empty() {
        let tx_size = if plane_index == 0 {
            block_mode.tx_size
        } else {
            scale_tx_size(block_mode.tx_size, subsampling_x, subsampling_y)
        };
        if subsampling_x == 0 && subsampling_y == 0 {
            iter_transform_blocks_with_tx_size(
                plane_index,
                x,
                y,
                block_mode.block_size,
                tx_size,
                decoder.mi_cols << 2,
                decoder.mi_rows << 2,
            )
            .collect()
        } else {
            iter_transform_blocks_with_tx_size(
                plane_index,
                block_x,
                block_y,
                plane_block_size,
                tx_size,
                plane_width,
                plane_height,
            )
            .collect()
        }
    } else {
        let mut mapped = Vec::with_capacity(block_mode.transform_blocks.len());
        for transform in &block_mode.transform_blocks {
            let local_x = transform.x.saturating_sub(x);
            let local_y = transform.y.saturating_sub(y);
            let chroma_tx_size = if plane_index == 0 {
                transform.tx_size
            } else {
                scale_tx_size(transform.tx_size, subsampling_x, subsampling_y)
            };
            let mapped_transform = TransformBlock {
                plane: plane_index,
                x: if plane_index == 0 {
                    transform.x
                } else {
                    block_x
                        + ((local_x >> subsampling_x) / chroma_tx_size.width())
                            * chroma_tx_size.width()
                },
                y: if plane_index == 0 {
                    transform.y
                } else {
                    block_y
                        + ((local_y >> subsampling_y) / chroma_tx_size.height())
                            * chroma_tx_size.height()
                },
                tx_size: chroma_tx_size,
            };
            if mapped_transform.x >= plane.layout.width || mapped_transform.y >= plane.layout.height
            {
                continue;
            }
            if !mapped.iter().any(|existing: &TransformBlock| {
                existing.x == mapped_transform.x
                    && existing.y == mapped_transform.y
                    && existing.tx_size == mapped_transform.tx_size
            }) {
                mapped.push(mapped_transform);
            }
        }
        mapped
    };
    let unit_x = plane_block_origin(unit_x, block_mode.block_size.width(), subsampling_x);
    let unit_y = plane_block_origin(unit_y, block_mode.block_size.height(), subsampling_y);
    // A chroma reference made from a sub-8x8 luma block still carries a
    // legal 4x4 chroma residual block. Keep the minimum plane unit at
    // 4x4 instead of filtering that transform out as a 2-pixel unit.
    let unit_width = ceil_shift(unit_width, subsampling_x).max(4);
    let unit_height = ceil_shift(unit_height, subsampling_y).max(4);
    let transform_in_unit = |transform: &crate::av1::transform::TransformBlock| {
        transform.x >= unit_x
            && transform.x < unit_x.saturating_add(unit_width)
            && transform.y >= unit_y
            && transform.y < unit_y.saturating_add(unit_height)
    };
    if block_mode.skip {
        for transform in transforms
            .into_iter()
            .filter(|transform| transform_in_unit(transform))
        {
            if plane_index == 0 {
                decoder.set_luma_tx_type(transform, TxType::DctDct);
            }
            decoder.record_transform_boundary(
                transform,
                TxType::DctDct,
                0,
                block_mode.skip,
                block_mode.is_inter,
                block_mode.reference_frame,
                has_nonzero_mv,
                block_mode.y_mode,
                block_mode.uv_mode,
            );
            decoder.set_txb_entropy_context(transform, 0);
            let (top_right_available, bottom_left_available) =
                decoder.reconstructed_extension_availability(plane, transform)?;
            let prediction_len = transform.tx_size.width() * transform.tx_size.height();
            let prediction = &mut decoder.scratch.prediction[..prediction_len];
            predict_plane_block_into(
                plane,
                reference_plane,
                secondary_reference_plane,
                block_mode,
                frame,
                plane_index,
                prediction_mode,
                block_x,
                block_y,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                angle_delta,
                filter_intra_mode,
                sequence.color_config.bit_depth,
                sequence.enable_intra_edge_filter,
                smooth_neighbour,
                top_right_available,
                bottom_left_available,
                tile_left,
                tile_top,
                tile_right,
                tile_bottom,
                luma_plane,
                cfl_alpha_q3,
                block_mode.intra_block_copy_mv,
                subsampling_x,
                subsampling_y,
                prediction,
                &mut decoder.scratch.inter_intra,
                &obmc_neighbors,
                sub8_chroma.as_ref(),
                compound_mask_references.as_ref(),
                obmc_references.as_ref(),
            )?;
            write_plane_block(
                plane,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                prediction,
            )?;
            decoder.mark_reconstructed_transform(transform)?;
        }
        return Ok(());
    }

    for transform in transforms
        .into_iter()
        .filter(|transform| transform_in_unit(transform))
    {
        let txb_context = decoder.txb_context(plane_block_size, transform);
        let skip_cdf = decoder
            .cdf
            .txb_skip_cdf_mut(transform.tx_size.coeff_cdf_index(), txb_context.skip);
        #[cfg(test)]
        if std::env::var_os("AVIF_ENTROPY_TRACE").is_some() {
            let state = decoder.reader.state_snapshot();
            eprintln!(
                "entropy-trace txb-skip-before x={x} y={y} plane={plane_index} tx={:?} txb_skip={} dc_sign={} cdf={:?} range={} dif={} count={} tell={}",
                transform.tx_size,
                txb_context.skip,
                txb_context.dc_sign,
                skip_cdf,
                state.range,
                state.dif,
                state.count,
                state.tell,
            );
        }
        let all_zero_symbol = decoder.reader.read_symbol(skip_cdf)?;
        #[cfg(test)]
        if std::env::var_os("AVIF_ENTROPY_TRACE").is_some() {
            let state = decoder.reader.state_snapshot();
            eprintln!(
                "entropy-trace txb-skip-after x={x} y={y} plane={plane_index} symbol={all_zero_symbol} range={} dif={} count={} tell={}",
                state.range, state.dif, state.count, state.tell
            );
        }
        if all_zero_symbol != 0 {
            // Deblocking depends on transform boundaries even when the
            // coefficient block is entirely zero; retain the transform
            // geometry for the post-filter stage.
            if plane_index == 0 {
                decoder.set_luma_tx_type(transform, TxType::DctDct);
            }
            decoder.record_transform_boundary(
                transform,
                TxType::DctDct,
                0,
                block_mode.skip,
                block_mode.is_inter,
                block_mode.reference_frame,
                has_nonzero_mv,
                block_mode.y_mode,
                block_mode.uv_mode,
            );
            decoder.set_txb_entropy_context(transform, 0);
            let (top_right_available, bottom_left_available) =
                decoder.reconstructed_extension_availability(plane, transform)?;
            let prediction_len = transform.tx_size.width() * transform.tx_size.height();
            let prediction = &mut decoder.scratch.prediction[..prediction_len];
            predict_plane_block_into(
                plane,
                reference_plane,
                secondary_reference_plane,
                block_mode,
                frame,
                plane_index,
                prediction_mode,
                block_x,
                block_y,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                angle_delta,
                filter_intra_mode,
                sequence.color_config.bit_depth,
                sequence.enable_intra_edge_filter,
                smooth_neighbour,
                top_right_available,
                bottom_left_available,
                tile_left,
                tile_top,
                tile_right,
                tile_bottom,
                luma_plane,
                cfl_alpha_q3,
                block_mode.intra_block_copy_mv,
                subsampling_x,
                subsampling_y,
                prediction,
                &mut decoder.scratch.inter_intra,
                &obmc_neighbors,
                sub8_chroma.as_ref(),
                compound_mask_references.as_ref(),
                obmc_references.as_ref(),
            )?;
            write_plane_block(
                plane,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                prediction,
            )?;
            decoder.mark_reconstructed_transform(transform)?;
            continue;
        }

        let retain_coefficients = plane_index == 0 && decoded_luma.is_some();
        let tx_type_override = (plane_index > 0 && block_mode.is_inter).then(|| {
            let luma_x = x + ((transform.x.saturating_sub(block_x)) << subsampling_x);
            let luma_y = y + ((transform.y.saturating_sub(block_y)) << subsampling_y);
            inter_chroma_tx_type(
                frame,
                transform.tx_size,
                decoder.luma_tx_type_at(luma_x, luma_y),
            )
        });
        let decoded_transform = decoder.read_decoded_transform(
            frame,
            block_mode,
            transform,
            txb_context.dc_sign,
            tx_type_override,
        )?;
        if plane_index == 0 {
            decoder.set_luma_tx_type(transform, decoded_transform.tx_type);
        }
        decoder.set_txb_entropy_context(transform, decoded_transform.entropy_context);
        let (top_right_available, bottom_left_available) =
            decoder.reconstructed_extension_availability(plane, transform)?;
        let prediction_len = transform.tx_size.width() * transform.tx_size.height();
        let prediction = &mut decoder.scratch.prediction[..prediction_len];
        predict_plane_block_into(
            plane,
            reference_plane,
            secondary_reference_plane,
            block_mode,
            frame,
            plane_index,
            prediction_mode,
            block_x,
            block_y,
            transform.x,
            transform.y,
            transform.tx_size.width(),
            transform.tx_size.height(),
            angle_delta,
            filter_intra_mode,
            sequence.color_config.bit_depth,
            sequence.enable_intra_edge_filter,
            smooth_neighbour,
            top_right_available,
            bottom_left_available,
            tile_left,
            tile_top,
            tile_right,
            tile_bottom,
            luma_plane,
            cfl_alpha_q3,
            block_mode.intra_block_copy_mv,
            subsampling_x,
            subsampling_y,
            prediction,
            &mut decoder.scratch.inter_intra,
            &obmc_neighbors,
            sub8_chroma.as_ref(),
            compound_mask_references.as_ref(),
            obmc_references.as_ref(),
        )?;
        let block = decoded_transform.transform;
        let tx_type = decoded_transform.tx_type;
        let dequant = &mut decoder.scratch.dequant[..prediction_len];
        let reconstructed = &mut decoder.scratch.reconstruction[..prediction_len];
        let residual = &mut decoder.scratch.residual[..prediction_len];
        let reconstructed_transform = if frame.coded_lossless() {
            reconstruct_lossless_transform_block_parts_into(
                plane,
                block,
                tx_type,
                &decoded_transform.coefficients,
                quant_state.plane(transform.plane),
                prediction,
                sequence.color_config.bit_depth,
                dequant,
                reconstructed,
            )?
        } else {
            let qmatrix_level = frame.quantization.qmatrix_level(transform.plane);
            let qmatrix = frame
                .quantization
                .using_qmatrix
                .then_some((qmatrix_level, transform.plane))
                .filter(|(level, _)| *level < 15);
            reconstruct_transform_block_parts_into(
                plane,
                block,
                tx_type,
                &decoded_transform.coefficients,
                quant_state.plane(transform.plane),
                prediction,
                sequence.color_config.bit_depth,
                qmatrix,
                dequant,
                reconstructed,
                residual,
            )?
        };
        // The reconstruction already counted non-zero coefficients for its
        // zero-residual fast path. Reuse that result instead of scanning the
        // full transform a second time while recording post-filter state.
        decoder.record_transform_boundary(
            reconstructed_transform.block,
            reconstructed_transform.tx_type,
            reconstructed_transform.non_zero_coefficients,
            block_mode.skip,
            block_mode.is_inter,
            block_mode.reference_frame,
            has_nonzero_mv,
            block_mode.y_mode,
            block_mode.uv_mode,
        );
        decoder.mark_reconstructed_transform(transform)?;
        if retain_coefficients {
            if let Some(decoded_luma) = decoded_luma.as_deref_mut() {
                decoded_luma.push(decoded_transform);
            }
        } else {
            decoder.coefficient_scratch = decoded_transform.coefficients;
        }
    }

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "prediction arguments mirror the AV1 intra prediction inputs"
)]
#[allow(dead_code)]
pub(super) fn predict_block(
    plane: &PlaneBuffer,
    prediction_mode: PredictionMode,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    angle_delta: Option<i8>,
    filter_intra_mode: Option<usize>,
    bit_depth: u8,
    enable_intra_edge_filter: bool,
    smooth_neighbour: bool,
    top_right_available: usize,
    bottom_left_available: usize,
) -> Result<Vec<u16>, DecoderError> {
    let sample_count = width.checked_mul(height).ok_or_else(|| {
        DecoderError::InvalidParam("AV1 prediction dimensions overflow".to_string())
    })?;
    let mut output = vec![0; sample_count];
    predict_block_into(
        plane,
        prediction_mode,
        x,
        y,
        width,
        height,
        angle_delta,
        filter_intra_mode,
        bit_depth,
        enable_intra_edge_filter,
        smooth_neighbour,
        top_right_available,
        bottom_left_available,
        0,
        0,
        plane.layout.width,
        plane.layout.height,
        &mut output,
    )?;
    Ok(output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "prediction arguments mirror the AV1 intra prediction inputs"
)]
fn predict_block_into(
    plane: &PlaneBuffer,
    prediction_mode: PredictionMode,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    angle_delta: Option<i8>,
    filter_intra_mode: Option<usize>,
    bit_depth: u8,
    enable_intra_edge_filter: bool,
    smooth_neighbour: bool,
    top_right_available: usize,
    bottom_left_available: usize,
    tile_left: usize,
    tile_top: usize,
    tile_right: usize,
    tile_bottom: usize,
    output: &mut [u16],
) -> Result<(), DecoderError> {
    // AV1 prediction blocks can reach 128x128, so each directional edge needs
    // room for the sum of both dimensions (2 * MAX_SB_SIZE).
    const MAX_INTRA_EDGE_LEN: usize = 256;
    let edge_len = width
        .checked_add(height)
        .ok_or_else(|| DecoderError::InvalidParam("AV1 intra edge length overflows".to_string()))?;
    if edge_len > MAX_INTRA_EDGE_LEN {
        return Err(DecoderError::Unsupported(format!(
            "AV1 intra prediction edge length {edge_len} exceeds the supported maximum"
        )));
    }
    let sample_count = width.checked_mul(height).ok_or_else(|| {
        DecoderError::InvalidParam("AV1 prediction dimensions overflow".to_string())
    })?;
    if output.len() != sample_count {
        return Err(DecoderError::InvalidParam(
            "AV1 prediction output dimensions do not match block".to_string(),
        ));
    }
    let mut above_storage = [0u16; MAX_INTRA_EDGE_LEN];
    let mut left_storage = [0u16; MAX_INTRA_EDGE_LEN];
    let (above_available, left_available, edge_above_left) = read_intra_edges_into_with_bounds(
        plane,
        x,
        y,
        width,
        height,
        bit_depth,
        top_right_available,
        bottom_left_available,
        tile_left,
        tile_top,
        tile_right,
        tile_bottom,
        &mut above_storage,
        &mut left_storage,
    )?;
    let above = &mut above_storage[..edge_len];
    let left = &mut left_storage[..edge_len];
    let midpoint = 1u16 << (bit_depth - 1);
    let above_left = match (above_available, left_available) {
        (true, true) => edge_above_left,
        (true, false) => above[0],
        (false, true) => left[0],
        (false, false) => midpoint,
    };
    if !above_available && left_available {
        above.fill(left[0]);
    }
    if !left_available && above_available {
        left.fill(above[0]);
    }
    let edges = if prediction_mode == PredictionMode::Dc && filter_intra_mode.is_none() {
        IntraEdges {
            above: above_available.then_some(&*above),
            left: left_available.then_some(&*left),
            above_left: Some(above_left),
            bit_depth,
        }
    } else {
        IntraEdges {
            above: Some(&*above),
            left: Some(&*left),
            above_left: Some(above_left),
            bit_depth,
        }
    };
    if let Some(filter_intra_mode) = filter_intra_mode {
        return predict_filter_intra_into(filter_intra_mode, width, height, edges, output);
    }
    predict_intra_with_edge_filter_into(
        prediction_mode,
        angle_delta,
        width,
        height,
        edges,
        enable_intra_edge_filter,
        smooth_neighbour,
        output,
    )
}

fn ceil_shift(value: usize, shift: usize) -> usize {
    (value + ((1usize << shift) - 1)) >> shift
}

fn plane_block_origin(luma_origin: usize, luma_extent: usize, subsampling: usize) -> usize {
    if subsampling > 0 && luma_extent == 4 {
        luma_origin.saturating_sub(4) >> subsampling
    } else {
        luma_origin >> subsampling
    }
}

fn scale_tx_size(tx_size: TxSize, subsampling_x: usize, subsampling_y: usize) -> TxSize {
    let width = ceil_shift(tx_size.width(), subsampling_x).max(4);
    let height = ceil_shift(tx_size.height(), subsampling_y).max(4);
    TxSize::from_dimensions(width, height)
        .or({
            if width == 4 && height > 16 {
                Some(TxSize::Tx4x16)
            } else if height == 4 && width > 16 {
                Some(TxSize::Tx16x4)
            } else {
                None
            }
        })
        .unwrap_or(TxSize::Tx4x4)
}

fn plane_block_size(
    block_size: crate::av1::syntax::BlockSize,
    subsampling_x: usize,
    subsampling_y: usize,
) -> crate::av1::syntax::BlockSize {
    use crate::av1::syntax::BlockSize;

    if subsampling_x == 0 && subsampling_y == 0 {
        return block_size;
    }

    // AV1's scale_chroma_bsize enlarges the narrow 4-pixel luma shapes before
    // converting them to plane coordinates. This keeps a sub-8x8 luma block
    // represented by a legal 4x4 chroma block instead of a 2x4/4x2 shape.
    let scaled = match block_size {
        BlockSize::Block4x4 if subsampling_x == 1 && subsampling_y == 1 => BlockSize::Block8x8,
        BlockSize::Block4x4 if subsampling_x == 1 => BlockSize::Block8x4,
        BlockSize::Block4x4 if subsampling_y == 1 => BlockSize::Block4x8,
        BlockSize::Block4x8 if subsampling_x == 1 => BlockSize::Block8x8,
        BlockSize::Block4x8 if subsampling_y == 1 => BlockSize::Block4x8,
        BlockSize::Block8x4 if subsampling_x == 1 && subsampling_y == 1 => BlockSize::Block8x8,
        BlockSize::Block8x4 if subsampling_x == 1 => BlockSize::Block8x4,
        BlockSize::Block8x4 if subsampling_y == 1 => BlockSize::Block8x8,
        BlockSize::Block4x16 if subsampling_x == 1 => BlockSize::Block8x16,
        BlockSize::Block4x16 if subsampling_y == 1 => BlockSize::Block4x16,
        BlockSize::Block16x4 if subsampling_x == 1 && subsampling_y == 1 => BlockSize::Block16x8,
        BlockSize::Block16x4 if subsampling_x == 1 => BlockSize::Block16x4,
        BlockSize::Block16x4 if subsampling_y == 1 => BlockSize::Block16x8,
        _ => block_size,
    };
    BlockSize::from_dimensions(
        ceil_shift(scaled.width(), subsampling_x).max(4),
        ceil_shift(scaled.height(), subsampling_y).max(4),
    )
    .unwrap_or(BlockSize::Block4x4)
}

#[expect(
    clippy::too_many_arguments,
    reason = "plane prediction keeps palette, CFL, edge, and geometry inputs explicit"
)]
fn predict_plane_block_into(
    plane: &PlaneBuffer,
    reference_plane: Option<&PlaneBuffer>,
    secondary_reference_plane: Option<&PlaneBuffer>,
    block_mode: &BlockModeProbe,
    frame: &FrameHeader,
    plane_index: usize,
    prediction_mode: PredictionMode,
    block_x: usize,
    block_y: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    angle_delta: Option<i8>,
    filter_intra_mode: Option<usize>,
    bit_depth: u8,
    enable_intra_edge_filter: bool,
    smooth_neighbour: bool,
    top_right_available: usize,
    bottom_left_available: usize,
    tile_left: usize,
    tile_top: usize,
    tile_right: usize,
    tile_bottom: usize,
    luma_plane: Option<&PlaneBuffer>,
    cfl_alpha_q3: Option<i8>,
    intra_block_copy_mv: Option<(i32, i32)>,
    subsampling_x: usize,
    subsampling_y: usize,
    output: &mut [u16],
    inter_intra_scratch: &mut [u16],
    obmc_neighbors: &ObmcNeighbors,
    sub8_chroma: Option<&sub8_chroma::PredictionPlan>,
    compound_mask_references: Option<&compound_mask::References<'_>>,
    obmc_references: Option<&obmc_reference::Plan>,
) -> Result<(), DecoderError> {
    let sample_count = width.checked_mul(height).ok_or_else(|| {
        DecoderError::InvalidParam("AV1 prediction dimensions overflow".to_string())
    })?;
    if output.len() != sample_count {
        return Err(DecoderError::InvalidParam(
            "AV1 prediction output dimensions do not match block".to_string(),
        ));
    }
    // Filter choice follows the full prediction block even when this call
    // reconstructs only one of its transform subdivisions.
    let mask_geometry = MaskGeometry {
        x: x.saturating_sub(block_x),
        y: y.saturating_sub(block_y),
        width: (block_mode.block_size.width() >> subsampling_x).max(4),
        height: (block_mode.block_size.height() >> subsampling_y).max(4),
    };
    if let Some(plan) = sub8_chroma {
        return sub8_chroma::predict(
            plan,
            plane_index,
            x,
            y,
            width,
            height,
            subsampling_x,
            subsampling_y,
            bit_depth,
            output,
        );
    }
    if block_mode.is_inter {
        let slot = block_mode.reference_frame.ok_or_else(|| {
            DecoderError::Unsupported("AV1 inter block has no reference frame".to_string())
        })?;
        let mv = block_mode.motion_vector.ok_or_else(|| {
            DecoderError::Unsupported("AV1 inter block has no motion vector".to_string())
        })?;
        let reference = reference_plane.ok_or_else(|| {
            DecoderError::Unsupported(format!(
                "AV1 inter reference slot {slot} plane {plane_index} is unavailable"
            ))
        })?;
        if plane_index > 0
            && block_mode.compound_mask.is_some()
            && let (Some(secondary), Some(references)) =
                (secondary_reference_plane, compound_mask_references)
        {
            compound_mask::predict_chroma(
                references,
                reference,
                secondary,
                block_mode,
                frame,
                x,
                y,
                width,
                height,
                subsampling_x,
                subsampling_y,
                bit_depth,
                mask_geometry,
                output,
                inter_intra_scratch,
            );
            return Ok(());
        }
        predict_inter_block_into(
            reference,
            secondary_reference_plane,
            x,
            y,
            width,
            height,
            mv,
            block_mode.motion_vector_secondary,
            subsampling_x,
            subsampling_y,
            block_mode
                .interpolation_filter
                .unwrap_or((InterpolationFilter::Regular, InterpolationFilter::Regular)),
            block_mode.compound_weight,
            block_mode.compound_mask,
            bit_depth,
            mask_geometry,
            output,
        )?;
        let warp_allowed =
            mask_geometry.width >= 8 && mask_geometry.height >= 8 && frame.force_integer_mv != 1;
        let global_model = warp_allowed
            .then(|| global_motion_matrix(frame, block_mode.global_motion_index))
            .flatten();
        let secondary_global_model = warp_allowed
            .then(|| global_motion_matrix(frame, block_mode.global_motion_index_secondary))
            .flatten();
        if secondary_reference_plane.is_none() {
            if let Some(matrix) = global_model {
                let _ = predict_global_motion_block_anchored_into(
                    output,
                    reference,
                    x,
                    y,
                    width,
                    height,
                    subsampling_x,
                    subsampling_y,
                    bit_depth,
                    frame.allow_high_precision_mv,
                    matrix,
                    false,
                    x - mask_geometry.x,
                    y - mask_geometry.y,
                );
            }
        } else if let Some(secondary) = secondary_reference_plane {
            // GLOBAL_GLOBALMV carries two independent global models. Decode
            // each prediction into its own reusable buffer before applying
            // the already-decoded compound mask/weight.
            if global_model.is_some() || secondary_global_model.is_some() {
                let filters = block_mode
                    .interpolation_filter
                    .unwrap_or((InterpolationFilter::Regular, InterpolationFilter::Regular));
                predict_compound_global_or_translation_into(
                    output,
                    reference,
                    x,
                    y,
                    width,
                    height,
                    mv,
                    subsampling_x,
                    subsampling_y,
                    filters,
                    bit_depth,
                    mask_geometry,
                    frame.allow_high_precision_mv,
                    global_model,
                );
                let secondary_output = &mut inter_intra_scratch[..output.len()];
                predict_compound_global_or_translation_into(
                    secondary_output,
                    secondary,
                    x,
                    y,
                    width,
                    height,
                    block_mode.motion_vector_secondary.unwrap_or(mv),
                    subsampling_x,
                    subsampling_y,
                    filters,
                    bit_depth,
                    mask_geometry,
                    frame.allow_high_precision_mv,
                    secondary_global_model,
                );
                for row in 0..height {
                    for col in 0..width {
                        let index = row * width + col;
                        output[index] = blend_compound_intermediates(
                            output[index],
                            secondary_output[index],
                            block_mode.compound_weight,
                            block_mode.compound_mask,
                            bit_depth,
                            mask_geometry.x + col,
                            mask_geometry.y + row,
                            mask_geometry.width,
                            mask_geometry.height,
                        );
                    }
                }
            }
        }
        match block_mode.motion_mode {
            MotionMode::Simple => {}
            MotionMode::Obmc => apply_obmc_edge_blend(
                output,
                reference,
                block_x,
                block_y,
                x,
                y,
                width,
                height,
                (block_mode.block_size.width() >> subsampling_x).max(4),
                (block_mode.block_size.height() >> subsampling_y).max(4),
                subsampling_x,
                subsampling_y,
                obmc_neighbors,
                inter_intra_scratch,
                bit_depth,
                obmc_references,
                plane_index,
            )?,
            MotionMode::LocalWarp if secondary_reference_plane.is_none() && warp_allowed => {
                apply_local_warp_prediction(
                    output,
                    reference,
                    x,
                    y,
                    width,
                    height,
                    block_x,
                    block_y,
                    (block_mode.block_size.width() >> subsampling_x).max(4),
                    (block_mode.block_size.height() >> subsampling_y).max(4),
                    mv,
                    subsampling_x,
                    subsampling_y,
                    bit_depth,
                    block_mode.local_warp_samples,
                )?;
            }
            MotionMode::LocalWarp => {}
        }
        if let Some(mode) = block_mode.interintra_mode {
            interintra::apply(
                plane,
                output,
                mode,
                block_mode.interintra_wedge_index,
                block_mode.block_size.width(),
                block_mode.block_size.height(),
                block_x,
                block_y,
                x,
                y,
                width,
                height,
                mask_geometry,
                bit_depth,
                subsampling_x,
                subsampling_y,
                enable_intra_edge_filter,
                tile_left,
                tile_top,
                tile_right,
                tile_bottom,
                inter_intra_scratch,
            )?;
        }
    } else if let Some(mv) = intra_block_copy_mv {
        predict_intra_block_copy_into(
            plane,
            x,
            y,
            width,
            height,
            mv,
            subsampling_x,
            subsampling_y,
            output,
        )?;
    } else {
        let palette_prediction =
            if filter_intra_mode.is_none() && prediction_mode == PredictionMode::Dc {
                if plane_index == 0 {
                    block_mode
                        .palette
                        .y
                        .as_ref()
                        .map(|palette| (palette, 0, palette.colors.len()))
                } else {
                    block_mode.palette.uv.as_ref().map(|palette| {
                        let palette_size = palette.colors.len() / 2;
                        (
                            palette,
                            usize::from(plane_index == 2) * palette_size,
                            palette_size,
                        )
                    })
                }
            } else {
                None
            };
        if let Some((palette, color_offset, palette_size)) = palette_prediction
            && !palette.color_map.is_empty()
            && palette.map_width > 0
            && palette.map_height > 0
        {
            predict_palette_block_into(
                palette,
                color_offset,
                palette_size,
                block_x,
                block_y,
                x,
                y,
                width,
                height,
                output,
            );
        } else if prediction_mode == PredictionMode::Dc && filter_intra_mode.is_none() {
            predict_dc_block_into_with_bounds(
                plane,
                x,
                y,
                width,
                height,
                bit_depth,
                tile_left,
                tile_top,
                tile_right,
                tile_bottom,
                output,
            );
        } else {
            predict_block_into(
                plane,
                prediction_mode,
                x,
                y,
                width,
                height,
                angle_delta,
                filter_intra_mode,
                bit_depth,
                enable_intra_edge_filter,
                smooth_neighbour,
                top_right_available,
                bottom_left_available,
                tile_left,
                tile_top,
                tile_right,
                tile_bottom,
                output,
            )?;
        }
    }
    if let Some(alpha_q3) = cfl_alpha_q3 {
        let luma_plane = luma_plane.ok_or_else(|| {
            DecoderError::Bitstream("AV1 CFL prediction is missing its luma plane".to_string())
        })?;
        apply_cfl_prediction(
            output,
            luma_plane,
            x,
            y,
            width,
            height,
            alpha_q3,
            bit_depth,
            subsampling_x,
            subsampling_y,
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "OBMC prediction keeps frame, block, plane, and scratch geometry explicit"
)]
pub(super) fn apply_obmc_edge_blend(
    output: &mut [u16],
    reference: &PlaneBuffer,
    block_x: usize,
    block_y: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    block_width: usize,
    block_height: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    neighbors: &ObmcNeighbors,
    scratch: &mut [u16],
    bit_depth: u8,
    references: Option<&obmc_reference::Plan>,
    plane_index: usize,
) -> Result<(), DecoderError> {
    let has_neighbors = neighbors.above.iter().flatten().next().is_some()
        || neighbors.left.iter().flatten().next().is_some();
    if !has_neighbors {
        // AV1 only blends predictors from eligible causal neighbours. A frame
        // edge or an intra neighbour does not synthesize a replacement
        // predictor from the already reconstructed current frame.
        return Ok(());
    }
    let sample_count = width.checked_mul(height).ok_or_else(|| {
        DecoderError::InvalidParam("AV1 OBMC prediction size overflows".to_string())
    })?;
    if scratch.len() < sample_count {
        return Err(DecoderError::InvalidParam(
            "AV1 OBMC scratch buffer is too small".to_string(),
        ));
    }
    let overlap_x = (block_width.min(64) / 2).max(1);
    let overlap_y = (block_height.min(64) / 2).max(1);
    // AOM skips the above predictor for chroma plane blocks smaller than
    // 8x8 (4x4, 8x4 and 4x8), while retaining the left predictor.
    let skip_above =
        block_width <= 8 && block_height <= 8 && (block_width == 4 || block_height == 4);
    if !skip_above {
        for neighbor in neighbors.above.iter().flatten() {
            let reference = if let Some(plan) = references {
                plan.plane(neighbor.reference_frame, plane_index)?
            } else {
                reference
            };
            blend_obmc_neighbor_prediction(
                output,
                reference,
                x,
                y,
                width,
                height,
                block_x,
                block_y,
                block_width,
                block_height,
                overlap_x,
                overlap_y,
                neighbor,
                true,
                subsampling_x,
                subsampling_y,
                &mut scratch[..sample_count],
                bit_depth,
            )?;
        }
    }
    for neighbor in neighbors.left.iter().flatten() {
        let reference = if let Some(plan) = references {
            plan.plane(neighbor.reference_frame, plane_index)?
        } else {
            reference
        };
        blend_obmc_neighbor_prediction(
            output,
            reference,
            x,
            y,
            width,
            height,
            block_x,
            block_y,
            block_width,
            block_height,
            overlap_x,
            overlap_y,
            neighbor,
            false,
            subsampling_x,
            subsampling_y,
            &mut scratch[..sample_count],
            bit_depth,
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "OBMC neighbor blending keeps predictor and overlap geometry explicit"
)]
fn blend_obmc_neighbor_prediction(
    output: &mut [u16],
    reference: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    block_width: usize,
    block_height: usize,
    overlap_x: usize,
    overlap_y: usize,
    neighbor: &super::diagnostic::ObmcNeighbor,
    above: bool,
    subsampling_x: usize,
    subsampling_y: usize,
    scratch: &mut [u16],
    bit_depth: u8,
) -> Result<(), DecoderError> {
    let neighbor_x = neighbor.origin_x >> subsampling_x;
    let neighbor_y = neighbor.origin_y >> subsampling_y;
    let neighbor_width = ceil_shift(neighbor.width, subsampling_x);
    let neighbor_height = ceil_shift(neighbor.height, subsampling_y);
    // OBMC predicts a complete neighbor strip before blending it. The current
    // transform may cover only a slice of that strip; filter selection must
    // retain the strip dimensions (AV1 build_prediction_by_above/left_pred).
    let filter_dimensions = if above {
        (
            block_x
                .saturating_add(block_width)
                .min(neighbor_x.saturating_add(neighbor_width))
                .saturating_sub(block_x.max(neighbor_x)),
            overlap_y.max(4),
        )
    } else {
        (
            overlap_x.max(4),
            block_y
                .saturating_add(block_height)
                .min(neighbor_y.saturating_add(neighbor_height))
                .saturating_sub(block_y.max(neighbor_y)),
        )
    };
    let x_end = x.saturating_add(width);
    let y_end = y.saturating_add(height);
    let (predict_x, predict_y, predict_width, predict_height) = if above {
        let start_x = x.max(neighbor_x);
        let end_x = x_end.min(neighbor_x.saturating_add(neighbor_width));
        let start_y = y.max(block_y);
        let end_y = y_end.min(block_y.saturating_add(overlap_y));
        (
            start_x,
            start_y,
            end_x.saturating_sub(start_x),
            end_y.saturating_sub(start_y),
        )
    } else {
        let start_x = x.max(block_x);
        let end_x = x_end.min(block_x.saturating_add(overlap_x));
        let start_y = y.max(neighbor_y);
        let end_y = y_end.min(neighbor_y.saturating_add(neighbor_height));
        (
            start_x,
            start_y,
            end_x.saturating_sub(start_x),
            end_y.saturating_sub(start_y),
        )
    };
    if predict_width == 0 || predict_height == 0 {
        return Ok(());
    }
    let predicted_count = predict_width
        .checked_mul(predict_height)
        .ok_or_else(|| DecoderError::InvalidParam("AV1 OBMC strip size overflows".to_string()))?;
    predict_inter_block_into(
        reference,
        None,
        predict_x,
        predict_y,
        predict_width,
        predict_height,
        neighbor.motion_vector,
        None,
        subsampling_x,
        subsampling_y,
        neighbor.interpolation_filters,
        None,
        None,
        bit_depth,
        MaskGeometry {
            x: 0,
            y: 0,
            width: filter_dimensions.0,
            height: filter_dimensions.1,
        },
        &mut scratch[..predicted_count],
    )?;
    let max_value = (1_u32 << u32::from(bit_depth.min(16))) - 1;
    for row in 0..predict_height {
        let global_y = predict_y + row;
        for col in 0..predict_width {
            let global_x = predict_x + col;
            let output_index = (global_y - y) * width + global_x - x;
            let current_weight = if above {
                obmc_mask_value(overlap_y, global_y.saturating_sub(block_y))
            } else {
                obmc_mask_value(overlap_x, global_x.saturating_sub(block_x))
            };
            let neighbor_weight = u32::from(64_u8.saturating_sub(current_weight));
            let current = u32::from(output[output_index]);
            let predicted = u32::from(scratch[row * predict_width + col]);
            output[output_index] =
                ((current * u32::from(current_weight) + predicted * neighbor_weight + 32) >> 6)
                    .min(max_value) as u16;
        }
    }
    Ok(())
}

fn obmc_mask_value(length: usize, index: usize) -> u8 {
    const MASK_1: [u8; 1] = [64];
    const MASK_2: [u8; 2] = [45, 64];
    const MASK_4: [u8; 4] = [39, 50, 59, 64];
    const MASK_8: [u8; 8] = [36, 42, 48, 53, 57, 61, 64, 64];
    const MASK_16: [u8; 16] = [
        34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64,
    ];
    const MASK_32: [u8; 32] = [
        33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 60, 61,
        62, 64, 64, 64, 64, 64, 64, 64, 64,
    ];
    const MASK_64: [u8; 64] = [
        33, 34, 35, 35, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 44, 44, 45, 46, 47, 47, 48, 49, 50,
        51, 51, 51, 52, 52, 53, 54, 55, 56, 56, 56, 57, 57, 58, 58, 59, 60, 60, 60, 60, 61, 62, 62,
        62, 62, 62, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    ];
    let mask = match length {
        0 => &MASK_1[..],
        1 => &MASK_1[..],
        2 => &MASK_2[..],
        4 => &MASK_4[..],
        8 => &MASK_8[..],
        16 => &MASK_16[..],
        32 => &MASK_32[..],
        64 => &MASK_64[..],
        other => {
            debug_assert!(other <= 64);
            if other < 2 {
                &MASK_1[..]
            } else if other < 4 {
                &MASK_2[..]
            } else if other < 8 {
                &MASK_4[..]
            } else if other < 16 {
                &MASK_8[..]
            } else if other < 32 {
                &MASK_16[..]
            } else {
                &MASK_32[..]
            }
        }
    };
    mask[index.min(mask.len().saturating_sub(1))]
}

fn global_motion_matrix(frame: &FrameHeader, index: Option<usize>) -> Option<[i32; 6]> {
    let index = index?;
    let motion_type = frame.global_motion.types.get(index).copied()?;
    if !matches!(
        motion_type,
        GlobalMotionType::RotZoom | GlobalMotionType::Affine
    ) {
        return None;
    }
    frame.global_motion.matrices.get(index).copied()
}

/// Reconstruct a single-reference GLOBALMV block with its signalled
/// rotzoom/affine matrix.  Translation-only global motion intentionally stays
/// on the cheaper regular predictor because the block-centre MV is equivalent.
#[expect(
    clippy::too_many_arguments,
    reason = "global motion prediction keeps matrix and plane geometry explicit"
)]
#[cfg(test)]
fn predict_global_motion_block_into(
    output: &mut [u16],
    reference: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    bit_depth: u8,
    allow_high_precision_mv: bool,
    matrix: [i32; 6],
) -> bool {
    predict_global_motion_block_with_precision_into(
        output,
        reference,
        x,
        y,
        width,
        height,
        subsampling_x,
        subsampling_y,
        bit_depth,
        allow_high_precision_mv,
        matrix,
        false,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "global warp keeps prediction geometry and precision explicit"
)]
fn predict_global_motion_block_with_precision_into(
    output: &mut [u16],
    reference: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    bit_depth: u8,
    allow_high_precision_mv: bool,
    matrix: [i32; 6],
    compound: bool,
) -> bool {
    predict_global_motion_block_anchored_into(
        output,
        reference,
        x,
        y,
        width,
        height,
        subsampling_x,
        subsampling_y,
        bit_depth,
        allow_high_precision_mv,
        matrix,
        compound,
        x,
        y,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "global warp keeps prediction geometry and precision explicit"
)]
fn predict_global_motion_block_anchored_into(
    output: &mut [u16],
    reference: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    bit_depth: u8,
    allow_high_precision_mv: bool,
    matrix: [i32; 6],
    compound: bool,
    block_x: usize,
    block_y: usize,
) -> bool {
    let translation_shift = if allow_high_precision_mv { 3 } else { 4 };
    let Some(translation_x) = i64::from(matrix[0]).checked_shl(translation_shift) else {
        return false;
    };
    let Some(translation_y) = i64::from(matrix[1]).checked_shl(translation_shift) else {
        return false;
    };
    let params = LocalWarpParams {
        translation_x,
        translation_y,
        alpha: i64::from(matrix[2]),
        beta: i64::from(matrix[3]),
        gamma: i64::from(matrix[4]),
        delta: i64::from(matrix[5]),
    };
    let Some((warp_alpha, warp_beta, warp_gamma, warp_delta)) = setup_warp_shear(&params) else {
        return false;
    };
    let (inter_round0, inter_round1) = if bit_depth == 12 { (5, 9) } else { (3, 11) };
    let inter_round1 = if compound { 7 } else { inter_round1 };
    predict_warped_region_into(
        output,
        reference,
        x,
        y,
        width,
        height,
        block_x,
        block_y,
        subsampling_x,
        subsampling_y,
        &params,
        (warp_alpha, warp_beta, warp_gamma, warp_delta),
        inter_round0,
        inter_round1,
        bit_depth,
    );
    true
}

#[expect(
    clippy::too_many_arguments,
    reason = "local warp prediction keeps reference, block geometry, and filter state explicit"
)]
fn apply_local_warp_prediction(
    output: &mut [u16],
    reference: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    block_width: usize,
    block_height: usize,
    base_mv: (i32, i32),
    subsampling_x: usize,
    subsampling_y: usize,
    bit_depth: u8,
    samples: [Option<LocalWarpSample>; 8],
) -> Result<(), DecoderError> {
    let block_x_luma = i64::try_from(block_x)
        .map_err(|_| DecoderError::Bitstream("AV1 local warp block x overflows".to_string()))?
        .checked_shl(u32::try_from(subsampling_x).unwrap_or(0))
        .ok_or_else(|| DecoderError::Bitstream("AV1 local warp block x overflows".to_string()))?;
    let block_y_luma = i64::try_from(block_y)
        .map_err(|_| DecoderError::Bitstream("AV1 local warp block y overflows".to_string()))?
        .checked_shl(u32::try_from(subsampling_y).unwrap_or(0))
        .ok_or_else(|| DecoderError::Bitstream("AV1 local warp block y overflows".to_string()))?;
    let block_width_luma = block_width
        .checked_shl(u32::try_from(subsampling_x).unwrap_or(0))
        .unwrap_or(usize::MAX);
    let block_height_luma = block_height
        .checked_shl(u32::try_from(subsampling_y).unwrap_or(0))
        .unwrap_or(usize::MAX);
    let Some(params) = estimate_local_warp_params(
        block_x_luma,
        block_y_luma,
        block_width_luma,
        block_height_luma,
        base_mv,
        samples,
    ) else {
        return Ok(());
    };
    if params.alpha == 1 << 16 && params.beta == 0 && params.gamma == 0 && params.delta == 1 << 16 {
        // The translation predictor already produced the exact regular
        // interpolation result for the identity local model.
        return Ok(());
    }
    let Some((warp_alpha, warp_beta, warp_gamma, warp_delta)) = setup_warp_shear(&params) else {
        return Ok(());
    };
    let (inter_round0, inter_round1) = if bit_depth == 12 { (5, 9) } else { (3, 11) };
    predict_warped_region_into(
        output,
        reference,
        x,
        y,
        width,
        height,
        block_x,
        block_y,
        subsampling_x,
        subsampling_y,
        &params,
        (warp_alpha, warp_beta, warp_gamma, warp_delta),
        inter_round0,
        inter_round1,
        bit_depth,
    );
    Ok(())
}

// A warp's filter phases are relative to the prediction block's 8x8 tiles,
// independently of the transform fragments that request their samples.
#[expect(
    clippy::too_many_arguments,
    reason = "warp region keeps prediction and transform geometry explicit"
)]
fn predict_warped_region_into(
    output: &mut [u16],
    reference: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    params: &LocalWarpParams,
    shear: (i64, i64, i64, i64),
    inter_round0: u32,
    inter_round1: u32,
    bit_depth: u8,
) {
    let mut intermediate = [0_i64; 15 * 8];
    let mut tile = [0_u16; 64];
    let offset_x = x - block_x;
    let offset_y = y - block_y;
    for tile_y in ((offset_y & !7)..offset_y + height).step_by(8) {
        for tile_x in ((offset_x & !7)..offset_x + width).step_by(8) {
            predict_warped_block_into(
                &mut tile,
                reference,
                block_x + tile_x,
                block_y + tile_y,
                8,
                8,
                0,
                0,
                subsampling_x,
                subsampling_y,
                params,
                shear.0,
                shear.1,
                shear.2,
                shear.3,
                inter_round0,
                inter_round1,
                bit_depth,
                &mut intermediate,
            );
            let left = offset_x.max(tile_x);
            let right = (offset_x + width).min(tile_x + 8);
            let top = offset_y.max(tile_y);
            let bottom = (offset_y + height).min(tile_y + 8);
            for row in top..bottom {
                let src = (row - tile_y) * 8 + left - tile_x;
                let dst = (row - offset_y) * width + left - offset_x;
                output[dst..dst + right - left].copy_from_slice(&tile[src..src + right - left]);
            }
        }
    }
}

fn setup_warp_shear(params: &LocalWarpParams) -> Option<(i64, i64, i64, i64)> {
    const MODEL_BITS: i64 = 16;
    const PARAM_REDUCE_BITS: u32 = 6;
    const NONDIAG_CLAMP: i64 = 1 << 15;
    let alpha = (params.alpha - (1 << MODEL_BITS)).clamp(-NONDIAG_CLAMP, NONDIAG_CLAMP - 1);
    let beta = params.beta.clamp(-NONDIAG_CLAMP, NONDIAG_CLAMP - 1);
    if params.alpha <= 0 {
        return None;
    }
    let gamma = warp_divide(params.gamma, params.alpha, MODEL_BITS as u32)?
        .clamp(-NONDIAG_CLAMP, NONDIAG_CLAMP - 1);
    let delta = (params.delta
        - warp_divide(params.beta.checked_mul(params.gamma)?, params.alpha, 0)?
        - (1 << MODEL_BITS))
        .clamp(-NONDIAG_CLAMP, NONDIAG_CLAMP - 1);
    let reduce =
        |value: i64| round_div_power_of_two_signed(value, PARAM_REDUCE_BITS) << PARAM_REDUCE_BITS;
    let (alpha, beta, gamma, delta) = (reduce(alpha), reduce(beta), reduce(gamma), reduce(delta));
    if 4 * alpha.abs() + 7 * beta.abs() >= (1 << MODEL_BITS)
        || 4 * gamma.abs() + 4 * delta.abs() >= (1 << MODEL_BITS)
    {
        return None;
    }
    Some((alpha, beta, gamma, delta))
}

#[inline]
fn round_div_power_of_two_signed(value: i64, bits: u32) -> i64 {
    if bits == 0 {
        value
    } else {
        let half = 1_i64 << (bits - 1);
        if value < 0 {
            -((-value + half) >> bits)
        } else {
            (value + half) >> bits
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "AV1 warped block filtering keeps geometry and caller scratch explicit"
)]
fn predict_warped_block_into(
    output: &mut [u16],
    reference: &PlaneBuffer,
    origin_x: usize,
    origin_y: usize,
    output_width: usize,
    output_height: usize,
    row: usize,
    col: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    params: &LocalWarpParams,
    alpha: i64,
    beta: i64,
    gamma: i64,
    delta: i64,
    inter_round0: u32,
    inter_round1: u32,
    bit_depth: u8,
    intermediate: &mut [i64; 15 * 8],
) {
    const MODEL_BITS: u32 = 16;
    const DIFF_BITS: u32 = 10;
    const PIXEL_SHIFT: i64 = 64;
    let src_x = i64::try_from(origin_x + col + 4)
        .ok()
        .and_then(|value| value.checked_shl(u32::try_from(subsampling_x).ok()?))
        .unwrap_or(i64::MAX);
    let src_y = i64::try_from(origin_y + row + 4)
        .ok()
        .and_then(|value| value.checked_shl(u32::try_from(subsampling_y).ok()?))
        .unwrap_or(i64::MAX);
    let dst_x = params
        .alpha
        .saturating_mul(src_x)
        .saturating_add(params.beta.saturating_mul(src_y))
        .saturating_add(params.translation_x);
    let dst_y = params
        .gamma
        .saturating_mul(src_x)
        .saturating_add(params.delta.saturating_mul(src_y))
        .saturating_add(params.translation_y);
    let x4 = dst_x >> subsampling_x;
    let y4 = dst_y >> subsampling_y;
    let ix4 = floor_div_power_of_two(x4, MODEL_BITS);
    let iy4 = floor_div_power_of_two(y4, MODEL_BITS);
    let sx4 = ((x4 & ((1_i64 << MODEL_BITS) - 1)) - 4 * alpha - 4 * beta) & !63;
    let sy4 = ((y4 & ((1_i64 << MODEL_BITS) - 1)) - 4 * gamma - 4 * delta) & !63;
    let offset_bits_horiz = u32::from(bit_depth) + 7 - 1;
    let offset_bits_vert = u32::from(bit_depth) + 14 - inter_round0;
    for k in -7_i64..8 {
        let iy = (iy4 + k).clamp(0, reference.layout.height.saturating_sub(1) as i64) as usize;
        for l in -4_i64..4 {
            let sx = sx4 + alpha * (l + 4) + beta * (k + 4);
            let offset = (round_div_power_of_two_unsigned(sx, DIFF_BITS) + PIXEL_SHIFT)
                .clamp(0, 192) as usize;
            let coeffs = &AV1_WARPED_FILTERS[offset];
            let mut sum = 1_i64 << offset_bits_horiz;
            for (tap, coeff) in coeffs.iter().enumerate() {
                let ix = (ix4 + l - 3 + tap as i64)
                    .clamp(0, reference.layout.width.saturating_sub(1) as i64)
                    as usize;
                sum += i64::from(*coeff)
                    * i64::from(reference.samples[iy * reference.layout.width + ix]);
            }
            intermediate[(k + 7) as usize * 8 + (l + 4) as usize] =
                round_div_power_of_two_unsigned(sum, inter_round0);
        }
    }
    let chunk_height = output_height.saturating_sub(row).min(8);
    let chunk_width = output_width.saturating_sub(col).min(8);
    for out_row in 0..chunk_height {
        let k = out_row as i64 - 4;
        for out_col in 0..chunk_width {
            let l = out_col as i64 - 4;
            let sy = sy4 + delta * (k + 4) + gamma * (l + 4);
            let offset = (round_div_power_of_two_unsigned(sy, DIFF_BITS) + PIXEL_SHIFT)
                .clamp(0, 192) as usize;
            let coeffs = &AV1_WARPED_FILTERS[offset];
            let mut sum = 1_i64 << offset_bits_vert;
            for (tap, coeff) in coeffs.iter().enumerate() {
                sum += i64::from(*coeff)
                    * intermediate[(k + tap as i64 + 4) as usize * 8 + (l + 4) as usize];
            }
            let value = round_div_power_of_two_unsigned(sum, inter_round1);
            let pixel = if inter_round1 == 7 {
                // Compound warp uses the same biased convolution buffer as translation.
                debug_assert!((0..=i64::from(u16::MAX)).contains(&value));
                value as u16
            } else {
                (value - (1_i64 << (u32::from(bit_depth) - 1)) - (1_i64 << u32::from(bit_depth)))
                    .clamp(0, (1_i64 << bit_depth.min(16)) - 1) as u16
            };
            output[(row + out_row) * output_width + col + out_col] = pixel;
        }
    }
}

#[inline]
fn round_div_power_of_two_unsigned(value: i64, bits: u32) -> i64 {
    if bits == 0 {
        value
    } else {
        (value + (1_i64 << (bits - 1))) >> bits
    }
}

#[derive(Clone, Copy)]
struct LocalWarpParams {
    translation_x: i64,
    translation_y: i64,
    alpha: i64,
    beta: i64,
    gamma: i64,
    delta: i64,
}

fn estimate_local_warp_params(
    block_x_luma: i64,
    block_y_luma: i64,
    block_width_luma: usize,
    block_height_luma: usize,
    base_mv: (i32, i32),
    samples: [Option<LocalWarpSample>; 8],
) -> Option<LocalWarpParams> {
    const WARPEDMODEL_PREC_BITS: u32 = 16;
    const LS_MV_MAX: i64 = 256;
    const NONDIAG_CLAMP: i64 = 1 << 13;
    const TRANS_CLAMP: i64 = 1 << 23;
    let mid_x = block_x_luma.checked_add(i64::try_from(block_width_luma / 2).ok()?)? - 1;
    let mid_y = block_y_luma.checked_add(i64::try_from(block_height_luma / 2).ok()?)? - 1;
    let sux = mid_x.checked_mul(8)?;
    let suy = mid_y.checked_mul(8)?;
    let dux = sux.checked_add(i64::from(base_mv.1))?;
    let duy = suy.checked_add(i64::from(base_mv.0))?;
    let mut a00 = 0_i64;
    let mut a01 = 0_i64;
    let mut a11 = 0_i64;
    let mut bx0 = 0_i64;
    let mut bx1 = 0_i64;
    let mut by0 = 0_i64;
    let mut by1 = 0_i64;
    let mut used = 0usize;
    for sample in samples.into_iter().flatten() {
        let sx = i64::from(sample.source.1).checked_sub(sux)?;
        let sy = i64::from(sample.source.0).checked_sub(suy)?;
        let dx = i64::from(sample.destination.1).checked_sub(dux)?;
        let dy = i64::from(sample.destination.0).checked_sub(duy)?;
        if (sx - dx).abs() >= LS_MV_MAX || (sy - dy).abs() >= LS_MV_MAX {
            continue;
        }
        a00 = a00.checked_add(ls_product(sx, sx)?.checked_add(8)?)?;
        a01 = a01.checked_add(ls_product(sx, sy)?.checked_add(4)?)?;
        a11 = a11.checked_add(ls_product(sy, sy)?.checked_add(8)?)?;
        bx0 = bx0.checked_add(ls_product(sx, dx)?.checked_add(8)?)?;
        bx1 = bx1.checked_add(ls_product(sy, dx)?.checked_add(4)?)?;
        by0 = by0.checked_add(ls_product(sx, dy)?.checked_add(4)?)?;
        by1 = by1.checked_add(ls_product(sy, dy)?.checked_add(8)?)?;
        used += 1;
    }
    if used == 0 {
        return None;
    }
    let determinant = a00.checked_mul(a11)?.checked_sub(a01.checked_mul(a01)?)?;
    if determinant == 0 {
        return None;
    }
    let alpha = warp_divide(
        a11.checked_mul(bx0)?.checked_sub(a01.checked_mul(bx1)?)?,
        determinant,
        WARPEDMODEL_PREC_BITS,
    )?
    .clamp(
        (1 << WARPEDMODEL_PREC_BITS) - NONDIAG_CLAMP + 1,
        (1 << WARPEDMODEL_PREC_BITS) + NONDIAG_CLAMP - 1,
    );
    let beta = warp_divide(
        a00.checked_mul(bx1)?.checked_sub(a01.checked_mul(bx0)?)?,
        determinant,
        WARPEDMODEL_PREC_BITS,
    )?
    .clamp(-NONDIAG_CLAMP + 1, NONDIAG_CLAMP - 1);
    let gamma = warp_divide(
        a11.checked_mul(by0)?.checked_sub(a01.checked_mul(by1)?)?,
        determinant,
        WARPEDMODEL_PREC_BITS,
    )?
    .clamp(-NONDIAG_CLAMP + 1, NONDIAG_CLAMP - 1);
    let delta = warp_divide(
        a00.checked_mul(by1)?.checked_sub(a01.checked_mul(by0)?)?,
        determinant,
        WARPEDMODEL_PREC_BITS,
    )?
    .clamp(
        (1 << WARPEDMODEL_PREC_BITS) - NONDIAG_CLAMP + 1,
        (1 << WARPEDMODEL_PREC_BITS) + NONDIAG_CLAMP - 1,
    );
    let translation_x = (i64::from(base_mv.1) << (WARPEDMODEL_PREC_BITS - 3))
        .checked_sub(
            mid_x
                .checked_mul(alpha - (1 << WARPEDMODEL_PREC_BITS))?
                .checked_add(mid_y.checked_mul(beta)?)?,
        )?
        .clamp(-TRANS_CLAMP, TRANS_CLAMP - 1);
    let translation_y = (i64::from(base_mv.0) << (WARPEDMODEL_PREC_BITS - 3))
        .checked_sub(
            mid_x
                .checked_mul(gamma)?
                .checked_add(mid_y.checked_mul(delta - (1 << WARPEDMODEL_PREC_BITS))?)?,
        )?
        .clamp(-TRANS_CLAMP, TRANS_CLAMP - 1);
    Some(LocalWarpParams {
        translation_x,
        translation_y,
        alpha,
        beta,
        gamma,
        delta,
    })
}

fn ls_product(a: i64, b: i64) -> Option<i64> {
    a.checked_mul(b)?
        .checked_shr(2)?
        .checked_add(a)?
        .checked_add(b)
}

// AV1 ResolveDivisor quantizes the reciprocal to the normative 257-entry
// table. Exact division changes affine coefficients and sampled filter phases.
fn warp_divide(numerator: i64, denominator: i64, precision: u32) -> Option<i64> {
    let magnitude = denominator.unsigned_abs();
    if magnitude == 0 {
        return None;
    }
    let msb = 63 - magnitude.leading_zeros();
    let remainder = magnitude - (1u64 << msb);
    let index = if msb > 8 {
        let bits = msb - 8;
        (remainder + (1u64 << (bits - 1))) >> bits
    } else {
        remainder << (8 - msb)
    };
    // round(2^22 / (256+index)) reproduces every entry of AV1's div_lut.
    let divisor = 256 + index;
    let reciprocal = ((1u64 << 22) + divisor / 2) / divisor;
    let sign = if denominator < 0 { -1i128 } else { 1 };
    let value = i128::from(numerator) * i128::from(reciprocal) * sign;
    let shift = msb + 14;
    let result = if shift > precision {
        let bits = shift - precision;
        let half = 1i128 << (bits - 1);
        if value < 0 {
            -((-value + half) >> bits)
        } else {
            (value + half) >> bits
        }
    } else {
        value << (precision - shift)
    };
    i64::try_from(result).ok()
}

#[cfg(test)]
fn fixed_warp_to_plane_eighth(value: i64, subsampling: usize) -> i64 {
    let shift = 13_u32.saturating_add(u32::try_from(subsampling).unwrap_or(0));
    floor_div_power_of_two(value, shift)
}

fn floor_div_power_of_two(value: i64, shift: u32) -> i64 {
    if shift == 0 {
        return value;
    }
    let divisor = 1_i64.checked_shl(shift).unwrap_or(i64::MAX);
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder < 0 {
        quotient - 1
    } else {
        quotient
    }
}

fn predict_inter_block_into(
    reference: &PlaneBuffer,
    secondary_reference: Option<&PlaneBuffer>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    mv: (i32, i32),
    secondary_mv: Option<(i32, i32)>,
    subsampling_x: usize,
    subsampling_y: usize,
    interpolation_filters: (InterpolationFilter, InterpolationFilter),
    compound_weight: Option<u8>,
    compound_mask: Option<CompoundMask>,
    bit_depth: u8,
    mask_geometry: MaskGeometry,
    output: &mut [u16],
) -> Result<(), DecoderError> {
    let expected_len = width.checked_mul(height).ok_or_else(|| {
        DecoderError::Bitstream("AV1 inter prediction size overflows".to_string())
    })?;
    if output.len() != expected_len {
        return Err(DecoderError::Bitstream(
            "AV1 inter prediction buffer has an invalid size".to_string(),
        ));
    }
    let mv_x = motion_component_q4(mv.1, subsampling_x);
    let mv_y = motion_component_q4(mv.0, subsampling_y);
    let secondary_mv = secondary_mv.unwrap_or(mv);
    let secondary_mv_x = motion_component_q4(secondary_mv.1, subsampling_x);
    let secondary_mv_y = motion_component_q4(secondary_mv.0, subsampling_y);
    let integer_mv = mv_x % 16 == 0
        && mv_y % 16 == 0
        && (secondary_reference.is_none()
            || (secondary_mv_x % 16 == 0 && secondary_mv_y % 16 == 0));
    if integer_mv {
        let source_x = i64::try_from(x)
            .ok()
            .and_then(|value| value.checked_add(mv_x / 16))
            .ok_or_else(|| DecoderError::Bitstream("AV1 inter source x overflows".to_string()))?;
        let source_y = i64::try_from(y)
            .ok()
            .and_then(|value| value.checked_add(mv_y / 16))
            .ok_or_else(|| DecoderError::Bitstream("AV1 inter source y overflows".to_string()))?;
        let source_in_bounds = source_x >= 0
            && source_y >= 0
            && source_x.saturating_add(width as i64) <= reference.layout.width as i64
            && source_y.saturating_add(height as i64) <= reference.layout.height as i64;
        if !source_in_bounds {
            return predict_inter_block_into_fractional(
                reference,
                secondary_reference,
                x,
                y,
                width,
                height,
                mv,
                Some(secondary_mv),
                subsampling_x,
                subsampling_y,
                interpolation_filters,
                compound_weight,
                compound_mask,
                bit_depth,
                mask_geometry,
                output,
            );
        }
        let source_x = source_x as usize;
        let source_y = source_y as usize;
        if let Some(secondary) = secondary_reference {
            let secondary_source_x = i64::try_from(x)
                .ok()
                .and_then(|value| value.checked_add(secondary_mv_x / 16))
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 secondary source x overflows".to_string())
                })?;
            let secondary_source_y = i64::try_from(y)
                .ok()
                .and_then(|value| value.checked_add(secondary_mv_y / 16))
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 secondary source y overflows".to_string())
                })?;
            let secondary_right =
                secondary_source_x
                    .checked_add(width as i64)
                    .ok_or_else(|| {
                        DecoderError::Bitstream("AV1 secondary source x overflows".to_string())
                    })?;
            let secondary_bottom =
                secondary_source_y
                    .checked_add(height as i64)
                    .ok_or_else(|| {
                        DecoderError::Bitstream("AV1 secondary source y overflows".to_string())
                    })?;
            if secondary_source_x < 0
                || secondary_source_y < 0
                || secondary_right > secondary.layout.width as i64
                || secondary_bottom > secondary.layout.height as i64
            {
                return predict_inter_block_into_fractional(
                    reference,
                    secondary_reference,
                    x,
                    y,
                    width,
                    height,
                    mv,
                    Some(secondary_mv),
                    subsampling_x,
                    subsampling_y,
                    interpolation_filters,
                    compound_weight,
                    compound_mask,
                    bit_depth,
                    mask_geometry,
                    output,
                );
            }
            let secondary_source_x = secondary_source_x as usize;
            let secondary_source_y = secondary_source_y as usize;
            for row in 0..height {
                let source_start = (source_y + row) * reference.layout.width + source_x;
                let secondary_start =
                    (secondary_source_y + row) * secondary.layout.width + secondary_source_x;
                let target_start = row * width;
                for col in 0..width {
                    output[target_start + col] = blend_compound_prediction(
                        reference.samples[source_start + col],
                        secondary.samples[secondary_start + col],
                        compound_weight,
                        compound_mask,
                        bit_depth,
                        mask_geometry.x + col,
                        mask_geometry.y + row,
                        mask_geometry.width,
                        mask_geometry.height,
                    );
                }
            }
            return Ok(());
        }
        for row in 0..height {
            let source_start = (source_y + row) * reference.layout.width + source_x;
            let target_start = row * width;
            output[target_start..target_start + width]
                .copy_from_slice(&reference.samples[source_start..source_start + width]);
        }
        return Ok(());
    }

    // The integer path above is allocation-free. Fractional motion uses the
    // dimension-specific AV1 separable kernels; keeping the source
    // coordinate arrays per block avoids redoing fixed-point division for
    // every row/column sample.
    const MAX_INTER_BLOCK_DIMENSION: usize = 128;
    if width > MAX_INTER_BLOCK_DIMENSION || height > MAX_INTER_BLOCK_DIMENSION {
        return Err(DecoderError::Bitstream(
            "AV1 inter prediction block exceeds the supported dimension".to_string(),
        ));
    }
    let base_x = (i64::try_from(x).unwrap_or(i64::MAX) << 4)
        .checked_add(mv_x)
        .ok_or_else(|| DecoderError::Bitstream("AV1 inter source x overflows".to_string()))?;
    let base_y = (i64::try_from(y).unwrap_or(i64::MAX) << 4)
        .checked_add(mv_y)
        .ok_or_else(|| DecoderError::Bitstream("AV1 inter source y overflows".to_string()))?;
    let mut x0 = [0i64; MAX_INTER_BLOCK_DIMENSION];
    let mut fx = [0i64; MAX_INTER_BLOCK_DIMENSION];
    for col in 0..width {
        let fixed = base_x + col as i64 * 16;
        let source = floor_div_sixteen(fixed);
        x0[col] = source;
        fx[col] = fixed - source * 16;
    }
    let mut y0 = [0i64; MAX_INTER_BLOCK_DIMENSION];
    let mut fy = [0i64; MAX_INTER_BLOCK_DIMENSION];
    for row in 0..height {
        let fixed = base_y + row as i64 * 16;
        let source = floor_div_sixteen(fixed);
        y0[row] = source;
        fy[row] = fixed - source * 16;
    }
    if let Some(secondary) = secondary_reference {
        let secondary_base_x = (i64::try_from(x).unwrap_or(i64::MAX) << 4)
            .checked_add(secondary_mv_x)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 secondary inter source x overflows".to_string())
            })?;
        let secondary_base_y = (i64::try_from(y).unwrap_or(i64::MAX) << 4)
            .checked_add(secondary_mv_y)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 secondary inter source y overflows".to_string())
            })?;
        let mut secondary_x0 = [0i64; MAX_INTER_BLOCK_DIMENSION];
        let mut secondary_fx = [0i64; MAX_INTER_BLOCK_DIMENSION];
        for col in 0..width {
            let fixed = secondary_base_x + col as i64 * 16;
            let source = floor_div_sixteen(fixed);
            secondary_x0[col] = source;
            secondary_fx[col] = fixed - source * 16;
        }
        let mut secondary_y0 = [0i64; MAX_INTER_BLOCK_DIMENSION];
        let mut secondary_fy = [0i64; MAX_INTER_BLOCK_DIMENSION];
        for row in 0..height {
            let fixed = secondary_base_y + row as i64 * 16;
            let source = floor_div_sixteen(fixed);
            secondary_y0[row] = source;
            secondary_fy[row] = fixed - source * 16;
        }
        for row in 0..height {
            for col in 0..width {
                output[row * width + col] = blend_compound_intermediates(
                    predict_compound_sample(
                        reference,
                        x0[col],
                        y0[row],
                        fx[col],
                        fy[row],
                        interpolation_filters,
                        bit_depth,
                        (mask_geometry.width, mask_geometry.height),
                    ),
                    predict_compound_sample(
                        secondary,
                        secondary_x0[col],
                        secondary_y0[row],
                        secondary_fx[col],
                        secondary_fy[row],
                        interpolation_filters,
                        bit_depth,
                        (mask_geometry.width, mask_geometry.height),
                    ),
                    compound_weight,
                    compound_mask,
                    bit_depth,
                    mask_geometry.x + col,
                    mask_geometry.y + row,
                    mask_geometry.width,
                    mask_geometry.height,
                );
            }
        }
    } else {
        for row in 0..height {
            for col in 0..width {
                output[row * width + col] = predict_inter_sample(
                    reference,
                    x0[col],
                    y0[row],
                    fx[col],
                    fy[row],
                    interpolation_filters,
                    bit_depth,
                    (mask_geometry.width, mask_geometry.height),
                );
            }
        }
    }
    Ok(())
}

const WEDGE_MASTER_EVEN: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 11,
    27, 46, 58, 62, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64,
];
const WEDGE_MASTER_ODD: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 6,
    18, 37, 53, 60, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64,
];
const WEDGE_MASTER_VERTICAL: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7,
    21, 43, 57, 62, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64,
];

#[derive(Clone, Copy)]
struct WedgeCode {
    direction: u8,
    x_offset: u8,
    y_offset: u8,
}

const WEDGE_CODEBOOK_HGTW: [WedgeCode; 16] = [
    WedgeCode {
        direction: 0,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 5,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 0,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 0,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 1,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 6,
        y_offset: 4,
    },
];
const WEDGE_CODEBOOK_HLTW: [WedgeCode; 16] = [
    WedgeCode {
        direction: 0,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 5,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 5,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 5,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 0,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 0,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 1,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 6,
        y_offset: 4,
    },
];
const WEDGE_CODEBOOK_HEQW: [WedgeCode; 16] = [
    WedgeCode {
        direction: 0,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 5,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 5,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 0,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 0,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 1,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 6,
        y_offset: 4,
    },
];

fn wedge_code(width: usize, height: usize, index: usize) -> Option<(WedgeCode, bool)> {
    let (codes, sign_flips) = match (width, height) {
        (8, 32) => (
            &WEDGE_CODEBOOK_HGTW,
            &[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1][..],
        ),
        (32, 8) => (
            &WEDGE_CODEBOOK_HLTW,
            &[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1][..],
        ),
        (8, 16) | (16, 32) => (
            &WEDGE_CODEBOOK_HGTW,
            &[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1][..],
        ),
        (16, 8) | (32, 16) => (
            &WEDGE_CODEBOOK_HLTW,
            &[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1][..],
        ),
        (8, 8) | (16, 16) | (32, 32) => (
            &WEDGE_CODEBOOK_HEQW,
            &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1][..],
        ),
        _ => return None,
    };
    codes
        .get(index)
        .copied()
        .zip(sign_flips.get(index).copied().map(|v| v != 0))
}

#[inline]
fn wedge_oblique63(row: usize, column: usize) -> u8 {
    let shift = 16_i32 - i32::try_from(row.div_ceil(2)).unwrap_or(i32::MAX);
    let source = if row & 1 == 0 {
        &WEDGE_MASTER_EVEN
    } else {
        &WEDGE_MASTER_ODD
    };
    let source_index = (i32::try_from(column).unwrap_or(i32::MAX) - shift).clamp(0, 63);
    source[usize::try_from(source_index).unwrap_or(63)]
}

#[inline]
fn wedge_master_value(direction: u8, inverse: bool, row: usize, column: usize) -> u8 {
    let row = row.min(63);
    let column = column.min(63);
    let (value, base_inverse) = match direction {
        0 => (wedge_oblique63(column, row), false),
        1 => (wedge_oblique63(row, column), false),
        2 => (wedge_oblique63(row, 63 - column), true),
        3 => (wedge_oblique63(column, 63 - row), true),
        4 => (WEDGE_MASTER_VERTICAL[row], false),
        5 => (WEDGE_MASTER_VERTICAL[column], false),
        _ => return 32,
    };
    if inverse ^ base_inverse {
        64 - value
    } else {
        value
    }
}

#[inline]
fn wedge_mask_value(
    width: usize,
    height: usize,
    index: u8,
    inverse: bool,
    x: usize,
    y: usize,
) -> Option<u8> {
    let (code, sign_flip) = wedge_code(width, height, usize::from(index))?;
    let woff = (usize::from(code.x_offset) * width) >> 3;
    let hoff = (usize::from(code.y_offset) * height) >> 3;
    Some(wedge_master_value(
        code.direction,
        inverse ^ sign_flip,
        32 - hoff + y,
        32 - woff + x,
    ))
}

// All weights operate on unrounded convolution values. The bias is common
// to both predictors and is removed once, immediately before final rounding.
#[inline]
fn blend_compound_intermediates(
    first: u16,
    second: u16,
    primary_weight: Option<u8>,
    compound_mask: Option<CompoundMask>,
    bit_depth: u8,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> u16 {
    let bits = compound_round_bits(bit_depth);
    let mask = match compound_mask {
        Some(CompoundMask::DifferenceWeighted { inverse }) => Some(compound_mask::difference_mask(
            first, second, bit_depth, inverse,
        )),
        Some(CompoundMask::Wedge { index, inverse }) => {
            wedge_mask_value(width, height, index, inverse, x, y)
        }
        None => None,
    };
    let value = if let Some(weight) = mask.or(primary_weight) {
        (i64::from(first) * i64::from(weight) + i64::from(second) * i64::from(64 - weight)) >> 6
    } else {
        (i64::from(first) + i64::from(second)) >> 1
    };
    round_div_power_of_two_unsigned(value - compound_offset(bit_depth), bits)
        .clamp(0, (1_i64 << bit_depth.min(16)) - 1) as u16
}

fn predict_compound_global_or_translation_into(
    output: &mut [u16],
    reference: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    mv: (i32, i32),
    subsampling_x: usize,
    subsampling_y: usize,
    filters: (InterpolationFilter, InterpolationFilter),
    bit_depth: u8,
    geometry: MaskGeometry,
    allow_high_precision_mv: bool,
    matrix: Option<[i32; 6]>,
) {
    if let Some(matrix) = matrix
        && predict_global_motion_block_anchored_into(
            output,
            reference,
            x,
            y,
            width,
            height,
            subsampling_x,
            subsampling_y,
            bit_depth,
            allow_high_precision_mv,
            matrix,
            true,
            x - geometry.x,
            y - geometry.y,
        )
    {
        return;
    }
    predict_compound_reference_into(
        reference,
        x,
        y,
        width,
        height,
        mv,
        subsampling_x,
        subsampling_y,
        filters,
        bit_depth,
        geometry,
        output,
    );
}

fn predict_compound_reference_into(
    reference: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    mv: (i32, i32),
    subsampling_x: usize,
    subsampling_y: usize,
    filters: (InterpolationFilter, InterpolationFilter),
    bit_depth: u8,
    geometry: MaskGeometry,
    output: &mut [u16],
) {
    let mx = motion_component_q4(mv.1, subsampling_x);
    let my = motion_component_q4(mv.0, subsampling_y);
    for row in 0..height {
        let fixed_y = ((y + row) as i64) * 16 + my;
        let sy = floor_div_sixteen(fixed_y);
        for col in 0..width {
            let fixed_x = ((x + col) as i64) * 16 + mx;
            let sx = floor_div_sixteen(fixed_x);
            output[row * width + col] = predict_compound_sample(
                reference,
                sx,
                sy,
                fixed_x - sx * 16,
                fixed_y - sy * 16,
                filters,
                bit_depth,
                (geometry.width, geometry.height),
            );
        }
    }
}

#[inline]
fn blend_compound_prediction(
    first: u16,
    second: u16,
    primary_weight: Option<u8>,
    compound_mask: Option<CompoundMask>,
    bit_depth: u8,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> u16 {
    let bits = compound_round_bits(bit_depth);
    let offset = compound_offset(bit_depth);
    blend_compound_intermediates(
        ((i64::from(first) << bits) + offset) as u16,
        ((i64::from(second) << bits) + offset) as u16,
        primary_weight,
        compound_mask,
        bit_depth,
        x,
        y,
        width,
        height,
    )
}

fn predict_inter_block_into_fractional(
    reference: &PlaneBuffer,
    secondary_reference: Option<&PlaneBuffer>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    mv: (i32, i32),
    secondary_mv: Option<(i32, i32)>,
    subsampling_x: usize,
    subsampling_y: usize,
    interpolation_filters: (InterpolationFilter, InterpolationFilter),
    compound_weight: Option<u8>,
    compound_mask: Option<CompoundMask>,
    bit_depth: u8,
    mask_geometry: MaskGeometry,
    output: &mut [u16],
) -> Result<(), DecoderError> {
    let mv_x = motion_component_q4(mv.1, subsampling_x);
    let mv_y = motion_component_q4(mv.0, subsampling_y);
    let secondary_mv = secondary_mv.unwrap_or(mv);
    let secondary_mv_x = motion_component_q4(secondary_mv.1, subsampling_x);
    let secondary_mv_y = motion_component_q4(secondary_mv.0, subsampling_y);
    for row in 0..height {
        let y_fixed = (i64::try_from(y).unwrap_or(i64::MAX) << 4)
            .checked_add(mv_y)
            .ok_or_else(|| DecoderError::Bitstream("AV1 inter source y overflows".to_string()))?
            + (row as i64 * 16);
        let secondary_y_fixed = (i64::try_from(y).unwrap_or(i64::MAX) << 4)
            .checked_add(secondary_mv_y)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 secondary inter source y overflows".to_string())
            })?
            + (row as i64 * 16);
        let y0 = floor_div_sixteen(y_fixed);
        let fy = y_fixed - y0 * 16;
        let secondary_y0 = floor_div_sixteen(secondary_y_fixed);
        let secondary_fy = secondary_y_fixed - secondary_y0 * 16;
        for col in 0..width {
            let x_fixed = (i64::try_from(x).unwrap_or(i64::MAX) << 4)
                .checked_add(mv_x)
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 inter source x overflows".to_string())
                })?
                + (col as i64 * 16);
            let secondary_x_fixed = (i64::try_from(x).unwrap_or(i64::MAX) << 4)
                .checked_add(secondary_mv_x)
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 secondary inter source x overflows".to_string())
                })?
                + (col as i64 * 16);
            let x0 = floor_div_sixteen(x_fixed);
            let fx = x_fixed - x0 * 16;
            let secondary_x0 = floor_div_sixteen(secondary_x_fixed);
            let secondary_fx = secondary_x_fixed - secondary_x0 * 16;
            let first = if secondary_reference.is_some() {
                predict_compound_sample(
                    reference,
                    x0,
                    y0,
                    fx,
                    fy,
                    interpolation_filters,
                    bit_depth,
                    (mask_geometry.width, mask_geometry.height),
                )
            } else {
                predict_inter_sample(
                    reference,
                    x0,
                    y0,
                    fx,
                    fy,
                    interpolation_filters,
                    bit_depth,
                    (mask_geometry.width, mask_geometry.height),
                )
            };
            output[row * width + col] = secondary_reference
                .map(|secondary| {
                    blend_compound_intermediates(
                        first,
                        predict_compound_sample(
                            secondary,
                            secondary_x0,
                            secondary_y0,
                            secondary_fx,
                            secondary_fy,
                            interpolation_filters,
                            bit_depth,
                            (mask_geometry.width, mask_geometry.height),
                        ),
                        compound_weight,
                        compound_mask,
                        bit_depth,
                        mask_geometry.x + col,
                        mask_geometry.y + row,
                        mask_geometry.width,
                        mask_geometry.height,
                    )
                })
                .unwrap_or(first);
        }
    }
    Ok(())
}

const AV1_REGULAR_SUBPEL_FILTERS: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, -6, 126, 8, -2, 0, 0],
    [0, 2, -10, 122, 18, -4, 0, 0],
    [0, 2, -12, 116, 28, -8, 2, 0],
    [0, 2, -14, 110, 38, -10, 2, 0],
    [0, 2, -14, 102, 48, -12, 2, 0],
    [0, 2, -16, 94, 58, -12, 2, 0],
    [0, 2, -14, 84, 66, -12, 2, 0],
    [0, 2, -14, 76, 76, -14, 2, 0],
    [0, 2, -12, 66, 84, -14, 2, 0],
    [0, 2, -12, 58, 94, -16, 2, 0],
    [0, 2, -12, 48, 102, -14, 2, 0],
    [0, 2, -10, 38, 110, -14, 2, 0],
    [0, 2, -8, 28, 116, -12, 2, 0],
    [0, 0, -4, 18, 122, -10, 2, 0],
    [0, 0, -2, 8, 126, -6, 2, 0],
];

const AV1_SMOOTH_SUBPEL_FILTERS: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, 28, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, -2, 16, 54, 48, 12, 0, 0],
    [0, -2, 14, 52, 52, 14, -2, 0],
    [0, 0, 12, 48, 54, 16, -2, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 28, 2, 0],
];

const AV1_SHARP_SUBPEL_FILTERS: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-2, 2, -6, 126, 8, -2, 2, 0],
    [-2, 6, -12, 124, 16, -6, 4, -2],
    [-2, 8, -18, 120, 26, -10, 6, -2],
    [-4, 10, -22, 116, 38, -14, 6, -2],
    [-4, 10, -22, 108, 48, -18, 8, -2],
    [-4, 10, -24, 100, 60, -20, 8, -2],
    [-4, 10, -24, 90, 70, -22, 10, -2],
    [-4, 12, -24, 80, 80, -24, 12, -4],
    [-2, 10, -22, 70, 90, -24, 10, -4],
    [-2, 8, -20, 60, 100, -24, 10, -4],
    [-2, 8, -18, 48, 108, -22, 10, -4],
    [-2, 6, -14, 38, 116, -22, 10, -4],
    [-2, 6, -10, 26, 120, -18, 8, -2],
    [-2, 4, -6, 16, 124, -12, 6, -2],
    [0, 2, -2, 8, 126, -6, 2, -2],
];

const AV1_BILINEAR_SUBPEL_FILTERS: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 0, 120, 8, 0, 0, 0],
    [0, 0, 0, 112, 16, 0, 0, 0],
    [0, 0, 0, 104, 24, 0, 0, 0],
    [0, 0, 0, 96, 32, 0, 0, 0],
    [0, 0, 0, 88, 40, 0, 0, 0],
    [0, 0, 0, 80, 48, 0, 0, 0],
    [0, 0, 0, 72, 56, 0, 0, 0],
    [0, 0, 0, 64, 64, 0, 0, 0],
    [0, 0, 0, 56, 72, 0, 0, 0],
    [0, 0, 0, 48, 80, 0, 0, 0],
    [0, 0, 0, 40, 88, 0, 0, 0],
    [0, 0, 0, 32, 96, 0, 0, 0],
    [0, 0, 0, 24, 104, 0, 0, 0],
    [0, 0, 0, 16, 112, 0, 0, 0],
    [0, 0, 0, 8, 120, 0, 0, 0],
];

// AV1 subpel-filter tables for prediction dimensions of four pixels or less.
// Regular and Sharp share the reduced-tap bank (AOM av1/common/filter.h).
const AV1_SMALL_REGULAR_SUBPEL_FILTERS: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -4, 126, 8, -2, 0, 0],
    [0, 0, -8, 122, 18, -4, 0, 0],
    [0, 0, -10, 116, 28, -6, 0, 0],
    [0, 0, -12, 110, 38, -8, 0, 0],
    [0, 0, -12, 102, 48, -10, 0, 0],
    [0, 0, -14, 94, 58, -10, 0, 0],
    [0, 0, -12, 84, 66, -10, 0, 0],
    [0, 0, -12, 76, 76, -12, 0, 0],
    [0, 0, -10, 66, 84, -12, 0, 0],
    [0, 0, -10, 58, 94, -14, 0, 0],
    [0, 0, -10, 48, 102, -12, 0, 0],
    [0, 0, -8, 38, 110, -12, 0, 0],
    [0, 0, -6, 28, 116, -10, 0, 0],
    [0, 0, -4, 18, 122, -8, 0, 0],
    [0, 0, -2, 8, 126, -4, 0, 0],
];

const AV1_SMALL_SMOOTH_SUBPEL_FILTERS: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 30, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, 0, 14, 54, 48, 12, 0, 0],
    [0, 0, 12, 52, 52, 12, 0, 0],
    [0, 0, 12, 48, 54, 14, 0, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 30, 0, 0],
];

#[inline]
fn interpolation_kernel(
    filter: InterpolationFilter,
    phase: usize,
    dimension: usize,
) -> &'static [i16; 8] {
    if dimension <= 4 {
        match filter {
            InterpolationFilter::Regular
            | InterpolationFilter::Switchable
            | InterpolationFilter::Sharp => {
                return &AV1_SMALL_REGULAR_SUBPEL_FILTERS[phase];
            }
            InterpolationFilter::Smooth => return &AV1_SMALL_SMOOTH_SUBPEL_FILTERS[phase],
            InterpolationFilter::Bilinear => {}
        }
    }
    match filter {
        InterpolationFilter::Regular | InterpolationFilter::Switchable => {
            &AV1_REGULAR_SUBPEL_FILTERS[phase]
        }
        InterpolationFilter::Smooth => &AV1_SMOOTH_SUBPEL_FILTERS[phase],
        InterpolationFilter::Sharp => &AV1_SHARP_SUBPEL_FILTERS[phase],
        InterpolationFilter::Bilinear => &AV1_BILINEAR_SUBPEL_FILTERS[phase],
    }
}

#[inline]
pub(super) fn predict_inter_sample(
    plane: &PlaneBuffer,
    source_x: i64,
    source_y: i64,
    subpel_x: i64,
    subpel_y: i64,
    interpolation_filters: (InterpolationFilter, InterpolationFilter),
    bit_depth: u8,
    prediction_dimensions: (usize, usize),
) -> u16 {
    // Bilinear coefficients are multiples of 8, so round_0=3 is exact at
    // 8/10 bits and the four-sample shortcut preserves the normative result.
    // At 12 bits round_0=5 is material, so use the staged convolution below.
    if bit_depth != 12
        && interpolation_filters == (InterpolationFilter::Bilinear, InterpolationFilter::Bilinear)
    {
        return predict_inter_sample_bilinear(
            plane, source_x, source_y, subpel_x, subpel_y, bit_depth,
        );
    }
    let round_0 = if bit_depth == 12 { 5 } else { 3 };
    interpolate_inter_sample(
        plane,
        source_x,
        source_y,
        subpel_x,
        subpel_y,
        interpolation_filters,
        bit_depth,
        prediction_dimensions,
        14 - round_0,
    )
    .clamp(0, (1_i64 << bit_depth.min(16)) - 1) as u16
}

// Compound predictors keep fractional bits and filter overshoot in the same
// biased u16 representation as AV1's convolution buffer. Clip only after blending.
fn compound_round_bits(bit_depth: u8) -> u32 {
    if bit_depth == 12 { 2 } else { 4 }
}

fn compound_offset(bit_depth: u8) -> i64 {
    3_i64 << (u32::from(bit_depth) + compound_round_bits(bit_depth) - 1)
}

#[inline]
fn predict_compound_sample(
    plane: &PlaneBuffer,
    source_x: i64,
    source_y: i64,
    subpel_x: i64,
    subpel_y: i64,
    interpolation_filters: (InterpolationFilter, InterpolationFilter),
    bit_depth: u8,
    prediction_dimensions: (usize, usize),
) -> u16 {
    let value = interpolate_inter_sample(
        plane,
        source_x,
        source_y,
        subpel_x,
        subpel_y,
        interpolation_filters,
        bit_depth,
        prediction_dimensions,
        7,
    ) + compound_offset(bit_depth);
    debug_assert!((0..=i64::from(u16::MAX)).contains(&value));
    value as u16
}

#[inline]
fn interpolate_inter_sample(
    plane: &PlaneBuffer,
    source_x: i64,
    source_y: i64,
    subpel_x: i64,
    subpel_y: i64,
    interpolation_filters: (InterpolationFilter, InterpolationFilter),
    bit_depth: u8,
    prediction_dimensions: (usize, usize),
    round_1: u32,
) -> i64 {
    // AV1 separable convolution retains 7-round_0 fractional bits
    // between the separable passes. The 12-bit path increases round_0 so the
    // offset intermediate fits 16 bits. Rounding each pass by 7 loses precision.
    let round_0 = if bit_depth == 12 { 5 } else { 3 };
    let horizontal = interpolation_kernel(
        interpolation_filters.0,
        subpel_x as usize,
        prediction_dimensions.0,
    );
    let vertical = interpolation_kernel(
        interpolation_filters.1,
        subpel_y as usize,
        prediction_dimensions.1,
    );
    let mut intermediate = [0i64; 8];
    for (row, value) in intermediate.iter_mut().enumerate() {
        let sy = (source_y + row as i64 - 3).clamp(0, plane.layout.height.saturating_sub(1) as i64)
            as usize;
        let mut sum = 0i64;
        for (tap, coefficient) in horizontal.iter().enumerate() {
            let sx = (source_x + tap as i64 - 3)
                .clamp(0, plane.layout.width.saturating_sub(1) as i64)
                as usize;
            sum += i64::from(plane.samples[sy * plane.layout.width + sx]) * i64::from(*coefficient);
        }
        // The positive offsets used by AOM cancel exactly after the vertical
        // pass; this signed form avoids carrying them through the temporary.
        *value = round_div_power_of_two_unsigned(sum, round_0);
    }
    let mut sum = 0i64;
    for (tap, coefficient) in vertical.iter().enumerate() {
        sum += intermediate[tap] * i64::from(*coefficient);
    }
    round_div_power_of_two_unsigned(sum, round_1)
}

#[inline]
fn predict_inter_sample_bilinear(
    plane: &PlaneBuffer,
    source_x: i64,
    source_y: i64,
    subpel_x: i64,
    subpel_y: i64,
    bit_depth: u8,
) -> u16 {
    let sample = |x: i64, y: i64| {
        let x = x.clamp(0, plane.layout.width.saturating_sub(1) as i64) as usize;
        let y = y.clamp(0, plane.layout.height.saturating_sub(1) as i64) as usize;
        i64::from(plane.samples[y * plane.layout.width + x])
    };
    let top =
        sample(source_x, source_y) * (16 - subpel_x) + sample(source_x + 1, source_y) * subpel_x;
    let bottom = sample(source_x, source_y + 1) * (16 - subpel_x)
        + sample(source_x + 1, source_y + 1) * subpel_x;
    ((top * (16 - subpel_y) + bottom * subpel_y + 128) >> 8)
        .clamp(0, (1_i64 << bit_depth.min(16)) - 1) as u16
}

#[inline]
fn motion_component_q4(value: i32, subsampling: usize) -> i64 {
    if subsampling == 0 {
        i64::from(value) * 2
    } else {
        i64::from(value) / (1_i64 << (subsampling - 1))
    }
}

fn floor_div_sixteen(value: i64) -> i64 {
    if value >= 0 {
        value / 16
    } else {
        -((-value + 15) / 16)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "intra block copy geometry is explicit for bounds and subsampling checks"
)]
fn predict_intra_block_copy_into(
    plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    mv: (i32, i32),
    subsampling_x: usize,
    subsampling_y: usize,
    output: &mut [u16],
) -> Result<(), DecoderError> {
    let expected_len = width
        .checked_mul(height)
        .ok_or_else(|| DecoderError::Bitstream("intrabc prediction size overflows".to_string()))?;
    if output.len() != expected_len {
        return Err(DecoderError::Bitstream(
            "intrabc prediction buffer has an invalid size".to_string(),
        ));
    }
    let luma_x = i64::try_from(x)
        .ok()
        .and_then(|value| value.checked_shl(subsampling_x as u32))
        .ok_or_else(|| DecoderError::Bitstream("intrabc destination x overflows".to_string()))?;
    let luma_y = i64::try_from(y)
        .ok()
        .and_then(|value| value.checked_shl(subsampling_y as u32))
        .ok_or_else(|| DecoderError::Bitstream("intrabc destination y overflows".to_string()))?;
    let source_luma_x = luma_x
        .checked_add(i64::from(mv.1 / 8))
        .ok_or_else(|| DecoderError::Bitstream("intrabc source x overflows".to_string()))?;
    let source_luma_y = luma_y
        .checked_add(i64::from(mv.0 / 8))
        .ok_or_else(|| DecoderError::Bitstream("intrabc source y overflows".to_string()))?;
    let source_x = usize::try_from(source_luma_x >> subsampling_x)
        .map_err(|_| DecoderError::Bitstream("intrabc source x is negative".to_string()))?;
    let source_y = usize::try_from(source_luma_y >> subsampling_y)
        .map_err(|_| DecoderError::Bitstream("intrabc source y is negative".to_string()))?;
    let source_right = source_x
        .checked_add(width)
        .ok_or_else(|| DecoderError::Bitstream("intrabc source width overflows".to_string()))?;
    let source_bottom = source_y
        .checked_add(height)
        .ok_or_else(|| DecoderError::Bitstream("intrabc source height overflows".to_string()))?;
    if source_right > plane.layout.width || source_bottom > plane.layout.height {
        return Err(DecoderError::Bitstream(
            "intrabc source block exceeds plane bounds".to_string(),
        ));
    }
    for row in 0..height {
        let source_start = (source_y + row) * plane.layout.width + source_x;
        let output_start = row * width;
        output[output_start..output_start + width]
            .copy_from_slice(&plane.samples[source_start..source_start + width]);
    }
    Ok(())
}

fn predict_dc_block_into_with_bounds(
    plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    tile_left: usize,
    tile_top: usize,
    tile_right: usize,
    tile_bottom: usize,
    output: &mut [u16],
) {
    let tile_right = tile_right.min(plane.layout.width).max(tile_left);
    let tile_bottom = tile_bottom.min(plane.layout.height).max(tile_top);
    let above_available = y > tile_top && tile_right > tile_left;
    let left_available = x > tile_left && tile_bottom > tile_top;
    let value = match (above_available, left_available) {
        (true, true) => {
            let above_sum: u32 = (0..width)
                .map(|offset| {
                    u32::from(
                        plane.samples
                            [(y - 1) * plane.layout.width + (x + offset).min(tile_right - 1)],
                    )
                })
                .sum();
            let left_sum: u32 = (0..height)
                .map(|offset| {
                    u32::from(
                        plane.samples
                            [((y + offset).min(tile_bottom - 1)) * plane.layout.width + x - 1],
                    )
                })
                .sum();
            (above_sum + left_sum + ((width + height) as u32 >> 1)) / (width + height) as u32
        }
        (true, false) => {
            let sum: u32 = (0..width)
                .map(|offset| {
                    u32::from(
                        plane.samples
                            [(y - 1) * plane.layout.width + (x + offset).min(tile_right - 1)],
                    )
                })
                .sum();
            (sum + (width as u32 >> 1)) >> width.trailing_zeros()
        }
        (false, true) => {
            let sum: u32 = (0..height)
                .map(|offset| {
                    u32::from(
                        plane.samples
                            [((y + offset).min(tile_bottom - 1)) * plane.layout.width + x - 1],
                    )
                })
                .sum();
            (sum + (height as u32 >> 1)) >> height.trailing_zeros()
        }
        (false, false) => 1u32 << (bit_depth - 1),
    };
    let value = value.min((1u32 << bit_depth) - 1) as u16;
    output.fill(value);
}

#[cfg(test)]
fn predict_dc_block_into(
    plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    output: &mut [u16],
) {
    predict_dc_block_into_with_bounds(
        plane,
        x,
        y,
        width,
        height,
        bit_depth,
        0,
        0,
        plane.layout.width,
        plane.layout.height,
        output,
    );
}

#[cfg(test)]
fn predict_dc_block(
    plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
) -> Vec<u16> {
    let above_available = y > 0 && plane.layout.width > 0;
    let left_available = x > 0 && plane.layout.height > 0;
    let value = match (above_available, left_available) {
        (true, true) => {
            let above_sum: u32 = (0..width)
                .map(|offset| {
                    u32::from(
                        plane.samples[(y - 1) * plane.layout.width
                            + (x + offset).min(plane.layout.width - 1)],
                    )
                })
                .sum();
            let left_sum: u32 = (0..height)
                .map(|offset| {
                    u32::from(
                        plane.samples[((y + offset).min(plane.layout.height - 1))
                            * plane.layout.width
                            + x
                            - 1],
                    )
                })
                .sum();
            (above_sum + left_sum + ((width + height) as u32 >> 1)) / (width + height) as u32
        }
        (true, false) => {
            let sum: u32 = (0..width)
                .map(|offset| {
                    u32::from(
                        plane.samples[(y - 1) * plane.layout.width
                            + (x + offset).min(plane.layout.width - 1)],
                    )
                })
                .sum();
            (sum + (width as u32 >> 1)) >> width.trailing_zeros()
        }
        (false, true) => {
            let sum: u32 = (0..height)
                .map(|offset| {
                    u32::from(
                        plane.samples[((y + offset).min(plane.layout.height - 1))
                            * plane.layout.width
                            + x
                            - 1],
                    )
                })
                .sum();
            (sum + (height as u32 >> 1)) >> height.trailing_zeros()
        }
        (false, false) => 1u32 << (bit_depth - 1),
    };
    let maximum = (1u32 << bit_depth) - 1;
    vec![value.min(maximum) as u16; width * height]
}

#[expect(
    clippy::too_many_arguments,
    reason = "CFL parameters mirror the normative subsampled prediction inputs"
)]
pub(super) fn apply_cfl_prediction(
    prediction: &mut [u16],
    luma_plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    alpha_q3: i8,
    bit_depth: u8,
    subsampling_x: usize,
    subsampling_y: usize,
) -> Result<(), DecoderError> {
    let sample_count = width.checked_mul(height).ok_or_else(|| {
        DecoderError::InvalidParam("AV1 CFL transform dimensions are too large".to_string())
    })?;
    if prediction.len() != sample_count || sample_count == 0 {
        return Err(DecoderError::InvalidParam(
            "AV1 CFL prediction dimensions are invalid".to_string(),
        ));
    }
    let luma_width = luma_plane.layout.width;
    let luma_height = luma_plane.layout.height;
    if luma_width == 0 || luma_height == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 CFL luma plane is empty".to_string(),
        ));
    }

    let luma_q3_at = |row: usize, col: usize| {
        let source_x = (x + col) << subsampling_x;
        let source_y = (y + row) << subsampling_y;
        let mut luma_sum = 0u32;
        for luma_row in 0..(1usize << subsampling_y) {
            let source_y = (source_y + luma_row).min(luma_height - 1);
            for luma_col in 0..(1usize << subsampling_x) {
                let source_x = (source_x + luma_col).min(luma_width - 1);
                luma_sum += u32::from(luma_plane.samples[source_y * luma_width + source_x]);
            }
        }
        // AV1 7.11.5 retains the subsampled sum at Q3 precision. Rounding
        // to an integer luma value first loses fractional AC information.
        (luma_sum << (3 - subsampling_x - subsampling_y)) as i32
    };
    let mut sum = 0i64;
    for row in 0..height {
        for col in 0..width {
            sum += i64::from(luma_q3_at(row, col));
        }
    }
    let average_q3 = ((sum + sample_count as i64 / 2) / sample_count as i64) as i32;
    let maximum = (1i32 << bit_depth) - 1;
    for (index, destination) in prediction.iter_mut().enumerate() {
        let row = index / width;
        let col = index % width;
        let value_q3 = luma_q3_at(row, col);
        let scaled = round_power_of_two_signed(i32::from(alpha_q3) * (value_q3 - average_q3), 6);
        *destination = (i32::from(*destination) + scaled).clamp(0, maximum) as u16;
    }
    Ok(())
}

fn round_power_of_two_signed(value: i32, bits: u32) -> i32 {
    let rounding = 1i32 << (bits - 1);
    if value < 0 {
        -((-value + rounding) >> bits)
    } else {
        (value + rounding) >> bits
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "palette prediction keeps source-map and destination geometry explicit"
)]
#[cfg(test)]
fn predict_palette_block(
    palette: &PalettePlaneInfo,
    color_offset: usize,
    palette_size: usize,
    block_x: usize,
    block_y: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Vec<u16> {
    let palette_size = palette_size.min(PALETTE_MAX_SIZE);
    let mut prediction = Vec::with_capacity(width * height);
    for row in 0..height {
        let map_row = (y + row).saturating_sub(block_y);
        for col in 0..width {
            let map_col = (x + col).saturating_sub(block_x);
            let map_index = map_row.min(palette.map_height - 1) * palette.map_width
                + map_col.min(palette.map_width - 1);
            let color_index = usize::from(palette.color_map[map_index]).min(palette_size - 1);
            prediction.push(palette.colors[color_offset + color_index]);
        }
    }
    prediction
}

#[expect(
    clippy::too_many_arguments,
    reason = "palette prediction keeps source-map and destination geometry explicit"
)]
fn predict_palette_block_into(
    palette: &PalettePlaneInfo,
    color_offset: usize,
    palette_size: usize,
    block_x: usize,
    block_y: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    output: &mut [u16],
) {
    let palette_size = palette_size.min(PALETTE_MAX_SIZE);
    let mut index = 0;
    for row in 0..height {
        let map_row = (y + row).saturating_sub(block_y);
        for col in 0..width {
            let map_col = (x + col).saturating_sub(block_x);
            let map_index = map_row.min(palette.map_height - 1) * palette.map_width
                + map_col.min(palette.map_width - 1);
            let color_index = usize::from(palette.color_map[map_index]).min(palette_size - 1);
            output[index] = palette.colors[color_offset + color_index];
            index += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_convolution_and_blends_match_aom_8_10_12_bit_oracles() {
        // Scalar AOM 9440aeedd097343b763dfe1c75af72c99f4a3175:
        // av1_highbd_dist_wtd_convolve_2d_c + blend_a64_d16_mask_c.
        let cases = [
            (
                8,
                [9720, 9700, 6812, 6856, 8008, 8128, 8456, 9744],
                [9444, 6651, 9440, 7930, 5879, 8117, 9328, 8618],
                [
                    [215, 127, 124, 78, 50, 124, 172, 190],
                    [218, 163, 93, 65, 75, 124, 162, 203],
                    [217, 178, 83, 68, 79, 124, 164, 201],
                    [213, 76, 165, 89, 21, 124, 179, 179],
                    [224, 222, 42, 46, 114, 124, 148, 218],
                ],
            ),
            (
                10,
                [38880, 38800, 27248, 27424, 32032, 32512, 33824, 38976],
                [37772, 26600, 37756, 31716, 23512, 32464, 37308, 34468],
                [
                    [859, 508, 495, 312, 200, 495, 687, 759],
                    [872, 651, 372, 262, 300, 495, 646, 812],
                    [867, 710, 331, 270, 316, 495, 656, 803],
                    [852, 305, 660, 354, 83, 494, 717, 715],
                    [894, 889, 167, 182, 458, 496, 592, 874],
                ],
            ),
            (
                12,
                [38880, 38800, 27248, 27424, 32032, 32512, 33824, 38976],
                [37772, 26600, 37756, 31716, 23512, 32464, 37308, 34468],
                [
                    [3438, 2031, 1982, 1249, 799, 1978, 2748, 3037],
                    [3489, 2603, 1489, 1047, 1198, 1980, 2584, 3248],
                    [3468, 2841, 1325, 1081, 1265, 1979, 2625, 3213],
                    [3407, 1221, 2638, 1416, 333, 1977, 2870, 2860],
                    [3576, 3556, 668, 729, 1831, 1984, 2366, 3494],
                ],
            ),
        ];
        for (bit_depth, expected_a, expected_b, expected_outputs) in cases {
            let layout = crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 32,
                height: 32,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 1024,
            };
            let a = PlaneBuffer {
                layout,
                samples: (0..1024)
                    .map(|i| (((i * 37 + i / 32 * 53) % 256) << (bit_depth - 8)) as u16)
                    .collect(),
            };
            let b = PlaneBuffer {
                layout,
                samples: (0..1024)
                    .map(|i| (((i * 71 + i / 32 * 19 + 43) % 256) << (bit_depth - 8)) as u16)
                    .collect(),
            };
            let filters = (InterpolationFilter::Sharp, InterpolationFilter::Sharp);
            for col in 0..8 {
                let first = predict_compound_sample(
                    &a,
                    8 + col as i64,
                    8,
                    8,
                    8,
                    filters,
                    bit_depth,
                    (8, 8),
                );
                let second = predict_compound_sample(
                    &b,
                    8 + col as i64,
                    8,
                    6,
                    10,
                    filters,
                    bit_depth,
                    (8, 8),
                );
                assert_eq!(first, expected_a[col], "bd={bit_depth} A col={col}");
                assert_eq!(second, expected_b[col], "bd={bit_depth} B col={col}");
                let modes = [
                    (None, None),
                    (Some(44), None),
                    (
                        None,
                        Some(CompoundMask::DifferenceWeighted { inverse: false }),
                    ),
                    (
                        None,
                        Some(CompoundMask::DifferenceWeighted { inverse: true }),
                    ),
                    (
                        None,
                        Some(CompoundMask::Wedge {
                            index: 0,
                            inverse: false,
                        }),
                    ),
                ];
                for (mode, (weight, mask)) in modes.into_iter().enumerate() {
                    assert_eq!(
                        blend_compound_intermediates(
                            first, second, weight, mask, bit_depth, col, 0, 8, 8
                        ),
                        expected_outputs[mode][col],
                        "bd={bit_depth} mode={mode} col={col}"
                    );
                }
            }
        }
    }

    #[test]
    fn compound_prediction_retains_negative_filter_overshoot_until_blending() {
        let layout = crate::av1::decode::PlaneLayout {
            plane: 0,
            width: 32,
            height: 32,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 1024,
        };
        let mut reference = PlaneBuffer {
            layout,
            samples: vec![0; 1024],
        };
        reference.samples[8 * 32 + 7] = 255;
        let secondary = PlaneBuffer {
            layout,
            samples: vec![128; 1024],
        };
        let mut output = [0; 64];
        predict_inter_block_into(
            &reference,
            Some(&secondary),
            8,
            8,
            8,
            8,
            (0, 4),
            Some((0, 0)),
            0,
            0,
            (InterpolationFilter::Regular, InterpolationFilter::Regular),
            None,
            None,
            8,
            MaskGeometry {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            &mut output,
        )
        .unwrap();
        // libaom dist_wtd_convolve: first predictor = -446/16, second=128.
        // Clipping the first predictor before averaging incorrectly gives 64.
        assert_eq!(output[0], 50);
    }

    #[test]
    fn transform_prediction_reads_reconstructed_neighbor() {
        let layout = crate::av1::decode::PlaneLayout {
            plane: 0,
            width: 8,
            height: 4,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 32,
        };
        let mut samples = vec![0; 32];
        for (row, value) in [10, 20, 30, 40].into_iter().enumerate() {
            samples[row * 8 + 3] = value;
        }
        let plane = PlaneBuffer { layout, samples };

        let prediction = predict_block(
            &plane,
            PredictionMode::Horizontal,
            4,
            0,
            4,
            4,
            None,
            None,
            8,
            false,
            false,
            0,
            0,
        )
        .unwrap();

        assert_eq!(
            prediction,
            vec![
                10, 10, 10, 10, //
                20, 20, 20, 20, //
                30, 30, 30, 30, //
                40, 40, 40, 40,
            ]
        );
    }

    #[test]
    fn direct_dc_prediction_matches_edge_reader() {
        let layout = crate::av1::decode::PlaneLayout {
            plane: 0,
            width: 8,
            height: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 64,
        };
        let plane = PlaneBuffer {
            layout,
            samples: (0..64).map(|value| (value * 3) as u16).collect(),
        };
        for (x, y, width, height) in [(0, 0, 4, 4), (4, 0, 4, 4), (0, 4, 4, 4), (4, 4, 4, 4)] {
            let expected = predict_block(
                &plane,
                PredictionMode::Dc,
                x,
                y,
                width,
                height,
                None,
                None,
                8,
                false,
                false,
                width,
                height,
            )
            .unwrap();
            assert_eq!(predict_dc_block(&plane, x, y, width, height, 8), expected);
        }
    }

    #[test]
    fn direct_dc_prediction_writes_caller_buffer() {
        let layout = crate::av1::decode::PlaneLayout {
            plane: 0,
            width: 4,
            height: 4,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 16,
        };
        let plane = PlaneBuffer {
            layout,
            samples: (0..16).map(|value| value as u16).collect(),
        };
        let expected = predict_dc_block(&plane, 2, 2, 2, 2, 8);
        let mut output = [0; 4];
        predict_dc_block_into(&plane, 2, 2, 2, 2, 8, &mut output);
        assert_eq!(output, expected.as_slice());
    }

    #[test]
    fn transform_prediction_can_mask_partition_unavailable_edge_extensions() {
        let layout = crate::av1::decode::PlaneLayout {
            plane: 0,
            width: 6,
            height: 4,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 24,
        };
        let plane = PlaneBuffer {
            layout,
            samples: (0..24).collect(),
        };

        let unmasked = predict_block(
            &plane,
            PredictionMode::D45,
            1,
            2,
            2,
            2,
            None,
            None,
            8,
            false,
            false,
            2,
            2,
        )
        .unwrap();
        let masked = predict_block(
            &plane,
            PredictionMode::D45,
            1,
            2,
            2,
            2,
            None,
            None,
            8,
            false,
            false,
            0,
            2,
        )
        .unwrap();

        assert_ne!(masked, unmasked);
        assert_eq!(unmasked, vec![8, 9, 9, 10]);
        assert_eq!(masked, vec![8, 8, 8, 8]);
    }

    #[test]
    fn palette_prediction_uses_per_pixel_color_map() {
        let color_map = (0..64).map(|index| (index % 3) as u8).collect();
        let palette = PalettePlaneInfo {
            colors: vec![10, 20, 30],
            color_map,
            map_width: 8,
            map_height: 8,
        };

        let prediction = predict_palette_block(&palette, 0, 3, 0, 0, 0, 0, 8, 8);

        assert_eq!(prediction.len(), 64);
        for row in 0..8 {
            for col in 0..8 {
                let expected = match (row * 8 + col) % 3 {
                    0 => 10,
                    1 => 20,
                    _ => 30,
                };
                assert_eq!(prediction[row * 8 + col], expected);
            }
        }
    }

    #[test]
    fn palette_prediction_uses_chroma_color_offset() {
        let color_map = (0..4).flat_map(|_| [0, 0, 0, 0, 1, 1, 1, 1]).collect();
        let palette = PalettePlaneInfo {
            colors: vec![100, 200, 300, 400],
            color_map,
            map_width: 8,
            map_height: 4,
        };

        let prediction = predict_palette_block(&palette, 2, 2, 0, 0, 0, 0, 8, 4);

        assert_eq!(prediction.len(), 32);
        for row in 0..4 {
            for col in 0..8 {
                let expected = if col < 4 { 300 } else { 400 };
                assert_eq!(prediction[row * 8 + col], expected);
            }
        }
    }

    #[test]
    fn palette_prediction_writes_caller_buffer() {
        let palette = PalettePlaneInfo {
            colors: vec![10, 20, 30],
            color_map: vec![0, 1, 2, 1],
            map_width: 2,
            map_height: 2,
        };
        let mut output = [0; 4];
        predict_palette_block_into(&palette, 0, 3, 0, 0, 0, 0, 2, 2, &mut output);
        assert_eq!(output, [10, 20, 30, 20]);
    }

    #[test]
    fn inter_prediction_copies_an_integer_reference_block() {
        let reference = PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 4,
                height: 3,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 12,
            },
            samples: (0..12).collect(),
        };
        let mut output = [0; 4];
        predict_inter_block_into(
            &reference,
            None,
            1,
            1,
            2,
            2,
            (0, 0),
            None,
            0,
            0,
            (InterpolationFilter::Regular, InterpolationFilter::Regular),
            None,
            None,
            8,
            MaskGeometry {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [5, 6, 9, 10]);
    }

    #[test]
    fn compound_prediction_uses_secondary_motion_vector() {
        let reference = PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 4,
                height: 3,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 12,
            },
            samples: (0..12).collect(),
        };
        let secondary = PlaneBuffer {
            layout: reference.layout,
            samples: (100..112).collect(),
        };
        let mut output = [0; 4];
        predict_inter_block_into(
            &reference,
            Some(&secondary),
            1,
            1,
            2,
            2,
            (0, 0),
            Some((0, 8)),
            0,
            0,
            (InterpolationFilter::Regular, InterpolationFilter::Regular),
            None,
            None,
            8,
            MaskGeometry {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [56, 57, 60, 61]);
    }

    #[test]
    fn compound_prediction_applies_distance_weight() {
        assert_eq!(
            blend_compound_prediction(10, 100, Some(48), None, 8, 0, 0, 2, 2),
            33
        );
        assert_eq!(
            blend_compound_prediction(10, 100, None, None, 8, 0, 0, 2, 2),
            55
        );
        assert_eq!(
            blend_compound_prediction(
                0,
                255,
                None,
                Some(CompoundMask::DifferenceWeighted { inverse: false }),
                8,
                0,
                0,
                2,
                2,
            ),
            44
        );
    }

    #[test]
    fn every_wedge_mask_matches_aom() {
        // Actual av1_get_contiguous_soft_mask bytes from AOM 9440aeedd097343b763dfe1c75af72c99f4a3175.
        // Each hash covers all 16 indices, both signs, and every pixel.
        for (width, height, expected) in [
            (8, 8, 1974917667661160085u64),
            (8, 16, 4364072839248270917),
            (16, 8, 17765138988367674869),
            (16, 16, 5976315135349940925),
            (16, 32, 12970946847464165333),
            (32, 16, 15208080273227059917),
            (32, 32, 1611978096444511037),
            (8, 32, 16583822929344630061),
            (32, 8, 4133912236952539205),
        ] {
            let mut hash = 14695981039346656037u64;
            for index in 0..16 {
                for inverse in [false, true] {
                    for y in 0..height {
                        for x in 0..width {
                            let value =
                                wedge_mask_value(width, height, index, inverse, x, y).unwrap();
                            hash = (hash ^ u64::from(value)).wrapping_mul(1099511628211);
                        }
                    }
                }
            }
            assert_eq!(hash, expected, "{width}x{height}");
        }
    }

    #[test]
    fn wedge_prediction_uses_dimension_specific_mask() {
        let first = wedge_mask_value(8, 8, 0, false, 0, 0).unwrap();
        let opposite = wedge_mask_value(8, 8, 0, false, 7, 7).unwrap();
        assert!(first <= 64);
        assert!(opposite <= 64);
        assert_ne!(first, opposite);
        assert_eq!(
            blend_compound_prediction(
                0,
                64,
                None,
                Some(CompoundMask::Wedge {
                    index: 0,
                    inverse: false,
                }),
                8,
                0,
                0,
                8,
                8,
            ),
            u16::from(64 - first)
        );
    }

    #[test]
    fn obmc_without_eligible_neighbors_keeps_predictor() {
        let reference = PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 8,
                height: 8,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 64,
            },
            samples: vec![0; 64],
        };
        let mut output = [123_u16; 64];
        let mut scratch = [0_u16; 64];
        apply_obmc_edge_blend(
            &mut output,
            &reference,
            0,
            0,
            0,
            0,
            8,
            8,
            8,
            8,
            0,
            0,
            &ObmcNeighbors::default(),
            &mut scratch,
            8,
            None,
            0,
        )
        .unwrap();
        assert!(output.iter().all(|sample| *sample == 123));
    }

    #[test]
    fn obmc_masks_follow_aom_tables() {
        assert_eq!(obmc_mask_value(4, 0), 39);
        assert_eq!(obmc_mask_value(4, 1), 50);
        assert_eq!(obmc_mask_value(8, 5), 61);
        assert_eq!(obmc_mask_value(32, 24), 64);
        assert_eq!(obmc_mask_value(64, 0), 33);
    }

    #[test]
    fn local_warp_projection_and_shear_match_aom_single_sample() {
        let mut samples = [None; 8];
        samples[0] = Some(LocalWarpSample {
            source: (56, 184),
            destination: (56, 184),
        });
        let p = estimate_local_warp_params(8, 16, 8, 8, (32, -12), samples).unwrap();
        // Actual av1_find_projection: points relative to block origin are
        // (120,-72); absolute coordinates above match the decoder MI grid.
        assert_eq!(
            [
                p.translation_x,
                p.translation_y,
                p.alpha,
                p.beta,
                p.gamma,
                p.delta
            ],
            [-70110, 196616, 69730, -3912, -8191, 73727]
        );
        assert_eq!(setup_warp_shear(&p), Some((4224, -3904, -7680, 7744)));
    }

    #[test]
    fn local_warp_estimation_matches_aom_quantized_flat_samples() {
        let base_mv = (8_i32, 16_i32);
        let mid_x = 15_i64 * 8;
        let mid_y = 15_i64 * 8;
        let mut samples = [None; 8];
        let flat_samples = [(-32_i64, -32_i64), (-32, 32), (32, -32), (32, 32)].map(|(dy, dx)| {
            Some(LocalWarpSample {
                source: ((mid_y + dy) as i32, (mid_x + dx) as i32),
                destination: (
                    (mid_y + dy + i64::from(base_mv.0)) as i32,
                    (mid_x + dx + i64::from(base_mv.1)) as i32,
                ),
            })
        });
        samples[..4].copy_from_slice(&flat_samples);
        let params = estimate_local_warp_params(8, 8, 16, 16, base_mv, samples).unwrap();
        // Actual AOM av1_find_projection uses a quantized reciprocal even for
        // a flat translation; mathematically exact division is not normative.
        assert_eq!(
            [
                params.translation_x,
                params.translation_y,
                params.alpha,
                params.beta,
                params.gamma,
                params.delta
            ],
            [130412, 64876, 65580, 0, 0, 65580]
        );
        assert_eq!(setup_warp_shear(&params), Some((64, 0, 0, 64)));
    }

    #[test]
    fn local_warp_estimation_fits_cross_axis_tilt() {
        let mid_x = 15_i64 * 8;
        let mid_y = 15_i64 * 8;
        let mut samples = [None; 8];
        let tilted_samples =
            [(-32_i64, -32_i64), (-32, 32), (32, -32), (32, 32)].map(|(dy, dx)| {
                Some(LocalWarpSample {
                    source: ((mid_y + dy) as i32, (mid_x + dx) as i32),
                    destination: ((mid_y + dy + dx / 8) as i32, (mid_x + dx + dy / 8) as i32),
                })
            });
        samples[..4].copy_from_slice(&tilted_samples);
        let params = estimate_local_warp_params(8, 8, 16, 16, (0, 0), samples).unwrap();
        assert!(params.beta.abs() > 0);
        assert!(params.gamma.abs() > 0);
        assert_eq!(fixed_warp_to_plane_eighth(-1, 0), -1);
    }

    #[test]
    fn warped_16x16_matches_aom_identity_and_affine_at_all_depths() {
        // AOM 9440aeedd097343b763dfe1c75af72c99f4a3175
        // av1_highbd_warp_affine_c: 16x16 output at (8,8) from 32x32 source.
        // FNV-1a includes every u16 sample, with no fixture/tool dependency.
        let hashes = [
            [3364059266267454437u64, 17396598565094874530],
            [9344816254078712485, 18342698394889052376],
            [2910353262732798072, 7441486440368512482],
        ];
        for (depth_index, bit_depth) in [8u8, 10, 12].into_iter().enumerate() {
            let reference = PlaneBuffer {
                layout: crate::av1::decode::PlaneLayout {
                    plane: 0,
                    width: 32,
                    height: 32,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    sample_count: 1024,
                },
                samples: (0..1024)
                    .map(|i| (((i * 37 + i / 32 * 53) % 256) as u16) << (bit_depth - 8))
                    .collect(),
            };
            for affine in [false, true] {
                let params = if affine {
                    LocalWarpParams {
                        translation_x: 32768,
                        translation_y: 16384,
                        alpha: 66048,
                        beta: 256,
                        gamma: 128,
                        delta: 65920,
                    }
                } else {
                    LocalWarpParams {
                        translation_x: 0,
                        translation_y: 0,
                        alpha: 65536,
                        beta: 0,
                        gamma: 0,
                        delta: 65536,
                    }
                };
                let (alpha, beta, gamma, delta) = if affine {
                    (512, 256, 128, 384)
                } else {
                    (0, 0, 0, 0)
                };
                let (round0, round1) = if bit_depth == 12 { (5, 9) } else { (3, 11) };
                let mut output = [0u16; 256];
                let mut scratch = [0i64; 120];
                for row in (0..16).step_by(8) {
                    for col in (0..16).step_by(8) {
                        predict_warped_block_into(
                            &mut output,
                            &reference,
                            8,
                            8,
                            16,
                            16,
                            row,
                            col,
                            0,
                            0,
                            &params,
                            alpha,
                            beta,
                            gamma,
                            delta,
                            round0,
                            round1,
                            bit_depth,
                            &mut scratch,
                        );
                    }
                }
                let hash = output
                    .iter()
                    .fold(14695981039346656037u64, |hash, &sample| {
                        (hash ^ u64::from(sample)).wrapping_mul(1099511628211)
                    });
                assert_eq!(
                    hash,
                    hashes[depth_index][usize::from(affine)],
                    "bit_depth={bit_depth} affine={affine}"
                );
            }
        }
    }

    #[test]
    fn warped_filter_bank_matches_av1_shape_and_normalization() {
        assert_eq!(AV1_WARPED_FILTERS.len(), 193);
        for row in AV1_WARPED_FILTERS {
            assert_eq!(row.iter().map(|value| i32::from(*value)).sum::<i32>(), 128);
        }
        assert_eq!(AV1_WARPED_FILTERS[0], [0, 0, 127, 1, 0, 0, 0, 0]);
        assert_eq!(AV1_WARPED_FILTERS[192], [0, 0, 0, 0, 2, 127, -1, 0]);
    }

    #[test]
    fn warped_prediction_reuses_intermediate_scratch_without_changing_output() {
        let reference = PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 32,
                height: 32,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 32 * 32,
            },
            samples: (0..(32 * 32)).map(|value| (value % 256) as u16).collect(),
        };
        let params = LocalWarpParams {
            translation_x: 0,
            translation_y: 0,
            alpha: 1 << 16,
            beta: 0,
            gamma: 0,
            delta: 1 << 16,
        };
        let mut reused = [0_i64; 15 * 8];
        let mut first = [0_u16; 64];
        predict_warped_block_into(
            &mut first,
            &reference,
            0,
            0,
            8,
            8,
            0,
            0,
            0,
            0,
            &params,
            0,
            0,
            0,
            0,
            3,
            11,
            8,
            &mut reused,
        );
        let mut second = [0_u16; 64];
        predict_warped_block_into(
            &mut second,
            &reference,
            0,
            0,
            8,
            8,
            0,
            0,
            0,
            0,
            &params,
            0,
            0,
            0,
            0,
            3,
            11,
            8,
            &mut reused,
        );
        let mut fresh = [0_i64; 15 * 8];
        let mut expected = [0_u16; 64];
        predict_warped_block_into(
            &mut expected,
            &reference,
            0,
            0,
            8,
            8,
            0,
            0,
            0,
            0,
            &params,
            0,
            0,
            0,
            0,
            3,
            11,
            8,
            &mut fresh,
        );
        assert_eq!(first, expected);
        assert_eq!(second, expected);
    }

    #[test]
    fn inter_prediction_regular_filters_fractional_motion() {
        let reference = PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 4,
                height: 4,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 16,
            },
            samples: (0..16).collect(),
        };
        let mut output = [0; 4];
        predict_inter_block_into(
            &reference,
            None,
            0,
            0,
            2,
            2,
            (4, 4),
            None,
            0,
            0,
            (InterpolationFilter::Regular, InterpolationFilter::Regular),
            None,
            None,
            8,
            MaskGeometry {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            &mut output,
        )
        .unwrap();
        // libaom av1_convolve_2d_sr_c with replicated frame edges and
        // dimension-specific 4-tap regular filters produces [2,3,6,8].
        assert_eq!(output, [2, 3, 6, 8]);
    }

    #[test]
    fn inter_prediction_clips_filter_overshoot_to_bit_depth() {
        let reference = PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 16,
                height: 16,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 256,
            },
            samples: (0..256)
                .map(|index| [0, 255, 0, 255, 255, 0, 0, 0][index % 8])
                .collect(),
        };
        let max = predict_inter_sample(
            &reference,
            3,
            4,
            1,
            0,
            (InterpolationFilter::Regular, InterpolationFilter::Regular),
            8,
            (16, 16),
        );
        assert_eq!(max, 255);
    }

    #[test]
    fn inter_prediction_bilinear_filters_fractional_motion() {
        let reference = PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 4,
                height: 4,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 16,
            },
            samples: (0..16).collect(),
        };
        let mut output = [0; 4];
        predict_inter_block_into(
            &reference,
            None,
            0,
            0,
            2,
            2,
            (4, 4),
            None,
            0,
            0,
            (InterpolationFilter::Bilinear, InterpolationFilter::Bilinear),
            None,
            None,
            8,
            MaskGeometry {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [3, 4, 7, 8]);
    }
}

#[cfg(test)]
mod compound_warp_oracle_tests {
    use super::*;

    #[test]
    fn compound_warp_raw_buffers_and_equal_blends_match_libaom() {
        // av1_highbd_warp_affine_c, get_conv_params_no_round(...,compound=1),
        // 32x32 reference, origin(8,8), 16x16 output; FNV-1a over all 256 u16
        // values (not their byte encoding). Both raw biased convolution values
        // and the official second-reference equal blend are checked.
        let expected = [
            [
                (12428620813020704647u64, 16878257489973090916u64),
                (7296945116231023171u64, 11203030156346682354u64),
            ],
            [
                (9987412218138574116u64, 567118965557096033u64),
                (13995544837461872796u64, 13885753161763910192u64),
            ],
            [
                (9987412218138574116u64, 11928232938922546197u64),
                (13995544837461872796u64, 3505714676248549099u64),
            ],
        ];
        let hash = |pixels: &[u16]| {
            pixels.iter().fold(14695981039346656037u64, |h, p| {
                (h ^ u64::from(*p)).wrapping_mul(1099511628211)
            })
        };
        for (depth_index, bit_depth) in [8u8, 10, 12].into_iter().enumerate() {
            let reference = |second: bool| PlaneBuffer {
                layout: crate::av1::decode::PlaneLayout {
                    plane: 0,
                    width: 32,
                    height: 32,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    sample_count: 1024,
                },
                samples: (0..1024)
                    .map(|i| {
                        let value = if second {
                            i * 19 + i / 32 * 17 + 23
                        } else {
                            i * 37 + i / 32 * 53
                        };
                        ((value % 256) as u16) << (bit_depth - 8)
                    })
                    .collect(),
            };
            let first = reference(false);
            let second = reference(true);
            // Header translations are Q13 for high-precision MV. AOM's full
            // Q16 affine translations are (32768,16384), hence (4096,2048).
            for (mode, matrix) in [
                [0, 0, 65536, 0, 0, 65536],
                [4096, 2048, 66048, 256, 128, 65920],
            ]
            .into_iter()
            .enumerate()
            {
                let mut a = [0; 256];
                let mut b = [0; 256];
                assert!(predict_global_motion_block_with_precision_into(
                    &mut a, &first, 8, 8, 16, 16, 0, 0, bit_depth, true, matrix, true
                ));
                assert!(predict_global_motion_block_with_precision_into(
                    &mut b, &second, 8, 8, 16, 16, 0, 0, bit_depth, true, matrix, true
                ));
                assert_eq!(
                    hash(&a),
                    expected[depth_index][mode].0,
                    "raw compound warp bd={bit_depth} mode={mode}"
                );
                for (pixel, other) in a.iter_mut().zip(b) {
                    *pixel = blend_compound_intermediates(
                        *pixel, other, None, None, bit_depth, 0, 0, 16, 16,
                    );
                }
                assert_eq!(
                    hash(&a),
                    expected[depth_index][mode].1,
                    "equal compound warp bd={bit_depth} mode={mode}"
                );
            }
        }
    }
}

#[cfg(test)]
mod negative_shear_warp_oracle_tests {
    use super::*;

    #[test]
    fn negative_shear_full_warp_matches_aom_8_10_12_bit_oracles() {
        // Scalar libaom av1_get_shear_params + av1_highbd_warp_affine_c,
        // origin(8,8), full16x16 output, no external fixture or runtime tools.
        // Matrix0 is the actual AVIS local projection at block(8,16).
        let matrices = [
            [-70110, 196616, 69730, -3912, -8191, 73727],
            [8192, -4096, 65024, -384, -256, 65280],
        ];
        let shears = [(4224, -3904, -7680, 7744), (-512, -384, -256, -256)];
        let expected = [
            [8586656794483808355u64, 15309225790323642852u64],
            [11128722576567324103u64, 4205857561272084126u64],
            [11323588007346037997u64, 2868876082466028956u64],
        ];
        for (di, bit_depth) in [8u8, 10, 12].into_iter().enumerate() {
            let reference = PlaneBuffer {
                layout: crate::av1::decode::PlaneLayout {
                    plane: 0,
                    width: 32,
                    height: 32,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    sample_count: 1024,
                },
                samples: (0..1024)
                    .map(|i| (((i * 37 + i / 32 * 53) % 256) as u16) << (bit_depth - 8))
                    .collect(),
            };
            for (mi, matrix) in matrices.into_iter().enumerate() {
                let params = LocalWarpParams {
                    translation_x: matrix[0],
                    translation_y: matrix[1],
                    alpha: matrix[2],
                    beta: matrix[3],
                    gamma: matrix[4],
                    delta: matrix[5],
                };
                let shear = setup_warp_shear(&params).expect("AOM accepts this affine model");
                assert_eq!(shear, shears[mi], "AOM negative shear matrix{mi}");
                let mut output = [0; 256];
                let mut intermediate = [0i64; 15 * 8];
                let (round0, round1) = if bit_depth == 12 { (5, 9) } else { (3, 11) };
                for row in [0, 8] {
                    for col in [0, 8] {
                        predict_warped_block_into(
                            &mut output,
                            &reference,
                            8,
                            8,
                            16,
                            16,
                            row,
                            col,
                            0,
                            0,
                            &params,
                            shear.0,
                            shear.1,
                            shear.2,
                            shear.3,
                            round0,
                            round1,
                            bit_depth,
                            &mut intermediate,
                        );
                    }
                }
                let hash = output.iter().fold(14695981039346656037u64, |h, p| {
                    (h ^ u64::from(*p)).wrapping_mul(1099511628211)
                });
                assert_eq!(
                    hash, expected[di][mi],
                    "AOM full256pixels bd={bit_depth} matrix{mi}"
                );
            }
        }
    }
}
#[cfg(test)]
mod warp_eligibility_prediction_tests {
    use super::*;
    use crate::av1::syntax::BlockSize;

    fn frame() -> FrameHeader {
        let info = crate::container::parse_avif(include_bytes!(
            "../../../tests/avis-subsampled-palette.avif"
        ))
        .unwrap();
        let sample = &info.sequence_sample_payloads[0];
        let sequence = crate::av1::parse_sequence_header(
            crate::obu::find_obu_payload(sample, crate::obu::ObuType::SequenceHeader)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let mut frame = crate::av1::parse_frame_header(
            crate::obu::find_obu_payload(sample, crate::obu::ObuType::Frame)
                .unwrap()
                .unwrap(),
            &sequence,
        )
        .unwrap();
        frame.force_integer_mv = 0;
        frame.allow_high_precision_mv = true;
        frame.global_motion.types[0] = GlobalMotionType::Affine;
        frame.global_motion.matrices[0] = [4096, 2048, 66048, 256, 128, 65920];
        frame
    }

    fn mode(global: bool) -> BlockModeProbe {
        let mut samples = [None; 8];
        samples[0] = Some(LocalWarpSample {
            source: (56, 184),
            destination: (56, 184),
        });
        BlockModeProbe {
            tile_id: 0,
            block_size: BlockSize::Block8x8,
            segment_id: 0,
            qindex: 98,
            delta_lf: [0; 4],
            skip_context: 0,
            skip_symbol: 0,
            skip: false,
            skip_mode: false,
            is_inter: true,
            reference_frame: Some(0),
            reference_frame_secondary: None,
            motion_vector: Some((32, -12)),
            motion_vector_secondary: None,
            global_motion_index: global.then_some(0),
            global_motion_index_secondary: None,
            motion_mode: if global {
                MotionMode::Simple
            } else {
                MotionMode::LocalWarp
            },
            interintra_mode: None,
            interintra_wedge_index: None,
            local_warp_neighbors: [None; 4],
            local_warp_samples: samples,
            interpolation_filter: Some((
                InterpolationFilter::Regular,
                InterpolationFilter::Regular,
            )),
            compound_weight: None,
            compound_mask: None,
            use_intrabc: false,
            intra_block_copy_mv: None,
            cdef_idx: None,
            y_above_context: 0,
            y_left_context: 0,
            y_mode_symbol: 0,
            y_mode: PredictionMode::Dc,
            angle_delta_y: None,
            y_smooth_neighbour: false,
            filter_intra_mode: None,
            uv_mode_symbol: None,
            uv_mode: None,
            angle_delta_uv: None,
            uv_smooth_neighbour: false,
            palette: crate::av1::tile_decode::PaletteBlockInfo { y: None, uv: None },
            tx_size_context: None,
            tx_size_symbol: None,
            tx_size: TxSize::Tx4x4,
            transform_blocks: Vec::new(),
            bit_position_after: 0,
        }
    }

    fn reference() -> PlaneBuffer {
        PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 32,
                height: 32,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 1024,
            },
            samples: (0..1024)
                .map(|i| ((i * 37 + i / 32 * 53) % 256) as u16)
                .collect(),
        }
    }

    fn predict(
        reference: &PlaneBuffer,
        frame: &FrameHeader,
        mode: &BlockModeProbe,
        plane: usize,
        ss: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> Vec<u16> {
        predict_region(reference, frame, mode, plane, ss, x, y, x, y, w, h, 8)
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_region(
        reference: &PlaneBuffer,
        frame: &FrameHeader,
        mode: &BlockModeProbe,
        plane: usize,
        ss: usize,
        block_x: usize,
        block_y: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        bit_depth: u8,
    ) -> Vec<u16> {
        let mut output = vec![0; w * h];
        let mut scratch = [0; 4096];
        predict_plane_block_into(
            reference,
            Some(reference),
            None,
            mode,
            frame,
            plane,
            PredictionMode::Dc,
            block_x,
            block_y,
            x,
            y,
            w,
            h,
            None,
            None,
            bit_depth,
            false,
            false,
            0,
            0,
            0,
            0,
            32,
            32,
            None,
            None,
            None,
            ss,
            ss,
            &mut output,
            &mut scratch,
            &ObmcNeighbors::default(),
            None,
            None,
            None,
        )
        .unwrap();
        output
    }

    fn translation(mode: &BlockModeProbe) -> BlockModeProbe {
        let mut result = mode.clone();
        result.global_motion_index = None;
        result.motion_mode = MotionMode::Simple;
        result
    }

    #[test]
    fn sub8_chroma_bypasses_local_and_global_warp_in_actual_plane_prediction() {
        // AOM av1_init_warp_params rejects either prediction dimension<8.
        let frame = frame();
        let reference = reference();
        for global in [false, true] {
            let mode = mode(global);
            let baseline = predict(&reference, &frame, &translation(&mode), 1, 1, 4, 8, 4, 4);
            assert_eq!(
                predict(&reference, &frame, &mode, 1, 1, 4, 8, 4, 4),
                baseline
            );
            // Reproduce the pre-gate caller's unconditional warp to ensure
            // this fixture would fail if the eligibility guard were removed.
            let mut previous = baseline.clone();
            if global {
                assert!(predict_global_motion_block_into(
                    &mut previous,
                    &reference,
                    4,
                    8,
                    4,
                    4,
                    1,
                    1,
                    8,
                    true,
                    frame.global_motion.matrices[0]
                ));
            } else {
                apply_local_warp_prediction(
                    &mut previous,
                    &reference,
                    4,
                    8,
                    4,
                    4,
                    4,
                    8,
                    4,
                    4,
                    (32, -12),
                    1,
                    1,
                    8,
                    mode.local_warp_samples,
                )
                .unwrap();
            }
            assert_ne!(previous, baseline, "pre-gate local/global={global}");
        }
    }

    #[test]
    fn full8_prediction_still_warps_a_4x4_transform_fragment() {
        // Eligibility follows the full8x8 prediction, not this4x4 fragment.
        let frame = frame();
        let reference = reference();
        for global in [false, true] {
            let mode = mode(global);
            let baseline = predict(&reference, &frame, &translation(&mode), 0, 0, 8, 16, 4, 4);
            let actual = predict(&reference, &frame, &mode, 0, 0, 8, 16, 4, 4);
            assert_ne!(actual, baseline, "eligible local/global={global}");
            let mut expected = baseline.clone();
            if global {
                assert!(predict_global_motion_block_into(
                    &mut expected,
                    &reference,
                    8,
                    16,
                    4,
                    4,
                    0,
                    0,
                    8,
                    true,
                    frame.global_motion.matrices[0]
                ));
            } else {
                apply_local_warp_prediction(
                    &mut expected,
                    &reference,
                    8,
                    16,
                    4,
                    4,
                    8,
                    16,
                    8,
                    8,
                    (32, -12),
                    0,
                    0,
                    8,
                    mode.local_warp_samples,
                )
                .unwrap();
            }
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn force_integer_mv_bypasses_local_and_global_warp_in_actual_plane_prediction() {
        // AOM av1_init_warp_params also rejects cur_frame_force_integer_mv.
        let mut frame = frame();
        frame.force_integer_mv = 1;
        frame.allow_high_precision_mv = false;
        let reference = reference();
        for global in [false, true] {
            let mut mode = mode(global);
            mode.motion_vector = Some((32, -16));
            let baseline = predict(&reference, &frame, &translation(&mode), 0, 0, 8, 16, 8, 8);
            assert_eq!(
                predict(&reference, &frame, &mode, 0, 0, 8, 16, 8, 8),
                baseline
            );
        }
    }
    #[test]
    fn all_warp_transform_fragments_match_aom_full_prediction() {
        // libaom av1_highbd_warp_affine_c: full8x8 at8,16, raw affine
        // [-70110,196616,69730,-3912,-8191,73727]; global syntax rounds
        // translation_x to -70112 and yields these same64 samples.
        let expected = [
            14178864780517097510_u64,
            7958934973007843195,
            13145496955326679774,
        ];
        for (depth, expected) in [8_u8, 10, 12].into_iter().zip(expected) {
            let mut frame = frame();
            frame.global_motion.matrices[0] = [-8764, 24577, 69730, -3912, -8191, 73727];
            let mut reference = reference();
            for sample in &mut reference.samples {
                *sample <<= depth - 8;
            }
            for global in [false, true] {
                let mode = mode(global);
                let whole =
                    predict_region(&reference, &frame, &mode, 0, 0, 8, 16, 8, 16, 8, 8, depth);
                let hash = whole.iter().fold(14695981039346656037_u64, |h, &v| {
                    (h ^ u64::from(v)).wrapping_mul(1099511628211)
                });
                assert_eq!(
                    hash, expected,
                    "full prediction depth{depth} global{global}"
                );
                let mut assembled = [0_u16; 64];
                for oy in [0, 4] {
                    for ox in [0, 4] {
                        let part = predict_region(
                            &reference,
                            &frame,
                            &mode,
                            0,
                            0,
                            8,
                            16,
                            8 + ox,
                            16 + oy,
                            4,
                            4,
                            depth,
                        );
                        for row in 0..4 {
                            assembled[(oy + row) * 8 + ox..(oy + row) * 8 + ox + 4]
                                .copy_from_slice(&part[row * 4..row * 4 + 4]);
                        }
                    }
                }
                assert_eq!(
                    assembled.as_slice(),
                    whole,
                    "transform fragments depth{depth} global{global}"
                );
            }
        }
    }

    #[test]
    fn compound_global_fragments_preserve_aom_raw_precision() {
        let expected = [
            4834513913386005089_u64,
            12884725509075423797,
            12884725509075423797,
        ];
        for (depth, expected) in [8_u8, 10, 12].into_iter().zip(expected) {
            let mut reference = reference();
            for sample in &mut reference.samples {
                *sample <<= depth - 8;
            }
            let matrix = [-8764, 24577, 69730, -3912, -8191, 73727];
            let mut whole = [0_u16; 64];
            assert!(predict_global_motion_block_with_precision_into(
                &mut whole, &reference, 8, 16, 8, 8, 0, 0, depth, true, matrix, true
            ));
            let hash = whole.iter().fold(14695981039346656037_u64, |h, &v| {
                (h ^ u64::from(v)).wrapping_mul(1099511628211)
            });
            assert_eq!(hash, expected, "AOM raw compound depth{depth}");
            let mut assembled = [0_u16; 64];
            for oy in [0, 4] {
                for ox in [0, 4] {
                    let mut part = [0_u16; 16];
                    predict_compound_global_or_translation_into(
                        &mut part,
                        &reference,
                        8 + ox,
                        16 + oy,
                        4,
                        4,
                        (0, 0),
                        0,
                        0,
                        (InterpolationFilter::Regular, InterpolationFilter::Regular),
                        depth,
                        MaskGeometry {
                            x: ox,
                            y: oy,
                            width: 8,
                            height: 8,
                        },
                        true,
                        Some(matrix),
                    );
                    for row in 0..4 {
                        assembled[(oy + row) * 8 + ox..(oy + row) * 8 + ox + 4]
                            .copy_from_slice(&part[row * 4..row * 4 + 4]);
                    }
                }
            }
            assert_eq!(assembled, whole, "raw compound fragments depth{depth}");
        }
    }
}

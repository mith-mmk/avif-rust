use super::{
    BlockModeProbe, DecodedTransform, PalettePlaneInfo, TileDecoder, coefficient_entropy_context,
};
use crate::DecoderError;
use crate::av1::decode::{FrameBuffers, FrameDecodePlan, PlaneBuffer};
use crate::av1::frame::FrameHeader;
use crate::av1::predict::{IntraEdges, predict_filter_intra, predict_intra_with_edge_filter_into};
use crate::av1::quant::QuantState;
use crate::av1::reconstruct::{read_intra_edges_into, write_plane_block};
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::{PredictionMode, TxSize, TxType};
use crate::av1::tile_decode::palette::PALETTE_MAX_SIZE;
use crate::av1::transform::{
    iter_transform_blocks_with_tx_size, reconstruct_lossless_transform_block_parts_into,
    reconstruct_transform_block_parts_into,
};

#[expect(
    clippy::too_many_arguments,
    reason = "AV1 plane reconstruction keeps syntax, geometry, and mutable buffers explicit"
)]
pub(super) fn decode_plane_block_unit(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
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
    let block_x = plane_block_origin(x, block_mode.block_size.width(), subsampling_x);
    let block_y = plane_block_origin(y, block_mode.block_size.height(), subsampling_y);
    let plane_block_size = if plane_index == 0 {
        block_mode.block_size
    } else {
        plane_block_size(block_mode.block_size, subsampling_x, subsampling_y)
    };
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
    let tx_size = if plane_index > 0 && !frame.coded_lossless() {
        // Chroma uses the largest rectangular transform for its scaled plane
        // block. Scaling the luma transform independently would turn, for
        // example, a 16x8 luma block into a 4x4 chroma transform, while AV1
        // requires the 8x4 plane transform for 4:2:0.
        match plane_block_size.largest_supported_rect_tx_size() {
            // AOM limits chroma transforms with a 64-pixel dimension to the
            // corresponding 32-pixel form, while preserving other rectangles.
            TxSize::Tx64x64 | TxSize::Tx64x32 | TxSize::Tx32x64 => TxSize::Tx32x32,
            TxSize::Tx64x16 => TxSize::Tx32x16,
            TxSize::Tx16x64 => TxSize::Tx16x32,
            tx_size => tx_size,
        }
    } else {
        if plane_index == 0 {
            block_mode.tx_size
        } else {
            scale_tx_size(block_mode.tx_size, subsampling_x, subsampling_y)
        }
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
    if block_mode.is_inter && reference_plane.is_none() {
        return Err(DecoderError::Unsupported(format!(
            "AV1 inter reference plane {plane_index} is unavailable"
        )));
    }
    let transforms = if subsampling_x == 0 && subsampling_y == 0 {
        iter_transform_blocks_with_tx_size(
            plane_index,
            x,
            y,
            block_mode.block_size,
            tx_size,
            decoder.mi_cols << 2,
            decoder.mi_rows << 2,
        )
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
        for transform in transforms.filter(|transform| transform_in_unit(transform)) {
            decoder.record_transform_boundary(transform, TxType::DctDct, 0);
            decoder.set_txb_entropy_context(transform, 0);
            let (top_right_available, bottom_left_available) =
                decoder.reconstructed_extension_availability(plane, transform)?;
            let prediction_len = transform.tx_size.width() * transform.tx_size.height();
            let prediction = &mut decoder.prediction_scratch[..prediction_len];
            predict_plane_block_into(
                plane,
                reference_plane,
                secondary_reference_plane,
                block_mode,
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
                luma_plane,
                cfl_alpha_q3,
                block_mode.intra_block_copy_mv,
                subsampling_x,
                subsampling_y,
                prediction,
            )?;
            write_plane_block(
                plane,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                &prediction,
            )?;
            decoder.mark_reconstructed_transform(transform)?;
        }
        return Ok(());
    }

    for transform in transforms.filter(|transform| transform_in_unit(transform)) {
        let txb_context = decoder.txb_context(plane_block_size, transform);
        let skip_cdf = decoder
            .cdf
            .txb_skip_cdf_mut(transform.tx_size.coeff_cdf_index(), txb_context.skip);
        let all_zero_symbol = decoder.reader.read_symbol(skip_cdf)?;
        if all_zero_symbol != 0 {
            // Deblocking depends on transform boundaries even when the
            // coefficient block is entirely zero; retain the transform
            // geometry for the post-filter stage.
            decoder.record_transform_boundary(transform, TxType::DctDct, 0);
            decoder.set_txb_entropy_context(transform, 0);
            let (top_right_available, bottom_left_available) =
                decoder.reconstructed_extension_availability(plane, transform)?;
            let prediction_len = transform.tx_size.width() * transform.tx_size.height();
            let prediction = &mut decoder.prediction_scratch[..prediction_len];
            predict_plane_block_into(
                plane,
                reference_plane,
                secondary_reference_plane,
                block_mode,
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
                luma_plane,
                cfl_alpha_q3,
                block_mode.intra_block_copy_mv,
                subsampling_x,
                subsampling_y,
                prediction,
            )?;
            write_plane_block(
                plane,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                &prediction,
            )?;
            decoder.mark_reconstructed_transform(transform)?;
            continue;
        }

        let retain_coefficients = plane_index == 0 && decoded_luma.is_some();
        let decoded_transform =
            decoder.read_decoded_transform(frame, block_mode, transform, txb_context.dc_sign)?;
        decoder.set_txb_entropy_context(
            transform,
            coefficient_entropy_context(&decoded_transform.coefficients),
        );
        let (top_right_available, bottom_left_available) =
            decoder.reconstructed_extension_availability(plane, transform)?;
        let prediction_len = transform.tx_size.width() * transform.tx_size.height();
        let prediction = &mut decoder.prediction_scratch[..prediction_len];
        predict_plane_block_into(
            plane,
            reference_plane,
            secondary_reference_plane,
            block_mode,
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
            luma_plane,
            cfl_alpha_q3,
            block_mode.intra_block_copy_mv,
            subsampling_x,
            subsampling_y,
            prediction,
        )?;
        let block = decoded_transform.transform;
        let tx_type = decoded_transform.tx_type;
        let dequant = &mut decoder.dequant_scratch[..prediction_len];
        let reconstructed = &mut decoder.reconstruction_scratch[..prediction_len];
        let residual = &mut decoder.residual_scratch[..prediction_len];
        let reconstructed_transform = if frame.coded_lossless() {
            reconstruct_lossless_transform_block_parts_into(
                plane,
                block,
                tx_type,
                &decoded_transform.coefficients,
                quant_state.plane(transform.plane),
                &prediction,
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
                &prediction,
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
    let (above_available, left_available, edge_above_left) = read_intra_edges_into(
        plane,
        x,
        y,
        width,
        height,
        bit_depth,
        top_right_available,
        bottom_left_available,
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
        let prediction = predict_filter_intra(filter_intra_mode, width, height, edges)?;
        output.copy_from_slice(&prediction);
        return Ok(());
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
        BlockSize::Block8x4 if subsampling_x == 1 => BlockSize::Block8x4,
        BlockSize::Block8x4 if subsampling_y == 1 => BlockSize::Block8x8,
        BlockSize::Block4x16 if subsampling_x == 1 => BlockSize::Block8x16,
        BlockSize::Block4x16 if subsampling_y == 1 => BlockSize::Block4x16,
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
    luma_plane: Option<&PlaneBuffer>,
    cfl_alpha_q3: Option<i8>,
    intra_block_copy_mv: Option<(i32, i32)>,
    subsampling_x: usize,
    subsampling_y: usize,
    output: &mut [u16],
) -> Result<(), DecoderError> {
    let sample_count = width.checked_mul(height).ok_or_else(|| {
        DecoderError::InvalidParam("AV1 prediction dimensions overflow".to_string())
    })?;
    if output.len() != sample_count {
        return Err(DecoderError::InvalidParam(
            "AV1 prediction output dimensions do not match block".to_string(),
        ));
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
            output,
        )?;
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
            predict_dc_block_into(plane, x, y, width, height, bit_depth, output);
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
    let mv_x = i64::from(mv.1) / (1_i64 << subsampling_x);
    let mv_y = i64::from(mv.0) / (1_i64 << subsampling_y);
    let secondary_mv = secondary_mv.unwrap_or(mv);
    let secondary_mv_x = i64::from(secondary_mv.1) / (1_i64 << subsampling_x);
    let secondary_mv_y = i64::from(secondary_mv.0) / (1_i64 << subsampling_y);
    let integer_mv = mv_x % 8 == 0
        && mv_y % 8 == 0
        && (secondary_reference.is_none() || (secondary_mv_x % 8 == 0 && secondary_mv_y % 8 == 0));
    if integer_mv {
        let source_x = i64::try_from(x)
            .ok()
            .and_then(|value| value.checked_add(mv_x / 8))
            .ok_or_else(|| DecoderError::Bitstream("AV1 inter source x overflows".to_string()))?;
        let source_y = i64::try_from(y)
            .ok()
            .and_then(|value| value.checked_add(mv_y / 8))
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
                output,
            );
        }
        let source_x = source_x as usize;
        let source_y = source_y as usize;
        if let Some(secondary) = secondary_reference {
            let secondary_source_x = i64::try_from(x)
                .ok()
                .and_then(|value| value.checked_add(secondary_mv_x / 8))
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 secondary source x overflows".to_string())
                })?;
            let secondary_source_y = i64::try_from(y)
                .ok()
                .and_then(|value| value.checked_add(secondary_mv_y / 8))
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
                    output[target_start + col] = average_prediction(
                        reference.samples[source_start + col],
                        secondary.samples[secondary_start + col],
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
    // normative AV1 regular 8-tap separable kernel; keeping the source
    // coordinate arrays per block avoids redoing fixed-point division for
    // every row/column sample.
    const MAX_INTER_BLOCK_DIMENSION: usize = 128;
    if width > MAX_INTER_BLOCK_DIMENSION || height > MAX_INTER_BLOCK_DIMENSION {
        return Err(DecoderError::Bitstream(
            "AV1 inter prediction block exceeds the supported dimension".to_string(),
        ));
    }
    let secondary_mv_x = i64::from(secondary_mv.1) / (1_i64 << subsampling_x);
    let secondary_mv_y = i64::from(secondary_mv.0) / (1_i64 << subsampling_y);
    let base_x = (i64::try_from(x).unwrap_or(i64::MAX) << 3)
        .checked_add(mv_x)
        .ok_or_else(|| DecoderError::Bitstream("AV1 inter source x overflows".to_string()))?;
    let base_y = (i64::try_from(y).unwrap_or(i64::MAX) << 3)
        .checked_add(mv_y)
        .ok_or_else(|| DecoderError::Bitstream("AV1 inter source y overflows".to_string()))?;
    let mut x0 = [0i64; MAX_INTER_BLOCK_DIMENSION];
    let mut fx = [0i64; MAX_INTER_BLOCK_DIMENSION];
    for col in 0..width {
        let fixed = base_x + col as i64 * 8;
        let source = floor_div_eight(fixed);
        x0[col] = source;
        fx[col] = fixed - source * 8;
    }
    let mut y0 = [0i64; MAX_INTER_BLOCK_DIMENSION];
    let mut fy = [0i64; MAX_INTER_BLOCK_DIMENSION];
    for row in 0..height {
        let fixed = base_y + row as i64 * 8;
        let source = floor_div_eight(fixed);
        y0[row] = source;
        fy[row] = fixed - source * 8;
    }
    if let Some(secondary) = secondary_reference {
        let secondary_base_x = (i64::try_from(x).unwrap_or(i64::MAX) << 3)
            .checked_add(secondary_mv_x)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 secondary inter source x overflows".to_string())
            })?;
        let secondary_base_y = (i64::try_from(y).unwrap_or(i64::MAX) << 3)
            .checked_add(secondary_mv_y)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 secondary inter source y overflows".to_string())
            })?;
        let mut secondary_x0 = [0i64; MAX_INTER_BLOCK_DIMENSION];
        let mut secondary_fx = [0i64; MAX_INTER_BLOCK_DIMENSION];
        for col in 0..width {
            let fixed = secondary_base_x + col as i64 * 8;
            let source = floor_div_eight(fixed);
            secondary_x0[col] = source;
            secondary_fx[col] = fixed - source * 8;
        }
        let mut secondary_y0 = [0i64; MAX_INTER_BLOCK_DIMENSION];
        let mut secondary_fy = [0i64; MAX_INTER_BLOCK_DIMENSION];
        for row in 0..height {
            let fixed = secondary_base_y + row as i64 * 8;
            let source = floor_div_eight(fixed);
            secondary_y0[row] = source;
            secondary_fy[row] = fixed - source * 8;
        }
        for row in 0..height {
            for col in 0..width {
                output[row * width + col] = average_prediction(
                    predict_inter_sample(reference, x0[col], y0[row], fx[col], fy[row]),
                    predict_inter_sample(
                        secondary,
                        secondary_x0[col],
                        secondary_y0[row],
                        secondary_fx[col],
                        secondary_fy[row],
                    ),
                );
            }
        }
    } else {
        for row in 0..height {
            for col in 0..width {
                output[row * width + col] =
                    predict_inter_sample(reference, x0[col], y0[row], fx[col], fy[row]);
            }
        }
    }
    Ok(())
}

#[inline]
fn average_prediction(first: u16, second: u16) -> u16 {
    ((u32::from(first) + u32::from(second) + 1) / 2) as u16
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
    output: &mut [u16],
) -> Result<(), DecoderError> {
    let mv_x = i64::from(mv.1) / (1_i64 << subsampling_x);
    let mv_y = i64::from(mv.0) / (1_i64 << subsampling_y);
    let secondary_mv = secondary_mv.unwrap_or(mv);
    let secondary_mv_x = i64::from(secondary_mv.1) / (1_i64 << subsampling_x);
    let secondary_mv_y = i64::from(secondary_mv.0) / (1_i64 << subsampling_y);
    for row in 0..height {
        let y_fixed = (i64::try_from(y).unwrap_or(i64::MAX) << 3)
            .checked_add(mv_y)
            .ok_or_else(|| DecoderError::Bitstream("AV1 inter source y overflows".to_string()))?
            + (row as i64 * 8);
        let secondary_y_fixed = (i64::try_from(y).unwrap_or(i64::MAX) << 3)
            .checked_add(secondary_mv_y)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 secondary inter source y overflows".to_string())
            })?
            + (row as i64 * 8);
        let y0 = floor_div_eight(y_fixed);
        let fy = y_fixed - y0 * 8;
        let secondary_y0 = floor_div_eight(secondary_y_fixed);
        let secondary_fy = secondary_y_fixed - secondary_y0 * 8;
        for col in 0..width {
            let x_fixed = (i64::try_from(x).unwrap_or(i64::MAX) << 3)
                .checked_add(mv_x)
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 inter source x overflows".to_string())
                })?
                + (col as i64 * 8);
            let secondary_x_fixed = (i64::try_from(x).unwrap_or(i64::MAX) << 3)
                .checked_add(secondary_mv_x)
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 secondary inter source x overflows".to_string())
                })?
                + (col as i64 * 8);
            let x0 = floor_div_eight(x_fixed);
            let fx = x_fixed - x0 * 8;
            let secondary_x0 = floor_div_eight(secondary_x_fixed);
            let secondary_fx = secondary_x_fixed - secondary_x0 * 8;
            let first = predict_inter_sample(reference, x0, y0, fx, fy);
            output[row * width + col] = secondary_reference
                .map(|secondary| {
                    average_prediction(
                        first,
                        predict_inter_sample(
                            secondary,
                            secondary_x0,
                            secondary_y0,
                            secondary_fx,
                            secondary_fy,
                        ),
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

#[inline]
fn round_filter_sum(sum: i64) -> i64 {
    if sum >= 0 {
        (sum + 64) >> 7
    } else {
        -(((-sum) + 64) >> 7)
    }
}

#[inline]
fn predict_inter_sample(
    plane: &PlaneBuffer,
    source_x: i64,
    source_y: i64,
    subpel_x: i64,
    subpel_y: i64,
) -> u16 {
    let horizontal = AV1_REGULAR_SUBPEL_FILTERS[subpel_x as usize];
    let vertical = AV1_REGULAR_SUBPEL_FILTERS[subpel_y as usize];
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
        *value = round_filter_sum(sum);
    }
    let mut sum = 0i64;
    for (tap, coefficient) in vertical.iter().enumerate() {
        sum += intermediate[tap] * i64::from(*coefficient);
    }
    round_filter_sum(sum).clamp(0, i64::from(u16::MAX)) as u16
}

fn floor_div_eight(value: i64) -> i64 {
    if value >= 0 {
        value / 8
    } else {
        -((-value + 7) / 8)
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

fn predict_dc_block_into(
    plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    output: &mut [u16],
) {
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
    let value = value.min((1u32 << bit_depth) - 1) as u16;
    output.fill(value);
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
        let sample_count = 1usize << (subsampling_x + subsampling_y);
        for luma_row in 0..(1usize << subsampling_y) {
            let source_y = (source_y + luma_row).min(luma_height - 1);
            for luma_col in 0..(1usize << subsampling_x) {
                let source_x = (source_x + luma_col).min(luma_width - 1);
                luma_sum += u32::from(luma_plane.samples[source_y * luma_width + source_x]);
            }
        }
        let value = (luma_sum + (sample_count as u32 / 2)) / sample_count as u32;
        (value as i32) << 3
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
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [56, 57, 60, 61]);
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
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [1, 2, 5, 6]);
    }
}

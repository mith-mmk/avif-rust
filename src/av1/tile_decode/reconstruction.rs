use super::{
    BlockModeProbe, CompoundMask, DecodedTransform, InterIntraMode, LocalWarpSample, MotionMode,
    PalettePlaneInfo, TileDecoder, coefficient_entropy_context,
};
use crate::DecoderError;
use crate::av1::decode::{FrameBuffers, FrameDecodePlan, PlaneBuffer};
use crate::av1::frame::{FrameHeader, InterpolationFilter};
use crate::av1::predict::{IntraEdges, predict_filter_intra, predict_intra_with_edge_filter_into};
use crate::av1::quant::QuantState;
use crate::av1::reconstruct::{read_intra_edges_into, write_plane_block};
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::{PredictionMode, TxSize, TxType};
use crate::av1::tile_decode::palette::PALETTE_MAX_SIZE;
use crate::av1::tile_decode::warped_filter::AV1_WARPED_FILTERS;
use crate::av1::transform::{
    iter_transform_blocks_with_tx_size, reconstruct_lossless_transform_block_parts_into,
    reconstruct_transform_block_parts_into,
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
    let mask_geometry = MaskGeometry {
        x: x.saturating_sub(block_x),
        y: y.saturating_sub(block_y),
        width: (block_mode.block_size.width() >> subsampling_x).max(4),
        height: (block_mode.block_size.height() >> subsampling_y).max(4),
    };
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
            block_mode
                .interpolation_filter
                .unwrap_or((InterpolationFilter::Regular, InterpolationFilter::Regular)),
            block_mode.compound_weight,
            block_mode.compound_mask,
            bit_depth,
            mask_geometry,
            output,
        )?;
        match block_mode.motion_mode {
            MotionMode::Simple => {}
            MotionMode::Obmc => apply_obmc_edge_blend(
                output,
                plane,
                block_x,
                block_y,
                x,
                y,
                width,
                height,
                (block_mode.block_size.width() >> subsampling_x).max(4),
                (block_mode.block_size.height() >> subsampling_y).max(4),
                bit_depth,
            ),
            MotionMode::LocalWarp if secondary_reference_plane.is_none() => {
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
        if let Some(interintra_mode) = block_mode.interintra_mode {
            let mut intra_prediction = vec![0_u16; output.len()];
            let prediction_mode = match interintra_mode {
                InterIntraMode::Dc => PredictionMode::Dc,
                InterIntraMode::Vertical => PredictionMode::Vertical,
                InterIntraMode::Horizontal => PredictionMode::Horizontal,
                InterIntraMode::Smooth => PredictionMode::Smooth,
            };
            if prediction_mode == PredictionMode::Dc {
                predict_dc_block_into(plane, x, y, width, height, bit_depth, &mut intra_prediction);
            } else {
                predict_block_into(
                    plane,
                    prediction_mode,
                    x,
                    y,
                    width,
                    height,
                    None,
                    None,
                    bit_depth,
                    enable_intra_edge_filter,
                    false,
                    top_right_available,
                    bottom_left_available,
                    &mut intra_prediction,
                )?;
            }
            let wedge_index = block_mode.interintra_wedge_index;
            for (sample_index, sample) in output.iter_mut().enumerate() {
                let mask = wedge_index
                    .and_then(|wedge_idx| {
                        wedge_mask_value(
                            mask_geometry.width,
                            mask_geometry.height,
                            wedge_idx,
                            false,
                            mask_geometry.x + sample_index % width,
                            mask_geometry.y + sample_index / width,
                        )
                    })
                    .unwrap_or(32);
                *sample = ((u32::from(*sample) * u32::from(mask)
                    + u32::from(intra_prediction[sample_index]) * u32::from(64 - mask)
                    + 32)
                    >> 6) as u16;
            }
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

#[inline]
fn apply_obmc_edge_blend(
    output: &mut [u16],
    plane: &PlaneBuffer,
    block_x: usize,
    block_y: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    block_width: usize,
    block_height: usize,
    bit_depth: u8,
) {
    let overlap_x = (block_width / 4).clamp(4, 32);
    let overlap_y = (block_height / 4).clamp(4, 32);
    let max_value = (1_u32 << u32::from(bit_depth.min(16))) - 1;
    for row in 0..height {
        let global_y = y.saturating_add(row);
        for col in 0..width {
            let global_x = x.saturating_add(col);
            let mut value = u32::from(output[row * width + col]);
            if block_y > 0 && global_y < block_y.saturating_add(overlap_y) {
                let boundary_y = block_y - 1;
                let neighbor = u32::from(
                    plane.samples[boundary_y.min(plane.layout.height.saturating_sub(1))
                        * plane.layout.width
                        + global_x.min(plane.layout.width.saturating_sub(1))],
                );
                let current_weight = (((global_y - block_y + 1) * 64) / (overlap_y + 1)) as u32;
                value = (neighbor * (64 - current_weight) + value * current_weight + 32) >> 6;
            }
            if block_x > 0 && global_x < block_x.saturating_add(overlap_x) {
                let boundary_x = block_x - 1;
                let neighbor = u32::from(
                    plane.samples[global_y.min(plane.layout.height.saturating_sub(1))
                        * plane.layout.width
                        + boundary_x.min(plane.layout.width.saturating_sub(1))],
                );
                let current_weight = (((global_x - block_x + 1) * 64) / (overlap_x + 1)) as u32;
                value = (neighbor * (64 - current_weight) + value * current_weight + 32) >> 6;
            }
            output[row * width + col] = value.min(max_value) as u16;
        }
    }
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
    for row in (0..height).step_by(8) {
        for col in (0..width).step_by(8) {
            predict_warped_block_into(
                output,
                reference,
                x,
                y,
                width,
                height,
                row,
                col,
                subsampling_x,
                subsampling_y,
                &params,
                warp_alpha,
                warp_beta,
                warp_gamma,
                warp_delta,
                inter_round0,
                inter_round1,
                bit_depth,
            );
        }
    }
    Ok(())
}

fn setup_warp_shear(params: &LocalWarpParams) -> Option<(i64, i64, i64, i64)> {
    const MODEL_BITS: i64 = 16;
    const PARAM_REDUCE_BITS: u32 = 6;
    const NONDIAG_CLAMP: i64 = 1 << 15;
    let alpha = (params.alpha - (1 << MODEL_BITS)).clamp(-NONDIAG_CLAMP, NONDIAG_CLAMP - 1);
    let beta = params.beta.clamp(-NONDIAG_CLAMP, NONDIAG_CLAMP - 1);
    if params.alpha == 0 {
        return None;
    }
    let gamma = round_div_signed(params.gamma << MODEL_BITS, params.alpha)?
        .clamp(-NONDIAG_CLAMP, NONDIAG_CLAMP - 1);
    let delta = (params.delta
        - round_div_signed(params.beta.checked_mul(params.gamma)?, params.alpha)?
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
            (value - half) >> bits
        } else {
            (value + half) >> bits
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "AV1 warped block filtering keeps geometry and rounding explicit"
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
    let mut intermediate = [0_i64; 15 * 8];
    for k in -7_i64..8 {
        let iy = (iy4 + k).clamp(0, reference.layout.height.saturating_sub(1) as i64) as usize;
        for l in -4_i64..4 {
            let sx = sx4 + alpha * l + beta * (k + 4);
            let offset =
                (round_div_power_of_two_signed(sx, DIFF_BITS) + PIXEL_SHIFT).clamp(0, 192) as usize;
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
    let output_height = output_height.saturating_sub(row).min(8);
    let output_width = output_width.saturating_sub(col).min(8);
    for out_row in 0..output_height {
        let k = out_row as i64 - 4;
        for out_col in 0..output_width {
            let l = out_col as i64 - 4;
            let sy = sy4 + delta * (k + 4) + gamma * l;
            let offset =
                (round_div_power_of_two_signed(sy, DIFF_BITS) + PIXEL_SHIFT).clamp(0, 192) as usize;
            let coeffs = &AV1_WARPED_FILTERS[offset];
            let mut sum = 1_i64 << offset_bits_vert;
            for (tap, coeff) in coeffs.iter().enumerate() {
                sum += i64::from(*coeff)
                    * intermediate[(k + tap as i64 + 4) as usize * 8 + (l + 4) as usize];
            }
            let pixel = (round_div_power_of_two_unsigned(sum, inter_round1)
                - (1_i64 << (u32::from(bit_depth) - 1))
                - (1_i64 << u32::from(bit_depth)))
            .clamp(0, (1_i64 << bit_depth.min(16)) - 1) as u16;
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
    let alpha = round_div_signed(
        a11.checked_mul(bx0)?
            .checked_sub(a01.checked_mul(bx1)?)?
            .checked_shl(WARPEDMODEL_PREC_BITS)?,
        determinant,
    )?
    .clamp(
        (1 << WARPEDMODEL_PREC_BITS) - NONDIAG_CLAMP + 1,
        (1 << WARPEDMODEL_PREC_BITS) + NONDIAG_CLAMP - 1,
    );
    let beta = round_div_signed(
        a00.checked_mul(bx1)?
            .checked_sub(a01.checked_mul(bx0)?)?
            .checked_shl(WARPEDMODEL_PREC_BITS)?,
        determinant,
    )?
    .clamp(-NONDIAG_CLAMP + 1, NONDIAG_CLAMP - 1);
    let gamma = round_div_signed(
        a11.checked_mul(by0)?
            .checked_sub(a01.checked_mul(by1)?)?
            .checked_shl(WARPEDMODEL_PREC_BITS)?,
        determinant,
    )?
    .clamp(-NONDIAG_CLAMP + 1, NONDIAG_CLAMP - 1);
    let delta = round_div_signed(
        a00.checked_mul(by1)?
            .checked_sub(a01.checked_mul(by0)?)?
            .checked_shl(WARPEDMODEL_PREC_BITS)?,
        determinant,
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

fn round_div_signed(numerator: i64, denominator: i64) -> Option<i64> {
    if denominator == 0 {
        return None;
    }
    let half = denominator.abs() / 2;
    let adjusted = if (numerator < 0) ^ (denominator < 0) {
        numerator.checked_sub(half)?
    } else {
        numerator.checked_add(half)?
    };
    adjusted.checked_div(denominator)
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
                output[row * width + col] = blend_compound_prediction(
                    predict_inter_sample(
                        reference,
                        x0[col],
                        y0[row],
                        fx[col],
                        fy[row],
                        interpolation_filters,
                    ),
                    predict_inter_sample(
                        secondary,
                        secondary_x0[col],
                        secondary_y0[row],
                        secondary_fx[col],
                        secondary_fy[row],
                        interpolation_filters,
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
                );
            }
        }
    }
    Ok(())
}

#[inline]
fn average_prediction(first: u16, second: u16) -> u16 {
    ((u32::from(first) + u32::from(second) + 1) / 2) as u16
}

const WEDGE_MASTER_EVEN: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4,
    11, 27, 46, 58, 62, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64,
];
const WEDGE_MASTER_ODD: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
    6, 18, 37, 53, 60, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64,
];
const WEDGE_MASTER_VERTICAL: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7,
    21, 43, 57, 62, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64,
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
        (8, 16) | (16, 32) | (8, 32) => (
            &WEDGE_CODEBOOK_HGTW,
            &[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1][..],
        ),
        (16, 8) | (32, 16) | (32, 8) => (
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
    let shift = 16_i32 - i32::try_from((row + 1) / 2).unwrap_or(i32::MAX);
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
        3 => (wedge_oblique63(63 - column, row), true),
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
    if let Some(CompoundMask::DifferenceWeighted { inverse }) = compound_mask {
        let shift = u32::from(bit_depth.saturating_sub(8));
        let diff = u32::from(first.abs_diff(second)) >> shift;
        let mask = (38 + diff / 16).min(64);
        let mask = if inverse { 64 - mask } else { mask };
        return ((u32::from(first) * mask + u32::from(second) * (64 - mask) + 32) >> 6) as u16;
    }
    if let Some(CompoundMask::Wedge { index, inverse }) = compound_mask
        && let Some(mask) = wedge_mask_value(width, height, index, inverse, x, y)
    {
        return ((u32::from(first) * u32::from(mask)
            + u32::from(second) * u32::from(64 - mask)
            + 32)
            >> 6) as u16;
    }
    let Some(primary_weight) = primary_weight else {
        return average_prediction(first, second);
    };
    let secondary_weight = 64 - u32::from(primary_weight);
    ((u32::from(first) * u32::from(primary_weight) + u32::from(second) * secondary_weight + 32)
        >> 6) as u16
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
            let first = predict_inter_sample(reference, x0, y0, fx, fy, interpolation_filters);
            output[row * width + col] = secondary_reference
                .map(|secondary| {
                    blend_compound_prediction(
                        first,
                        predict_inter_sample(
                            secondary,
                            secondary_x0,
                            secondary_y0,
                            secondary_fx,
                            secondary_fy,
                            interpolation_filters,
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

#[inline]
fn interpolation_kernel(filter: InterpolationFilter, phase: usize) -> &'static [i16; 8] {
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
    interpolation_filters: (InterpolationFilter, InterpolationFilter),
) -> u16 {
    if interpolation_filters == (InterpolationFilter::Bilinear, InterpolationFilter::Bilinear) {
        return predict_inter_sample_bilinear(plane, source_x, source_y, subpel_x, subpel_y);
    }
    let horizontal = interpolation_kernel(interpolation_filters.0, subpel_x as usize);
    let vertical = interpolation_kernel(interpolation_filters.1, subpel_y as usize);
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

#[inline]
fn predict_inter_sample_bilinear(
    plane: &PlaneBuffer,
    source_x: i64,
    source_y: i64,
    subpel_x: i64,
    subpel_y: i64,
) -> u16 {
    let sample = |x: i64, y: i64| {
        let x = x.clamp(0, plane.layout.width.saturating_sub(1) as i64) as usize;
        let y = y.clamp(0, plane.layout.height.saturating_sub(1) as i64) as usize;
        i64::from(plane.samples[y * plane.layout.width + x])
    };
    let top =
        sample(source_x, source_y) * (8 - subpel_x) + sample(source_x + 1, source_y) * subpel_x;
    let bottom = sample(source_x, source_y + 1) * (8 - subpel_x)
        + sample(source_x + 1, source_y + 1) * subpel_x;
    ((top * (8 - subpel_y) + bottom * subpel_y + 32) >> 6).clamp(0, i64::from(u16::MAX)) as u16
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
    fn obmc_blends_only_the_block_overlap_edges() {
        let plane = PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 10,
                height: 10,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 100,
            },
            samples: vec![0; 100],
        };
        let mut output = [64_u16; 64];
        apply_obmc_edge_blend(&mut output, &plane, 1, 1, 1, 1, 8, 8, 8, 8, 8);
        assert!(output[0] < 64);
        assert_eq!(output[7 * 8 + 7], 64);
    }

    #[test]
    fn local_warp_estimation_preserves_translation_for_flat_samples() {
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
        assert_eq!(params.alpha, 1 << 16);
        assert_eq!(params.beta, 0);
        assert_eq!(params.gamma, 0);
        assert_eq!(params.delta, 1 << 16);
        assert_eq!(params.translation_x, i64::from(base_mv.1) << 13);
        assert_eq!(params.translation_y, i64::from(base_mv.0) << 13);
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
    fn warped_filter_bank_matches_av1_shape_and_normalization() {
        assert_eq!(AV1_WARPED_FILTERS.len(), 193);
        for row in AV1_WARPED_FILTERS {
            assert_eq!(row.iter().map(|value| i32::from(*value)).sum::<i32>(), 128);
        }
        assert_eq!(AV1_WARPED_FILTERS[0], [0, 0, 127, 1, 0, 0, 0, 0]);
        assert_eq!(AV1_WARPED_FILTERS[192], [0, 0, 0, 0, 2, 127, -1, 0]);
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
        assert_eq!(output, [1, 2, 5, 6]);
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

use super::{
    BlockModeProbe, DecodedTransform, PalettePlaneInfo, TileDecoder, coefficient_entropy_context,
};
use crate::DecoderError;
use crate::av1::decode::{FrameBuffers, FrameDecodePlan, PlaneBuffer};
use crate::av1::frame::FrameHeader;
use crate::av1::predict::{IntraEdges, predict_filter_intra, predict_intra_with_edge_filter};
use crate::av1::quant::QuantState;
use crate::av1::reconstruct::{read_intra_edges_with_extension_availability, write_plane_block};
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::{PredictionMode, TxSize, TxType};
use crate::av1::tile_decode::palette::PALETTE_MAX_SIZE;
use crate::av1::transform::{
    QuantizedTransform, plan_transform_blocks_with_tx_size, reconstruct_lossless_transform_block,
    reconstruct_transform_block,
};

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
) -> Result<Vec<DecodedTransform>, DecoderError> {
    let layout = plan.planes.get(plane_index).ok_or_else(|| {
        DecoderError::Bitstream(format!("AV1 plane {plane_index} decode plan is missing"))
    })?;
    if layout.subsampling_x != 0 || layout.subsampling_y != 0 {
        return Err(DecoderError::Unsupported(
            "AV1 subsampled chroma block reconstruction is not supported yet".to_string(),
        ));
    }
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
    let tx_size = if plane_index > 0 && !frame.quantization.coded_lossless() {
        match block_mode.block_size.largest_supported_rect_tx_size() {
            // AOM limits chroma transforms with a 64-pixel dimension to the
            // corresponding 32-pixel form, while preserving other rectangles.
            TxSize::Tx64x64 | TxSize::Tx64x32 | TxSize::Tx32x64 => TxSize::Tx32x32,
            TxSize::Tx64x16 => TxSize::Tx32x16,
            TxSize::Tx16x64 => TxSize::Tx16x32,
            tx_size => tx_size,
        }
    } else {
        block_mode.tx_size
    };
    let transforms = plan_transform_blocks_with_tx_size(
        plane_index,
        x,
        y,
        block_mode.block_size,
        tx_size,
        decoder.mi_cols << 2,
        decoder.mi_rows << 2,
    )
    .into_iter()
    .filter(|transform| {
        transform.x >= unit_x
            && transform.x < unit_x.saturating_add(unit_width)
            && transform.y >= unit_y
            && transform.y < unit_y.saturating_add(unit_height)
    })
    .collect::<Vec<_>>();
    for transform in &transforms {
        decoder.record_transform_boundary(*transform, TxType::DctDct, 0);
    }
    if block_mode.skip {
        for transform in &transforms {
            decoder.set_txb_entropy_context(*transform, 0);
            let (top_right_available, bottom_left_available) =
                decoder.reconstructed_extension_availability(plane, *transform)?;
            let prediction = predict_plane_block(
                plane,
                block_mode,
                plane_index,
                prediction_mode,
                x,
                y,
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
            )?;
            write_plane_block(
                plane,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                &prediction,
            )?;
            decoder.mark_reconstructed_transform(*transform)?;
        }
        return Ok(Vec::new());
    }

    let mut decoded = Vec::new();
    for transform in transforms {
        let txb_context = decoder.txb_context(block_mode.block_size, transform);
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
            let prediction = predict_plane_block(
                plane,
                block_mode,
                plane_index,
                prediction_mode,
                x,
                y,
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

        let decoded_transform =
            decoder.read_decoded_transform(frame, block_mode, transform, txb_context.dc_sign)?;
        decoder.record_transform_boundary(
            decoded_transform.transform,
            decoded_transform.tx_type,
            decoded_transform
                .coefficients
                .iter()
                .filter(|coefficient| **coefficient != 0)
                .count(),
        );
        decoder.set_txb_entropy_context(
            transform,
            coefficient_entropy_context(&decoded_transform.coefficients),
        );
        let (top_right_available, bottom_left_available) =
            decoder.reconstructed_extension_availability(plane, transform)?;
        let prediction = predict_plane_block(
            plane,
            block_mode,
            plane_index,
            prediction_mode,
            x,
            y,
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
        )?;
        let quantized = QuantizedTransform {
            block: decoded_transform.transform,
            tx_type: decoded_transform.tx_type,
            coefficients: decoded_transform.coefficients.clone(),
        };
        if frame.quantization.coded_lossless() {
            reconstruct_lossless_transform_block(
                plane,
                &quantized,
                quant_state.plane(transform.plane),
                &prediction,
                sequence.color_config.bit_depth,
            )?;
        } else {
            reconstruct_transform_block(
                plane,
                &quantized,
                quant_state.plane(transform.plane),
                &prediction,
                sequence.color_config.bit_depth,
            )?;
        }
        decoder.mark_reconstructed_transform(transform)?;
        decoded.push(decoded_transform);
    }

    Ok(decoded)
}

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
    let mut edges = read_intra_edges_with_extension_availability(
        plane,
        x,
        y,
        width,
        height,
        bit_depth,
        top_right_available,
        bottom_left_available,
    );
    let midpoint = 1u16 << (bit_depth - 1);
    let above_left = match (edges.above_available, edges.left_available) {
        (true, true) => edges.above_left,
        (true, false) => edges.above[0],
        (false, true) => edges.left[0],
        (false, false) => midpoint,
    };
    if !edges.above_available && edges.left_available {
        edges.above.fill(edges.left[0]);
    }
    if !edges.left_available && edges.above_available {
        edges.left.fill(edges.above[0]);
    }
    let edges = if prediction_mode == PredictionMode::Dc && filter_intra_mode.is_none() {
        IntraEdges {
            above: edges.above_available.then_some(edges.above.as_slice()),
            left: edges.left_available.then_some(edges.left.as_slice()),
            above_left: Some(above_left),
            bit_depth,
        }
    } else {
        IntraEdges {
            above: Some(&edges.above),
            left: Some(&edges.left),
            above_left: Some(above_left),
            bit_depth,
        }
    };
    if let Some(filter_intra_mode) = filter_intra_mode {
        return predict_filter_intra(filter_intra_mode, width, height, edges);
    }
    predict_intra_with_edge_filter(
        prediction_mode,
        angle_delta,
        width,
        height,
        edges,
        enable_intra_edge_filter,
        smooth_neighbour,
    )
}

fn predict_plane_block(
    plane: &PlaneBuffer,
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
) -> Result<Vec<u16>, DecoderError> {
    if filter_intra_mode.is_none() && prediction_mode == PredictionMode::Dc {
        let palette_prediction = if plane_index == 0 {
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
        };
        if let Some((palette, color_offset, palette_size)) = palette_prediction {
            if !palette.color_map.is_empty() && palette.map_width > 0 && palette.map_height > 0 {
                return Ok(predict_palette_block(
                    palette,
                    color_offset,
                    palette_size,
                    block_x,
                    block_y,
                    x,
                    y,
                    width,
                    height,
                ));
            }
        }
    }
    let mut prediction = predict_block(
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
    )?;
    if let Some(alpha_q3) = cfl_alpha_q3 {
        let luma_plane = luma_plane.ok_or_else(|| {
            DecoderError::Bitstream("AV1 CFL prediction is missing its luma plane".to_string())
        })?;
        apply_cfl_prediction(
            &mut prediction,
            luma_plane,
            x,
            y,
            width,
            height,
            alpha_q3,
            bit_depth,
        )?;
    }
    Ok(prediction)
}

pub(super) fn apply_cfl_prediction(
    prediction: &mut [u16],
    luma_plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    alpha_q3: i8,
    bit_depth: u8,
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

    let mut luma_q3 = Vec::with_capacity(sample_count);
    let mut sum = 0i64;
    for row in 0..height {
        let source_y = (y + row).min(luma_height - 1);
        for col in 0..width {
            let source_x = (x + col).min(luma_width - 1);
            let value_q3 = i32::from(luma_plane.samples[source_y * luma_width + source_x]) << 3;
            luma_q3.push(value_q3);
            sum += i64::from(value_q3);
        }
    }
    let average_q3 = ((sum + sample_count as i64 / 2) / sample_count as i64) as i32;
    let maximum = (1i32 << bit_depth) - 1;
    for (destination, value_q3) in prediction.iter_mut().zip(luma_q3) {
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
}

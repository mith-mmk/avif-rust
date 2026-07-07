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
use crate::av1::syntax::{PredictionMode, TxSize};
use crate::av1::tile_decode::palette::PALETTE_MAX_SIZE;
use crate::av1::transform::{
    QuantizedTransform, plan_transform_blocks_with_tx_size, reconstruct_transform_block,
};

pub(super) fn decode_plane_block(
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
    x: usize,
    y: usize,
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
    let plane = buffers.planes.get_mut(plane_index).ok_or_else(|| {
        DecoderError::Bitstream(format!("AV1 plane {plane_index} buffer is missing"))
    })?;
    let tx_size = if plane_index > 0 && block_mode.tx_size == TxSize::Tx64x64 {
        TxSize::Tx32x32
    } else {
        block_mode.tx_size
    };
    let transforms = plan_transform_blocks_with_tx_size(
        plane_index,
        x,
        y,
        block_mode.block_size,
        tx_size,
        layout.width,
        layout.height,
    );
    if block_mode.skip {
        for transform in &transforms {
            decoder.set_txb_entropy_context(*transform, 0);
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
                true,
                true,
            )?;
            write_plane_block(
                plane,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                &prediction,
            )?;
        }
        return Ok(Vec::new());
    }

    let mut decoded = Vec::new();
    for transform in transforms {
        let txb_context = decoder.txb_context(block_mode.block_size, transform);
        let all_zero_symbol = decoder.reader.read_symbol(
            decoder
                .cdf
                .txb_skip_cdf_mut(transform.tx_size.coeff_cdf_index(), txb_context.skip),
        )?;
        if all_zero_symbol != 0 {
            decoder.set_txb_entropy_context(transform, 0);
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
                true,
                true,
            )?;
            write_plane_block(
                plane,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                &prediction,
            )?;
            continue;
        }

        let decoded_transform =
            decoder.read_decoded_transform(frame, block_mode, transform, txb_context.dc_sign)?;
        decoder.set_txb_entropy_context(
            transform,
            coefficient_entropy_context(&decoded_transform.coefficients),
        );
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
            true,
            true,
        )?;
        let quantized = QuantizedTransform {
            block: decoded_transform.transform,
            tx_type: decoded_transform.tx_type,
            coefficients: decoded_transform.coefficients.clone(),
        };
        reconstruct_transform_block(
            plane,
            &quantized,
            quant_state.plane(transform.plane),
            &prediction,
            sequence.color_config.bit_depth,
        )?;
        decoded.push(decoded_transform);
    }

    Ok(decoded)
}

fn predict_block(
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
    top_right_available: bool,
    bottom_left_available: bool,
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
    top_right_available: bool,
    bottom_left_available: bool,
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
    predict_block(
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
    )
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
        let map_row = (y + row).saturating_sub(block_y) / 4;
        for col in 0..width {
            let map_col = (x + col).saturating_sub(block_x) / 4;
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
            true,
            true,
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
            true,
            true,
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
            false,
            true,
        )
        .unwrap();

        assert_ne!(masked, unmasked);
        assert_eq!(unmasked, vec![8, 9, 9, 10]);
        assert_eq!(masked, vec![8, 8, 8, 8]);
    }

    #[test]
    fn palette_prediction_expands_color_map_cells() {
        let palette = PalettePlaneInfo {
            colors: vec![10, 20, 30],
            color_map: vec![
                0, 1, //
                2, 1,
            ],
            map_width: 2,
            map_height: 2,
        };

        let prediction = predict_palette_block(&palette, 0, 3, 0, 0, 0, 0, 8, 8);

        assert_eq!(prediction.len(), 64);
        for row in 0..8 {
            for col in 0..8 {
                let expected = match (row / 4, col / 4) {
                    (0, 0) => 10,
                    (0, 1) => 20,
                    (1, 0) => 30,
                    (1, 1) => 20,
                    _ => unreachable!(),
                };
                assert_eq!(prediction[row * 8 + col], expected);
            }
        }
    }

    #[test]
    fn palette_prediction_uses_chroma_color_offset() {
        let palette = PalettePlaneInfo {
            colors: vec![100, 200, 300, 400],
            color_map: vec![0, 1],
            map_width: 2,
            map_height: 1,
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

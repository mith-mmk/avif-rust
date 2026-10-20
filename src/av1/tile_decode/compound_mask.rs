//! Chroma compound blending downsamples the full-resolution luma mask.
use super::*;

pub(super) struct References<'a> {
    pub primary: &'a PlaneBuffer,
    pub secondary: &'a PlaneBuffer,
}

pub(super) fn predict_chroma(
    references: &References<'_>,
    primary: &PlaneBuffer,
    secondary: &PlaneBuffer,
    mode: &BlockModeProbe,
    frame: &FrameHeader,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    bit_depth: u8,
    geometry: MaskGeometry,
    output: &mut [u16],
    scratch: &mut [u16],
) {
    let mv = mode.motion_vector.unwrap_or((0, 0));
    let secondary_mv = mode.motion_vector_secondary.unwrap_or(mv);
    let filters = mode
        .interpolation_filter
        .unwrap_or((InterpolationFilter::Regular, InterpolationFilter::Regular));
    let primary_model = global_motion_matrix(frame, mode.global_motion_index);
    let secondary_model = global_motion_matrix(frame, mode.global_motion_index_secondary);
    predict_compound_global_or_translation_into(
        output,
        primary,
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
        frame.allow_high_precision_mv,
        primary_model,
    );
    let secondary_output = &mut scratch[..output.len()];
    predict_compound_global_or_translation_into(
        secondary_output,
        secondary,
        x,
        y,
        width,
        height,
        secondary_mv,
        subsampling_x,
        subsampling_y,
        filters,
        bit_depth,
        geometry,
        frame.allow_high_precision_mv,
        secondary_model,
    );
    let block_x = (x - geometry.x) << subsampling_x;
    let block_y = (y - geometry.y) << subsampling_y;
    for row in 0..height {
        for col in 0..width {
            let mut mask_sum = 0u32;
            for dy in 0..(1 << subsampling_y) {
                for dx in 0..(1 << subsampling_x) {
                    let lx = ((geometry.x + col) << subsampling_x) + dx;
                    let ly = ((geometry.y + row) << subsampling_y) + dy;
                    let mask = match mode.compound_mask {
                        Some(CompoundMask::Wedge { index, inverse }) => wedge_mask_value(
                            mode.block_size.width(),
                            mode.block_size.height(),
                            index,
                            inverse,
                            lx,
                            ly,
                        )
                        .unwrap_or(32),
                        Some(CompoundMask::DifferenceWeighted { inverse }) => {
                            let a = luma_sample(
                                references.primary,
                                block_x,
                                block_y,
                                lx,
                                ly,
                                mv,
                                filters,
                                bit_depth,
                                mode.block_size,
                                frame.allow_high_precision_mv,
                                primary_model,
                            );
                            let b = luma_sample(
                                references.secondary,
                                block_x,
                                block_y,
                                lx,
                                ly,
                                secondary_mv,
                                filters,
                                bit_depth,
                                mode.block_size,
                                frame.allow_high_precision_mv,
                                secondary_model,
                            );
                            difference_mask(a, b, bit_depth, inverse)
                        }
                        None => 32,
                    };
                    mask_sum += u32::from(mask);
                }
            }
            let shift = subsampling_x + subsampling_y;
            let mask = ((mask_sum + (1 << shift) / 2) >> shift) as u8;
            let index = row * width + col;
            output[index] = blend_compound_intermediates(
                output[index],
                secondary_output[index],
                Some(mask),
                None,
                bit_depth,
                0,
                0,
                0,
                0,
            );
        }
    }
}

pub(super) fn difference_mask(a: u16, b: u16, bit_depth: u8, inverse: bool) -> u8 {
    let diff = round_div_power_of_two_unsigned(
        i64::from(a.abs_diff(b)),
        compound_round_bits(bit_depth) + u32::from(bit_depth.saturating_sub(8)),
    );
    let mask = (38 + diff / 16).min(64) as u8;
    if inverse { 64 - mask } else { mask }
}

fn luma_sample(
    reference: &PlaneBuffer,
    block_x: usize,
    block_y: usize,
    x: usize,
    y: usize,
    mv: (i32, i32),
    filters: (InterpolationFilter, InterpolationFilter),
    bit_depth: u8,
    block_size: crate::av1::syntax::BlockSize,
    allow_high_precision_mv: bool,
    matrix: Option<[i32; 6]>,
) -> u16 {
    if let Some(matrix) = matrix {
        // Warped filter phases are anchored to the prediction block's 8x8
        // chunks; a 1x1 warp would choose a different centre and phase.
        let mut tile = [0u16; 64];
        if predict_global_motion_block_with_precision_into(
            &mut tile,
            reference,
            block_x + (x & !7),
            block_y + (y & !7),
            8,
            8,
            0,
            0,
            bit_depth,
            allow_high_precision_mv,
            matrix,
            true,
        ) {
            return tile[(y & 7) * 8 + (x & 7)];
        }
    }
    let fx = ((block_x + x) as i64) * 16 + motion_component_q4(mv.1, 0);
    let fy = ((block_y + y) as i64) * 16 + motion_component_q4(mv.0, 0);
    let sx = floor_div_sixteen(fx);
    let sy = floor_div_sixteen(fy);
    predict_compound_sample(
        reference,
        sx,
        sy,
        fx - sx * 16,
        fy - sy * 16,
        filters,
        bit_depth,
        (block_size.width(), block_size.height()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::tile_decode::PaletteBlockInfo;
    use crate::av1::{BlockSize, TxSize, UvPredictionMode};

    fn mode(mask: CompoundMask) -> BlockModeProbe {
        BlockModeProbe {
            tile_id: 0,
            block_size: BlockSize::Block8x8,
            segment_id: 0,
            qindex: 0,
            delta_lf: [0; 4],
            skip_context: 0,
            skip_symbol: 0,
            skip: false,
            skip_mode: false,
            is_inter: true,
            reference_frame: Some(0),
            reference_frame_secondary: Some(1),
            motion_vector: Some((4, 4)),
            motion_vector_secondary: Some((5, 3)),
            global_motion_index: None,
            global_motion_index_secondary: None,
            motion_mode: MotionMode::Simple,
            interintra_mode: None,
            interintra_wedge_index: None,
            local_warp_neighbors: [None; 4],
            local_warp_samples: [None; 8],
            interpolation_filter: Some((InterpolationFilter::Sharp, InterpolationFilter::Sharp)),
            compound_weight: None,
            compound_mask: Some(mask),
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
            uv_mode: Some(UvPredictionMode::Intra(PredictionMode::Dc)),
            angle_delta_uv: None,
            uv_smooth_neighbour: false,
            palette: PaletteBlockInfo { y: None, uv: None },
            tx_size_context: None,
            tx_size_symbol: None,
            tx_size: TxSize::Tx4x4,
            transform_blocks: Vec::new(),
            bit_position_after: 0,
        }
    }

    #[test]
    fn chroma_masks_match_aom_for_every_depth_and_subsampling() {
        // Actual highbd convolution + luma diff mask + d16 mask blend from
        // AOM 9440aeedd097343b763dfe1c75af72c99f4a3175. Each FNV hash
        // covers every chroma output sample, including 4:4:0 low-level paths.
        let cases: &[(u8, usize, usize, u8, u64)] = &[
            (8, 0, 0, 0, 12952281262659421497),
            (8, 0, 0, 1, 916112544731195609),
            (8, 0, 0, 2, 6940392356380544805),
            (8, 0, 1, 0, 8336534094058237159),
            (8, 0, 1, 1, 14364485304353684960),
            (8, 0, 1, 2, 13098795629212534603),
            (8, 1, 0, 0, 14777552286438751255),
            (8, 1, 0, 1, 1854805101476929622),
            (8, 1, 0, 2, 13284654542194361954),
            (8, 1, 1, 0, 12265825018464125358),
            (8, 1, 1, 1, 91642095950277221),
            (8, 1, 1, 2, 6119537534682725959),
            (10, 0, 0, 0, 1695092254072003261),
            (10, 0, 0, 1, 14052870388473586224),
            (10, 0, 0, 2, 18115070429591975165),
            (10, 0, 1, 0, 18355471042372273540),
            (10, 0, 1, 1, 13616561454840445817),
            (10, 0, 1, 2, 1066473481563081068),
            (10, 1, 0, 0, 15855221642565851398),
            (10, 1, 0, 1, 9954433340592386319),
            (10, 1, 0, 2, 9860495442780570268),
            (10, 1, 1, 0, 502458611570379599),
            (10, 1, 1, 1, 13378261796986598794),
            (10, 1, 1, 2, 11713635726732125963),
            (12, 0, 0, 0, 4434114942617751062),
            (12, 0, 0, 1, 14380578626703775042),
            (12, 0, 0, 2, 17129270542693937755),
            (12, 0, 1, 0, 7715827242234217322),
            (12, 0, 1, 1, 1468447481676501166),
            (12, 0, 1, 2, 10986902977630997412),
            (12, 1, 0, 0, 8241638988744206882),
            (12, 1, 0, 1, 14985068665488985157),
            (12, 1, 0, 2, 5473592972477931316),
            (12, 1, 1, 0, 16480125074788682132),
            (12, 1, 1, 1, 11986816282732071878),
            (12, 1, 1, 2, 16882662934098748084),
        ];
        let info =
            crate::container::parse_avif(include_bytes!("../../../tests/WML2Viewer.avif")).unwrap();
        let seq = crate::obu::find_obu_payload(
            &info.primary_item_payload,
            crate::obu::ObuType::SequenceHeader,
        )
        .unwrap()
        .unwrap();
        let sequence = crate::av1::parse_sequence_header(seq).unwrap();
        let payload =
            crate::obu::find_obu_payload(&info.primary_item_payload, crate::obu::ObuType::Frame)
                .unwrap()
                .unwrap();
        let frame = crate::av1::parse_frame_header(payload, &sequence).unwrap();
        for &(bd, sx, sy, kind, expected) in cases {
            let make_plane = |secondary: bool| PlaneBuffer {
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
                        let v = if secondary {
                            (i * 71 + i / 32 * 19 + 43) % 256
                        } else {
                            (i * 37 + i / 32 * 53) % 256
                        };
                        (v as u16) << (bd - 8)
                    })
                    .collect(),
            };
            let a = make_plane(false);
            let b = make_plane(true);
            let refs = References {
                primary: &a,
                secondary: &b,
            };
            let mask = match kind {
                0 => CompoundMask::DifferenceWeighted { inverse: false },
                1 => CompoundMask::DifferenceWeighted { inverse: true },
                _ => CompoundMask::Wedge {
                    index: 0,
                    inverse: false,
                },
            };
            let mode = mode(mask);
            let w = 8 >> sx;
            let h = 8 >> sy;
            let x = 8 >> sx;
            let y = 8 >> sy;
            let mut whole = vec![0; w * h];
            let mut scratch = vec![0; w * h];
            predict_chroma(
                &refs,
                &a,
                &b,
                &mode,
                &frame,
                x,
                y,
                w,
                h,
                sx,
                sy,
                bd,
                MaskGeometry {
                    x: 0,
                    y: 0,
                    width: w,
                    height: h,
                },
                &mut whole,
                &mut scratch,
            );
            let hash = whole.iter().fold(14695981039346656037u64, |h, &v| {
                (h ^ u64::from(v)).wrapping_mul(1099511628211)
            });
            assert_eq!(hash, expected, "bd{bd} sx{sx} sy{sy} mask{kind}");
            let mut divided = vec![0; w * h];
            for yy in (0..h).step_by(4) {
                for xx in (0..w).step_by(4) {
                    let mut tile = [0; 16];
                    let mut scratch = [0; 16];
                    predict_chroma(
                        &refs,
                        &a,
                        &b,
                        &mode,
                        &frame,
                        x + xx,
                        y + yy,
                        4,
                        4,
                        sx,
                        sy,
                        bd,
                        MaskGeometry {
                            x: xx,
                            y: yy,
                            width: w,
                            height: h,
                        },
                        &mut tile,
                        &mut scratch,
                    );
                    for r in 0..4 {
                        divided[(yy + r) * w + xx..(yy + r) * w + xx + 4]
                            .copy_from_slice(&tile[r * 4..r * 4 + 4]);
                    }
                }
            }
            assert_eq!(
                divided, whole,
                "subdivision bd{bd} sx{sx} sy{sy} mask{kind}"
            );
        }
    }
}

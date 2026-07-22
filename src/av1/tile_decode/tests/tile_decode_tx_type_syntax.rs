use super::tx_type_syntax::{
    filter_intra_mode_to_tx_cdf_mode, fixed_tx_type, intra_ext_tx_set_context,
};
use crate::av1::frame::{
    CdefParams, DeltaLfParams, DeltaQParams, FrameHeader, FrameType, GlobalMotionParams,
    LoopFilterParams, QuantizationParams, RestorationParams, SegmentationParams, TxMode,
};
use crate::av1::syntax::{TxSize, TxType};
use crate::av1::tile::TileInfo;
use crate::av1::transform::TransformBlock;

#[test]
fn intra_ext_tx_set_context_uses_set2_for_tx16() {
    assert_eq!(intra_ext_tx_set_context(false, TxSize::Tx4x4), Some((1, 0)));
    assert_eq!(intra_ext_tx_set_context(false, TxSize::Tx8x8), Some((1, 1)));
    assert_eq!(
        intra_ext_tx_set_context(false, TxSize::Tx16x16),
        Some((2, 2))
    );
    assert_eq!(
        intra_ext_tx_set_context(false, TxSize::Tx4x16),
        Some((1, 0))
    );
    assert_eq!(
        intra_ext_tx_set_context(false, TxSize::Tx8x32),
        Some((1, 1))
    );
    assert_eq!(
        intra_ext_tx_set_context(false, TxSize::Tx16x32),
        Some((2, 2))
    );
    assert_eq!(intra_ext_tx_set_context(true, TxSize::Tx4x4), Some((2, 0)));
    assert_eq!(intra_ext_tx_set_context(true, TxSize::Tx8x8), Some((2, 1)));
    assert_eq!(intra_ext_tx_set_context(true, TxSize::Tx4x16), Some((2, 0)));
    assert_eq!(intra_ext_tx_set_context(true, TxSize::Tx8x32), Some((2, 1)));
    assert_eq!(intra_ext_tx_set_context(false, TxSize::Tx32x32), None);
}

#[test]
fn filter_intra_mode_selects_normative_tx_cdf_mode() {
    assert_eq!(filter_intra_mode_to_tx_cdf_mode(0).unwrap(), 0);
    assert_eq!(filter_intra_mode_to_tx_cdf_mode(1).unwrap(), 1);
    assert_eq!(filter_intra_mode_to_tx_cdf_mode(2).unwrap(), 2);
    assert_eq!(filter_intra_mode_to_tx_cdf_mode(3).unwrap(), 6);
    assert_eq!(filter_intra_mode_to_tx_cdf_mode(4).unwrap(), 0);
    assert!(filter_intra_mode_to_tx_cdf_mode(5).is_err());
}

#[test]
fn fixed_tx_type_uses_dct_for_non_lossless_chroma_and_large_blocks() {
    let frame = sample_frame(0);
    let chroma = TransformBlock {
        plane: 1,
        x: 0,
        y: 0,
        tx_size: TxSize::Tx4x4,
    };
    let large = TransformBlock {
        plane: 0,
        x: 0,
        y: 0,
        tx_size: TxSize::Tx32x32,
    };

    assert_eq!(fixed_tx_type(&frame, chroma), Some(TxType::DctDct));
    assert_eq!(fixed_tx_type(&frame, large), Some(TxType::DctDct));

    let frame = sample_frame(20);
    let luma_small = TransformBlock {
        plane: 0,
        x: 0,
        y: 0,
        tx_size: TxSize::Tx4x4,
    };
    assert_eq!(fixed_tx_type(&frame, luma_small), None);
}

#[test]
fn fixed_tx_type_uses_dct_for_coded_lossless_luma() {
    let frame = sample_frame(0);
    let transform = TransformBlock {
        plane: 0,
        x: 0,
        y: 0,
        tx_size: TxSize::Tx4x4,
    };

    assert_eq!(fixed_tx_type(&frame, transform), Some(TxType::DctDct));
}

fn sample_frame(base_q_idx: u8) -> FrameHeader {
    FrameHeader {
        frame_type: FrameType::Key,
        show_existing_frame: false,
        show_frame: true,
        showable_frame: false,
        error_resilient_mode: true,
        disable_cdf_update: false,
        allow_screen_content_tools: false,
        force_integer_mv: 2,
        frame_size_override_flag: false,
        order_hint: 0,
        primary_ref_frame: 7,
        refresh_frame_flags: 0xff,
        reference_frame_indices: [0; 7],
        frame_refs_short_signaling: false,
        frame_id: None,
        allow_high_precision_mv: false,
        is_filter_switchable: false,
        is_motion_mode_switchable: false,
        use_ref_frame_mvs: false,
        reference_select: false,
        skip_mode_present: false,
        allow_warped_motion: false,
        global_motion: GlobalMotionParams::default(),
        frame_width: 64,
        frame_height: 64,
        upscaled_width: 64,
        render_width: 64,
        render_height: 64,
        allow_intrabc: false,
        disable_frame_end_update_cdf: false,
        tile_info: TileInfo {
            uniform_tile_spacing: true,
            tile_cols: 1,
            tile_rows: 1,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            tile_size_bytes: 0,
            context_update_tile_id: 0,
            mi_col_starts: vec![0, 16],
            mi_row_starts: vec![0, 16],
        },
        base_q_idx,
        quantization: QuantizationParams {
            base_q_idx,
            delta_q_y_dc: 0,
            delta_q_u_dc: 0,
            delta_q_u_ac: 0,
            delta_q_v_dc: 0,
            delta_q_v_ac: 0,
            using_qmatrix: false,
            qm_y: 15,
            qm_u: 15,
            qm_v: 15,
        },
        segmentation: SegmentationParams::default(),
        delta_q: DeltaQParams {
            present: false,
            res: 0,
        },
        delta_lf: DeltaLfParams {
            present: false,
            res: 0,
            multi: false,
        },
        loop_filter: LoopFilterParams {
            levels: [0; 4],
            sharpness: 0,
            delta_enabled: false,
            delta_update: false,
            ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            mode_deltas: [0; 2],
        },
        cdef: CdefParams {
            enabled: false,
            damping: 0,
            bits: 0,
            strengths: [crate::av1::frame::CdefStrength {
                y_pri: 0,
                y_sec: 0,
                uv_pri: 0,
                uv_sec: 0,
            }; 8],
        },
        restoration: RestorationParams {
            uses_lr: false,
            lr_type: [0; 3],
            unit_shift: 0,
            uv_unit_shift: 0,
        },
        film_grain: None,
        tx_mode: TxMode::Only4x4,
        reduced_tx_set: false,
        uncompressed_header_bits: 0,
        payload_after_header_offset: 0,
    }
}

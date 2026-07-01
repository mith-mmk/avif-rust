use super::cdf::CdfContext;
use super::entropy::EntropyDecoder;
use super::frame::{FrameHeader, RestorationParams};
use super::syntax::{BlockSize, TxSize, TxType};
use super::transform::{TransformBlock, coefficient_scan};
use crate::DecoderError;

mod block_syntax;
mod coefficient;
mod coefficient_context;
mod context_grid;
mod context_state;
mod decode_flow;
mod diagnostic;
mod palette;
mod partition_syntax;
mod public_api;
mod reconstruction;
mod residual_decode;
mod residual_preview;
mod residual_probe;
mod restoration_syntax;
mod syntax_helpers;
mod tx_type_syntax;

#[cfg(test)]
#[allow(unused_imports)]
use coefficient_context::{
    BR_CDF_SIZE, COEFF_BR_CDF_ROUNDS, COEFF_CONTEXT_BITS, COEFFICIENT_LEVEL_MASK,
    MAX_BASE_BR_RANGE, NUM_BASE_LEVELS, clamp_coefficient_level, coeff_base_context_1d,
    coeff_base_context_2d, coeff_base_eob_context, coeff_base_non_zero_count, coeff_br_context_1d,
    coeff_br_context_2d, eob_base_from_pt, eob_multisize, eob_tx_class_context, first_signed_coeff,
};
use coefficient_context::{
    TxbContext, coefficient_entropy_context, set_txb_entropy_context, txb_context,
};
pub use diagnostic::{
    BlockModeProbe, DecodedBlockPrefix, DecodedLumaBlock, DecodedTransform, PartitionProbe,
    ResidualProbe, TileEntropyState,
};
use diagnostic::{CoeffBaseProbe, CoeffBaseRead, CoeffBrProbe, CoeffSignRead};
pub use public_api::{
    decode_first_luma_block, decode_first_luma_transform, decode_luma_root_block_prefix,
    decode_luma_root_blocks, prepare_tile_entropy, probe_first_block_residuals,
    probe_tile_block_modes, probe_tile_partitions,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlaneEntropyContexts {
    above: Vec<u8>,
    left: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PalettePlaneInfo {
    colors: Vec<u16>,
    color_map: Vec<u8>,
    map_width: usize,
    map_height: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteBlockInfo {
    y: Option<PalettePlaneInfo>,
    uv: Option<PalettePlaneInfo>,
}

pub struct TileDecoder<'a> {
    reader: EntropyDecoder<'a>,
    cdf: CdfContext,
    mi_cols: usize,
    mi_rows: usize,
    y_mode_grid: Vec<Option<usize>>,
    y_palette_size_grid: Vec<Option<usize>>,
    uv_palette_size_grid: Vec<Option<usize>>,
    y_palette_colors_grid: Vec<Option<Vec<u16>>>,
    u_palette_colors_grid: Vec<Option<Vec<u16>>>,
    y_smooth_grid: Vec<Option<bool>>,
    uv_smooth_grid: Vec<Option<bool>>,
    skip_grid: Vec<Option<bool>>,
    above_partition_context: Vec<u8>,
    left_partition_context: Vec<u8>,
    cdef_transmitted: [bool; 4],
    above_txfm_context: Vec<usize>,
    left_txfm_context: Vec<usize>,
    plane_entropy_contexts: [PlaneEntropyContexts; 3],
    restoration: RestorationParams,
    wiener_refs: [[[i16; 3]; 2]; 3],
    sgrproj_refs: [[i16; 2]; 3],
}

impl<'a> TileDecoder<'a> {
    pub fn new(payload: &'a [u8], frame: &FrameHeader) -> Result<Self, DecoderError> {
        let mi_cols = (usize::try_from(frame.frame_width)
            .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?
            + 3)
            >> 2;
        let mi_rows = (usize::try_from(frame.frame_height).map_err(|_| {
            DecoderError::InvalidParam("AV1 frame height is too large".to_string())
        })? + 3)
            >> 2;
        Ok(Self {
            reader: EntropyDecoder::new(payload, frame.disable_cdf_update)?,
            cdf: CdfContext::new(frame.base_q_idx),
            mi_cols,
            mi_rows,
            y_mode_grid: vec![None; mi_cols * mi_rows],
            y_palette_size_grid: vec![None; mi_cols * mi_rows],
            uv_palette_size_grid: vec![None; mi_cols * mi_rows],
            y_palette_colors_grid: vec![None; mi_cols * mi_rows],
            u_palette_colors_grid: vec![None; mi_cols * mi_rows],
            y_smooth_grid: vec![None; mi_cols * mi_rows],
            uv_smooth_grid: vec![None; mi_cols * mi_rows],
            skip_grid: vec![None; mi_cols * mi_rows],
            above_partition_context: vec![0; mi_cols],
            left_partition_context: vec![0; mi_rows],
            cdef_transmitted: [false; 4],
            above_txfm_context: vec![0; mi_cols],
            left_txfm_context: vec![0; mi_rows],
            plane_entropy_contexts: std::array::from_fn(|_| PlaneEntropyContexts {
                above: vec![0; mi_cols],
                left: vec![0; mi_rows],
            }),
            restoration: frame.restoration,
            wiener_refs: [[[3, -7, 15]; 2]; 3],
            sgrproj_refs: [[-32, 31]; 3],
        })
    }

    pub(super) fn txb_context(
        &self,
        block_size: BlockSize,
        transform: TransformBlock,
    ) -> TxbContext {
        let contexts = &self.plane_entropy_contexts[transform.plane];
        txb_context(block_size, transform, &contexts.above, &contexts.left)
    }

    pub(super) fn set_txb_entropy_context(&mut self, transform: TransformBlock, value: u8) {
        let contexts = &mut self.plane_entropy_contexts[transform.plane];
        set_txb_entropy_context(transform, value, &mut contexts.above, &mut contexts.left);
    }
}

#[cfg(test)]
#[path = "tests/tile_decode_coeff.rs"]
mod coeff_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::transform::plan_transform_blocks_with_tx_size;
    use crate::av1::{
        Partition, PredictionMode, UvPredictionMode, alloc_frame_buffers, build_still_decode_plan,
        parse_frame_header, parse_sequence_header, parse_tile_group,
    };
    use crate::container::parse_avif;
    use crate::obu::{ObuType, find_obu_payload};

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

    #[test]
    fn reads_sample_root_partition_symbol() {
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
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let tile_payload = &tile_group.tiles[0];
        let payload = &frame_payload[tile_payload.offset..tile_payload.offset + tile_payload.len];
        let mut decoder = TileDecoder::new(payload, &frame).unwrap();

        let probe = decoder
            .read_root_partition(&plan.tiles[0], &sequence)
            .unwrap();

        assert_eq!(probe.tile_id, 0);
        assert_eq!(probe.block_size, BlockSize::Block128x128);
        assert_eq!(probe.symbol, 3);
        assert_eq!(probe.partition, Partition::Split);
        assert!(probe.bit_position_after >= 15);
    }

    #[test]
    fn reads_sample_first_block_mode_symbols() {
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
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();

        let probes =
            probe_tile_block_modes(frame_payload, &tile_group, &sequence, &frame, &plan).unwrap();

        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].tile_id, 0);
        assert_eq!(probes[0].block_size, BlockSize::Block64x64);
        assert_eq!(probes[0].skip_symbol, 0);
        assert_eq!(probes[0].cdef_idx, Some(0));
        assert_eq!(probes[0].y_mode_symbol, 0);
        assert_eq!(probes[0].y_mode, PredictionMode::Dc);
        assert_eq!(probes[0].uv_mode_symbol, Some(0));
        assert_eq!(
            probes[0].uv_mode,
            Some(UvPredictionMode::Intra(PredictionMode::Dc))
        );
        assert_eq!(probes[0].tx_size_symbol, Some(0));
        assert_eq!(probes[0].tx_size, TxSize::Tx64x64);
        assert!(probes[0].bit_position_after > 15);
    }

    #[test]
    fn plans_sample_first_block_transforms() {
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
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let probes =
            probe_tile_block_modes(frame_payload, &tile_group, &sequence, &frame, &plan).unwrap();

        let transforms = plan_transform_blocks_with_tx_size(
            0,
            0,
            0,
            probes[0].block_size,
            probes[0].tx_size,
            plan.width,
            plan.height,
        );

        assert_eq!(transforms.len(), 1);
        assert!(transforms.iter().all(|tx| tx.plane == 0));
        assert!(transforms.iter().all(|tx| tx.tx_size == probes[0].tx_size));
    }

    #[test]
    fn probes_sample_first_block_residual_plan() {
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
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let probes =
            probe_first_block_residuals(frame_payload, &tile_group, &sequence, &frame, &plan)
                .unwrap();

        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].tile_id, 0);
        assert_eq!(probes[0].block_size, BlockSize::Block64x64);
        let first_tx_size = probes[0]
            .first_tx_size
            .expect("sample first transform size should be known");
        let transform_count = (probes[0].block_size.width() / first_tx_size.width())
            * (probes[0].block_size.height() / first_tx_size.height());
        let tx_sample_count = first_tx_size.sample_count();
        assert_eq!(probes[0].transform_count, transform_count);
        if probes[0].skipped {
            assert_eq!(probes[0].zero_transform_count, transform_count);
            assert_eq!(probes[0].txb_skip_context, None);
            assert_eq!(probes[0].all_zero_symbol, None);
            assert_eq!(probes[0].first_non_zero_transform_index, None);
            assert_eq!(probes[0].first_non_zero_transform, None);
            assert_eq!(probes[0].first_non_zero_tx_size, None);
            assert!(!probes[0].tx_type_read);
            assert_eq!(probes[0].tx_type_set, None);
            assert_eq!(probes[0].tx_type_symbol, None);
            assert_eq!(probes[0].tx_type, None);
            assert_eq!(probes[0].coeff_base_eob_context, None);
            assert_eq!(probes[0].coeff_base_eob_symbol, None);
            assert_eq!(probes[0].coeff_base_eob_level, None);
            assert_eq!(probes[0].regular_coeff_base_count, None);
            assert_eq!(probes[0].regular_coeff_base_decoded_count, None);
            assert_eq!(probes[0].coeff_base_non_zero_count, None);
            assert_eq!(probes[0].coeff_base_range_count, None);
            assert_eq!(probes[0].coeff_br_decoded_count, None);
            assert_eq!(probes[0].first_coeff_br_scan_index, None);
            assert_eq!(probes[0].first_coeff_br_context, None);
            assert_eq!(probes[0].first_coeff_br_symbol, None);
            assert_eq!(probes[0].first_coeff_br_level, None);
            assert_eq!(probes[0].sign_decoded_count, None);
            assert_eq!(probes[0].dc_sign_context, None);
            assert_eq!(probes[0].dc_sign_symbol, None);
            assert_eq!(probes[0].first_ac_sign_scan_index, None);
            assert_eq!(probes[0].first_ac_sign_bit, None);
            assert_eq!(probes[0].golomb_decoded_count, None);
            assert_eq!(probes[0].first_golomb_scan_index, None);
            assert_eq!(probes[0].first_golomb_value, None);
            assert_eq!(probes[0].signed_coeff_non_zero_count, None);
            assert_eq!(probes[0].first_signed_coeff_scan_index, None);
            assert_eq!(probes[0].first_signed_coeff_position, None);
            assert_eq!(probes[0].first_signed_coeff_value, None);
            assert_eq!(probes[0].dequant_non_zero_count, None);
            assert_eq!(probes[0].first_dequant_coeff_position, None);
            assert_eq!(probes[0].first_dequant_coeff_value, None);
            assert_eq!(probes[0].residual_preview_tx_type, None);
            assert_eq!(probes[0].residual_preview_sample_count, None);
            assert_eq!(probes[0].first_residual_preview_sample, None);
            assert_eq!(probes[0].first_coeff_base_scan_index, None);
            assert_eq!(probes[0].first_coeff_base_context, None);
            assert_eq!(probes[0].first_coeff_base_symbol, None);
            assert_eq!(probes[0].first_coeff_base_level, None);
            assert_eq!(probes[0].first_quantized_coefficients, None);
        } else {
            assert!(probes[0].txb_skip_context.unwrap() <= 1);
            assert!(probes[0].all_zero_symbol.unwrap() <= 1);
            assert_eq!(
                probes[0].zero_transform_count,
                probes[0]
                    .first_non_zero_transform_index
                    .unwrap_or(transform_count)
            );
            if probes[0].first_non_zero_transform_index.is_none() {
                assert_eq!(probes[0].eob_multisize, None);
                assert_eq!(probes[0].eob_pt_symbol, None);
                assert_eq!(probes[0].eob_base, None);
                assert_eq!(probes[0].eob_extra_symbol, None);
                assert_eq!(probes[0].eob, None);
                assert_eq!(probes[0].first_non_zero_transform, None);
                assert!(!probes[0].tx_type_read);
                assert_eq!(probes[0].tx_type_set, None);
                assert_eq!(probes[0].tx_type_symbol, None);
                assert_eq!(probes[0].tx_type, None);
                assert_eq!(probes[0].coeff_base_eob_context, None);
                assert_eq!(probes[0].coeff_base_eob_symbol, None);
                assert_eq!(probes[0].coeff_base_eob_level, None);
                assert_eq!(probes[0].regular_coeff_base_count, None);
                assert_eq!(probes[0].regular_coeff_base_decoded_count, None);
                assert_eq!(probes[0].coeff_base_non_zero_count, None);
                assert_eq!(probes[0].coeff_base_range_count, None);
                assert_eq!(probes[0].coeff_br_decoded_count, None);
                assert_eq!(probes[0].first_coeff_br_scan_index, None);
                assert_eq!(probes[0].first_coeff_br_context, None);
                assert_eq!(probes[0].first_coeff_br_symbol, None);
                assert_eq!(probes[0].first_coeff_br_level, None);
                assert_eq!(probes[0].sign_decoded_count, None);
                assert_eq!(probes[0].dc_sign_context, None);
                assert_eq!(probes[0].dc_sign_symbol, None);
                assert_eq!(probes[0].first_ac_sign_scan_index, None);
                assert_eq!(probes[0].first_ac_sign_bit, None);
                assert_eq!(probes[0].golomb_decoded_count, None);
                assert_eq!(probes[0].first_golomb_scan_index, None);
                assert_eq!(probes[0].first_golomb_value, None);
                assert_eq!(probes[0].signed_coeff_non_zero_count, None);
                assert_eq!(probes[0].first_signed_coeff_scan_index, None);
                assert_eq!(probes[0].first_signed_coeff_position, None);
                assert_eq!(probes[0].first_signed_coeff_value, None);
                assert_eq!(probes[0].dequant_non_zero_count, None);
                assert_eq!(probes[0].first_dequant_coeff_position, None);
                assert_eq!(probes[0].first_dequant_coeff_value, None);
                assert_eq!(probes[0].residual_preview_tx_type, None);
                assert_eq!(probes[0].residual_preview_sample_count, None);
                assert_eq!(probes[0].first_residual_preview_sample, None);
                assert_eq!(probes[0].first_coeff_base_scan_index, None);
                assert_eq!(probes[0].first_coeff_base_context, None);
                assert_eq!(probes[0].first_coeff_base_symbol, None);
                assert_eq!(probes[0].first_coeff_base_level, None);
                assert_eq!(probes[0].first_quantized_coefficients, None);
            } else {
                assert!(probes[0].first_non_zero_transform_index.unwrap() < transform_count);
                assert_eq!(
                    probes[0].first_non_zero_transform.unwrap().tx_size,
                    first_tx_size
                );
                assert_eq!(probes[0].first_non_zero_tx_size, Some(first_tx_size));
                assert_eq!(
                    probes[0].eob_multisize,
                    Some(eob_multisize(probes[0].first_non_zero_transform.unwrap()))
                );
                assert!(probes[0].eob_pt_symbol.unwrap() < 11);
                assert_eq!(
                    probes[0].eob_pt.unwrap(),
                    probes[0].eob_pt_symbol.unwrap() + 1
                );
                assert!(probes[0].eob_base.unwrap() > 0);
                assert_eq!(
                    probes[0].eob_extra_context,
                    probes[0].eob_pt.filter(|pt| *pt >= 3).map(|pt| pt - 3)
                );
                assert!(probes[0].eob_extra_symbol.unwrap_or(0) <= 1);
                assert_eq!(
                    probes[0].eob_extra_literal_bits,
                    Some(probes[0].eob_pt.unwrap().saturating_sub(3))
                );
                assert!(probes[0].eob.unwrap() >= probes[0].eob_base.unwrap());
                assert!(probes[0].eob.unwrap() <= tx_sample_count);
                assert!(!probes[0].tx_type_read);
                assert_eq!(probes[0].tx_type_set, None);
                assert_eq!(probes[0].tx_type_symbol, None);
                assert_eq!(probes[0].tx_type, Some(TxType::DctDct));
                assert!(probes[0].coeff_base_eob_context.unwrap() < 4);
                assert!(probes[0].coeff_base_eob_symbol.unwrap() < 3);
                assert_eq!(
                    probes[0].coeff_base_eob_level.unwrap(),
                    probes[0].coeff_base_eob_symbol.unwrap() + 1
                );
                assert_eq!(
                    probes[0].regular_coeff_base_count,
                    Some(probes[0].eob.unwrap() - 1)
                );
                assert_eq!(
                    probes[0].regular_coeff_base_decoded_count,
                    probes[0].regular_coeff_base_count
                );
                assert!(probes[0].coeff_base_non_zero_count.unwrap() >= 1);
                assert!(probes[0].coeff_base_non_zero_count.unwrap() <= probes[0].eob.unwrap());
                assert!(
                    probes[0].coeff_base_range_count.unwrap()
                        <= probes[0].coeff_base_non_zero_count.unwrap()
                );
                assert!(
                    probes[0].coeff_br_decoded_count.unwrap()
                        >= probes[0].coeff_base_range_count.unwrap()
                );
                assert_eq!(
                    probes[0].sign_decoded_count,
                    probes[0].coeff_base_non_zero_count
                );
                assert_eq!(
                    probes[0].signed_coeff_non_zero_count,
                    probes[0].coeff_base_non_zero_count
                );
                assert!(probes[0].first_signed_coeff_scan_index.unwrap() < probes[0].eob.unwrap());
                assert!(probes[0].first_signed_coeff_position.unwrap() < tx_sample_count);
                assert_ne!(probes[0].first_signed_coeff_value.unwrap(), 0);
                assert_eq!(
                    probes[0].dequant_non_zero_count,
                    probes[0].signed_coeff_non_zero_count
                );
                assert!(probes[0].first_dequant_coeff_position.unwrap() < tx_sample_count);
                assert_ne!(probes[0].first_dequant_coeff_value.unwrap(), 0);
                if matches!(
                    probes[0].tx_type,
                    Some(
                        TxType::DctDct
                            | TxType::Identity
                            | TxType::VerticalDct
                            | TxType::HorizontalDct
                    )
                ) {
                    assert_eq!(probes[0].residual_preview_tx_type, probes[0].tx_type);
                    assert_eq!(
                        probes[0].residual_preview_sample_count,
                        Some(tx_sample_count)
                    );
                    assert!(probes[0].first_residual_preview_sample.is_some());
                } else {
                    assert_eq!(probes[0].residual_preview_tx_type, None);
                    assert_eq!(probes[0].residual_preview_sample_count, None);
                    assert_eq!(probes[0].first_residual_preview_sample, None);
                }
                if probes[0].dc_sign_symbol.is_some() {
                    assert!(probes[0].dc_sign_context.unwrap() < 3);
                    assert!(probes[0].dc_sign_symbol.unwrap() <= 1);
                }
                assert!(
                    probes[0].golomb_decoded_count.unwrap()
                        <= probes[0].sign_decoded_count.unwrap()
                );
                if probes[0].sign_decoded_count.unwrap()
                    > usize::from(probes[0].dc_sign_symbol.is_some())
                {
                    assert!(probes[0].first_ac_sign_scan_index.unwrap() < probes[0].eob.unwrap());
                    assert!(probes[0].first_ac_sign_bit.unwrap() <= 1);
                } else {
                    assert_eq!(probes[0].first_ac_sign_scan_index, None);
                    assert_eq!(probes[0].first_ac_sign_bit, None);
                }
                if probes[0].golomb_decoded_count.unwrap() > 0 {
                    assert!(probes[0].first_golomb_scan_index.unwrap() < probes[0].eob.unwrap());
                    assert!(probes[0].first_golomb_value.is_some());
                } else {
                    assert_eq!(probes[0].first_golomb_scan_index, None);
                    assert_eq!(probes[0].first_golomb_value, None);
                }
                if probes[0].coeff_base_range_count.unwrap() > 0 {
                    assert!(probes[0].first_coeff_br_scan_index.unwrap() < probes[0].eob.unwrap());
                    assert!(probes[0].first_coeff_br_position.unwrap() < tx_sample_count);
                    assert!(probes[0].first_coeff_br_context.unwrap() < 21);
                    assert!(probes[0].first_coeff_br_symbol.unwrap() < 4);
                    assert!(probes[0].first_coeff_br_level.unwrap() >= 3);
                } else {
                    assert_eq!(probes[0].first_coeff_br_scan_index, None);
                    assert_eq!(probes[0].first_coeff_br_context, None);
                    assert_eq!(probes[0].first_coeff_br_symbol, None);
                    assert_eq!(probes[0].first_coeff_br_level, None);
                }
                if probes[0].regular_coeff_base_count.unwrap() > 0 {
                    assert_eq!(
                        probes[0].first_coeff_base_scan_index,
                        Some(probes[0].eob.unwrap() - 2)
                    );
                    assert!(probes[0].first_coeff_base_position.unwrap() < tx_sample_count);
                    assert!(probes[0].first_coeff_base_context.unwrap() < 42);
                    assert!(probes[0].first_coeff_base_reference_magnitude.unwrap() <= 15);
                    assert!(probes[0].first_coeff_base_symbol.unwrap() < 4);
                    assert_eq!(
                        probes[0].first_coeff_base_level,
                        probes[0].first_coeff_base_symbol
                    );
                }
                assert_eq!(
                    probes[0]
                        .first_quantized_coefficients
                        .as_ref()
                        .unwrap()
                        .len(),
                    tx_sample_count
                );
                let coefficients = probes[0].first_quantized_coefficients.as_ref().unwrap();
                assert_eq!(coefficients[0], -468);
                assert_eq!(coefficients.iter().filter(|value| **value != 0).count(), 1);
            }
        }
        assert_eq!(probes[0].first_tx_size, Some(first_tx_size));
    }

    #[test]
    fn decodes_sample_first_luma_transform_into_frame_buffer() {
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
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let mut buffers = alloc_frame_buffers(&plan).unwrap();
        buffers.planes[0].samples.fill(u16::MAX);

        let residual = decode_first_luma_transform(
            frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
        )
        .unwrap();

        assert_eq!(residual.tile_id, 0);
        assert!(residual.first_tx_size.is_some());
        if residual.first_non_zero_transform.is_none() {
            assert_eq!(residual.zero_transform_count, residual.transform_count);
        } else {
            assert!(
                buffers.planes[0]
                    .samples
                    .iter()
                    .any(|sample| *sample != u16::MAX)
            );
        }
    }

    #[test]
    fn decodes_sample_first_luma_block_transforms_into_frame_buffer() {
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
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let mut buffers = alloc_frame_buffers(&plan).unwrap();
        for plane in &mut buffers.planes {
            plane.samples.fill(u16::MAX);
        }

        let decoded = decode_first_luma_block(
            frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
        )
        .unwrap();

        assert!(
            decoded
                .iter()
                .all(|transform| transform.transform.plane == 0)
        );
        assert!(
            buffers
                .planes
                .iter()
                .all(|plane| { plane.samples.iter().any(|sample| *sample != u16::MAX) })
        );
    }

    #[test]
    fn decodes_sample_luma_root_block_prefix_with_split_children() {
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
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let mut buffers = alloc_frame_buffers(&plan).unwrap();

        let prefix = decode_luma_root_block_prefix(
            frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
            8,
        )
        .unwrap();
        let blocks = prefix.blocks;

        assert_eq!(blocks.len(), 8);
        assert_eq!((blocks[0].x, blocks[0].y), (0, 0));
        assert_eq!((blocks[1].x, blocks[1].y), (64, 0));
        assert!(blocks.iter().any(|block| !block.transforms.is_empty()));
        assert!(buffers.planes[1].samples.iter().any(|sample| *sample != 0));
        assert!(buffers.planes[2].samples.iter().any(|sample| *sample != 0));
        assert_eq!(prefix.next_unsupported, None);
    }

    #[test]
    fn decodes_sample_prefix_through_palette_blocks() {
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
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let mut buffers = alloc_frame_buffers(&plan).unwrap();

        let prefix = decode_luma_root_block_prefix(
            frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
            4096,
        )
        .unwrap();

        assert_eq!(prefix.blocks.len(), 2037);
        assert_eq!(prefix.next_unsupported, None);
        assert!(
            buffers
                .planes
                .iter()
                .all(|plane| { plane.samples.iter().any(|sample| *sample != 0) })
        );
    }

    #[test]
    fn coeff_base_context_2d_matches_square_offset_rules() {
        let mut quant = vec![0; super::super::syntax::TxSize::Tx32x32.sample_count()];

        assert_eq!(
            coeff_base_context_2d(super::super::syntax::TxSize::Tx32x32, 0, &quant).unwrap(),
            (0, 0)
        );

        quant[2] = 3;
        assert_eq!(
            coeff_base_context_2d(super::super::syntax::TxSize::Tx32x32, 1, &quant).unwrap(),
            (3, 3)
        );

        assert_eq!(
            coeff_base_context_2d(super::super::syntax::TxSize::Tx32x32, 4 * 32 + 4, &quant)
                .unwrap(),
            (21, 0)
        );
    }

    #[test]
    fn txb_context_uses_neighbor_levels_and_dc_signs() {
        let transform = TransformBlock {
            plane: 0,
            x: 4,
            y: 4,
            tx_size: TxSize::Tx4x4,
        };
        let mut above = vec![0; 8];
        let mut left = vec![0; 8];

        assert_eq!(
            txb_context(BlockSize::Block8x8, transform, &above, &left),
            TxbContext {
                skip: 1,
                dc_sign: 0
            }
        );

        above[1] = 4 | (2 << COEFF_CONTEXT_BITS);
        left[1] = 2 | (1 << COEFF_CONTEXT_BITS);
        assert_eq!(
            txb_context(BlockSize::Block8x8, transform, &above, &left),
            TxbContext {
                skip: 5,
                dc_sign: 0
            }
        );

        left[1] = 0;
        assert_eq!(
            txb_context(BlockSize::Block8x8, transform, &above, &left),
            TxbContext {
                skip: 3,
                dc_sign: 2
            }
        );
    }

    #[test]
    fn chroma_txb_skip_context_uses_non_zero_neighbors_and_block_area() {
        let transform = TransformBlock {
            plane: 1,
            x: 0,
            y: 0,
            tx_size: TxSize::Tx4x4,
        };
        assert_eq!(
            txb_context(BlockSize::Block4x4, transform, &[1], &[0]).skip,
            8
        );
        assert_eq!(
            txb_context(BlockSize::Block8x8, transform, &[1], &[1]).skip,
            12
        );
    }

    #[test]
    fn coefficient_entropy_context_caps_level_and_encodes_dc_sign() {
        assert_eq!(coefficient_entropy_context(&[0, 0]), 0);
        assert_eq!(coefficient_entropy_context(&[2, 3]), 5 | 16);
        assert_eq!(coefficient_entropy_context(&[-10, 4]), 7 | 8);

        let transform = TransformBlock {
            plane: 0,
            x: 4,
            y: 8,
            tx_size: TxSize::Tx8x8,
        };
        let mut above = vec![0; 8];
        let mut left = vec![0; 8];
        set_txb_entropy_context(transform, 23, &mut above, &mut left);
        assert_eq!(&above[1..3], &[23, 23]);
        assert_eq!(&left[2..4], &[23, 23]);
    }

    #[test]
    fn eob_context_distinguishes_2d_and_directional_transforms() {
        assert_eq!(eob_tx_class_context(TxType::DctDct), 0);
        assert_eq!(eob_tx_class_context(TxType::Identity), 0);
        assert_eq!(eob_tx_class_context(TxType::VerticalDct), 1);
        assert_eq!(eob_tx_class_context(TxType::HorizontalDct), 1);
    }

    #[test]
    fn coeff_br_context_2d_matches_square_tx_rules() {
        let mut quant = vec![0; super::super::syntax::TxSize::Tx32x32.sample_count()];

        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 0, &quant).unwrap(),
            0
        );

        quant[1] = 3;
        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 0, &quant).unwrap(),
            2
        );

        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 32 + 1, &quant).unwrap(),
            7
        );

        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 4 * 32 + 4, &quant).unwrap(),
            14
        );
    }

    #[test]
    fn directional_coefficient_contexts_follow_aom_1d_axes() {
        let tx_size = super::super::syntax::TxSize::Tx8x8;
        let mut quant = vec![0; tx_size.sample_count()];
        quant[2] = 3;
        quant[16] = 2;

        assert_eq!(
            coeff_base_context_1d(tx_size, TxType::VerticalDct, 1, &quant).unwrap(),
            (33, 3)
        );
        assert_eq!(
            coeff_base_context_1d(tx_size, TxType::HorizontalDct, 0, &quant).unwrap(),
            (0, 2)
        );
        assert_eq!(
            coeff_br_context_1d(tx_size, TxType::VerticalDct, 0, &quant).unwrap(),
            2
        );
        assert_eq!(
            coeff_br_context_1d(tx_size, TxType::HorizontalDct, 8, &quant).unwrap(),
            15
        );
    }

    #[test]
    fn coefficient_level_is_clamped_to_av1_twenty_bit_range() {
        assert_eq!(clamp_coefficient_level(0), 0);
        assert_eq!(clamp_coefficient_level(COEFFICIENT_LEVEL_MASK), 0x0f_ffff);
        assert_eq!(clamp_coefficient_level(1 << 20), 0);
        assert_eq!(clamp_coefficient_level((1 << 20) + 7), 7);
    }
}

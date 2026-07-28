use super::coefficient_context::eob_multisize;
use super::*;
use crate::av1::{
    BlockSize, TxType, build_still_decode_plan, parse_frame_header, parse_sequence_header,
    parse_tile_group,
};
use crate::container::parse_avif;
use crate::obu::{ObuType, find_obu_payload};

#[test]
fn probes_sample_first_block_residual_plan() {
    let Some(data) = crate::test_support::wml2viewer_avif() else {
        return;
    };
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
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
        probe_first_block_residuals(frame_payload, &tile_group, &sequence, &frame, &plan).unwrap();

    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].tile_id, 0);
    assert_eq!(probes[0].block_size, BlockSize::Block64x64);
    let first_tx_size = probes[0]
        .first_tx_size
        .expect("sample first transform size should be known");
    let transform_count = (probes[0].block_size.width() / first_tx_size.width())
        * (probes[0].block_size.height() / first_tx_size.height());
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
            assert_non_zero_residual_probe(&probes[0], first_tx_size, transform_count);
        }
    }
    assert_eq!(probes[0].first_tx_size, Some(first_tx_size));
}

fn assert_non_zero_residual_probe(
    probe: &ResidualProbe,
    first_tx_size: TxSize,
    transform_count: usize,
) {
    let tx_sample_count = first_tx_size.sample_count();
    assert!(probe.first_non_zero_transform_index.unwrap() < transform_count);
    assert_eq!(
        probe.first_non_zero_transform.unwrap().tx_size,
        first_tx_size
    );
    assert_eq!(probe.first_non_zero_tx_size, Some(first_tx_size));
    assert_eq!(
        probe.eob_multisize,
        Some(eob_multisize(probe.first_non_zero_transform.unwrap()))
    );
    assert!(probe.eob_pt_symbol.unwrap() < 11);
    assert_eq!(probe.eob_pt.unwrap(), probe.eob_pt_symbol.unwrap() + 1);
    assert!(probe.eob_base.unwrap() > 0);
    assert_eq!(
        probe.eob_extra_context,
        probe.eob_pt.filter(|pt| *pt >= 3).map(|pt| pt - 3)
    );
    assert!(probe.eob_extra_symbol.unwrap_or(0) <= 1);
    assert_eq!(
        probe.eob_extra_literal_bits,
        Some(probe.eob_pt.unwrap().saturating_sub(3))
    );
    assert!(probe.eob.unwrap() >= probe.eob_base.unwrap());
    assert!(probe.eob.unwrap() <= tx_sample_count);
    assert!(!probe.tx_type_read);
    assert_eq!(probe.tx_type_set, None);
    assert_eq!(probe.tx_type_symbol, None);
    assert_eq!(probe.tx_type, Some(TxType::DctDct));
    assert!(probe.coeff_base_eob_context.unwrap() < 4);
    assert!(probe.coeff_base_eob_symbol.unwrap() < 3);
    assert_eq!(
        probe.coeff_base_eob_level.unwrap(),
        probe.coeff_base_eob_symbol.unwrap() + 1
    );
    assert_eq!(probe.regular_coeff_base_count, Some(probe.eob.unwrap() - 1));
    assert_eq!(
        probe.regular_coeff_base_decoded_count,
        probe.regular_coeff_base_count
    );
    assert!(probe.coeff_base_non_zero_count.unwrap() >= 1);
    assert!(probe.coeff_base_non_zero_count.unwrap() <= probe.eob.unwrap());
    assert!(probe.coeff_base_range_count.unwrap() <= probe.coeff_base_non_zero_count.unwrap());
    assert!(probe.coeff_br_decoded_count.unwrap() >= probe.coeff_base_range_count.unwrap());
    assert_eq!(probe.sign_decoded_count, probe.coeff_base_non_zero_count);
    assert_eq!(
        probe.signed_coeff_non_zero_count,
        probe.coeff_base_non_zero_count
    );
    assert!(probe.first_signed_coeff_scan_index.unwrap() < probe.eob.unwrap());
    assert!(probe.first_signed_coeff_position.unwrap() < tx_sample_count);
    assert_ne!(probe.first_signed_coeff_value.unwrap(), 0);
    assert_eq!(
        probe.dequant_non_zero_count,
        probe.signed_coeff_non_zero_count
    );
    assert!(probe.first_dequant_coeff_position.unwrap() < tx_sample_count);
    assert_ne!(probe.first_dequant_coeff_value.unwrap(), 0);
    if matches!(
        probe.tx_type,
        Some(TxType::DctDct | TxType::Identity | TxType::VerticalDct | TxType::HorizontalDct)
    ) {
        assert_eq!(probe.residual_preview_tx_type, probe.tx_type);
        assert_eq!(probe.residual_preview_sample_count, Some(tx_sample_count));
        assert!(probe.first_residual_preview_sample.is_some());
    } else {
        assert_eq!(probe.residual_preview_tx_type, None);
        assert_eq!(probe.residual_preview_sample_count, None);
        assert_eq!(probe.first_residual_preview_sample, None);
    }
    if let Some(dc_sign_symbol) = probe.dc_sign_symbol {
        assert!(probe.dc_sign_context.unwrap() < 3);
        assert!(dc_sign_symbol <= 1);
    }
    assert!(probe.golomb_decoded_count.unwrap() <= probe.sign_decoded_count.unwrap());
    if probe.sign_decoded_count.unwrap() > usize::from(probe.dc_sign_symbol.is_some()) {
        assert!(probe.first_ac_sign_scan_index.unwrap() < probe.eob.unwrap());
        assert!(probe.first_ac_sign_bit.unwrap() <= 1);
    } else {
        assert_eq!(probe.first_ac_sign_scan_index, None);
        assert_eq!(probe.first_ac_sign_bit, None);
    }
    if probe.golomb_decoded_count.unwrap() > 0 {
        assert!(probe.first_golomb_scan_index.unwrap() < probe.eob.unwrap());
        assert!(probe.first_golomb_value.is_some());
    } else {
        assert_eq!(probe.first_golomb_scan_index, None);
        assert_eq!(probe.first_golomb_value, None);
    }
    if probe.coeff_base_range_count.unwrap() > 0 {
        assert!(probe.first_coeff_br_scan_index.unwrap() < probe.eob.unwrap());
        assert!(probe.first_coeff_br_position.unwrap() < tx_sample_count);
        assert!(probe.first_coeff_br_context.unwrap() < 21);
        assert!(probe.first_coeff_br_symbol.unwrap() < 4);
        assert!(probe.first_coeff_br_level.unwrap() >= 3);
    } else {
        assert_eq!(probe.first_coeff_br_scan_index, None);
        assert_eq!(probe.first_coeff_br_context, None);
        assert_eq!(probe.first_coeff_br_symbol, None);
        assert_eq!(probe.first_coeff_br_level, None);
    }
    if probe.regular_coeff_base_count.unwrap() > 0 {
        assert_eq!(
            probe.first_coeff_base_scan_index,
            Some(probe.eob.unwrap() - 2)
        );
        assert!(probe.first_coeff_base_position.unwrap() < tx_sample_count);
        assert!(probe.first_coeff_base_context.unwrap() < 42);
        assert!(probe.first_coeff_base_reference_magnitude.unwrap() <= 15);
        assert!(probe.first_coeff_base_symbol.unwrap() < 4);
        assert_eq!(probe.first_coeff_base_level, probe.first_coeff_base_symbol);
    }
    assert_eq!(
        probe.first_quantized_coefficients.as_ref().unwrap().len(),
        tx_sample_count
    );
    let coefficients = probe.first_quantized_coefficients.as_ref().unwrap();
    assert_eq!(coefficients[0], -468);
    assert_eq!(coefficients.iter().filter(|value| **value != 0).count(), 1);
}

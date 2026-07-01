use super::BlockModeProbe;
use super::coefficient::CoefficientRead;
use super::coefficient_context::TxbContext;
use super::diagnostic::ResidualProbe;
use super::diagnostic::{ResidualPreview, TxTypeProbe};
use crate::av1::syntax::{TxSize, TxType};
use crate::av1::transform::TransformBlock;

pub(super) fn empty_residual_probe(
    tile_id: u32,
    block_mode: &BlockModeProbe,
    skipped: bool,
    transform_count: usize,
    zero_transform_count: usize,
    first_tx_size: Option<TxSize>,
    first_transform_all_zero: bool,
) -> ResidualProbe {
    scanned_residual_probe(
        ResidualProbeContext {
            tile_id,
            skipped,
            transform_count,
            first_tx_size,
            bit_position_after: block_mode.bit_position_after,
        },
        block_mode,
        FirstNonZeroTransformScan::empty(zero_transform_count, first_transform_all_zero),
        ResidualProbeFields::default(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ResidualProbeContext {
    pub(super) tile_id: u32,
    pub(super) skipped: bool,
    pub(super) transform_count: usize,
    pub(super) first_tx_size: Option<TxSize>,
    pub(super) bit_position_after: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct FirstNonZeroTransformScan {
    pub(super) txb_skip_context: Option<usize>,
    pub(super) all_zero_symbol: Option<usize>,
    pub(super) first_transform_all_zero: bool,
    pub(super) zero_transform_count: usize,
    pub(super) first_non_zero_transform: Option<TransformBlock>,
    pub(super) first_non_zero_transform_index: Option<usize>,
    pub(super) first_non_zero_txb_context: Option<TxbContext>,
}

impl FirstNonZeroTransformScan {
    fn empty(zero_transform_count: usize, first_transform_all_zero: bool) -> Self {
        Self {
            zero_transform_count,
            first_transform_all_zero,
            ..Self::default()
        }
    }

    pub(super) fn scanning() -> Self {
        Self {
            first_transform_all_zero: true,
            ..Self::default()
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct ResidualProbeFields {
    pub(super) eob_multisize: Option<usize>,
    pub(super) eob_pt_symbol: Option<usize>,
    pub(super) eob_pt: Option<usize>,
    pub(super) eob_base: Option<usize>,
    pub(super) eob_extra_context: Option<usize>,
    pub(super) eob_extra_symbol: Option<usize>,
    pub(super) eob_extra_literal_bits: Option<usize>,
    pub(super) eob: Option<usize>,
    pub(super) tx_type_read: bool,
    pub(super) tx_type_set: Option<usize>,
    pub(super) tx_type_symbol: Option<usize>,
    pub(super) tx_type: Option<TxType>,
    pub(super) coeff_base_eob_context: Option<usize>,
    pub(super) coeff_base_eob_symbol: Option<usize>,
    pub(super) coeff_base_eob_level: Option<usize>,
    pub(super) regular_coeff_base_count: Option<usize>,
    pub(super) regular_coeff_base_decoded_count: Option<usize>,
    pub(super) coeff_base_non_zero_count: Option<usize>,
    pub(super) coeff_base_range_count: Option<usize>,
    pub(super) coeff_br_decoded_count: Option<usize>,
    pub(super) first_coeff_br_scan_index: Option<usize>,
    pub(super) first_coeff_br_position: Option<usize>,
    pub(super) first_coeff_br_context: Option<usize>,
    pub(super) first_coeff_br_symbol: Option<usize>,
    pub(super) first_coeff_br_level: Option<usize>,
    pub(super) sign_decoded_count: Option<usize>,
    pub(super) dc_sign_context: Option<usize>,
    pub(super) dc_sign_symbol: Option<usize>,
    pub(super) first_ac_sign_scan_index: Option<usize>,
    pub(super) first_ac_sign_bit: Option<usize>,
    pub(super) golomb_decoded_count: Option<usize>,
    pub(super) first_golomb_scan_index: Option<usize>,
    pub(super) first_golomb_value: Option<usize>,
    pub(super) signed_coeff_non_zero_count: Option<usize>,
    pub(super) first_signed_coeff_scan_index: Option<usize>,
    pub(super) first_signed_coeff_position: Option<usize>,
    pub(super) first_signed_coeff_value: Option<i32>,
    pub(super) dequant_non_zero_count: Option<usize>,
    pub(super) first_dequant_coeff_position: Option<usize>,
    pub(super) first_dequant_coeff_value: Option<i32>,
    pub(super) residual_preview_tx_type: Option<TxType>,
    pub(super) residual_preview_sample_count: Option<usize>,
    pub(super) first_residual_preview_sample: Option<i32>,
    pub(super) first_coeff_base_scan_index: Option<usize>,
    pub(super) first_coeff_base_position: Option<usize>,
    pub(super) first_coeff_base_context: Option<usize>,
    pub(super) first_coeff_base_reference_magnitude: Option<usize>,
    pub(super) first_coeff_base_symbol: Option<usize>,
    pub(super) first_coeff_base_level: Option<usize>,
    pub(super) first_quantized_coefficients: Option<Vec<i32>>,
}

impl ResidualProbeFields {
    pub(super) fn from_reads(
        tx_type_probe: TxTypeProbe,
        coefficient_read: CoefficientRead,
        residual_preview: Option<ResidualPreview>,
    ) -> Self {
        let coeff_base_read = coefficient_read.base;
        Self {
            eob_multisize: Some(coefficient_read.eob_multisize),
            eob_pt_symbol: Some(coefficient_read.eob_pt_symbol),
            eob_pt: Some(coefficient_read.eob_pt),
            eob_base: Some(coefficient_read.eob_base),
            eob_extra_context: coefficient_read.eob_extra_context,
            eob_extra_symbol: coefficient_read.eob_extra_symbol,
            eob_extra_literal_bits: Some(coefficient_read.eob_extra_literal_bits),
            eob: Some(coefficient_read.eob),
            tx_type_read: tx_type_probe.read,
            tx_type_set: tx_type_probe.set,
            tx_type_symbol: tx_type_probe.symbol,
            tx_type: Some(tx_type_probe.tx_type),
            coeff_base_eob_context: Some(coefficient_read.coeff_base_eob_context),
            coeff_base_eob_symbol: Some(coefficient_read.coeff_base_eob_symbol),
            coeff_base_eob_level: Some(coefficient_read.coeff_base_eob_level),
            regular_coeff_base_count: Some(coeff_base_read.probe.remaining_count),
            regular_coeff_base_decoded_count: Some(coeff_base_read.probe.decoded_count),
            coeff_base_non_zero_count: Some(coeff_base_read.non_zero_count),
            coeff_base_range_count: Some(coeff_base_read.base_range_count),
            coeff_br_decoded_count: Some(coeff_base_read.coeff_br_symbol_count),
            first_coeff_br_scan_index: coeff_base_read.first_coeff_br.map(|first| first.scan_index),
            first_coeff_br_position: coeff_base_read.first_coeff_br.map(|first| first.position),
            first_coeff_br_context: coeff_base_read.first_coeff_br.map(|first| first.context),
            first_coeff_br_symbol: coeff_base_read.first_coeff_br.map(|first| first.symbol),
            first_coeff_br_level: coeff_base_read
                .first_coeff_br
                .map(|first| first.level_after_symbol),
            sign_decoded_count: Some(coeff_base_read.signs.sign_count),
            dc_sign_context: coeff_base_read.signs.dc_sign_context,
            dc_sign_symbol: coeff_base_read.signs.dc_sign_symbol,
            first_ac_sign_scan_index: coeff_base_read.signs.first_ac_sign_scan_index,
            first_ac_sign_bit: coeff_base_read.signs.first_ac_sign_bit,
            golomb_decoded_count: Some(coeff_base_read.signs.golomb_count),
            first_golomb_scan_index: coeff_base_read.signs.first_golomb_scan_index,
            first_golomb_value: coeff_base_read.signs.first_golomb_value,
            signed_coeff_non_zero_count: Some(coeff_base_read.signed_non_zero_count),
            first_signed_coeff_scan_index: coeff_base_read
                .first_signed_coeff
                .map(|first| first.scan_index),
            first_signed_coeff_position: coeff_base_read
                .first_signed_coeff
                .map(|first| first.position),
            first_signed_coeff_value: coeff_base_read.first_signed_coeff.map(|first| first.value),
            dequant_non_zero_count: residual_preview
                .as_ref()
                .map(|preview| preview.dequant_non_zero_count),
            first_dequant_coeff_position: residual_preview
                .as_ref()
                .and_then(|preview| preview.first_dequant_coeff)
                .map(|first| first.position),
            first_dequant_coeff_value: residual_preview
                .as_ref()
                .and_then(|preview| preview.first_dequant_coeff)
                .map(|first| first.value),
            residual_preview_tx_type: residual_preview.as_ref().map(|preview| preview.tx_type),
            residual_preview_sample_count: residual_preview
                .as_ref()
                .map(|preview| preview.residual_sample_count),
            first_residual_preview_sample: residual_preview
                .as_ref()
                .and_then(|preview| preview.first_residual_sample),
            first_coeff_base_scan_index: coeff_base_read.probe.scan_index,
            first_coeff_base_position: coeff_base_read.probe.position,
            first_coeff_base_context: coeff_base_read.probe.context,
            first_coeff_base_reference_magnitude: coeff_base_read.probe.reference_magnitude,
            first_coeff_base_symbol: coeff_base_read.probe.symbol,
            first_coeff_base_level: coeff_base_read.probe.level,
            first_quantized_coefficients: Some(coeff_base_read.base_levels),
        }
    }
}

pub(super) fn scanned_residual_probe(
    context: ResidualProbeContext,
    block_mode: &BlockModeProbe,
    scan: FirstNonZeroTransformScan,
    fields: ResidualProbeFields,
) -> ResidualProbe {
    ResidualProbe {
        tile_id: context.tile_id,
        block_size: block_mode.block_size,
        skipped: context.skipped,
        transform_count: context.transform_count,
        zero_transform_count: scan.zero_transform_count,
        first_tx_size: context.first_tx_size,
        first_non_zero_transform_index: scan.first_non_zero_transform_index,
        first_non_zero_transform: scan.first_non_zero_transform,
        first_non_zero_tx_size: scan
            .first_non_zero_transform
            .map(|transform| transform.tx_size),
        tx_type_read: fields.tx_type_read,
        tx_type_set: fields.tx_type_set,
        tx_type_symbol: fields.tx_type_symbol,
        tx_type: fields.tx_type,
        txb_skip_context: scan.txb_skip_context,
        all_zero_symbol: scan.all_zero_symbol,
        first_transform_all_zero: scan.first_transform_all_zero,
        eob_multisize: fields.eob_multisize,
        eob_pt_symbol: fields.eob_pt_symbol,
        eob_pt: fields.eob_pt,
        eob_base: fields.eob_base,
        eob_extra_context: fields.eob_extra_context,
        eob_extra_symbol: fields.eob_extra_symbol,
        eob_extra_literal_bits: fields.eob_extra_literal_bits,
        eob: fields.eob,
        coeff_base_eob_context: fields.coeff_base_eob_context,
        coeff_base_eob_symbol: fields.coeff_base_eob_symbol,
        coeff_base_eob_level: fields.coeff_base_eob_level,
        regular_coeff_base_count: fields.regular_coeff_base_count,
        regular_coeff_base_decoded_count: fields.regular_coeff_base_decoded_count,
        coeff_base_non_zero_count: fields.coeff_base_non_zero_count,
        coeff_base_range_count: fields.coeff_base_range_count,
        coeff_br_decoded_count: fields.coeff_br_decoded_count,
        first_coeff_br_scan_index: fields.first_coeff_br_scan_index,
        first_coeff_br_position: fields.first_coeff_br_position,
        first_coeff_br_context: fields.first_coeff_br_context,
        first_coeff_br_symbol: fields.first_coeff_br_symbol,
        first_coeff_br_level: fields.first_coeff_br_level,
        sign_decoded_count: fields.sign_decoded_count,
        dc_sign_context: fields.dc_sign_context,
        dc_sign_symbol: fields.dc_sign_symbol,
        first_ac_sign_scan_index: fields.first_ac_sign_scan_index,
        first_ac_sign_bit: fields.first_ac_sign_bit,
        golomb_decoded_count: fields.golomb_decoded_count,
        first_golomb_scan_index: fields.first_golomb_scan_index,
        first_golomb_value: fields.first_golomb_value,
        signed_coeff_non_zero_count: fields.signed_coeff_non_zero_count,
        first_signed_coeff_scan_index: fields.first_signed_coeff_scan_index,
        first_signed_coeff_position: fields.first_signed_coeff_position,
        first_signed_coeff_value: fields.first_signed_coeff_value,
        dequant_non_zero_count: fields.dequant_non_zero_count,
        first_dequant_coeff_position: fields.first_dequant_coeff_position,
        first_dequant_coeff_value: fields.first_dequant_coeff_value,
        residual_preview_tx_type: fields.residual_preview_tx_type,
        residual_preview_sample_count: fields.residual_preview_sample_count,
        first_residual_preview_sample: fields.first_residual_preview_sample,
        first_coeff_base_scan_index: fields.first_coeff_base_scan_index,
        first_coeff_base_position: fields.first_coeff_base_position,
        first_coeff_base_context: fields.first_coeff_base_context,
        first_coeff_base_reference_magnitude: fields.first_coeff_base_reference_magnitude,
        first_coeff_base_symbol: fields.first_coeff_base_symbol,
        first_coeff_base_level: fields.first_coeff_base_level,
        first_quantized_coefficients: fields.first_quantized_coefficients,
        bit_position_after: context.bit_position_after,
    }
}

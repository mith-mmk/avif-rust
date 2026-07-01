use super::diagnostic::{DequantCoeffProbe, ResidualPreview};
use crate::DecoderError;
use crate::av1::quant::{QuantState, dequantize_coefficients};
use crate::av1::syntax::TxType;
use crate::av1::tile_decode::coefficient::CoefficientRead;
use crate::av1::transform::{TransformBlock, inverse_transform};

pub(super) fn build_probe_residual_preview(
    transform: TransformBlock,
    coefficient_read: &CoefficientRead,
    quant_state: QuantState,
    bit_depth: u8,
    tx_type: TxType,
) -> Result<Option<ResidualPreview>, DecoderError> {
    let coeff_base_read = &coefficient_read.base;
    debug_assert_eq!(
        coeff_base_read.base_levels.len(),
        transform.tx_size.sample_count()
    );
    build_residual_preview(
        transform,
        &coeff_base_read.base_levels,
        quant_state,
        bit_depth,
        tx_type,
    )
}

pub(super) fn build_residual_preview(
    transform: TransformBlock,
    coefficients: &[i32],
    quant_state: QuantState,
    bit_depth: u8,
    tx_type: TxType,
) -> Result<Option<ResidualPreview>, DecoderError> {
    if coefficients.len() != transform.tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 residual preview coefficient count does not match transform size".to_string(),
        ));
    }
    if !matches!(
        tx_type,
        TxType::DctDct
            | TxType::AdstDct
            | TxType::DctAdst
            | TxType::AdstAdst
            | TxType::Identity
            | TxType::VerticalDct
            | TxType::HorizontalDct
    ) {
        return Ok(None);
    }
    let plane_quant = quant_state.plane(transform.plane);
    let dequant = dequantize_coefficients(
        coefficients,
        plane_quant,
        bit_depth,
        transform.tx_size.dq_denom(),
    );
    let first_dequant_coeff = dequant
        .iter()
        .copied()
        .enumerate()
        .find_map(|(position, value)| {
            (value != 0).then_some(DequantCoeffProbe { position, value })
        });
    let dequant_non_zero_count = dequant.iter().filter(|value| **value != 0).count();
    let residual = inverse_transform(tx_type, transform.tx_size, &dequant, bit_depth)?;
    let first_residual_sample = residual.first().copied();
    Ok(Some(ResidualPreview {
        tx_type,
        dequant_non_zero_count,
        first_dequant_coeff,
        residual_sample_count: residual.len(),
        first_residual_sample,
    }))
}

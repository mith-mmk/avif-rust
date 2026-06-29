use super::coefficient_context::{
    BR_CDF_SIZE, COEFF_BR_CDF_ROUNDS, MAX_BASE_BR_RANGE, NUM_BASE_LEVELS, clamp_coefficient_level,
    coeff_base_context_1d, coeff_base_context_2d, coeff_base_eob_context,
    coeff_base_non_zero_count, coeff_br_context_1d, coeff_br_context_2d, eob_base_from_pt,
    eob_tx_class_context, first_signed_coeff,
};
use super::{
    CoeffBaseProbe, CoeffBaseRead, CoeffBrProbe, CoeffSignRead, DecoderError, EntropyDecoder,
    TxSize, TxType, coefficient_scan,
};
use crate::av1::cdf::CdfContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CoefficientSymbol {
    EobPoint {
        multisize: usize,
        plane_type: usize,
        tx_class: usize,
    },
    EobExtra {
        tx_size_context: usize,
        plane_type: usize,
        context: usize,
    },
    BaseEob {
        tx_size_context: usize,
        plane_type: usize,
        context: usize,
    },
    Base {
        tx_size_context: usize,
        plane_type: usize,
        context: usize,
    },
    BaseRange {
        tx_size_context: usize,
        plane_type: usize,
        context: usize,
    },
    DcSign {
        plane_type: usize,
        context: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CoefficientLiteral {
    EobExtra { index: usize },
    AcSign { scan_index: usize },
    GolombPrefix { length: usize },
    GolombSuffix { index: usize },
}

pub(super) trait CoefficientTokenSource {
    fn read_symbol(&mut self, symbol: CoefficientSymbol) -> Result<usize, DecoderError>;
    fn read_literal(&mut self, literal: CoefficientLiteral) -> Result<usize, DecoderError>;
}

pub(super) struct EntropyCoefficientSource<'a, 'data> {
    reader: &'a mut EntropyDecoder<'data>,
    cdf: &'a mut CdfContext,
}

impl<'a, 'data> EntropyCoefficientSource<'a, 'data> {
    pub(super) fn new(reader: &'a mut EntropyDecoder<'data>, cdf: &'a mut CdfContext) -> Self {
        Self { reader, cdf }
    }
}

impl CoefficientTokenSource for EntropyCoefficientSource<'_, '_> {
    fn read_symbol(&mut self, symbol: CoefficientSymbol) -> Result<usize, DecoderError> {
        let cdf = match symbol {
            CoefficientSymbol::EobPoint {
                multisize,
                plane_type,
                tx_class,
            } => self.cdf.eob_pt_cdf_mut(multisize, plane_type, tx_class),
            CoefficientSymbol::EobExtra {
                tx_size_context,
                plane_type,
                context,
            } => self
                .cdf
                .eob_extra_cdf_mut(tx_size_context, plane_type, context),
            CoefficientSymbol::BaseEob {
                tx_size_context,
                plane_type,
                context,
            } => self
                .cdf
                .coeff_base_eob_cdf_mut(tx_size_context, plane_type, context),
            CoefficientSymbol::Base {
                tx_size_context,
                plane_type,
                context,
            } => self
                .cdf
                .coeff_base_cdf_mut(tx_size_context, plane_type, context),
            CoefficientSymbol::BaseRange {
                tx_size_context,
                plane_type,
                context,
            } => self
                .cdf
                .coeff_br_cdf_mut(tx_size_context, plane_type, context),
            CoefficientSymbol::DcSign {
                plane_type,
                context,
            } => self.cdf.dc_sign_cdf_mut(plane_type, context),
        };
        self.reader.read_symbol(cdf)
    }

    fn read_literal(&mut self, literal: CoefficientLiteral) -> Result<usize, DecoderError> {
        let label = match literal {
            CoefficientLiteral::EobExtra { .. } => "AV1 eob_extra_bit",
            CoefficientLiteral::AcSign { .. } => "AV1 coeff_sign_bit",
            CoefficientLiteral::GolombPrefix { .. } => "AV1 coeff_golomb_prefix",
            CoefficientLiteral::GolombSuffix { .. } => "AV1 coeff_golomb_suffix",
        };
        self.reader
            .read_literal(1)
            .map(|value| value as usize)
            .map_err(|err| DecoderError::Bitstream(format!("{label}: {err}")))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CoefficientRead {
    pub eob_multisize: usize,
    pub eob_pt_symbol: usize,
    pub eob_pt: usize,
    pub eob_base: usize,
    pub eob_extra_context: Option<usize>,
    pub eob_extra_symbol: Option<usize>,
    pub eob_extra_literal_bits: usize,
    pub eob: usize,
    pub coeff_base_eob_context: usize,
    pub coeff_base_eob_symbol: usize,
    pub coeff_base_eob_level: usize,
    pub base: CoeffBaseRead,
}

pub(super) fn decode_coefficients<S: CoefficientTokenSource>(
    source: &mut S,
    tx_size: TxSize,
    tx_type: TxType,
    plane_type: usize,
    dc_sign_context: usize,
) -> Result<CoefficientRead, DecoderError> {
    let eob_multisize = usize::from(tx_size.width_log2().min(5) + tx_size.height_log2().min(5) - 4);
    let eob_pt_symbol = source.read_symbol(CoefficientSymbol::EobPoint {
        multisize: eob_multisize,
        plane_type,
        tx_class: eob_tx_class_context(tx_type),
    })?;
    let eob_pt = eob_pt_symbol + 1;
    let eob_base = eob_base_from_pt(eob_pt);
    let (eob_extra_context, eob_extra_symbol, eob_extra_literal_bits, eob) =
        read_eob_extra(source, tx_size, plane_type, eob_pt, eob_base)?;
    if eob == 0 || eob > coefficient_scan(tx_size, tx_type).len() {
        return Err(DecoderError::Bitstream(format!(
            "AV1 eob {eob} is invalid for {tx_size:?}"
        )));
    }
    let coeff_base_eob_context = coeff_base_eob_context(tx_size, eob - 1);
    let coeff_base_eob_symbol = source.read_symbol(CoefficientSymbol::BaseEob {
        tx_size_context: tx_size.coeff_cdf_index(),
        plane_type,
        context: coeff_base_eob_context,
    })?;
    let coeff_base_eob_level = coeff_base_eob_symbol + 1;
    let base = read_regular_coeff_bases(
        source,
        tx_size,
        tx_type,
        plane_type,
        eob,
        coeff_base_eob_level,
        dc_sign_context,
    )?;
    Ok(CoefficientRead {
        eob_multisize,
        eob_pt_symbol,
        eob_pt,
        eob_base,
        eob_extra_context,
        eob_extra_symbol,
        eob_extra_literal_bits,
        eob,
        coeff_base_eob_context,
        coeff_base_eob_symbol,
        coeff_base_eob_level,
        base,
    })
}

fn read_eob_extra<S: CoefficientTokenSource>(
    source: &mut S,
    tx_size: TxSize,
    plane_type: usize,
    eob_pt: usize,
    eob_base: usize,
) -> Result<(Option<usize>, Option<usize>, usize, usize), DecoderError> {
    if eob_pt < 3 {
        return Ok((None, None, 0, eob_base));
    }
    let context = eob_pt - 3;
    let symbol = source.read_symbol(CoefficientSymbol::EobExtra {
        tx_size_context: tx_size.coeff_cdf_index(),
        plane_type,
        context,
    })?;
    let literal_bits = eob_pt - 3;
    let mut eob = eob_base + (symbol << literal_bits);
    for index in 0..literal_bits {
        let bit = source.read_literal(CoefficientLiteral::EobExtra { index })?;
        eob += bit << (literal_bits - 1 - index);
    }
    Ok((Some(context), Some(symbol), literal_bits, eob))
}

fn read_regular_coeff_bases<S: CoefficientTokenSource>(
    source: &mut S,
    tx_size: TxSize,
    tx_type: TxType,
    plane_type: usize,
    eob: usize,
    eob_level: usize,
    dc_sign_context: usize,
) -> Result<CoeffBaseRead, DecoderError> {
    let scan = coefficient_scan(tx_size, tx_type);
    let remaining_count = eob - 1;
    let mut quant = vec![0i32; tx_size.sample_count()];
    let mut base_range_count = 0;
    let mut coeff_br_symbol_count = 0;
    let mut first_coeff_br = None;
    let eob_position = scan[eob - 1];
    let eob_level = read_coeff_br_range(
        source,
        tx_size,
        tx_type,
        plane_type,
        eob - 1,
        eob_position,
        eob_level,
        &quant,
        &mut base_range_count,
        &mut coeff_br_symbol_count,
        &mut first_coeff_br,
    )?;
    quant[eob_position] = eob_level as i32;

    let mut first = None;
    let mut decoded_count = 0;
    for scan_index in (0..eob - 1).rev() {
        let position = scan[scan_index];
        let (context, reference_magnitude) = match tx_type {
            TxType::VerticalDct | TxType::HorizontalDct => {
                coeff_base_context_1d(tx_size, tx_type, position, &quant)?
            }
            _ => coeff_base_context_2d(tx_size, position, &quant)?,
        };
        let symbol = source.read_symbol(CoefficientSymbol::Base {
            tx_size_context: tx_size.coeff_cdf_index(),
            plane_type,
            context,
        })?;
        let level = read_coeff_br_range(
            source,
            tx_size,
            tx_type,
            plane_type,
            scan_index,
            position,
            symbol,
            &quant,
            &mut base_range_count,
            &mut coeff_br_symbol_count,
            &mut first_coeff_br,
        )?;
        quant[position] = level as i32;
        decoded_count += 1;
        if first.is_none() {
            first = Some((scan_index, position, context, reference_magnitude, symbol));
        }
    }

    let probe = first.map_or(
        CoeffBaseProbe {
            remaining_count,
            decoded_count: 0,
            scan_index: None,
            position: None,
            context: None,
            reference_magnitude: None,
            symbol: None,
            level: None,
        },
        |(scan_index, position, context, reference_magnitude, symbol)| CoeffBaseProbe {
            remaining_count,
            decoded_count,
            scan_index: Some(scan_index),
            position: Some(position),
            context: Some(context),
            reference_magnitude: Some(reference_magnitude),
            symbol: Some(symbol),
            level: Some(symbol),
        },
    );
    let non_zero_count = coeff_base_non_zero_count(&quant);
    let signs =
        read_coeff_signs_and_golomb(source, plane_type, dc_sign_context, eob, &scan, &mut quant)?;
    let signed_non_zero_count = coeff_base_non_zero_count(&quant);
    let first_signed_coeff = first_signed_coeff(eob, &scan, &quant)?;
    Ok(CoeffBaseRead {
        probe,
        base_levels: quant,
        non_zero_count,
        base_range_count,
        coeff_br_symbol_count,
        first_coeff_br,
        signs,
        signed_non_zero_count,
        first_signed_coeff,
    })
}

#[allow(clippy::too_many_arguments)]
fn read_coeff_br_range<S: CoefficientTokenSource>(
    source: &mut S,
    tx_size: TxSize,
    tx_type: TxType,
    plane_type: usize,
    scan_index: usize,
    position: usize,
    base_level: usize,
    quant: &[i32],
    base_range_count: &mut usize,
    symbol_count: &mut usize,
    first: &mut Option<CoeffBrProbe>,
) -> Result<usize, DecoderError> {
    if base_level <= NUM_BASE_LEVELS {
        return Ok(base_level);
    }
    *base_range_count += 1;
    let context = match tx_type {
        TxType::VerticalDct | TxType::HorizontalDct => {
            coeff_br_context_1d(tx_size, tx_type, position, quant)?
        }
        _ => coeff_br_context_2d(tx_size, position, quant)?,
    };
    let mut level = base_level;
    for _ in 0..COEFF_BR_CDF_ROUNDS {
        let symbol = source.read_symbol(CoefficientSymbol::BaseRange {
            tx_size_context: tx_size.coeff_cdf_index(),
            plane_type,
            context,
        })?;
        level += symbol;
        *symbol_count += 1;
        if first.is_none() {
            *first = Some(CoeffBrProbe {
                scan_index,
                position,
                context,
                symbol,
                level_after_symbol: level,
            });
        }
        if symbol < BR_CDF_SIZE - 1 {
            break;
        }
    }
    Ok(level)
}

fn read_coeff_signs_and_golomb<S: CoefficientTokenSource>(
    source: &mut S,
    plane_type: usize,
    dc_context: usize,
    eob: usize,
    scan: &[usize],
    levels: &mut [i32],
) -> Result<CoeffSignRead, DecoderError> {
    let mut result = CoeffSignRead {
        sign_count: 0,
        dc_sign_context: None,
        dc_sign_symbol: None,
        first_ac_sign_scan_index: None,
        first_ac_sign_bit: None,
        golomb_count: 0,
        first_golomb_scan_index: None,
        first_golomb_value: None,
    };
    for scan_index in 0..eob {
        let position = scan[scan_index];
        let mut level = levels[position].unsigned_abs() as usize;
        if level == 0 {
            continue;
        }
        let sign = if scan_index == 0 {
            let symbol = source.read_symbol(CoefficientSymbol::DcSign {
                plane_type,
                context: dc_context,
            })?;
            result.dc_sign_context = Some(dc_context);
            result.dc_sign_symbol = Some(symbol);
            symbol
        } else {
            let bit = source.read_literal(CoefficientLiteral::AcSign { scan_index })?;
            if result.first_ac_sign_scan_index.is_none() {
                result.first_ac_sign_scan_index = Some(scan_index);
                result.first_ac_sign_bit = Some(bit);
            }
            bit
        };
        result.sign_count += 1;
        if level >= MAX_BASE_BR_RANGE {
            let golomb = read_golomb(source)?;
            level += golomb;
            result.golomb_count += 1;
            if result.first_golomb_scan_index.is_none() {
                result.first_golomb_scan_index = Some(scan_index);
                result.first_golomb_value = Some(golomb);
            }
        }
        level = clamp_coefficient_level(level);
        levels[position] = if sign != 0 {
            -(level as i32)
        } else {
            level as i32
        };
    }
    Ok(result)
}

pub(super) fn read_golomb<S: CoefficientTokenSource>(
    source: &mut S,
) -> Result<usize, DecoderError> {
    let mut value = 1usize;
    let mut length = 0usize;
    loop {
        length += 1;
        if length > 20 {
            return Err(DecoderError::Bitstream(
                "AV1 coeff golomb length exceeds 20 bits".to_string(),
            ));
        }
        if source.read_literal(CoefficientLiteral::GolombPrefix { length })? != 0 {
            break;
        }
    }
    for index in 0..length - 1 {
        value = (value << 1) | source.read_literal(CoefficientLiteral::GolombSuffix { index })?;
    }
    Ok(value - 1)
}

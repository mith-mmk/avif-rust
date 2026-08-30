use super::coefficient_context::{
    BR_CDF_SIZE, COEFF_BR_CDF_ROUNDS, MAX_BASE_BR_RANGE, NUM_BASE_LEVELS, clamp_coefficient_level,
    coeff_base_context_1d, coeff_base_context_2d, coeff_base_eob_context,
    coeff_base_non_zero_count, coeff_br_context_1d, coeff_br_context_2d, eob_base_from_pt,
    eob_tx_class_context, first_signed_coeff, is_directional_tx_type,
};
use super::{
    CoeffBaseProbe, CoeffBaseRead, CoeffBrProbe, CoeffSignRead, DecoderError, EntropyDecoder,
    TxSize, TxType,
};
use crate::av1::cdf::CdfContext;
use crate::av1::syntax::TX_TYPES;
use crate::av1::transform::{coefficient_scan, coefficient_scan_into};
use crate::allocation::{capacity_bytes, fresh_replacement};

const COEFFICIENT_SCAN_ENTRY_COUNT: usize = 19 * TX_TYPES;

fn coefficient_scan_len(tx_size: TxSize) -> usize {
    tx_size.width().min(32) * tx_size.height().min(32)
}

const ALL_TX_SIZES: [TxSize; 19] = [
    TxSize::Tx4x4,
    TxSize::Tx8x8,
    TxSize::Tx16x16,
    TxSize::Tx32x32,
    TxSize::Tx64x64,
    TxSize::Tx4x8,
    TxSize::Tx8x4,
    TxSize::Tx8x16,
    TxSize::Tx16x8,
    TxSize::Tx16x32,
    TxSize::Tx32x16,
    TxSize::Tx32x64,
    TxSize::Tx64x32,
    TxSize::Tx4x16,
    TxSize::Tx16x4,
    TxSize::Tx8x32,
    TxSize::Tx32x8,
    TxSize::Tx16x64,
    TxSize::Tx64x16,
];

#[derive(Debug)]
pub(super) struct CoefficientScanCache {
    entries: Vec<Option<Vec<usize>>>,
    strict_lazy_limit: Option<usize>,
    strict_lazy_bytes: usize,
}

impl CoefficientScanCache {
    pub(super) fn new() -> Self {
        Self {
            entries: (0..COEFFICIENT_SCAN_ENTRY_COUNT).map(|_| None).collect(),
            strict_lazy_limit: None,
            strict_lazy_bytes: 0,
        }
    }

    /// Constructs the cache for the bounded decoder.  The outer entry table is
    /// admitted before it becomes reachable; transform scans remain lazy.
    /// `strict_lazy_limit` is a pre-admitted byte allowance for those later
    /// entries, not a request to materialize every transform.
    pub(super) fn new_strict(strict_lazy_limit: usize) -> Result<Self, DecoderError> {
        let replacement = fresh_replacement::<Option<Vec<usize>>>(
            COEFFICIENT_SCAN_ENTRY_COUNT,
            "native AV1 coefficient scan cache",
        )?;
        let mut entries = replacement.into_vec();
        if entries.capacity() < COEFFICIENT_SCAN_ENTRY_COUNT {
            return Err(DecoderError::InvalidParam(
                "native AV1 coefficient scan cache allocation is too small".to_string(),
            ));
        }
        if entries.capacity() > COEFFICIENT_SCAN_ENTRY_COUNT {
            return Err(DecoderError::InvalidParam(
                "native AV1 coefficient scan cache allocation exceeds plan".to_string(),
            ));
        }
        entries.resize(COEFFICIENT_SCAN_ENTRY_COUNT, None);
        Ok(Self {
            entries,
            strict_lazy_limit: Some(strict_lazy_limit),
            strict_lazy_bytes: 0,
        })
    }

    pub(super) fn strict_lazy_bytes() -> Result<usize, DecoderError> {
        ALL_TX_SIZES.iter().try_fold(0usize, |total, &tx_size| {
            let entry = coefficient_scan_len(tx_size)
                .checked_mul(std::mem::size_of::<usize>())
                .and_then(|bytes| bytes.checked_mul(TX_TYPES))
                .ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 coefficient scan cache size overflows".to_string(),
                    )
                })?;
            total.checked_add(entry).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 coefficient scan cache size overflows".to_string(),
                )
            })
        })
    }

    pub(super) fn strict_outer_bytes() -> Result<usize, DecoderError> {
        capacity_bytes::<Option<Vec<usize>>>(
            COEFFICIENT_SCAN_ENTRY_COUNT,
            "coefficient scan cache",
        )
    }

    pub(super) fn actual_bytes(&self) -> Result<usize, DecoderError> {
        let outer = capacity_bytes::<Option<Vec<usize>>>(
            self.entries.capacity(),
            "coefficient scan cache",
        )?;
        self.entries.iter().try_fold(outer, |total, entry| {
            let inner = entry.as_ref().map_or(0, |scan| {
                scan.capacity()
                    .checked_mul(std::mem::size_of::<usize>())
                    .unwrap_or(usize::MAX)
            });
            total.checked_add(inner).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 coefficient scan cache size overflows".to_string(),
                )
            })
        })
    }

    pub(super) fn get(&mut self, tx_size: TxSize, tx_type: TxType) -> &[usize] {
        let index = usize::from(tx_size as u8) * TX_TYPES + usize::from(tx_type as u8);
        if self.entries[index].is_none() {
            self.entries[index] = Some(coefficient_scan(tx_size, tx_type));
        }
        self.entries[index]
            .as_deref()
            .expect("scan cache entry was just inserted")
    }

    pub(super) fn get_strict(
        &mut self,
        tx_size: TxSize,
        tx_type: TxType,
    ) -> Result<&[usize], DecoderError> {
        let index = usize::from(tx_size as u8) * TX_TYPES + usize::from(tx_type as u8);
        if self.entries[index].is_none() {
            let requested = coefficient_scan_len(tx_size)
                .checked_mul(std::mem::size_of::<usize>())
                .ok_or_else(|| {
                    DecoderError::InvalidParam(
                        "native AV1 coefficient scan entry size overflows".to_string(),
                    )
                })?;
            let limit = self.strict_lazy_limit.ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 strict coefficient scan cache is not configured".to_string(),
                )
            })?;
            let requested_total = self.strict_lazy_bytes.checked_add(requested).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 coefficient scan cache size overflows".to_string(),
                )
            })?;
            if requested_total > limit {
                return Err(DecoderError::InvalidParam(
                    "native AV1 coefficient scan entry exceeds admitted budget".to_string(),
                ));
            }
            let replacement = fresh_replacement::<usize>(
                coefficient_scan_len(tx_size),
                "native AV1 coefficient scan entry",
            )?;
            let actual = replacement.actual_bytes();
            let actual_total = self.strict_lazy_bytes.checked_add(actual).ok_or_else(|| {
                DecoderError::InvalidParam(
                    "native AV1 coefficient scan cache size overflows".to_string(),
                )
            })?;
            if actual_total > limit {
                return Err(DecoderError::InvalidParam(
                    "native AV1 coefficient scan entry capacity exceeds admitted budget".to_string(),
                ));
            }
            let mut scan = replacement.into_vec();
            coefficient_scan_into(tx_size, tx_type, &mut scan);
            self.strict_lazy_bytes = actual_total;
            self.entries[index] = Some(scan);
        }
        Ok(self.entries[index]
            .as_deref()
            .expect("strict scan cache entry was just inserted"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_cache_keeps_scan_entries_lazy_and_reconciles_actual_bytes() {
        let outer = CoefficientScanCache::strict_outer_bytes().unwrap();
        let entry = coefficient_scan_len(TxSize::Tx4x4) * std::mem::size_of::<usize>();
        let mut cache = CoefficientScanCache::new_strict(entry).unwrap();
        assert_eq!(cache.actual_bytes().unwrap(), outer);

        let scan = cache
            .get_strict(TxSize::Tx4x4, TxType::DctDct)
            .unwrap();
        assert_eq!(scan.len(), coefficient_scan_len(TxSize::Tx4x4));
        assert_eq!(cache.actual_bytes().unwrap(), outer + entry);
    }

    #[test]
    fn strict_cache_rejects_one_byte_under_and_allows_retry() {
        let entry = coefficient_scan_len(TxSize::Tx4x4) * std::mem::size_of::<usize>();
        let mut under = CoefficientScanCache::new_strict(entry - 1).unwrap();
        assert!(under
            .get_strict(TxSize::Tx4x4, TxType::DctDct)
            .is_err());
        assert_eq!(under.strict_lazy_bytes, 0);

        let mut retry = CoefficientScanCache::new_strict(entry).unwrap();
        assert!(retry
            .get_strict(TxSize::Tx4x4, TxType::DctDct)
            .is_ok());
        assert_eq!(retry.strict_lazy_bytes, entry);
    }

    #[test]
    fn strict_cache_requires_each_lazy_entry_to_fit_remaining_allowance() {
        let first = coefficient_scan_len(TxSize::Tx4x4) * std::mem::size_of::<usize>();
        let second = coefficient_scan_len(TxSize::Tx8x8) * std::mem::size_of::<usize>();
        let mut cache = CoefficientScanCache::new_strict(first + second - 1).unwrap();
        cache
            .get_strict(TxSize::Tx4x4, TxType::DctDct)
            .unwrap();
        assert!(cache
            .get_strict(TxSize::Tx8x8, TxType::DctDct)
            .is_err());
        assert_eq!(cache.strict_lazy_bytes, first);
    }

    #[test]
    fn strict_cache_rejects_actual_outer_overcapacity_and_retries() {
        let _observation = crate::test_allocation_observer::Observation::begin(1, false);
        let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(
            "native AV1 coefficient scan cache",
            1 << 20,
        );
        let lazy = CoefficientScanCache::strict_lazy_bytes().unwrap();
        assert!(CoefficientScanCache::new_strict(lazy).is_err());
        drop(_extra);
        let cache = CoefficientScanCache::new_strict(lazy).unwrap();
        assert_eq!(
            cache.actual_bytes().unwrap(),
            CoefficientScanCache::strict_outer_bytes().unwrap()
        );
    }
}

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
        #[cfg(test)]
        if std::env::var_os("AVIF_COEFF_TRACE").is_some()
            && matches!(
                symbol,
                CoefficientSymbol::EobPoint { .. } | CoefficientSymbol::EobExtra { .. }
            )
        {
            let state = self.reader.state_snapshot();
            eprintln!(
                "entropy-trace coeff-symbol-before symbol={symbol:?} cdf={cdf:?} range={} dif={} count={} tell={}",
                state.range, state.dif, state.count, state.tell
            );
        }
        let value = self.reader.read_symbol(cdf)?;
        #[cfg(test)]
        if std::env::var_os("AVIF_COEFF_TRACE").is_some()
            && matches!(
                symbol,
                CoefficientSymbol::EobPoint { .. } | CoefficientSymbol::EobExtra { .. }
            )
        {
            let state = self.reader.state_snapshot();
            eprintln!(
                "entropy-trace coeff-symbol-after symbol={symbol:?} value={value} range={} dif={} count={} tell={}",
                state.range, state.dif, state.count, state.tell
            );
        }
        Ok(value)
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

#[allow(dead_code)]
pub(super) fn decode_coefficients<S: CoefficientTokenSource>(
    source: &mut S,
    tx_size: TxSize,
    tx_type: TxType,
    plane_type: usize,
    dc_sign_context: usize,
) -> Result<CoefficientRead, DecoderError> {
    let mut scratch = Vec::new();
    decode_coefficients_with_scratch(
        source,
        tx_size,
        tx_type,
        plane_type,
        dc_sign_context,
        &mut scratch,
    )
}

pub(super) fn decode_coefficients_with_scratch<S: CoefficientTokenSource>(
    source: &mut S,
    tx_size: TxSize,
    tx_type: TxType,
    plane_type: usize,
    dc_sign_context: usize,
    scratch: &mut Vec<i32>,
) -> Result<CoefficientRead, DecoderError> {
    let scan = coefficient_scan(tx_size, tx_type);
    decode_coefficients_with_scan(
        source,
        tx_size,
        tx_type,
        plane_type,
        dc_sign_context,
        &scan,
        scratch,
    )
}

pub(super) fn decode_coefficients_with_scan<S: CoefficientTokenSource>(
    source: &mut S,
    tx_size: TxSize,
    tx_type: TxType,
    plane_type: usize,
    dc_sign_context: usize,
    scan: &[usize],
    scratch: &mut Vec<i32>,
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
    if eob == 0 || eob > scan.len() {
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
        scan,
        scratch,
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
    scan: &[usize],
    scratch: &mut Vec<i32>,
) -> Result<CoeffBaseRead, DecoderError> {
    let remaining_count = eob - 1;
    scratch.resize(tx_size.sample_count(), 0);
    scratch.fill(0);
    let quant = scratch.as_mut_slice();
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
        quant,
        &mut base_range_count,
        &mut coeff_br_symbol_count,
        &mut first_coeff_br,
    )?;
    quant[eob_position] = eob_level as i32;

    let mut first = None;
    let mut decoded_count = 0;
    for scan_index in (0..eob - 1).rev() {
        let position = scan[scan_index];
        let (context, reference_magnitude) = match is_directional_tx_type(tx_type) {
            true => coeff_base_context_1d(tx_size, tx_type, position, quant)?,
            false => coeff_base_context_2d(tx_size, position, quant)?,
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
            quant,
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
    let non_zero_count = coeff_base_non_zero_count(quant);
    let signs = read_coeff_signs_and_golomb(source, plane_type, dc_sign_context, eob, scan, quant)?;
    let signed_non_zero_count = coeff_base_non_zero_count(quant);
    let first_signed_coeff = first_signed_coeff(eob, scan, quant)?;
    let base_levels = std::mem::take(scratch);
    Ok(CoeffBaseRead {
        probe,
        base_levels,
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
    let context = match is_directional_tx_type(tx_type) {
        true => coeff_br_context_1d(tx_size, tx_type, position, quant)?,
        false => coeff_br_context_2d(tx_size, position, quant)?,
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
    for (scan_index, &position) in scan.iter().enumerate().take(eob) {
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

use super::diagnostic::SignedCoeffProbe;
use crate::DecoderError;
use crate::av1::syntax::{BlockSize, TxSize, TxType};
use crate::av1::transform::TransformBlock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TxbContext {
    pub(super) skip: usize,
    pub(super) dc_sign: usize,
}

pub(super) const COEFF_CONTEXT_BITS: u8 = 3;
const COEFF_CONTEXT_MASK: u8 = (1 << COEFF_CONTEXT_BITS) - 1;
const TXB_SKIP_CONTEXTS: [[usize; 5]; 5] = [
    [1, 2, 2, 2, 3],
    [2, 4, 4, 4, 5],
    [2, 4, 4, 4, 5],
    [2, 4, 4, 4, 5],
    [3, 5, 5, 5, 6],
];

pub(super) fn txb_context(
    block_size: BlockSize,
    transform: TransformBlock,
    above: &[u8],
    left: &[u8],
) -> TxbContext {
    let col = transform.x >> 2;
    let row = transform.y >> 2;
    let width_units = transform.tx_size.width() >> 2;
    let height_units = transform.tx_size.height() >> 2;
    let above_contexts = above
        .get(col..col.saturating_add(width_units).min(above.len()))
        .unwrap_or(&[]);
    let left_contexts = left
        .get(row..row.saturating_add(height_units).min(left.len()))
        .unwrap_or(&[]);

    let dc_sign_sum = above_contexts
        .iter()
        .chain(left_contexts)
        .map(|value| match value >> COEFF_CONTEXT_BITS {
            1 => -1,
            2 => 1,
            _ => 0,
        })
        .sum::<i32>();
    let dc_sign = match dc_sign_sum.cmp(&0) {
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 2,
    };

    let skip = if transform.plane == 0 {
        if block_size.width() == transform.tx_size.width()
            && block_size.height() == transform.tx_size.height()
        {
            0
        } else {
            let top = above_contexts
                .iter()
                .fold(0, |value, context| value | context)
                & COEFF_CONTEXT_MASK;
            let left = left_contexts
                .iter()
                .fold(0, |value, context| value | context)
                & COEFF_CONTEXT_MASK;
            TXB_SKIP_CONTEXTS[usize::from(top.min(4))][usize::from(left.min(4))]
        }
    } else {
        let base = usize::from(above_contexts.iter().any(|value| *value != 0))
            + usize::from(left_contexts.iter().any(|value| *value != 0));
        let offset = if block_size.width() * block_size.height()
            > transform.tx_size.width() * transform.tx_size.height()
        {
            10
        } else {
            7
        };
        base + offset
    };

    TxbContext { skip, dc_sign }
}

pub(super) fn coefficient_entropy_context(coefficients: &[i32]) -> u8 {
    let mut context = coefficients
        .iter()
        .map(|coefficient| coefficient.unsigned_abs() as u64)
        .sum::<u64>()
        .min(u64::from(COEFF_CONTEXT_MASK)) as u8;
    if let Some(dc) = coefficients.first() {
        if *dc < 0 {
            context |= 1 << COEFF_CONTEXT_BITS;
        } else if *dc > 0 {
            context += 2 << COEFF_CONTEXT_BITS;
        }
    }
    context
}

pub(super) fn set_txb_entropy_context(
    transform: TransformBlock,
    value: u8,
    above: &mut [u8],
    left: &mut [u8],
) {
    let col = transform.x >> 2;
    let row = transform.y >> 2;
    let width_units = transform.tx_size.width() >> 2;
    let height_units = transform.tx_size.height() >> 2;
    let above_end = col.saturating_add(width_units).min(above.len());
    if let Some(contexts) = above.get_mut(col..above_end) {
        contexts.fill(value);
    }
    let left_end = row.saturating_add(height_units).min(left.len());
    if let Some(contexts) = left.get_mut(row..left_end) {
        contexts.fill(value);
    }
}

pub(super) fn eob_tx_class_context(tx_type: TxType) -> usize {
    usize::from(matches!(
        tx_type,
        TxType::VerticalDct | TxType::HorizontalDct
    ))
}

#[cfg(test)]
pub(super) fn eob_multisize(transform: TransformBlock) -> usize {
    usize::from(transform.tx_size.width_log2().min(5) + transform.tx_size.height_log2().min(5) - 4)
}

pub(super) fn eob_base_from_pt(eob_pt: usize) -> usize {
    if eob_pt < 2 {
        eob_pt
    } else {
        (1 << (eob_pt - 2)) + 1
    }
}

pub(super) fn coeff_base_eob_context(tx_size: TxSize, scan_index: usize) -> usize {
    let samples = coeff_scan_sample_count(tx_size);
    if scan_index == 0 {
        0
    } else if scan_index <= samples / 8 {
        1
    } else if scan_index <= samples / 4 {
        2
    } else {
        3
    }
}

fn coeff_scan_sample_count(tx_size: TxSize) -> usize {
    tx_size.width().min(32) * tx_size.height().min(32)
}

pub(super) fn coeff_base_non_zero_count(base_levels: &[i32]) -> usize {
    let mut non_zero_count = 0usize;
    for level in base_levels.iter().copied() {
        let magnitude = level.unsigned_abs() as usize;
        if magnitude != 0 {
            non_zero_count += 1;
        }
    }
    non_zero_count
}

pub(super) fn first_signed_coeff(
    eob: usize,
    scan: &[usize],
    coefficients: &[i32],
) -> Result<Option<SignedCoeffProbe>, DecoderError> {
    if eob == 0 || eob > scan.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 signed coefficient eob exceeds scan".to_string(),
        ));
    }
    for scan_index in 0..eob {
        let position = scan[scan_index];
        let value = coefficients[position];
        if value != 0 {
            return Ok(Some(SignedCoeffProbe {
                scan_index,
                position,
                value,
            }));
        }
    }
    Ok(None)
}

pub(super) const NUM_BASE_LEVELS: usize = 2;
const COEFF_BASE_RANGE: usize = 12;
pub(super) const BR_CDF_SIZE: usize = 4;
pub(super) const COEFF_BR_CDF_ROUNDS: usize = COEFF_BASE_RANGE / (BR_CDF_SIZE - 1);
pub(super) const MAX_BASE_BR_RANGE: usize = NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1;
const BR_LEVEL_CAP: usize = COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1;
pub(super) const COEFFICIENT_LEVEL_MASK: usize = (1 << 20) - 1;

fn coeff_context_coords(tx_size: TxSize, position: usize) -> (usize, usize) {
    let width = tx_size.width();
    let height = tx_size.height();
    if width == height {
        (position / width, position % width)
    } else {
        // Rectangular AV1 coefficient positions use the column-major layout
        // used by AOM's padded level buffer (bhl is the height log2).
        (position % height, position / height)
    }
}

fn directional_coeff_coords(tx_size: TxSize, position: usize) -> (usize, usize) {
    // get_txb_bhl() uses the adjusted transform for 64-wide variants.  The
    // scan and levels buffer therefore split the coefficient index at the
    // (possibly adjusted) transform height.
    let bhl = match tx_size {
        TxSize::Tx64x64 | TxSize::Tx64x32 | TxSize::Tx32x64 => 5,
        TxSize::Tx64x16 => 4,
        TxSize::Tx16x64 => 5,
        _ => usize::from(tx_size.height_log2()),
    };
    let row_mask = (1usize << bhl) - 1;
    (position & row_mask, position >> bhl)
}

fn directional_coeff_position(tx_size: TxSize, row: usize, col: usize) -> usize {
    let height = match tx_size {
        TxSize::Tx64x64 | TxSize::Tx64x32 | TxSize::Tx32x64 => 32,
        TxSize::Tx64x16 => 16,
        TxSize::Tx16x64 => 32,
        _ => tx_size.height(),
    };
    col * height + row
}

fn coeff_context_position(tx_size: TxSize, row: usize, col: usize) -> usize {
    if tx_size.width() == tx_size.height() {
        row * tx_size.width() + col
    } else {
        col * tx_size.height() + row
    }
}

pub(super) fn clamp_coefficient_level(level: usize) -> usize {
    level & COEFFICIENT_LEVEL_MASK
}

const MAG_REF_OFFSET_WITH_TX_CLASS_2D: [(usize, usize); 3] = [(0, 1), (1, 0), (1, 1)];
const SIG_REF_DIFF_OFFSET_2D: [(usize, usize); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];
pub(super) fn coeff_base_context_2d(
    tx_size: TxSize,
    position: usize,
    quant: &[i32],
) -> Result<(usize, usize), DecoderError> {
    if quant.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_base context quant buffer size does not match transform size".to_string(),
        ));
    }
    if position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_base context position exceeds transform size".to_string(),
        ));
    }

    let width = tx_size.width();
    let height = tx_size.height();
    let (row, col) = coeff_context_coords(tx_size, position);
    let mut magnitude = 0usize;
    for (row_offset, col_offset) in SIG_REF_DIFF_OFFSET_2D {
        let ref_row = row + row_offset;
        let ref_col = col + col_offset;
        if ref_row < height && ref_col < width {
            magnitude += quant[coeff_context_position(tx_size, ref_row, ref_col)]
                .unsigned_abs()
                .min(3) as usize;
        }
    }

    if row == 0 && col == 0 {
        return Ok((0, magnitude));
    }

    let context_delta = ((magnitude + 1) >> 1).min(4);
    let offset = if width < height && row < 2 {
        11
    } else if width > height && col < 2 {
        16
    } else if row + col < 2 {
        1
    } else if row + col < 4 {
        6
    } else {
        21
    };
    Ok((context_delta + offset, magnitude))
}

pub(super) fn coeff_base_context_1d(
    tx_size: TxSize,
    tx_type: TxType,
    position: usize,
    quant: &[i32],
) -> Result<(usize, usize), DecoderError> {
    if quant.len() != tx_size.sample_count() || position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 1D coeff_base context input is invalid".to_string(),
        ));
    }
    let width = tx_size.width();
    let height = tx_size.height();
    let (row, col) = directional_coeff_coords(tx_size, position);
    // AV1's levels buffer is column-major.  The two directional classes use
    // the same first two neighbours, then continue along their transform
    // axis (see AOM's get_nz_mag()).
    let offsets: [(usize, usize); 5] = match tx_type {
        TxType::VerticalDct => [(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)],
        TxType::HorizontalDct => [(0, 1), (1, 0), (0, 2), (0, 3), (0, 4)],
        _ => {
            return Err(DecoderError::InvalidParam(
                "AV1 1D coeff_base context requires a directional transform".to_string(),
            ));
        }
    };
    let magnitude = offsets
        .into_iter()
        .filter_map(|(dy, dx)| {
            let y = row + dy;
            let x = col + dx;
            (y < height && x < width).then(|| {
                quant[directional_coeff_position(tx_size, y, x)]
                    .unsigned_abs()
                    .min(3) as usize
            })
        })
        .sum::<usize>();
    let delta = ((magnitude + 1) >> 1).min(4);
    let axis = if tx_type == TxType::HorizontalDct {
        col
    } else {
        row
    };
    let offset = if axis == 0 {
        26
    } else if axis == 1 {
        31
    } else {
        36
    };
    Ok((offset + delta, magnitude))
}

pub(super) fn coeff_br_context_2d(
    tx_size: TxSize,
    position: usize,
    quant: &[i32],
) -> Result<usize, DecoderError> {
    if quant.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_br context quant buffer size does not match transform size".to_string(),
        ));
    }
    if position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_br context position exceeds transform size".to_string(),
        ));
    }

    let width = tx_size.width();
    let height = tx_size.height();
    let (row, col) = coeff_context_coords(tx_size, position);
    let mut magnitude = 0usize;
    for (row_offset, col_offset) in MAG_REF_OFFSET_WITH_TX_CLASS_2D {
        let ref_row = row + row_offset;
        let ref_col = col + col_offset;
        if ref_row < height && ref_col < width {
            magnitude += (quant[coeff_context_position(tx_size, ref_row, ref_col)].unsigned_abs()
                as usize)
                .min(BR_LEVEL_CAP);
        }
    }

    let magnitude_context = ((magnitude + 1) >> 1).min(6);
    if position == 0 {
        Ok(magnitude_context)
    } else if row < 2 && col < 2 {
        Ok(magnitude_context + 7)
    } else {
        Ok(magnitude_context + 14)
    }
}

pub(super) fn coeff_br_context_1d(
    tx_size: TxSize,
    tx_type: TxType,
    position: usize,
    quant: &[i32],
) -> Result<usize, DecoderError> {
    if quant.len() != tx_size.sample_count() || position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 1D coeff_br context input is invalid".to_string(),
        ));
    }
    let width = tx_size.width();
    let height = tx_size.height();
    let (row, col) = directional_coeff_coords(tx_size, position);
    let offsets = match tx_type {
        TxType::VerticalDct => [(0, 1), (1, 0), (2, 0)],
        TxType::HorizontalDct => [(0, 1), (1, 0), (0, 2)],
        _ => {
            return Err(DecoderError::InvalidParam(
                "AV1 1D coeff_br context requires a directional transform".to_string(),
            ));
        }
    };
    let magnitude = offsets
        .into_iter()
        .filter_map(|(dy, dx)| {
            let y = row + dy;
            let x = col + dx;
            (y < height && x < width).then(|| {
                (quant[directional_coeff_position(tx_size, y, x)].unsigned_abs() as usize)
                    .min(BR_LEVEL_CAP)
            })
        })
        .sum::<usize>();
    let delta = ((magnitude + 1) >> 1).min(6);
    if position == 0 {
        Ok(delta)
    } else if (tx_type == TxType::HorizontalDct && col == 0)
        || (tx_type == TxType::VerticalDct && row == 0)
    {
        Ok(delta + 7)
    } else {
        Ok(delta + 14)
    }
}

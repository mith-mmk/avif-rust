mod data {
    include!("qmatrix_data.rs");
}

use super::syntax::{TxSize, TxType};

const QM_TOTAL_SIZE: usize = 3344;

/// Return the normative inverse quantizer-matrix value for one coefficient.
///
/// Levels 0..14 use the normative libaom tables. Level 15 is the flat matrix
/// and is represented by `None`.
pub(crate) fn inverse_value(
    level: u8,
    plane: usize,
    tx_size: TxSize,
    tx_type: TxType,
    coefficient_index: usize,
) -> Option<u8> {
    if level >= 15 || !is_two_dimensional(tx_type) {
        return None;
    }
    let adjusted = adjusted_tx_size(tx_size);
    let offset = matrix_offset(adjusted);
    let matrix = if plane == 0 {
        &data::IWT_MATRIX_LUMA[level as usize][offset..offset + adjusted.sample_count()]
    } else {
        &data::IWT_MATRIX_CHROMA[level as usize][offset..offset + adjusted.sample_count()]
    };
    let matrix_index = inverse_storage_matrix_index(tx_size, tx_type, coefficient_index);
    Some(matrix.get(matrix_index).copied().unwrap_or(32))
}

fn is_two_dimensional(tx_type: TxType) -> bool {
    matches!(
        tx_type,
        TxType::DctDct | TxType::AdstDct | TxType::DctAdst | TxType::AdstAdst
    )
}

fn adjusted_tx_size(tx_size: TxSize) -> TxSize {
    match tx_size {
        TxSize::Tx64x64 | TxSize::Tx64x32 | TxSize::Tx32x64 => TxSize::Tx32x32,
        TxSize::Tx64x16 => TxSize::Tx32x16,
        TxSize::Tx16x64 => TxSize::Tx16x32,
        tx_size => tx_size,
    }
}

fn matrix_offset(tx_size: TxSize) -> usize {
    match tx_size {
        TxSize::Tx4x4 => 0,
        TxSize::Tx8x8 => 16,
        TxSize::Tx16x16 => 80,
        TxSize::Tx32x32 => 336,
        TxSize::Tx4x8 => 1360,
        TxSize::Tx8x4 => 1392,
        TxSize::Tx8x16 => 1424,
        TxSize::Tx16x8 => 1552,
        TxSize::Tx16x32 => 1680,
        TxSize::Tx32x16 => 2192,
        TxSize::Tx4x16 => 2704,
        TxSize::Tx16x4 => 2768,
        TxSize::Tx8x32 => 2832,
        TxSize::Tx32x8 => 3088,
        TxSize::Tx64x64 | TxSize::Tx32x64 | TxSize::Tx64x32 | TxSize::Tx64x16 | TxSize::Tx16x64 => {
            unreachable!("adjusted transform size must be canonical")
        }
    }
}

fn inverse_storage_matrix_index(
    tx_size: TxSize,
    tx_type: TxType,
    coefficient_index: usize,
) -> usize {
    if matches!(tx_size, TxSize::Tx64x32 | TxSize::Tx32x64) && tx_type == TxType::DctDct {
        // The coded coefficient region is a 32x32 scan stored in the
        // rectangular inverse-transform buffer. Recover the pre-remap raster
        // position used to index the matrix.
        let height = tx_size.height();
        let row = coefficient_index % height;
        let column = coefficient_index / height;
        if row >= 32 || column >= 32 {
            return QM_TOTAL_SIZE;
        }
        let coded_position = column * 32 + row;
        let coded_scan = super::transform::zig_zag_scan(TxSize::Tx32x32);
        let source_scan = super::transform::zig_zag_scan(tx_size);
        coded_scan
            .iter()
            .position(|&position| position == coded_position)
            .and_then(|scan_index| source_scan.get(scan_index).copied())
            .unwrap_or(QM_TOTAL_SIZE)
    } else if tx_size.is_rectangular() {
        coefficient_index
    } else if needs_square_transpose(tx_size, tx_type) {
        let side = tx_size.width();
        let row = coefficient_index / side;
        let column = coefficient_index % side;
        column * side + row
    } else {
        coefficient_index
    }
}

fn needs_square_transpose(tx_size: TxSize, tx_type: TxType) -> bool {
    tx_size.is_rectangular()
        || (tx_type == TxType::DctDct
            && matches!(
                tx_size,
                TxSize::Tx4x4 | TxSize::Tx8x8 | TxSize::Tx16x16 | TxSize::Tx32x32 | TxSize::Tx64x64
            ))
        || matches!(
            tx_type,
            TxType::AdstDct | TxType::DctAdst | TxType::AdstAdst
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_zero_matrix_uses_canonical_offsets() {
        assert_eq!(matrix_offset(TxSize::Tx4x4), 0);
        assert_eq!(matrix_offset(TxSize::Tx32x32), 336);
        assert_eq!(matrix_offset(TxSize::Tx32x8), 3088);
        assert_eq!(
            matrix_offset(TxSize::Tx32x8) + TxSize::Tx32x8.sample_count(),
            QM_TOTAL_SIZE
        );
    }

    #[test]
    fn flat_level_is_not_returned_as_an_active_table() {
        assert_eq!(inverse_value(15, 0, TxSize::Tx8x8, TxType::DctDct, 0), None);
        assert_eq!(
            inverse_value(0, 0, TxSize::Tx8x8, TxType::Identity, 0),
            None
        );
        assert_eq!(
            inverse_value(0, 0, TxSize::Tx4x4, TxType::DctDct, 0),
            Some(32)
        );
        assert_eq!(
            inverse_value(14, 0, TxSize::Tx4x4, TxType::DctDct, 1),
            Some(31)
        );
    }
}

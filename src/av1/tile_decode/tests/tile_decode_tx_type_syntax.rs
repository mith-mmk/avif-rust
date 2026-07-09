use super::tx_type_syntax::{filter_intra_mode_to_tx_cdf_mode, intra_ext_tx_set_context};
use crate::av1::syntax::TxSize;

#[test]
fn intra_ext_tx_set_context_uses_set2_for_tx16() {
    assert_eq!(intra_ext_tx_set_context(false, TxSize::Tx4x4), Some((1, 0)));
    assert_eq!(intra_ext_tx_set_context(false, TxSize::Tx8x8), Some((1, 1)));
    assert_eq!(
        intra_ext_tx_set_context(false, TxSize::Tx16x16),
        Some((2, 2))
    );
    assert_eq!(intra_ext_tx_set_context(true, TxSize::Tx4x4), Some((2, 0)));
    assert_eq!(intra_ext_tx_set_context(true, TxSize::Tx8x8), Some((2, 1)));
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

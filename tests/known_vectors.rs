use avif_rust::av1::{BlockSize, CdfContext, Partition, TxSize, TxType};

#[test]
fn syntax_known_vectors_match_av1_tables() {
    assert_eq!(
        BlockSize::from_dimensions(128, 64),
        Some(BlockSize::Block128x64)
    );
    assert_eq!(
        BlockSize::from_dimensions(64, 128),
        Some(BlockSize::Block64x128)
    );
    assert_eq!(BlockSize::from_dimensions(2, 4), None);

    assert_eq!(BlockSize::Block4x4.partition_contexts(), (31, 31));
    assert_eq!(BlockSize::Block16x32.partition_contexts(), (28, 24));
    assert_eq!(BlockSize::Block64x128.partition_contexts(), (16, 0));
    assert_eq!(BlockSize::Block128x32.partition_contexts(), (0, 24));

    assert_eq!(
        Partition::from_symbol(BlockSize::Block128x128, 4),
        Some(Partition::HorizontalA)
    );
    assert_eq!(
        Partition::from_symbol(BlockSize::Block128x128, 7),
        Some(Partition::VerticalB)
    );
    assert_eq!(Partition::from_symbol(BlockSize::Block128x128, 8), None);

    assert_eq!(BlockSize::Block8x8.tx_size_category(), 0);
    assert_eq!(BlockSize::Block16x16.tx_size_category(), 1);
    assert_eq!(BlockSize::Block32x32.tx_size_category(), 2);
    assert_eq!(BlockSize::Block64x64.tx_size_category(), 3);
    assert_eq!(BlockSize::Block64x64.tx_size_from_depth(1), TxSize::Tx32x32);

    assert_eq!(
        TxType::from_intra_ext_tx_set1_symbol(0),
        Some(TxType::Identity)
    );
    assert_eq!(
        TxType::from_intra_ext_tx_set1_symbol(6),
        Some(TxType::DctAdst)
    );
    assert_eq!(TxType::from_intra_ext_tx_set1_symbol(7), None);
    assert_eq!(
        TxType::from_intra_ext_tx_set2_symbol(4),
        Some(TxType::DctAdst)
    );
    assert_eq!(TxType::from_intra_ext_tx_set2_symbol(5), None);
}

#[test]
fn cdf_known_vectors_match_default_av1_tables() {
    let context = CdfContext::default();

    assert_eq!(context.partition_w8[0][3], 32768);
    assert_eq!(context.partition_w16[0][9], 32768);
    assert_eq!(context.partition_w32[0][9], 32768);
    assert_eq!(context.partition_w64[0][9], 32768);
    assert_eq!(context.partition_w128[0][7], 32768);
    assert_eq!(context.skip[0][1], 32768);
    assert_eq!(context.txb_skip[0][0][1], 32768);
    assert_eq!(context.eob_pt_16[0][0][4], 32768);
    assert_eq!(context.eob_pt_1024[1][1][10], 32768);
    assert_eq!(context.coeff_base[4][1][22][3], 32768);
    assert_eq!(context.coeff_br[4][1][20][3], 32768);
    assert_eq!(context.dc_sign[1][2][1], 32768);
}

#[test]
fn cdf_q_context_vectors_select_expected_coefficient_tables() {
    assert_eq!(CdfContext::new(20).txb_skip[0][0], [31849, 32768, 0]);
    assert_eq!(CdfContext::new(21).txb_skip[0][0], [30371, 32768, 0]);
    assert_eq!(CdfContext::new(61).txb_skip[0][0], [29614, 32768, 0]);
    assert_eq!(CdfContext::new(121).txb_skip[0][0], [26887, 32768, 0]);

    assert_eq!(CdfContext::new(20).eob_pt_1024[0][0][0], 393);
    assert_eq!(CdfContext::new(21).eob_pt_1024[0][0][0], 696);
    assert_eq!(CdfContext::new(61).eob_pt_1024[0][0][0], 2784);
    assert_eq!(CdfContext::new(121).eob_pt_1024[0][0][0], 6698);

    assert_eq!(
        CdfContext::new(20).coeff_base[3][0][22],
        [23352, 31766, 32545, 32768, 0]
    );
    assert_eq!(
        CdfContext::new(121).coeff_base[3][0][22],
        [20618, 31487, 32544, 32768, 0]
    );
    assert_eq!(
        CdfContext::new(20).coeff_br[3][0][0],
        [2331, 3662, 5244, 32768, 0]
    );
    assert_eq!(
        CdfContext::new(121).coeff_br[3][0][0],
        [12162, 18785, 22648, 32768, 0]
    );
    assert_eq!(CdfContext::new(20).dc_sign[0][0], [16000, 32768, 0]);
    assert_eq!(CdfContext::new(121).dc_sign[1][2], [17280, 32768, 0]);
}

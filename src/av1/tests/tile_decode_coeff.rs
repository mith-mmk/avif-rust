use std::collections::VecDeque;

use super::coefficient::{
    CoefficientLiteral, CoefficientScanCache, CoefficientSymbol, CoefficientTokenSource,
    EntropyCoefficientSource, decode_coefficients, read_golomb,
};
use super::coefficient_context::{
    COEFFICIENT_LEVEL_MASK, TxbContext, clamp_coefficient_level, coeff_base_context_1d,
    coeff_base_context_2d, coeff_br_context_1d, coeff_br_context_2d, coefficient_entropy_context,
    eob_base_from_pt, eob_tx_class_context, set_txb_entropy_context, txb_context,
};
use super::{BlockSize, CdfContext, DecoderError, EntropyDecoder, TransformBlock, TxSize, TxType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Symbol(CoefficientSymbol, usize),
    Literal(CoefficientLiteral, usize),
}

struct ScriptedTokens {
    tokens: VecDeque<Token>,
}

#[test]
fn coefficient_scan_cache_reuses_scan_storage() {
    let mut cache = CoefficientScanCache::new();
    let (first_ptr, first_len) = {
        let first = cache.get(TxSize::Tx16x16, TxType::DctDct);
        (first.as_ptr(), first.len())
    };
    let second = cache.get(TxSize::Tx16x16, TxType::DctDct);

    assert_eq!(second.as_ptr(), first_ptr);
    assert_eq!(second.len(), first_len);
}

#[test]
fn coefficient_scan_cache_indexes_every_transform_variant() {
    let mut cache = CoefficientScanCache::new();
    let tx_sizes = [
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
    let tx_types = [
        TxType::DctDct,
        TxType::AdstDct,
        TxType::DctAdst,
        TxType::AdstAdst,
        TxType::Identity,
        TxType::VerticalDct,
        TxType::HorizontalDct,
    ];

    for tx_size in tx_sizes {
        for tx_type in tx_types {
            let scan = cache.get(tx_size, tx_type);
            assert!(!scan.is_empty(), "{tx_size:?} {tx_type:?}");
            assert!(scan.len() <= tx_size.sample_count());
        }
    }
}

impl ScriptedTokens {
    fn new(tokens: impl IntoIterator<Item = Token>) -> Self {
        Self {
            tokens: tokens.into_iter().collect(),
        }
    }

    fn finish(self) {
        assert!(self.tokens.is_empty(), "unread tokens: {:?}", self.tokens);
    }

    fn next(&mut self, expected: Token) -> usize {
        let actual = self
            .tokens
            .pop_front()
            .expect("coefficient token requested");
        assert_eq!(actual, expected);
        match actual {
            Token::Symbol(_, value) | Token::Literal(_, value) => value,
        }
    }
}

impl CoefficientTokenSource for ScriptedTokens {
    fn read_symbol(&mut self, symbol: CoefficientSymbol) -> Result<usize, DecoderError> {
        let value = self
            .tokens
            .front()
            .and_then(|token| match token {
                Token::Symbol(_, value) => Some(*value),
                Token::Literal(_, _) => None,
            })
            .expect("symbol token requested");
        Ok(self.next(Token::Symbol(symbol, value)))
    }

    fn read_literal(&mut self, literal: CoefficientLiteral) -> Result<usize, DecoderError> {
        let value = self
            .tokens
            .front()
            .and_then(|token| match token {
                Token::Literal(_, value) => Some(*value),
                Token::Symbol(_, _) => None,
            })
            .expect("literal token requested");
        Ok(self.next(Token::Literal(literal, value)))
    }
}

#[test]
fn golomb_values_match_aom_bitwriter_vectors() {
    const CASES: &[(usize, &[u8])] = &[
        (0, &[192, 0]),
        (1, &[72, 0, 0, 0, 0]),
        (2, &[104, 0, 0, 0, 0]),
        (5, &[52, 0, 0, 0, 0, 0]),
        (14, &[31, 0, 0, 0, 0, 0, 0, 0]),
        (31, &[4, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ];

    for &(expected, payload) in CASES {
        let mut reader = EntropyDecoder::new(payload, true).unwrap();
        let mut cdf = CdfContext::new(0);
        let mut source = EntropyCoefficientSource::new(&mut reader, &mut cdf);
        assert_eq!(read_golomb(&mut source).unwrap(), expected);
    }
}

#[test]
fn eob_point_groups_match_av1_group_starts() {
    assert_eq!(
        (1..=11).map(eob_base_from_pt).collect::<Vec<_>>(),
        vec![1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513]
    );
}

#[test]
fn coeff_base_context_2d_matches_square_offset_rules() {
    let mut quant = vec![0; TxSize::Tx32x32.sample_count()];

    assert_eq!(
        coeff_base_context_2d(TxSize::Tx32x32, 0, &quant).unwrap(),
        (0, 0)
    );

    quant[2] = 3;
    assert_eq!(
        coeff_base_context_2d(TxSize::Tx32x32, 1, &quant).unwrap(),
        (3, 3)
    );

    assert_eq!(
        coeff_base_context_2d(TxSize::Tx32x32, 4 * 32 + 4, &quant).unwrap(),
        (21, 0)
    );
}

#[test]
fn coeff_base_context_2d_uses_adjusted_tx32x64_raster_coordinates() {
    let mut quant = vec![0; TxSize::Tx32x64.sample_count()];
    // In the adjusted 32x32 levels buffer, position 1 is row 1/column 0.
    // Its first right neighbour is therefore the column-major slot 33.
    quant[33] = 3;

    assert_eq!(
        coeff_base_context_2d(TxSize::Tx32x64, 1, &quant).unwrap(),
        (13, 3)
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

    above[1] = 4 | (2 << super::coefficient_context::COEFF_CONTEXT_BITS);
    left[1] = 2 | (1 << super::coefficient_context::COEFF_CONTEXT_BITS);
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
    let mut quant = vec![0; TxSize::Tx32x32.sample_count()];

    assert_eq!(coeff_br_context_2d(TxSize::Tx32x32, 0, &quant).unwrap(), 0);

    quant[1] = 3;
    assert_eq!(coeff_br_context_2d(TxSize::Tx32x32, 0, &quant).unwrap(), 2);

    assert_eq!(
        coeff_br_context_2d(TxSize::Tx32x32, 32 + 1, &quant).unwrap(),
        7
    );

    assert_eq!(
        coeff_br_context_2d(TxSize::Tx32x32, 4 * 32 + 4, &quant).unwrap(),
        14
    );
}

#[test]
fn directional_coefficient_contexts_follow_aom_1d_axes() {
    let tx_size = TxSize::Tx8x8;
    let mut quant = vec![0; tx_size.sample_count()];
    quant[2] = 3;
    quant[16] = 2;

    assert_eq!(
        coeff_base_context_1d(tx_size, TxType::VerticalDct, 1, &quant).unwrap(),
        (33, 3)
    );
    assert_eq!(
        coeff_base_context_1d(tx_size, TxType::HorizontalDct, 0, &quant).unwrap(),
        (27, 2)
    );
    assert_eq!(
        coeff_base_context_1d(tx_size, TxType::VerticalDct, 0, &quant).unwrap(),
        (28, 3)
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

#[test]
fn scripted_transcript_covers_full_coefficient_state_machine() {
    use CoefficientLiteral::{AcSign, EobExtra, GolombPrefix, GolombSuffix};
    use CoefficientSymbol::{
        Base, BaseEob, BaseRange, DcSign, EobExtra as EobExtraSymbol, EobPoint,
    };
    use Token::{Literal, Symbol};

    let mut source = ScriptedTokens::new([
        Symbol(
            EobPoint {
                multisize: 0,
                plane_type: 0,
                tx_class: 0,
            },
            3,
        ),
        Symbol(
            EobExtraSymbol {
                tx_size_context: 0,
                plane_type: 0,
                context: 1,
            },
            0,
        ),
        Literal(EobExtra { index: 0 }, 1),
        Symbol(
            BaseEob {
                tx_size_context: 0,
                plane_type: 0,
                context: 3,
            },
            2,
        ),
        Symbol(
            BaseRange {
                tx_size_context: 0,
                plane_type: 0,
                context: 14,
            },
            3,
        ),
        Symbol(
            BaseRange {
                tx_size_context: 0,
                plane_type: 0,
                context: 14,
            },
            3,
        ),
        Symbol(
            BaseRange {
                tx_size_context: 0,
                plane_type: 0,
                context: 14,
            },
            3,
        ),
        Symbol(
            BaseRange {
                tx_size_context: 0,
                plane_type: 0,
                context: 14,
            },
            3,
        ),
        Symbol(
            Base {
                tx_size_context: 0,
                plane_type: 0,
                context: 6,
            },
            0,
        ),
        Symbol(
            Base {
                tx_size_context: 0,
                plane_type: 0,
                context: 6,
            },
            1,
        ),
        Symbol(
            Base {
                tx_size_context: 0,
                plane_type: 0,
                context: 2,
            },
            2,
        ),
        Symbol(
            Base {
                tx_size_context: 0,
                plane_type: 0,
                context: 3,
            },
            0,
        ),
        Symbol(
            Base {
                tx_size_context: 0,
                plane_type: 0,
                context: 0,
            },
            1,
        ),
        Symbol(
            DcSign {
                plane_type: 0,
                context: 2,
            },
            1,
        ),
        Literal(AcSign { scan_index: 2 }, 0),
        Literal(AcSign { scan_index: 3 }, 1),
        Literal(AcSign { scan_index: 5 }, 0),
        Literal(GolombPrefix { length: 1 }, 0),
        Literal(GolombPrefix { length: 2 }, 0),
        Literal(GolombPrefix { length: 3 }, 1),
        Literal(GolombSuffix { index: 0 }, 1),
        Literal(GolombSuffix { index: 1 }, 0),
    ]);

    let decoded = decode_coefficients(&mut source, TxSize::Tx4x4, TxType::DctDct, 0, 2).unwrap();
    source.finish();

    assert_eq!(decoded.eob, 6);
    assert_eq!(decoded.base.base_range_count, 1);
    assert_eq!(decoded.base.coeff_br_symbol_count, 4);
    assert_eq!(decoded.base.signs.sign_count, 4);
    assert_eq!(decoded.base.signs.golomb_count, 1);
    assert_eq!(decoded.base.signs.first_golomb_value, Some(5));
    assert_eq!(
        decoded.base.base_levels,
        vec![-1, 2, -1, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0]
    );
}

#[test]
fn aom_encoded_payload_decodes_to_reference_coefficient_vector() {
    // Generated with AOM's default q-context-0 CDFs and aom_write_symbol/bit.
    let payload = [27, 150, 246, 92, 224];
    let mut reader = EntropyDecoder::new(&payload, false).unwrap();
    let mut cdf = CdfContext::new(20);
    let mut source = EntropyCoefficientSource::new(&mut reader, &mut cdf);

    let decoded = decode_coefficients(&mut source, TxSize::Tx4x4, TxType::DctDct, 0, 2).unwrap();

    assert_eq!(decoded.eob, 6);
    assert_eq!(
        decoded.base.base_levels,
        vec![-1, 2, -1, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0]
    );
}

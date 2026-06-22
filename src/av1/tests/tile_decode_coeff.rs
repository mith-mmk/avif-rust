use std::collections::VecDeque;

use super::coefficient::{
    CoefficientLiteral, CoefficientSymbol, CoefficientTokenSource, EntropyCoefficientSource,
    decode_coefficients, read_golomb,
};
use super::{CdfContext, DecoderError, EntropyDecoder, TxSize, TxType, eob_base_from_pt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Symbol(CoefficientSymbol, usize),
    Literal(CoefficientLiteral, usize),
}

struct ScriptedTokens {
    tokens: VecDeque<Token>,
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

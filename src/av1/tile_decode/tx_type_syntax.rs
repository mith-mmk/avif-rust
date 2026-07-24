use super::TileDecoder;
use super::diagnostic::{BlockModeProbe, TxTypeProbe};
use crate::DecoderError;
use crate::av1::frame::{FrameHeader, FrameType};
use crate::av1::syntax::{PredictionMode, TxSize, TxType, UvPredictionMode};
use crate::av1::transform::TransformBlock;

impl<'a> TileDecoder<'a> {
    pub(super) fn read_intra_tx_type(
        &mut self,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        transform: TransformBlock,
    ) -> Result<TxTypeProbe, DecoderError> {
        if transform.plane == 0
            && matches!(frame.frame_type, FrameType::Inter | FrameType::Switch)
            && let Some(probe) = self.read_inter_tx_type(frame, transform)?
        {
            return Ok(probe);
        }
        if transform.plane > 0 {
            return Ok(TxTypeProbe {
                read: false,
                set: None,
                symbol: None,
                tx_type: chroma_intra_tx_type(frame, block_mode, transform.tx_size),
            });
        }
        if let Some(tx_type) = fixed_tx_type(frame, transform) {
            return Ok(TxTypeProbe {
                read: false,
                set: None,
                symbol: None,
                tx_type,
            });
        }
        let intra_mode = block_mode
            .filter_intra_mode
            .map(filter_intra_mode_to_tx_cdf_mode)
            .transpose()?
            .unwrap_or(block_mode.y_mode_symbol);
        let (set, tx_size_context) =
            intra_ext_tx_set_context(frame.reduced_tx_set, transform.tx_size).ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AV1 intra tx_type is not signaled for {:?}",
                    transform.tx_size
                ))
            })?;
        if set == 2 {
            let symbol = self.reader.read_symbol(
                self.cdf
                    .intra_ext_tx_set2_cdf_mut(tx_size_context, intra_mode),
            )?;
            let tx_type = TxType::from_intra_ext_tx_set2_symbol(symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AV1 intra tx_type set2 symbol {symbol} is invalid"
                ))
            })?;
            Ok(TxTypeProbe {
                read: true,
                set: Some(2),
                symbol: Some(symbol),
                tx_type,
            })
        } else {
            let symbol = self.reader.read_symbol(
                self.cdf
                    .intra_ext_tx_set1_cdf_mut(tx_size_context, intra_mode),
            )?;
            let tx_type = TxType::from_intra_ext_tx_set1_symbol(symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AV1 intra tx_type set1 symbol {symbol} is invalid"
                ))
            })?;
            Ok(TxTypeProbe {
                read: true,
                set: Some(1),
                symbol: Some(symbol),
                tx_type,
            })
        }
    }

    fn read_inter_tx_type(
        &mut self,
        frame: &FrameHeader,
        transform: TransformBlock,
    ) -> Result<Option<TxTypeProbe>, DecoderError> {
        // AV1 selects the inter transform set from the smaller and larger
        // square-up dimensions. Sizes above 32 are DCT-only and fall through
        // to fixed_tx_type; all other sets consume one luma CDF symbol.
        let tx_size_sqr = usize::from(
            transform
                .tx_size
                .width_log2()
                .min(transform.tx_size.height_log2())
                - 2,
        );
        let tx_size_sqr_up = transform
            .tx_size
            .width_log2()
            .max(transform.tx_size.height_log2())
            - 2;
        if tx_size_sqr_up > 3 {
            return Ok(None);
        }
        let set = if frame.reduced_tx_set || tx_size_sqr_up == 3 {
            3
        } else if tx_size_sqr == 2 {
            2
        } else {
            1
        };
        let (symbol, tx_type) = match set {
            1 => {
                let symbol = self
                    .reader
                    .read_symbol(self.cdf.inter_ext_tx_set1_cdf_mut(tx_size_sqr))?;
                let tx_type = TxType::from_inter_ext_tx_set1_symbol(symbol).ok_or_else(|| {
                    DecoderError::Bitstream(format!(
                        "AV1 inter tx_type set 1 symbol {symbol} is invalid"
                    ))
                })?;
                (symbol, tx_type)
            }
            2 => {
                let symbol = self
                    .reader
                    .read_symbol(self.cdf.inter_ext_tx_set2_cdf_mut())?;
                let tx_type = TxType::from_inter_ext_tx_set2_symbol(symbol).ok_or_else(|| {
                    DecoderError::Bitstream(format!(
                        "AV1 inter tx_type set 2 symbol {symbol} is invalid"
                    ))
                })?;
                (symbol, tx_type)
            }
            3 => {
                let symbol = self
                    .reader
                    .read_symbol(self.cdf.inter_ext_tx_set3_cdf_mut(tx_size_sqr))?;
                let tx_type = TxType::from_inter_ext_tx_set3_symbol(symbol).ok_or_else(|| {
                    DecoderError::Bitstream(format!(
                        "AV1 inter tx_type set 3 symbol {symbol} is invalid"
                    ))
                })?;
                (symbol, tx_type)
            }
            _ => unreachable!("inter transform set is bounded to 1..=3"),
        };
        Ok(Some(TxTypeProbe {
            read: true,
            set: Some(set),
            symbol: Some(symbol),
            tx_type,
        }))
    }
}

fn chroma_intra_tx_type(
    frame: &FrameHeader,
    block_mode: &BlockModeProbe,
    tx_size: TxSize,
) -> TxType {
    if frame.coded_lossless() || tx_size.width() >= 32 || tx_size.height() >= 32 {
        return TxType::DctDct;
    }
    let mode = match block_mode.uv_mode {
        Some(UvPredictionMode::Intra(mode)) => mode,
        Some(UvPredictionMode::Cfl) | None => PredictionMode::Dc,
    };
    match mode {
        PredictionMode::Dc => TxType::DctDct,
        PredictionMode::Vertical => TxType::AdstDct,
        PredictionMode::Horizontal => TxType::DctAdst,
        PredictionMode::D45 => TxType::DctDct,
        PredictionMode::D135 => TxType::AdstAdst,
        PredictionMode::D113 => TxType::AdstDct,
        PredictionMode::D157 | PredictionMode::D203 => TxType::DctAdst,
        PredictionMode::D67 => TxType::AdstDct,
        PredictionMode::Smooth => TxType::AdstAdst,
        PredictionMode::SmoothVertical => TxType::AdstDct,
        PredictionMode::SmoothHorizontal => TxType::DctAdst,
        PredictionMode::Paeth => TxType::AdstAdst,
    }
}

pub(super) fn fixed_tx_type(frame: &FrameHeader, transform: TransformBlock) -> Option<TxType> {
    if frame.coded_lossless()
        || transform.plane != 0
        || transform.tx_size.width() >= 32
        || transform.tx_size.height() >= 32
    {
        return Some(TxType::DctDct);
    }
    None
}

pub(super) fn intra_ext_tx_set_context(
    reduced_tx_set: bool,
    tx_size: TxSize,
) -> Option<(usize, usize)> {
    match tx_size {
        TxSize::Tx4x4 | TxSize::Tx4x8 | TxSize::Tx8x4 | TxSize::Tx4x16 | TxSize::Tx16x4 => {
            Some((if reduced_tx_set { 2 } else { 1 }, 0))
        }
        TxSize::Tx8x8 | TxSize::Tx8x16 | TxSize::Tx16x8 | TxSize::Tx8x32 | TxSize::Tx32x8 => {
            Some((if reduced_tx_set { 2 } else { 1 }, 1))
        }
        TxSize::Tx16x16 | TxSize::Tx16x32 | TxSize::Tx32x16 => Some((2, 2)),
        TxSize::Tx32x32
        | TxSize::Tx64x64
        | TxSize::Tx32x64
        | TxSize::Tx64x32
        | TxSize::Tx16x64
        | TxSize::Tx64x16 => None,
    }
}

pub(super) fn filter_intra_mode_to_tx_cdf_mode(
    filter_intra_mode: usize,
) -> Result<usize, DecoderError> {
    match filter_intra_mode {
        0 => Ok(0),
        1 => Ok(1),
        2 => Ok(2),
        3 => Ok(6),
        4 => Ok(0),
        _ => Err(DecoderError::Bitstream(format!(
            "AV1 filter intra mode {filter_intra_mode} is invalid"
        ))),
    }
}

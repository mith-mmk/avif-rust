use super::TileDecoder;
use super::diagnostic::{BlockModeProbe, TxTypeProbe};
use crate::DecoderError;
use crate::av1::frame::FrameHeader;
use crate::av1::syntax::{TxSize, TxType};
use crate::av1::transform::TransformBlock;

impl<'a> TileDecoder<'a> {
    pub(super) fn read_intra_tx_type(
        &mut self,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        transform: TransformBlock,
    ) -> Result<TxTypeProbe, DecoderError> {
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
}

pub(super) fn fixed_tx_type(frame: &FrameHeader, transform: TransformBlock) -> Option<TxType> {
    if frame.base_q_idx == 0
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
        TxSize::Tx4x4 => Some((if reduced_tx_set { 2 } else { 1 }, 0)),
        TxSize::Tx8x8 => Some((if reduced_tx_set { 2 } else { 1 }, 1)),
        TxSize::Tx16x16 => Some((2, 2)),
        TxSize::Tx32x32 | TxSize::Tx64x64 => None,
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

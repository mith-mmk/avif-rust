use super::{PartitionProbe, TileDecoder, partition_symbol};
use crate::DecoderError;
use crate::av1::decode::TileDecodePlan;
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::{BlockSize, Partition};

impl<'a> TileDecoder<'a> {
    pub fn read_root_partition(
        &mut self,
        tile: &TileDecodePlan,
        sequence: &SequenceHeader,
    ) -> Result<PartitionProbe, DecoderError> {
        self.read_restoration_units(sequence, tile.pixel_x, tile.pixel_y)?;
        self.read_partition(tile, BlockSize::Block128x128, tile.pixel_x, tile.pixel_y)
    }

    pub(super) fn read_first_leaf_partition(
        &mut self,
        tile: &TileDecodePlan,
        sequence: &SequenceHeader,
    ) -> Result<PartitionProbe, DecoderError> {
        let mut probe = self.read_root_partition(tile, sequence)?;
        loop {
            match probe.partition {
                Partition::None => return Ok(probe),
                Partition::Split => {
                    let subsize = probe.block_size.split_subsize().ok_or_else(|| {
                        DecoderError::Bitstream(format!(
                            "AV1 cannot split first leaf block {:?}",
                            probe.block_size
                        ))
                    })?;
                    probe = self.read_partition(tile, subsize, tile.pixel_x, tile.pixel_y)?;
                }
                partition => {
                    return Err(DecoderError::Unsupported(format!(
                        "AV1 first-leaf traversal for partition {partition:?} is not supported yet"
                    )));
                }
            }
        }
    }

    pub(super) fn read_partition(
        &mut self,
        tile: &TileDecodePlan,
        block_size: BlockSize,
        x: usize,
        y: usize,
    ) -> Result<PartitionProbe, DecoderError> {
        let context = self.partition_context(tile, x, y, block_size);
        let half_mi = block_size.width() / 8;
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        let has_rows = mi_row + half_mi < tile.mi_row_end as usize;
        let has_cols = mi_col + half_mi < tile.mi_col_end as usize;
        let (symbol, partition) = if !has_rows && !has_cols {
            (3, Partition::Split)
        } else if has_rows && has_cols {
            let symbol = self.reader.read_symbol(
                self.cdf
                    .partition_cdf_mut(block_size.width_mi_log2(), context),
            )?;
            let partition = Partition::from_symbol(block_size, symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AV1 partition symbol {symbol} is invalid for {block_size:?}"
                ))
            })?;
            (symbol, partition)
        } else {
            if block_size.width() <= 8 {
                return Err(DecoderError::Bitstream(format!(
                    "AV1 edge partition reached invalid block size {block_size:?}"
                )));
            }
            let source = self
                .cdf
                .partition_cdf_mut(block_size.width_mi_log2(), context)
                .to_vec();
            let mut cdf = restricted_partition_cdf(&source, has_rows);
            let symbol = self.reader.read_symbol(&mut cdf)?;
            let partition = if symbol == 0 {
                if has_rows {
                    Partition::Vertical
                } else {
                    Partition::Horizontal
                }
            } else {
                Partition::Split
            };
            (partition_symbol(partition), partition)
        };
        Ok(PartitionProbe {
            tile_id: tile.tile_id,
            block_size,
            context,
            symbol,
            partition,
            bit_position_after: self.reader.bit_position(),
        })
    }
}

fn restricted_partition_cdf(source: &[u16], has_rows: bool) -> [u16; 3] {
    let symbols = source.len().saturating_sub(1);
    let alike = if has_rows {
        [1usize, 3, 4, 5, 6, 8]
    } else {
        [2usize, 3, 4, 6, 7, 9]
    };
    let alike_probability = alike
        .into_iter()
        .filter(|symbol| *symbol < symbols)
        .map(|symbol| cdf_symbol_probability(source, symbol))
        .sum::<u16>();
    [32768u16.saturating_sub(alike_probability), 32768, 0]
}

fn cdf_symbol_probability(cdf: &[u16], symbol: usize) -> u16 {
    let lower = if symbol == 0 { 0 } else { cdf[symbol - 1] };
    cdf[symbol].saturating_sub(lower)
}

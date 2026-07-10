use super::{PartitionProbe, TileDecoder};
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
                partition => {
                    let subsize = first_partition_child_size(probe.block_size, partition)?;
                    probe = self.read_partition(tile, subsize, tile.pixel_x, tile.pixel_y)?;
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

pub(super) fn first_partition_child_size(
    block_size: BlockSize,
    partition: Partition,
) -> Result<BlockSize, DecoderError> {
    let child = match partition {
        Partition::None => Some(block_size),
        Partition::Horizontal => block_size.horizontal_subsize(),
        Partition::Vertical => block_size.vertical_subsize(),
        Partition::Split => block_size.split_subsize(),
        Partition::HorizontalA => block_size.split_subsize(),
        Partition::HorizontalB => block_size.horizontal_subsize(),
        Partition::VerticalA => block_size.split_subsize(),
        Partition::VerticalB => block_size.vertical_subsize(),
        Partition::Horizontal4 => block_size.horizontal_4_subsize(),
        Partition::Vertical4 => block_size.vertical_4_subsize(),
    };
    child.ok_or_else(|| {
        DecoderError::Unsupported(format!(
            "AV1 first partition child for {partition:?} is not supported for {block_size:?}"
        ))
    })
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

pub(super) fn partition_subsize(
    block_size: BlockSize,
    partition: Partition,
) -> Result<BlockSize, DecoderError> {
    let subsize = match partition {
        Partition::None => Some(block_size),
        Partition::Horizontal | Partition::HorizontalA | Partition::HorizontalB => {
            block_size.horizontal_subsize()
        }
        Partition::Vertical | Partition::VerticalA | Partition::VerticalB => {
            block_size.vertical_subsize()
        }
        Partition::Split => block_size.split_subsize(),
        Partition::Horizontal4 => block_size.horizontal_4_subsize(),
        Partition::Vertical4 => block_size.vertical_4_subsize(),
    };
    subsize.ok_or_else(|| {
        DecoderError::Bitstream(format!(
            "AV1 partition {partition:?} has no subsize for {block_size:?}"
        ))
    })
}

fn partition_symbol(partition: Partition) -> usize {
    match partition {
        Partition::None => 0,
        Partition::Horizontal => 1,
        Partition::Vertical => 2,
        Partition::Split => 3,
        Partition::HorizontalA => 4,
        Partition::HorizontalB => 5,
        Partition::VerticalA => 6,
        Partition::VerticalB => 7,
        Partition::Horizontal4 => 8,
        Partition::Vertical4 => 9,
    }
}

pub(super) fn partition_plane_context(above: u8, left: u8, block_size: BlockSize) -> usize {
    let bit = block_size.width_mi_log2().saturating_sub(1);
    usize::from((above >> bit) & 1) + usize::from((left >> bit) & 1) * 2
}

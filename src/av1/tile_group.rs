use super::bitstream::BitReader;
use super::tile::TileInfo;
use crate::DecoderError;
use crate::allocation::{AllocationClass, AllocationTicket, admit_fresh_with, fresh_replacement};
use crate::container::DecodeBudget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilePayload {
    pub tile_id: u32,
    pub offset: usize,
    pub len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileGroup {
    pub start_tile: u32,
    pub end_tile: u32,
    pub data_start_offset: usize,
    pub tiles: Vec<TilePayload>,
}

/// Ownership ticket for the tile table materialized by the strict native
/// `OBU_FRAME` path.  The public tile-group value deliberately stays
/// unchanged; this sidecar is retired only after the owning header is
/// physically dropped.
pub(crate) struct NativeTileGroupAllocation {
    tiles: AllocationTicket,
}

impl NativeTileGroupAllocation {
    fn new() -> Self {
        Self {
            tiles: AllocationTicket::new(AllocationClass::Frame),
        }
    }

    /// Attaches the already-admitted final tile table to this sidecar.  The
    /// table itself stays in the public `TileGroup`; this private ticket is
    /// released only after that table has been physically dropped.
    pub(crate) fn from_ticket(tiles: AllocationTicket) -> Self {
        Self { tiles }
    }

    pub(crate) fn release(&mut self, budget: &mut DecodeBudget) -> Result<(), DecoderError> {
        budget.release_token(&mut self.tiles)
    }
}

/// Allocation-free tile-group syntax summary.
///
/// Both legacy and strict-native paths use the same walker below.  In
/// particular, strict split-OBU assembly can validate every source group
/// before allocating its one final descriptor table and payload owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileGroupSyntax {
    pub(crate) start_tile: u32,
    pub(crate) end_tile: u32,
    pub(crate) data_start_offset: usize,
}

/// Walks one AV1 tile-group payload without materializing a tile vector.
pub(crate) fn walk_tile_group<F>(
    data: &[u8],
    start_bit_offset: usize,
    tile_info: &TileInfo,
    mut visit: F,
) -> Result<TileGroupSyntax, DecoderError>
where
    F: FnMut(TilePayload) -> Result<(), DecoderError>,
{
    let tile_count = tile_info
        .tile_cols
        .checked_mul(tile_info.tile_rows)
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile count overflows".to_string()))?;
    if tile_count == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 tile count is zero".to_string(),
        ));
    }

    let mut reader = BitReader::new_at(data, start_bit_offset)?;
    let (start_tile, end_tile) = if tile_count > 1 {
        if reader.read_bool("tile_start_and_end_present_flag")? {
            let tile_num_bits = (tile_info.tile_cols_log2 + tile_info.tile_rows_log2) as usize;
            (
                reader.read_bits(tile_num_bits, "tg_start")?,
                reader.read_bits(tile_num_bits, "tg_end")?,
            )
        } else {
            (0, tile_count - 1)
        }
    } else {
        (0, 0)
    };
    if start_tile > end_tile || end_tile >= tile_count {
        return Err(DecoderError::Bitstream(format!(
            "invalid AV1 tile group range {start_tile}..={end_tile} for {tile_count} tiles"
        )));
    }
    reader.byte_align_zero("tile_group")?;
    let data_start_offset = reader.byte_position_ceil();
    let mut offset = data_start_offset;
    for tile_id in start_tile..=end_tile {
        let len = if tile_id == end_tile {
            data.len().checked_sub(offset).ok_or_else(|| {
                DecoderError::Bitstream("AV1 tile payload offset overflow".to_string())
            })?
        } else {
            read_le_sized_int(data, &mut offset, tile_info.tile_size_bytes)?
                .checked_add(1)
                .ok_or_else(|| DecoderError::Bitstream("AV1 tile size overflow".to_string()))?
        };
        let end = offset
            .checked_add(len)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
        if end > data.len() {
            return Err(DecoderError::NotEnoughData(
                "AV1 tile payload extends beyond tile group".to_string(),
            ));
        }
        visit(TilePayload {
            tile_id,
            offset,
            len,
        })?;
        offset = end;
    }
    Ok(TileGroupSyntax {
        start_tile,
        end_tile,
        data_start_offset,
    })
}

pub fn parse_tile_group(
    data: &[u8],
    start_bit_offset: usize,
    tile_info: &TileInfo,
) -> Result<TileGroup, DecoderError> {
    let mut tiles = Vec::new();
    let syntax = walk_tile_group(data, start_bit_offset, tile_info, |tile| {
        tiles.push(tile);
        Ok(())
    })?;

    Ok(TileGroup {
        start_tile: syntax.start_tile,
        end_tile: syntax.end_tile,
        data_start_offset: syntax.data_start_offset,
        tiles,
    })
}

/// Bounded-native counterpart of [`parse_tile_group`].  It uses the exact
/// same syntax, but admits the final `TileGroup::tiles` owner before any
/// entries are written.  Keeping this as a separate path protects the legacy
/// parser from changes in allocation timing.
pub(crate) fn parse_tile_group_with_budget(
    data: &[u8],
    start_bit_offset: usize,
    tile_info: &TileInfo,
    budget: &mut DecodeBudget,
) -> Result<(TileGroup, NativeTileGroupAllocation), DecoderError> {
    let syntax = walk_tile_group(data, start_bit_offset, tile_info, |_| Ok(()))?;
    let tile_len = usize::try_from(syntax.end_tile - syntax.start_tile + 1)
        .map_err(|_| DecoderError::InvalidParam("AV1 tile count is too large".to_string()))?;

    let mut allocation = NativeTileGroupAllocation::new();
    let result = (|| {
        let (mut candidate, ticket) = admit_fresh_with(
            budget,
            tile_len,
            AllocationClass::Frame,
            "native AV1 tile group entries",
            fresh_replacement::<TilePayload>,
        )?;
        allocation.tiles = ticket;
        walk_tile_group(data, start_bit_offset, tile_info, |tile| {
            candidate.values_mut().push(tile);
            Ok(())
        })?;
        Ok(TileGroup {
            start_tile: syntax.start_tile,
            end_tile: syntax.end_tile,
            data_start_offset: syntax.data_start_offset,
            tiles: candidate.into_vec(),
        })
    })();
    match result {
        Ok(group) => Ok((group, allocation)),
        Err(error) => {
            allocation.release(budget)?;
            Err(error)
        }
    }
}

fn read_le_sized_int(
    data: &[u8],
    offset: &mut usize,
    byte_count: u8,
) -> Result<usize, DecoderError> {
    if byte_count == 0 || byte_count > 4 {
        return Err(DecoderError::Bitstream(format!(
            "unsupported AV1 tile size byte count {byte_count}"
        )));
    }
    let mut value = 0usize;
    for index in 0..byte_count {
        let byte = *data
            .get(*offset)
            .ok_or_else(|| DecoderError::NotEnoughData("AV1 tile size is truncated".to_string()))?;
        *offset += 1;
        value |= usize::from(byte) << (usize::from(index) * 8);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_truncated_tile_size_field() {
        let tile_info = TileInfo {
            uniform_tile_spacing: true,
            dependent_tiles: false,
            loop_filter_across_tiles: false,
            tile_cols: 2,
            tile_rows: 1,
            tile_cols_log2: 1,
            tile_rows_log2: 0,
            tile_size_bytes: 1,
            context_update_tile_id: 0,
            mi_col_starts: vec![0, 16, 32],
            mi_row_starts: vec![0, 16],
        };

        let err = parse_tile_group(&[0], 0, &tile_info).unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("tile size"))
        );
    }

    #[test]
    fn rejects_tile_payload_extending_beyond_tile_group() {
        let tile_info = TileInfo {
            uniform_tile_spacing: true,
            dependent_tiles: false,
            loop_filter_across_tiles: false,
            tile_cols: 2,
            tile_rows: 1,
            tile_cols_log2: 1,
            tile_rows_log2: 0,
            tile_size_bytes: 1,
            context_update_tile_id: 0,
            mi_col_starts: vec![0, 16, 32],
            mi_row_starts: vec![0, 16],
        };

        let err = parse_tile_group(&[0, 0xff], 0, &tile_info).unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("payload extends"))
        );
    }
}

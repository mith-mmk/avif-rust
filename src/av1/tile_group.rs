use super::bitstream::BitReader;
use super::tile::TileInfo;
use crate::DecoderError;

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

pub fn parse_tile_group(
    data: &[u8],
    start_bit_offset: usize,
    tile_info: &TileInfo,
) -> Result<TileGroup, DecoderError> {
    let tile_count = tile_info.tile_cols * tile_info.tile_rows;
    if tile_count == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 tile count is zero".to_string(),
        ));
    }

    let mut reader = BitReader::new_at(data, start_bit_offset)?;
    let (start_tile, end_tile) = if tile_count > 1 {
        if reader.read_bool("tile_start_and_end_present_flag")? {
            let tile_num_bits = (tile_info.tile_cols_log2 + tile_info.tile_rows_log2) as usize;
            let start_tile = reader.read_bits(tile_num_bits, "tg_start")?;
            let end_tile = reader.read_bits(tile_num_bits, "tg_end")?;
            (start_tile, end_tile)
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
    let mut tiles = Vec::with_capacity((end_tile - start_tile + 1) as usize);
    for tile_id in start_tile..=end_tile {
        let len = if tile_id == end_tile {
            data.len().checked_sub(offset).ok_or_else(|| {
                DecoderError::Bitstream("AV1 tile payload offset overflow".to_string())
            })?
        } else {
            let size_minus_one = read_le_sized_int(data, &mut offset, tile_info.tile_size_bytes)?;
            size_minus_one
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
        tiles.push(TilePayload {
            tile_id,
            offset,
            len,
        });
        offset = end;
    }

    Ok(TileGroup {
        start_tile,
        end_tile,
        data_start_offset,
        tiles,
    })
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

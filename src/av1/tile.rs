use super::bitstream::BitReader;
use super::sequence::SequenceHeader;
use super::syntax::mi_dimension;
use crate::DecoderError;

const MAX_TILE_WIDTH: u32 = 4096;
const MAX_TILE_AREA: u32 = 4096 * 2304;

#[derive(Debug, Clone, Copy)]
struct TileGeometry {
    sb_cols: u32,
    sb_rows: u32,
    mi_cols: u32,
    mi_rows: u32,
    sb_size_log2: u8,
}

#[derive(Debug, Clone, Copy)]
struct UniformTileLimits {
    min_log2_tile_cols: u8,
    max_log2_tile_cols: u8,
    min_log2_tile_rows: u8,
    max_log2_tile_rows: u8,
    min_log2_tiles: u8,
}

#[derive(Debug, Clone, Copy)]
struct NonUniformTileLimits {
    max_tile_width_sb: u32,
    max_tile_area_sb: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileInfo {
    pub uniform_tile_spacing: bool,
    pub dependent_tiles: bool,
    pub loop_filter_across_tiles: bool,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    pub tile_size_bytes: u8,
    pub context_update_tile_id: u32,
    pub mi_col_starts: Vec<u32>,
    pub mi_row_starts: Vec<u32>,
}

pub(crate) fn parse_tile_info(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    frame_width: u32,
    frame_height: u32,
) -> Result<TileInfo, DecoderError> {
    let mi_cols = mi_dimension(frame_width);
    let mi_rows = mi_dimension(frame_height);
    let sb_size_log2 = if sequence.use_128x128_superblock {
        5
    } else {
        4
    };
    let sb_cols = round_shift(mi_cols, sb_size_log2);
    let sb_rows = round_shift(mi_rows, sb_size_log2);
    let max_tile_width_sb = MAX_TILE_WIDTH >> (sb_size_log2 + 2);
    let max_tile_area_sb = MAX_TILE_AREA >> (2 * (sb_size_log2 + 2));
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols);
    let max_tile_height_sb = max_tile_area_sb.max(1) / sb_cols.max(1);
    let min_log2_tile_rows = tile_log2(max_tile_height_sb.max(1), sb_rows);
    let max_log2_tile_rows = tile_log2(1, sb_rows);
    let min_log2_tiles =
        min_log2_tile_cols.max(tile_log2(max_tile_area_sb.max(1), sb_rows * sb_cols));

    let uniform_tile_spacing = reader.read_bool("uniform_tile_spacing_flag")?;
    let geometry = TileGeometry {
        sb_cols,
        sb_rows,
        mi_cols,
        mi_rows,
        sb_size_log2,
    };
    let (tile_cols, tile_rows, tile_cols_log2, tile_rows_log2, mi_col_starts, mi_row_starts) =
        if uniform_tile_spacing {
            parse_uniform_tiles(
                reader,
                geometry,
                UniformTileLimits {
                    min_log2_tile_cols,
                    max_log2_tile_cols,
                    min_log2_tile_rows,
                    max_log2_tile_rows,
                    min_log2_tiles,
                },
            )?
        } else {
            parse_non_uniform_tiles(
                reader,
                geometry,
                NonUniformTileLimits {
                    max_tile_width_sb,
                    max_tile_area_sb,
                },
            )?
        };

    let tile_count = tile_cols * tile_rows;
    let context_update_tile_id = if tile_count > 1 {
        reader.read_bits(
            (tile_rows_log2 + tile_cols_log2) as usize,
            "context_update_tile_id",
        )?
    } else {
        0
    };
    // Reduced still-picture headers used by AVIF omit these dependency flags
    // even when their derived tile grid has multiple tiles. Full AV1 frame
    // headers carry the normative fields between the tile grid and sizes.
    let dependent_tiles = if !sequence.reduced_still_picture_header && tile_rows_log2 > 0 {
        reader.read_bool("dependent_tiles")?
    } else {
        false
    };
    let loop_filter_across_tiles =
        if !sequence.reduced_still_picture_header && (tile_rows_log2 > 0 || tile_cols_log2 > 0) {
            reader.read_bool("loop_filter_across_tiles")?
        } else {
            tile_count > 1
        };
    let tile_size_bytes = if tile_count > 1 {
        reader.read_bits(2, "tile_size_bytes_minus_1")? as u8 + 1
    } else {
        0
    };

    Ok(TileInfo {
        uniform_tile_spacing,
        dependent_tiles,
        loop_filter_across_tiles,
        tile_cols,
        tile_rows,
        tile_cols_log2,
        tile_rows_log2,
        tile_size_bytes,
        context_update_tile_id,
        mi_col_starts,
        mi_row_starts,
    })
}

#[allow(clippy::type_complexity)]
fn parse_uniform_tiles(
    reader: &mut BitReader<'_>,
    geometry: TileGeometry,
    limits: UniformTileLimits,
) -> Result<(u32, u32, u8, u8, Vec<u32>, Vec<u32>), DecoderError> {
    let mut tile_cols_log2 = limits.min_log2_tile_cols;
    while tile_cols_log2 < limits.max_log2_tile_cols
        && reader.read_bool("increment_tile_cols_log2")?
    {
        tile_cols_log2 += 1;
    }
    let tile_width_sb = round_shift(geometry.sb_cols, tile_cols_log2);
    let mi_col_starts = uniform_starts(
        geometry.sb_cols,
        geometry.mi_cols,
        geometry.sb_size_log2,
        tile_width_sb,
    );
    let tile_cols = (mi_col_starts.len() - 1) as u32;

    let min_log2_tile_rows = limits
        .min_log2_tile_rows
        .max(limits.min_log2_tiles.saturating_sub(tile_cols_log2));
    let mut tile_rows_log2 = min_log2_tile_rows;
    while tile_rows_log2 < limits.max_log2_tile_rows
        && reader.read_bool("increment_tile_rows_log2")?
    {
        tile_rows_log2 += 1;
    }
    let tile_height_sb = round_shift(geometry.sb_rows, tile_rows_log2);
    let mi_row_starts = uniform_starts(
        geometry.sb_rows,
        geometry.mi_rows,
        geometry.sb_size_log2,
        tile_height_sb,
    );
    let tile_rows = (mi_row_starts.len() - 1) as u32;
    Ok((
        tile_cols,
        tile_rows,
        tile_cols_log2,
        tile_rows_log2,
        mi_col_starts,
        mi_row_starts,
    ))
}

#[allow(clippy::type_complexity)]
fn parse_non_uniform_tiles(
    reader: &mut BitReader<'_>,
    geometry: TileGeometry,
    limits: NonUniformTileLimits,
) -> Result<(u32, u32, u8, u8, Vec<u32>, Vec<u32>), DecoderError> {
    let mut widest_tile_sb = 0;
    let mut mi_col_starts = Vec::new();
    let mut start_sb = 0;
    while start_sb < geometry.sb_cols {
        mi_col_starts.push(start_sb << geometry.sb_size_log2);
        let max_width = limits.max_tile_width_sb.min(geometry.sb_cols - start_sb);
        let size_sb = reader.read_ns(max_width, "width_in_sbs_minus_1")? + 1;
        widest_tile_sb = widest_tile_sb.max(size_sb);
        start_sb += size_sb;
    }
    mi_col_starts.push(geometry.mi_cols);
    let tile_cols = (mi_col_starts.len() - 1) as u32;
    let tile_cols_log2 = tile_log2(1, tile_cols);

    let max_tile_height_sb = (limits.max_tile_area_sb / widest_tile_sb.max(1)).max(1);
    let mut mi_row_starts = Vec::new();
    let mut start_sb = 0;
    while start_sb < geometry.sb_rows {
        mi_row_starts.push(start_sb << geometry.sb_size_log2);
        let max_height = max_tile_height_sb.min(geometry.sb_rows - start_sb);
        let size_sb = reader.read_ns(max_height, "height_in_sbs_minus_1")? + 1;
        start_sb += size_sb;
    }
    mi_row_starts.push(geometry.mi_rows);
    let tile_rows = (mi_row_starts.len() - 1) as u32;
    let tile_rows_log2 = tile_log2(1, tile_rows);

    Ok((
        tile_cols,
        tile_rows,
        tile_cols_log2,
        tile_rows_log2,
        mi_col_starts,
        mi_row_starts,
    ))
}

fn uniform_starts(sb_count: u32, mi_count: u32, sb_size_log2: u8, tile_size_sb: u32) -> Vec<u32> {
    let mut starts = Vec::new();
    let mut start_sb = 0;
    while start_sb < sb_count {
        starts.push(start_sb << sb_size_log2);
        start_sb += tile_size_sb;
    }
    starts.push(mi_count);
    starts
}

fn round_shift(value: u32, shift: u8) -> u32 {
    (value + (1u32 << shift) - 1) >> shift
}

fn tile_log2(blk_size: u32, target: u32) -> u8 {
    let mut k = 0u8;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

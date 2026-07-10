use super::{PaletteBlockInfo, PalettePlaneInfo, TileDecoder};
use crate::DecoderError;
use crate::av1::frame::FrameHeader;
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::{BlockSize, PredictionMode, UvPredictionMode};
use crate::av1::tile_decode::context_grid::{fill_mi_grid, fill_mi_grid_clone};
use crate::av1::tile_decode::syntax_helpers::ceil_log2;

pub(super) const PALETTE_MAX_SIZE: usize = 8;

const PALETTE_COLOR_CONTEXT_LOOKUP: [usize; 9] = [0, 0, 0, 0, 0, 4, 3, 2, 1];

pub(super) fn palette_color_index_context(
    color_map: &[u8],
    stride: usize,
    row: usize,
    col: usize,
    palette_size: usize,
) -> (usize, [usize; PALETTE_MAX_SIZE]) {
    let neighbours = [
        if col > 0 {
            Some(color_map[row * stride + col - 1] as usize)
        } else {
            None
        },
        if row > 0 && col > 0 {
            Some(color_map[(row - 1) * stride + col - 1] as usize)
        } else {
            None
        },
        if row > 0 {
            Some(color_map[(row - 1) * stride + col] as usize)
        } else {
            None
        },
    ];
    let mut scores = [0usize; PALETTE_MAX_SIZE];
    for (index, weight) in [2usize, 1, 2].into_iter().enumerate() {
        if let Some(color) = neighbours[index] {
            if color < palette_size {
                scores[color] += weight;
            }
        }
    }

    let mut color_order = [0usize; PALETTE_MAX_SIZE];
    for (index, entry) in color_order.iter_mut().enumerate() {
        *entry = index;
    }
    for index in 0..3.min(palette_size) {
        let mut max_score = scores[index];
        let mut max_index = index;
        for candidate in index + 1..palette_size {
            if scores[candidate] > max_score {
                max_score = scores[candidate];
                max_index = candidate;
            }
        }
        if max_index != index {
            let moved_score = scores[max_index];
            let moved_color = color_order[max_index];
            for shift in (index + 1..=max_index).rev() {
                scores[shift] = scores[shift - 1];
                color_order[shift] = color_order[shift - 1];
            }
            scores[index] = moved_score;
            color_order[index] = moved_color;
        }
    }

    let hash = scores[0] + 2 * scores[1] + 2 * scores[2];
    let context = PALETTE_COLOR_CONTEXT_LOOKUP[hash.min(PALETTE_COLOR_CONTEXT_LOOKUP.len() - 1)];
    (context, color_order)
}

pub(super) fn inv_recenter_finite_nonneg(n: usize, reference: usize, value: usize) -> usize {
    let inv_recenter = |r: usize, v: usize| {
        if v > (r << 1) {
            v
        } else if v & 1 == 0 {
            (v >> 1) + r
        } else {
            r - ((v + 1) >> 1)
        }
    };
    if (reference << 1) <= n {
        inv_recenter(reference, value)
    } else {
        n - 1 - inv_recenter(n - 1 - reference, value)
    }
}

pub(super) fn palette_colors_at_mi(
    grid: &[Option<Vec<u16>>],
    mi_cols: usize,
    mi_rows: usize,
    mi_col: usize,
    mi_row: usize,
) -> Option<&[u16]> {
    if mi_col >= mi_cols || mi_row >= mi_rows {
        return None;
    }
    grid[mi_row * mi_cols + mi_col].as_deref()
}

pub(super) fn merge_palette_cache(above: Option<&[u16]>, left: Option<&[u16]>) -> Vec<u16> {
    let mut cache = Vec::with_capacity(PALETTE_MAX_SIZE * 2);
    let mut above_index = 0usize;
    let mut left_index = 0usize;
    let above = above.unwrap_or(&[]);
    let left = left.unwrap_or(&[]);
    while above_index < above.len() && left_index < left.len() {
        let above_color = above[above_index];
        let left_color = left[left_index];
        if left_color < above_color {
            push_unique_palette_cache(&mut cache, left_color);
            left_index += 1;
        } else {
            push_unique_palette_cache(&mut cache, above_color);
            above_index += 1;
            if left_color == above_color {
                left_index += 1;
            }
        }
    }
    while above_index < above.len() {
        push_unique_palette_cache(&mut cache, above[above_index]);
        above_index += 1;
    }
    while left_index < left.len() {
        push_unique_palette_cache(&mut cache, left[left_index]);
        left_index += 1;
    }
    cache
}

fn push_unique_palette_cache(cache: &mut Vec<u16>, color: u16) {
    if cache.last().copied() != Some(color) {
        cache.push(color);
    }
}

pub(super) fn merge_cached_palette_colors(
    mut colors: Vec<u16>,
    cached_count: usize,
    palette_size: usize,
) -> Result<Vec<u16>, DecoderError> {
    if colors.len() != palette_size {
        return Err(DecoderError::Bitstream(format!(
            "AV1 palette color count {} does not match size {palette_size}",
            colors.len()
        )));
    }
    if cached_count == 0 {
        return Ok(colors);
    }
    let cached_colors = colors[..cached_count].to_vec();
    let transmitted_colors = colors[cached_count..].to_vec();
    let mut cache_index = 0usize;
    let mut transmitted_index = 0usize;
    for color in colors.iter_mut().take(palette_size) {
        if cache_index < cached_colors.len()
            && (transmitted_index >= transmitted_colors.len()
                || cached_colors[cache_index] <= transmitted_colors[transmitted_index])
        {
            *color = cached_colors[cache_index];
            cache_index += 1;
        } else {
            *color = transmitted_colors
                .get(transmitted_index)
                .copied()
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 palette transmitted color is missing".to_string())
                })?;
            transmitted_index += 1;
        }
    }
    Ok(colors)
}

impl<'a> TileDecoder<'a> {
    pub(super) fn read_palette_mode_info(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        block_size: BlockSize,
        x: usize,
        y: usize,
        y_mode: PredictionMode,
        uv_mode: Option<UvPredictionMode>,
    ) -> Result<PaletteBlockInfo, DecoderError> {
        let mut palette = PaletteBlockInfo { y: None, uv: None };
        if !frame.allow_screen_content_tools
            || block_size.width() < 8
            || block_size.height() < 8
            || block_size.width() > 64
            || block_size.height() > 64
        {
            self.set_palette_size_context(x, y, block_size, &palette);
            return Ok(palette);
        }
        let area_log2 = block_size.width().ilog2() as usize + block_size.height().ilog2() as usize;
        let block_size_context = area_log2.saturating_sub(6).min(6);
        if y_mode == PredictionMode::Dc {
            let neighbour_context = self.palette_y_mode_context(x, y);
            let selected = self.reader.read_symbol(
                self.cdf
                    .palette_y_mode_cdf_mut(block_size_context, neighbour_context),
            )? != 0;
            if selected {
                let y_size = self
                    .reader
                    .read_symbol(self.cdf.palette_y_size_cdf_mut(block_size_context))?
                    + 2;
                let color_cache = self.palette_color_cache(x, y, 0);
                let colors = self.read_palette_colors_y(
                    sequence.color_config.bit_depth,
                    y_size,
                    &color_cache,
                )?;
                palette.y = Some(PalettePlaneInfo {
                    colors,
                    color_map: Vec::new(),
                    map_width: 0,
                    map_height: 0,
                });
            }
        }
        if uv_mode == Some(UvPredictionMode::Intra(PredictionMode::Dc)) {
            let y_palette_context = usize::from(palette.y.is_some());
            let selected = self
                .reader
                .read_symbol(self.cdf.palette_uv_mode_cdf_mut(y_palette_context))?
                != 0;
            if selected {
                let uv_size = self
                    .reader
                    .read_symbol(self.cdf.palette_uv_size_cdf_mut(block_size_context))?
                    + 2;
                let color_cache = self.palette_color_cache(x, y, 1);
                let colors = self.read_palette_colors_uv(
                    sequence.color_config.bit_depth,
                    uv_size,
                    &color_cache,
                )?;
                palette.uv = Some(PalettePlaneInfo {
                    colors,
                    color_map: Vec::new(),
                    map_width: 0,
                    map_height: 0,
                });
            }
        }
        self.set_palette_size_context(x, y, block_size, &palette);
        Ok(palette)
    }

    fn read_palette_colors_y(
        &mut self,
        bit_depth: u8,
        palette_size: usize,
        color_cache: &[u16],
    ) -> Result<Vec<u16>, DecoderError> {
        if !(2..=PALETTE_MAX_SIZE).contains(&palette_size) {
            return Err(DecoderError::Bitstream(format!(
                "AV1 luma palette size {palette_size} is invalid"
            )));
        }
        let bit_depth = bit_depth as usize;
        let mut cached_colors = Vec::with_capacity(palette_size);
        for &color in color_cache {
            if cached_colors.len() >= palette_size {
                break;
            }
            if self.reader.read_literal(1)? != 0 {
                cached_colors.push(color);
            }
        }
        if cached_colors.len() >= palette_size {
            return Ok(cached_colors);
        }
        let cached_count = cached_colors.len();
        let mut colors = Vec::with_capacity(palette_size);
        colors.extend_from_slice(&cached_colors);
        let mut previous = self.reader.read_literal(bit_depth)? as usize;
        colors.push(previous as u16);
        if colors.len() < palette_size {
            let mut bits = bit_depth.saturating_sub(3) + self.reader.read_literal(2)? as usize;
            let mut range = (1usize << bit_depth).saturating_sub(previous + 1);
            while colors.len() < palette_size {
                let delta = self.reader.read_literal(bits)? as usize + 1;
                let current = (previous + delta).min((1usize << bit_depth) - 1);
                range = range.saturating_sub(current.saturating_sub(previous));
                previous = current;
                colors.push(current as u16);
                bits = bits.min(ceil_log2(range));
            }
        }
        merge_cached_palette_colors(colors, cached_count, palette_size)
    }

    fn read_palette_colors_uv(
        &mut self,
        bit_depth: u8,
        palette_size: usize,
        color_cache: &[u16],
    ) -> Result<Vec<u16>, DecoderError> {
        if !(2..=PALETTE_MAX_SIZE).contains(&palette_size) {
            return Err(DecoderError::Bitstream(format!(
                "AV1 chroma palette size {palette_size} is invalid"
            )));
        }
        let bit_depth = bit_depth as usize;
        let mut cached_u_colors = Vec::with_capacity(palette_size);
        for &color in color_cache {
            if cached_u_colors.len() >= palette_size {
                break;
            }
            if self.reader.read_literal(1)? != 0 {
                cached_u_colors.push(color);
            }
        }
        let cached_count = cached_u_colors.len();
        let mut u_colors = Vec::with_capacity(palette_size);
        u_colors.extend_from_slice(&cached_u_colors);
        if u_colors.len() < palette_size {
            let mut previous_u = self.reader.read_literal(bit_depth)? as usize;
            u_colors.push(previous_u as u16);
            if u_colors.len() < palette_size {
                let mut bits = bit_depth.saturating_sub(3) + self.reader.read_literal(2)? as usize;
                let mut range = (1usize << bit_depth).saturating_sub(previous_u);
                while u_colors.len() < palette_size {
                    let delta = self.reader.read_literal(bits)? as usize;
                    let current = (previous_u + delta).min((1usize << bit_depth) - 1);
                    range = range.saturating_sub(current.saturating_sub(previous_u));
                    previous_u = current;
                    u_colors.push(current as u16);
                    bits = bits.min(ceil_log2(range));
                }
            }
            u_colors = merge_cached_palette_colors(u_colors, cached_count, palette_size)?;
        }
        if u_colors.len() != palette_size {
            return Err(DecoderError::Bitstream(
                "AV1 palette U color count is invalid".to_string(),
            ));
        }
        let mut v_colors = Vec::with_capacity(palette_size);
        if self.reader.read_literal(1)? != 0 {
            let bits = bit_depth.saturating_sub(4) + self.reader.read_literal(2)? as usize;
            let mut previous_v = self.reader.read_literal(bit_depth)? as isize;
            v_colors.push(previous_v as u16);
            let max_value = 1isize << bit_depth;
            for _ in 1..palette_size {
                let mut delta = self.reader.read_literal(bits)? as isize;
                if delta != 0 && self.reader.read_literal(1)? != 0 {
                    delta = -delta;
                }
                previous_v += delta;
                if previous_v < 0 {
                    previous_v += max_value;
                }
                if previous_v >= max_value {
                    previous_v -= max_value;
                }
                v_colors.push(previous_v as u16);
            }
        } else {
            for _ in 0..palette_size {
                v_colors.push(self.reader.read_literal(bit_depth)? as u16);
            }
        }
        u_colors.extend(v_colors);
        Ok(u_colors)
    }

    fn palette_color_cache(&self, x: usize, y: usize, plane: usize) -> Vec<u16> {
        let grid = if plane == 0 {
            &self.y_palette_colors_grid
        } else {
            &self.u_palette_colors_grid
        };
        let above = if y >= 4 && y % 64 != 0 {
            palette_colors_at_mi(grid, self.mi_cols, self.mi_rows, x >> 2, (y >> 2) - 1)
        } else {
            None
        };
        let left = if x >= 4 {
            palette_colors_at_mi(grid, self.mi_cols, self.mi_rows, (x >> 2) - 1, y >> 2)
        } else {
            None
        };
        merge_palette_cache(above, left)
    }

    pub(super) fn read_palette_tokens(
        &mut self,
        sequence: &SequenceHeader,
        block_size: BlockSize,
        x: usize,
        y: usize,
        palette: &mut PaletteBlockInfo,
    ) -> Result<(), DecoderError> {
        if let Some(y_palette) = palette.y.as_mut() {
            let (color_map, map_width, map_height) = self.read_palette_color_map_tokens(
                0,
                block_size,
                x,
                y,
                y_palette.colors.len(),
                false,
                false,
            )?;
            y_palette.color_map = color_map;
            y_palette.map_width = map_width;
            y_palette.map_height = map_height;
        }
        if let Some(uv_palette) = palette.uv.as_mut() {
            let (color_map, map_width, map_height) = self.read_palette_color_map_tokens(
                1,
                block_size,
                x,
                y,
                uv_palette.colors.len() / 2,
                sequence.color_config.subsampling_x,
                sequence.color_config.subsampling_y,
            )?;
            uv_palette.color_map = color_map;
            uv_palette.map_width = map_width;
            uv_palette.map_height = map_height;
        }
        Ok(())
    }

    fn read_palette_color_map_tokens(
        &mut self,
        plane: usize,
        block_size: BlockSize,
        x: usize,
        y: usize,
        palette_size: usize,
        subsampling_x: bool,
        subsampling_y: bool,
    ) -> Result<(Vec<u8>, usize, usize), DecoderError> {
        let (plane_block_width, plane_block_height, cols, rows) = palette_map_dimensions(
            block_size,
            x,
            y,
            self.mi_cols,
            self.mi_rows,
            subsampling_x,
            subsampling_y,
        );

        let mut color_map = vec![0u8; plane_block_width * plane_block_height];
        color_map[0] = self.reader.read_uniform(palette_size)? as u8;
        let trace =
            std::env::var_os("AVIF_TRACE_WML2_MODES").is_some() && plane == 0 && x == 88 && y == 16;
        if trace {
            eprintln!(
                "Rust palette first={} state={:?}",
                color_map[0],
                self.reader.trace_state()
            );
        }
        for diagonal in 1..rows + cols - 1 {
            let start = diagonal.min(cols - 1);
            let end = diagonal.saturating_sub(rows - 1);
            for col in (end..=start).rev() {
                let row = diagonal - col;
                let (context, color_order) = palette_color_index_context(
                    &color_map,
                    plane_block_width,
                    row,
                    col,
                    palette_size,
                );
                let color_idx = self
                    .reader
                    .read_symbol(self.cdf.palette_color_index_cdf_mut(
                        plane,
                        palette_size,
                        context,
                    ))?;
                if trace && diagonal < 4 {
                    eprintln!(
                        "Rust palette diagonal={diagonal} col={col} ctx={context} idx={color_idx} order={:?} state={:?}",
                        &color_order[..palette_size],
                        self.reader.trace_state()
                    );
                }
                color_map[row * plane_block_width + col] = color_order[color_idx] as u8;
            }
        }
        Ok((color_map, plane_block_width, plane_block_height))
    }

    fn palette_y_mode_context(&self, x: usize, y: usize) -> usize {
        let above = if y >= 4 {
            self.palette_y_size_at_mi(x >> 2, (y >> 2).saturating_sub(1)) > 0
        } else {
            false
        };
        let left = if x >= 4 {
            self.palette_y_size_at_mi((x >> 2).saturating_sub(1), y >> 2) > 0
        } else {
            false
        };
        usize::from(above) + usize::from(left)
    }

    fn palette_y_size_at_mi(&self, mi_col: usize, mi_row: usize) -> usize {
        if mi_col >= self.mi_cols || mi_row >= self.mi_rows {
            return 0;
        }
        self.y_palette_size_grid[mi_row * self.mi_cols + mi_col].unwrap_or(0)
    }

    fn set_palette_size_context(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        palette: &PaletteBlockInfo,
    ) {
        fill_mi_grid(
            &mut self.y_palette_size_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            palette.y.as_ref().map_or(0, |palette| palette.colors.len()),
        );
        fill_mi_grid(
            &mut self.uv_palette_size_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            palette
                .uv
                .as_ref()
                .map_or(0, |palette| palette.colors.len() / 2),
        );
        fill_mi_grid_clone(
            &mut self.y_palette_colors_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            palette.y.as_ref().map(|palette| palette.colors.clone()),
        );
        fill_mi_grid_clone(
            &mut self.u_palette_colors_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            palette
                .uv
                .as_ref()
                .map(|palette| palette.colors[..palette.colors.len() / 2].to_vec()),
        );
    }
}

pub(super) fn palette_map_dimensions(
    block_size: BlockSize,
    x: usize,
    y: usize,
    mi_cols: usize,
    mi_rows: usize,
    subsampling_x: bool,
    subsampling_y: bool,
) -> (usize, usize, usize, usize) {
    let subsampling_x = usize::from(subsampling_x);
    let subsampling_y = usize::from(subsampling_y);
    let plane_block_width = block_size.width() >> subsampling_x;
    let plane_block_height = block_size.height() >> subsampling_y;
    let plane_x = x >> subsampling_x;
    let plane_y = y >> subsampling_y;
    let plane_frame_width = (mi_cols << 2).div_ceil(1 << subsampling_x);
    let plane_frame_height = (mi_rows << 2).div_ceil(1 << subsampling_y);
    let cols = plane_frame_width
        .saturating_sub(plane_x)
        .min(plane_block_width)
        .max(1);
    let rows = plane_frame_height
        .saturating_sub(plane_y)
        .min(plane_block_height)
        .max(1);
    (plane_block_width, plane_block_height, cols, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_cache_merges_above_left_and_transmitted_colors() {
        assert_eq!(
            merge_palette_cache(Some(&[10, 30, 50]), Some(&[20, 30, 40])),
            vec![10, 20, 30, 40, 50]
        );
        assert_eq!(
            merge_cached_palette_colors(vec![10, 30, 20, 40], 2, 4).unwrap(),
            vec![10, 20, 30, 40]
        );
        assert_eq!(
            merge_cached_palette_colors(vec![25, 15, 35], 1, 3).unwrap(),
            vec![15, 25, 35]
        );
    }
}

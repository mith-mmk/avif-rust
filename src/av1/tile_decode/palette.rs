use crate::DecoderError;

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

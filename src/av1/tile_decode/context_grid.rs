use crate::av1::syntax::{BlockSize, PredictionMode};

pub(super) fn intra_mode_context(mode: PredictionMode) -> usize {
    match mode {
        PredictionMode::Dc => 0,
        PredictionMode::Vertical => 1,
        PredictionMode::Horizontal => 2,
        PredictionMode::Smooth
        | PredictionMode::SmoothVertical
        | PredictionMode::SmoothHorizontal => 3,
        _ => 4,
    }
}

fn smooth_mode_at(
    grid: &[Option<bool>],
    mi_cols: usize,
    mi_rows: usize,
    mi_col: usize,
    mi_row: usize,
) -> bool {
    if mi_col >= mi_cols || mi_row >= mi_rows {
        return false;
    }
    grid[mi_row * mi_cols + mi_col].unwrap_or(false)
}

pub(super) fn has_smooth_neighbour(
    grid: &[Option<bool>],
    mi_cols: usize,
    mi_rows: usize,
    x: usize,
    y: usize,
) -> bool {
    let mi_col = x >> 2;
    let mi_row = y >> 2;
    let above = y >= 4 && smooth_mode_at(grid, mi_cols, mi_rows, mi_col, mi_row - 1);
    let left = x >= 4 && smooth_mode_at(grid, mi_cols, mi_rows, mi_col - 1, mi_row);
    above || left
}

pub(super) fn fill_mi_grid<T: Copy>(
    grid: &mut [Option<T>],
    mi_cols: usize,
    mi_rows: usize,
    x: usize,
    y: usize,
    block_size: BlockSize,
    value: T,
) {
    let start_col = x >> 2;
    let start_row = y >> 2;
    let end_col = ((x + block_size.width()).min(mi_cols << 2) + 3) >> 2;
    let end_row = ((y + block_size.height()).min(mi_rows << 2) + 3) >> 2;
    for mi_row in start_row..end_row.min(mi_rows) {
        for mi_col in start_col..end_col.min(mi_cols) {
            grid[mi_row * mi_cols + mi_col] = Some(value);
        }
    }
}

pub(super) fn fill_mi_grid_clone<T: Clone>(
    grid: &mut [Option<T>],
    mi_cols: usize,
    mi_rows: usize,
    x: usize,
    y: usize,
    block_size: BlockSize,
    value: Option<T>,
) {
    let start_col = x >> 2;
    let start_row = y >> 2;
    let end_col = ((x + block_size.width()).min(mi_cols << 2) + 3) >> 2;
    let end_row = ((y + block_size.height()).min(mi_rows << 2) + 3) >> 2;
    for mi_row in start_row..end_row.min(mi_rows) {
        for mi_col in start_col..end_col.min(mi_cols) {
            grid[mi_row * mi_cols + mi_col] = value.clone();
        }
    }
}

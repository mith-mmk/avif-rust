use super::context_grid::{fill_mi_grid, has_smooth_neighbour};
use crate::av1::syntax::{BlockSize, PredictionMode};

#[test]
fn smooth_mode_grid_tracks_above_and_left_neighbours() {
    let mut grid = vec![None; 16];
    fill_mi_grid(&mut grid, 4, 4, 4, 4, BlockSize::Block8x8, true);

    assert!(has_smooth_neighbour(&grid, 4, 4, 4, 12));
    assert!(has_smooth_neighbour(&grid, 4, 4, 12, 4));
    assert!(!has_smooth_neighbour(&grid, 4, 4, 0, 0));
    assert!(PredictionMode::Smooth.is_smooth());
    assert!(!PredictionMode::Vertical.is_smooth());
}

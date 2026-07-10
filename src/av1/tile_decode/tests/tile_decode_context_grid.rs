use super::context_grid::{fill_mi_grid, has_smooth_neighbour, intra_mode_context};
use crate::av1::syntax::{BlockSize, PredictionMode};

#[test]
fn intra_mode_context_matches_av1_table() {
    let modes = [
        PredictionMode::Dc,
        PredictionMode::Vertical,
        PredictionMode::Horizontal,
        PredictionMode::D45,
        PredictionMode::D135,
        PredictionMode::D113,
        PredictionMode::D157,
        PredictionMode::D203,
        PredictionMode::D67,
        PredictionMode::Smooth,
        PredictionMode::SmoothVertical,
        PredictionMode::SmoothHorizontal,
        PredictionMode::Paeth,
    ];
    let expected = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

    assert_eq!(modes.map(intra_mode_context), expected);
}
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

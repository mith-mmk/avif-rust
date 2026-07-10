use super::{PaletteBlockInfo, PalettePlaneInfo};
use crate::av1::syntax::BlockSize;
use crate::av1::tile_decode::palette::palette_map_dimensions;

#[test]
fn palette_map_dimensions_are_per_pixel_and_clip_at_frame_edges() {
    assert_eq!(
        palette_map_dimensions(BlockSize::Block8x8, 0, 0, 16, 16, false, false),
        (8, 8, 8, 8)
    );
    assert_eq!(
        palette_map_dimensions(BlockSize::Block8x8, 8, 8, 3, 3, false, false),
        (8, 8, 4, 4)
    );
    assert_eq!(
        palette_map_dimensions(BlockSize::Block16x8, 16, 8, 5, 3, true, true),
        (8, 4, 2, 2)
    );
}

#[test]
fn palette_diagnostic_accessors_expose_read_only_plane_metadata() {
    let y = PalettePlaneInfo {
        colors: vec![10, 20, 30],
        color_map: vec![0, 1, 2, 1],
        map_width: 2,
        map_height: 2,
    };
    let uv = PalettePlaneInfo {
        colors: vec![40, 50, 60, 70],
        color_map: Vec::new(),
        map_width: 0,
        map_height: 0,
    };
    let palette = PaletteBlockInfo {
        y: Some(y),
        uv: Some(uv),
    };

    assert!(palette.has_palette());
    assert!(palette.has_non_empty_color_map());

    let y = palette.y().expect("luma palette should be exposed");
    assert_eq!(y.colors(), &[10, 20, 30]);
    assert_eq!(y.color_map(), &[0, 1, 2, 1]);
    assert_eq!(y.map_width(), 2);
    assert_eq!(y.map_height(), 2);

    let uv = palette.uv().expect("chroma palette should be exposed");
    assert_eq!(uv.colors(), &[40, 50, 60, 70]);
    assert!(uv.color_map().is_empty());
    assert_eq!(uv.map_width(), 0);
    assert_eq!(uv.map_height(), 0);
}

#[test]
fn palette_diagnostic_helpers_distinguish_empty_maps_from_absent_palettes() {
    let absent = PaletteBlockInfo { y: None, uv: None };
    assert!(!absent.has_palette());
    assert!(!absent.has_non_empty_color_map());

    let empty_map = PaletteBlockInfo {
        y: Some(PalettePlaneInfo {
            colors: vec![1, 2],
            color_map: Vec::new(),
            map_width: 0,
            map_height: 0,
        }),
        uv: None,
    };
    assert!(empty_map.has_palette());
    assert!(!empty_map.has_non_empty_color_map());
}

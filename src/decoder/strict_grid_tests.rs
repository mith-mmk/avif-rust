use super::frame::reject_strict_derived_alpha;
use super::still::{finish_grid_frame_geometry, prepare_grid_cells};
use super::*;
use crate::container::{CleanAperture, ImageMirror, ImageRotation};

fn make_frame() -> DecodedFrame {
    let layout = PlaneLayout {
        plane: 0,
        width: 3,
        height: 2,
        subsampling_x: 0,
        subsampling_y: 0,
        sample_count: 6,
    };
    DecodedFrame {
        width: 3,
        height: 2,
        render_width: 3,
        render_height: 2,
        bit_depth: 8,
        color_config: ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: true,
            color_description: None,
            color_range: ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        },
        color_information: None,
        alpha_premultiplied: false,
        buffers: FrameBuffers {
            width: 3,
            height: 2,
            planes: vec![PlaneBuffer {
                layout,
                samples: vec![1, 2, 3, 4, 5, 6],
            }],
        },
    }
}

fn info() -> AvifInfo {
    AvifInfo {
        major_brand: *b"avif",
        compatible_brands: vec![*b"avif"],
        primary_item_id: Some(1),
        width: Some(3),
        height: Some(2),
        pixel_information: None,
        color_information: None,
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: Some(CleanAperture {
            width_n: 1,
            width_d: 1,
            height_n: 1,
            height_d: 1,
            horizontal_offset_n: 0,
            horizontal_offset_d: 1,
            vertical_offset_n: 0,
            vertical_offset_d: 1,
        }),
        rotation: Some(ImageRotation { angle: 1 }),
        mirror: Some(ImageMirror { axis: 0 }),
        av1_config: None,
        primary_item_payload: Vec::new(),
        sequence_sample_payloads: Vec::new(),
    }
}

#[test]
fn strict_grid_geometry_keeps_crop_rotation_and_mirror_as_metadata() {
    let mut frame = make_frame();
    let original = frame.clone();
    finish_grid_frame_geometry(&mut frame, &info(), false).unwrap();
    assert_eq!(frame, original);
}

#[test]
fn legacy_grid_geometry_policy_still_applies_explicitly() {
    let mut frame = make_frame();
    finish_grid_frame_geometry(&mut frame, &info(), true).unwrap();
    assert_ne!(frame, make_frame());
}

#[test]
fn strict_grid_does_not_normalize_mixed_monochrome_cells() {
    let mut color = make_frame();
    color.color_config.monochrome = false;
    let mut chroma_u = color.buffers.planes[0].clone();
    chroma_u.layout.plane = 1;
    let mut chroma_v = chroma_u.clone();
    chroma_v.layout.plane = 2;
    color.buffers.planes.extend([chroma_u, chroma_v]);

    let mut raw_cells = vec![make_frame(), color.clone()];
    prepare_grid_cells(&mut raw_cells, false).unwrap();
    assert!(raw_cells[0].color_config.monochrome);
    assert_eq!(raw_cells[0].buffers.planes.len(), 1);

    let mut legacy_cells = vec![make_frame(), color];
    prepare_grid_cells(&mut legacy_cells, true).unwrap();
    assert!(!legacy_cells[0].color_config.monochrome);
    assert_eq!(legacy_cells[0].buffers.planes.len(), 3);
}

#[test]
fn strict_derived_grid_and_sato_reject_both_alpha_descriptors() {
    for derived_kind in ["grid", "sato"] {
        for (has_alpha_grid, has_alpha_auxiliary) in [(true, false), (false, true)] {
            assert!(matches!(
                reject_strict_derived_alpha(
                    derived_kind,
                    has_alpha_grid,
                    has_alpha_auxiliary
                ),
                Err(DecoderError::Unsupported(message))
                    if message.contains(derived_kind)
            ));
        }
        reject_strict_derived_alpha(derived_kind, false, false).unwrap();
    }
}

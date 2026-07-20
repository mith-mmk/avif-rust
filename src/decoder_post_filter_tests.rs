use super::{
    DecodedFrame, apply_loop_filter_deltas, apply_loop_restoration_stage, cdef_strengths_disabled,
};
use crate::av1::{
    ColorConfig, ColorRange, FrameBuffers, PlaneBuffer, PlaneLayout, PostFilterState,
    RestorationUnit, wiener_filter_unit,
};

#[test]
fn segmentation_loop_filter_delta_is_applied_before_other_deltas() {
    assert_eq!(apply_loop_filter_deltas(10, false, 7, 9, 2, 5), 17);
    assert_eq!(apply_loop_filter_deltas(10, true, 1, 2, 2, 5), 20);
    assert_eq!(apply_loop_filter_deltas(2, false, 0, 0, 0, -9), 0);
    assert_eq!(apply_loop_filter_deltas(60, true, 0, 0, 0, 9), 63);
}

#[test]
fn zero_cdef_strengths_are_detected_without_filtering() {
    assert!(cdef_strengths_disabled(0, 0));
    assert!(!cdef_strengths_disabled(1, 0));
    assert!(!cdef_strengths_disabled(0, 1));
}

#[test]
fn restoration_units_share_the_same_cdef_source_snapshot() {
    let width = 16;
    let height = 8;
    let source = (0..width * height)
        .map(|index| ((index * 29 + index / width * 17) & 255) as u16)
        .collect::<Vec<_>>();
    let filters = [[[0, -3, 8], [0, 4, -7]], [[0, 5, -9], [0, -2, 6]]];
    let state = PostFilterState {
        restoration_units: filters
            .iter()
            .enumerate()
            .map(|(index, filters)| RestorationUnit {
                x: index * 8,
                y: 0,
                plane: 0,
                restoration_type: 1,
                wiener: Some(*filters),
                sgrproj: None,
                sgrproj_index: None,
            })
            .collect(),
        ..PostFilterState::default()
    };
    let mut expected = source.clone();
    for (index, filters) in filters.iter().enumerate() {
        let filtered = wiener_filter_unit(&source, width, height, index * 8, 0, 8, 8, *filters);
        for y in 0..height {
            let start = y * width + index * 8;
            expected[start..start + 8].copy_from_slice(&filtered[start..start + 8]);
        }
    }
    let layout = PlaneLayout {
        plane: 0,
        width,
        height,
        subsampling_x: 0,
        subsampling_y: 0,
        sample_count: source.len(),
    };
    let mut frame = DecodedFrame {
        width,
        height,
        render_width: width,
        render_height: height,
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
            width,
            height,
            planes: vec![PlaneBuffer {
                layout,
                samples: source,
            }],
        },
    };

    apply_loop_restoration_stage(&mut frame, &state, 8, &[1, 2]);

    assert_eq!(frame.buffers.planes[0].samples, expected);
}

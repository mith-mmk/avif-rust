use super::{
    DecodedFrame, apply_alpha_rows, apply_cdef_plane, apply_loop_filter_deltas,
    apply_loop_restoration_stage, cdef_has_active_strengths, cdef_strengths_disabled,
};
use crate::av1::CdefParams;
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
fn all_zero_cdef_indices_skip_the_whole_stage() {
    let mut cdef = CdefParams {
        enabled: true,
        bits: 2,
        ..CdefParams::default()
    };
    assert!(!cdef_has_active_strengths(&cdef));
    cdef.strengths[2].uv_sec = 1;
    assert!(cdef_has_active_strengths(&cdef));
}

#[test]
fn cdef_skips_a_plane_without_configured_strengths() {
    let source = (0..64).map(|value| value as u16).collect::<Vec<_>>();
    let mut plane = PlaneBuffer {
        layout: PlaneLayout {
            plane: 0,
            width: 8,
            height: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: source.len(),
        },
        samples: source.clone(),
    };
    let mut cdef = CdefParams {
        enabled: true,
        bits: 0,
        ..CdefParams::default()
    };
    cdef.strengths[0].uv_sec = 4;
    apply_cdef_plane(&mut plane, 0, false, false, cdef, &[(0, 0, 0, 0, 0)]);
    assert_eq!(plane.samples, source);
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

#[test]
fn restoration_runs_independently_for_multiple_planes() {
    let width = 16;
    let height = 8;
    let filters = [[[0, -3, 8], [0, 4, -7]], [[0, 5, -9], [0, -2, 6]]];
    let sources = (0..3)
        .map(|plane| {
            (0..width * height)
                .map(|index| ((index * 17 + plane * 31) & 255) as u16)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let state = PostFilterState {
        restoration_units: (0..3)
            .map(|plane| RestorationUnit {
                x: 0,
                y: 0,
                plane,
                restoration_type: 1,
                wiener: Some(filters[plane.min(1)]),
                sgrproj: None,
                sgrproj_index: None,
            })
            .collect(),
        ..PostFilterState::default()
    };
    let layout = |plane: usize| PlaneLayout {
        plane: plane as u8,
        width,
        height,
        subsampling_x: 0,
        subsampling_y: 0,
        sample_count: width * height,
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
            monochrome: false,
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
            planes: sources
                .iter()
                .enumerate()
                .map(|(plane, samples)| PlaneBuffer {
                    layout: layout(plane),
                    samples: samples.clone(),
                })
                .collect(),
        },
    };

    apply_loop_restoration_stage(&mut frame, &state, 8, &[1, 2]);

    for (plane, source) in sources.iter().enumerate() {
        let mut filter = filters[plane.min(1)];
        if plane > 0 {
            filter[0][0] = 0;
            filter[1][0] = 0;
        }
        let filtered = wiener_filter_unit(source, width, height, 0, 0, 8, 8, filter);
        let mut expected = source.clone();
        for row in 0..height {
            expected[row * width..row * width + 8]
                .copy_from_slice(&filtered[row * width..row * width + 8]);
        }
        assert_eq!(
            frame.buffers.planes[plane].samples, expected,
            "plane {plane} restoration mismatch"
        );
    }
}

#[test]
fn alpha_row_chunks_match_one_pass_with_subsampling() {
    let width = 8;
    let height = 4;
    let alpha_samples = vec![0, 64, 128, 255, 32, 96, 160, 224];
    let mut one_pass = vec![0u8; width * height * 4];
    apply_alpha_rows(&mut one_pass, 0, width, 4, 2, 1, 1, &alpha_samples, 8);
    let mut chunked = vec![0u8; width * height * 4];
    for row in 0..height {
        let start = row * width * 4;
        apply_alpha_rows(
            &mut chunked[start..start + width * 4],
            row,
            width,
            4,
            2,
            1,
            1,
            &alpha_samples,
            8,
        );
    }
    assert_eq!(chunked, one_pass);
}

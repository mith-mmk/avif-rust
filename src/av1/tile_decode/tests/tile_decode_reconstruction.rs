use super::*;
use crate::av1::decode::{PlaneBuffer, PlaneLayout};
use crate::av1::tile_decode::reconstruction::apply_cfl_prediction;
use crate::av1::{
    alloc_frame_buffers, build_still_decode_plan, parse_frame_header, parse_sequence_header,
    parse_tile_group,
};
use crate::container::parse_avif;
use crate::obu::{ObuType, find_obu_payload};

#[test]
fn cfl_prediction_applies_signed_q3_luma_ac() {
    let luma = PlaneBuffer {
        layout: PlaneLayout {
            plane: 0,
            width: 4,
            height: 4,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 16,
        },
        samples: (0..16).collect(),
    };
    let mut positive = vec![128; 16];
    apply_cfl_prediction(&mut positive, &luma, 0, 0, 4, 4, 8, 8, 0, 0).unwrap();
    assert_eq!(
        positive,
        vec![
            120, 121, 122, 123, 124, 125, 126, 127, 129, 130, 131, 132, 133, 134, 135, 136
        ]
    );

    let mut negative = vec![128; 16];
    apply_cfl_prediction(&mut negative, &luma, 0, 0, 4, 4, -8, 8, 0, 0).unwrap();
    assert_eq!(
        negative,
        vec![
            136, 135, 134, 133, 132, 131, 130, 129, 127, 126, 125, 124, 123, 122, 121, 120
        ]
    );

    let mut zero = vec![128; 16];
    apply_cfl_prediction(&mut zero, &luma, 0, 0, 4, 4, 0, 8, 0, 0).unwrap();
    assert_eq!(zero, vec![128; 16]);
}

#[test]
fn cfl_subsampled_prediction_keeps_fractional_luma_precision() {
    // The first pair/quad averages to 1/2. Q3 must retain 4 rather than
    // round it to one luma sample before computing AC. Fixed expectations
    // follow AV1 7.11.5, independently of encoder prediction code.
    for bits in [8, 10, 12] {
        for (sx, sy, height, samples) in [
            (1, 0, 1, vec![0, 1, 0, 0]),
            (1, 1, 2, vec![0, 1, 0, 0, 1, 0, 0, 0]),
        ] {
            let luma = PlaneBuffer {
                layout: PlaneLayout {
                    plane: 0,
                    width: 4,
                    height,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    sample_count: 4 * height,
                },
                samples,
            };
            for (alpha, expected) in [(16, [129, 127]), (-16, [127, 129])] {
                let mut prediction = [128; 2];
                apply_cfl_prediction(&mut prediction, &luma, 0, 0, 2, 1, alpha, bits, sx, sy)
                    .unwrap();
                assert_eq!(
                    prediction, expected,
                    "bits={bits} sampling={sx}/{sy} alpha={alpha}"
                );
            }
        }
    }
}

#[test]
fn decodes_sample_first_luma_transform_into_frame_buffer() {
    let Some(data) = read_sample_avif() else {
        return;
    };
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    let frame = parse_frame_header(frame_payload, &sequence).unwrap();
    let tile_group = parse_tile_group(
        frame_payload,
        frame.uncompressed_header_bits,
        &frame.tile_info,
    )
    .unwrap();
    let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
    let mut buffers = alloc_frame_buffers(&plan).unwrap();
    buffers.planes[0].samples.fill(u16::MAX);

    let residual = decode_first_luma_transform(
        frame_payload,
        &tile_group,
        &sequence,
        &frame,
        &plan,
        &mut buffers,
    )
    .unwrap();

    assert_eq!(residual.tile_id, 0);
    assert!(residual.first_tx_size.is_some());
    if residual.first_non_zero_transform.is_none() {
        assert_eq!(residual.zero_transform_count, residual.transform_count);
    } else {
        assert!(
            buffers.planes[0]
                .samples
                .iter()
                .any(|sample| *sample != u16::MAX)
        );
    }
}

#[test]
fn decodes_sample_first_luma_block_transforms_into_frame_buffer() {
    let Some(data) = read_sample_avif() else {
        return;
    };
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    let frame = parse_frame_header(frame_payload, &sequence).unwrap();
    let tile_group = parse_tile_group(
        frame_payload,
        frame.uncompressed_header_bits,
        &frame.tile_info,
    )
    .unwrap();
    let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
    let mut buffers = alloc_frame_buffers(&plan).unwrap();
    for plane in &mut buffers.planes {
        plane.samples.fill(u16::MAX);
    }

    let decoded = decode_first_luma_block(
        frame_payload,
        &tile_group,
        &sequence,
        &frame,
        &plan,
        &mut buffers,
    )
    .unwrap();

    assert!(
        decoded
            .iter()
            .all(|transform| transform.transform.plane == 0)
    );
    assert!(
        buffers
            .planes
            .iter()
            .all(|plane| { plane.samples.iter().any(|sample| *sample != u16::MAX) })
    );
}

#[test]
fn decodes_sample_luma_root_block_prefix_with_split_children() {
    let Some(data) = read_sample_avif() else {
        return;
    };
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    let frame = parse_frame_header(frame_payload, &sequence).unwrap();
    let tile_group = parse_tile_group(
        frame_payload,
        frame.uncompressed_header_bits,
        &frame.tile_info,
    )
    .unwrap();
    let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
    let mut buffers = alloc_frame_buffers(&plan).unwrap();

    let prefix = decode_luma_root_block_prefix(
        frame_payload,
        &tile_group,
        &sequence,
        &frame,
        &plan,
        &mut buffers,
        8,
    )
    .unwrap();
    let blocks = prefix.blocks;

    assert_eq!(blocks.len(), 8);
    assert_eq!((blocks[0].x, blocks[0].y), (0, 0));
    assert_eq!((blocks[1].x, blocks[1].y), (64, 0));
    assert!(blocks.iter().any(|block| !block.transforms.is_empty()));
    assert!(buffers.planes[1].samples.iter().any(|sample| *sample != 0));
    assert!(buffers.planes[2].samples.iter().any(|sample| *sample != 0));
    assert_eq!(prefix.next_unsupported, None);
}

#[test]
fn normal_prefix_decode_skips_diagnostic_block_retention() {
    let Some(data) = read_sample_avif() else {
        return;
    };
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    let frame = parse_frame_header(frame_payload, &sequence).unwrap();
    let tile_group = parse_tile_group(
        frame_payload,
        frame.uncompressed_header_bits,
        &frame.tile_info,
    )
    .unwrap();
    let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
    let mut buffers = alloc_frame_buffers(&plan).unwrap();

    let (prefix, _) =
        crate::av1::decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options(
            frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
            8,
            false,
            false,
        )
        .unwrap();

    assert!(prefix.blocks.is_empty());
    assert_eq!(prefix.next_unsupported, None);
    assert!(
        buffers
            .planes
            .iter()
            .all(|plane| { plane.samples.iter().any(|sample| *sample != 0) })
    );
}

#[test]
fn decodes_sample_prefix_through_palette_blocks() {
    let Some(data) = read_sample_avif() else {
        return;
    };
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    let frame = parse_frame_header(frame_payload, &sequence).unwrap();
    let tile_group = parse_tile_group(
        frame_payload,
        frame.uncompressed_header_bits,
        &frame.tile_info,
    )
    .unwrap();
    let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
    let mut buffers = alloc_frame_buffers(&plan).unwrap();

    let prefix = decode_luma_root_block_prefix(
        frame_payload,
        &tile_group,
        &sequence,
        &frame,
        &plan,
        &mut buffers,
        4096,
    )
    .unwrap();

    // Reaching a leaf that covers the clipped bottom-right corner confirms
    // traversal of the complete luma tree without pinning an intermediate
    // diagnostic block count while entropy conformance is still in progress.
    assert!(prefix.blocks.iter().any(|block| {
        block.x + block.block_size.width() >= plan.width
            && block.y + block.block_size.height() >= plan.height
    }));
    assert_eq!(prefix.next_unsupported, None);
    let palette_blocks = prefix
        .blocks
        .iter()
        .filter(|block| block.palette.has_palette())
        .count();
    assert!(palette_blocks > 0);
    assert!(
        prefix
            .blocks
            .iter()
            .any(|block| block.palette.has_non_empty_color_map())
    );
    assert!(prefix.blocks.iter().all(|block| {
        [block.palette.y(), block.palette.uv()]
            .into_iter()
            .flatten()
            .all(|palette| {
                !palette.colors().is_empty()
                    && palette.map_width() > 0
                    && palette.map_height() > 0
                    && palette.color_map().len() == palette.map_width() * palette.map_height()
            })
    }));
    assert!(
        buffers
            .planes
            .iter()
            .all(|plane| { plane.samples.iter().any(|sample| *sample != 0) })
    );
}

fn read_sample_avif() -> Option<Vec<u8>> {
    crate::test_support::wml2viewer_avif()
}

#[test]
fn small_inter_predictions_use_dimension_specific_four_tap_filters() {
    // AOM filter.h av1_sub_pel_filters_4[8] is [0,0,-12,76,76,-12,0,0].
    // A single 128-valued impulse at tap 2 therefore contributes -12; the
    // full regular and sharp kernels incorrectly contribute -14 and -24.
    // Add a DC pedestal so negative filter contributions are not clipped.
    let plane = PlaneBuffer {
        layout: PlaneLayout {
            plane: 0,
            width: 16,
            height: 16,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 256,
        },
        samples: (0..256)
            .map(|i| if i % 16 == 3 { 192 } else { 64 })
            .collect(),
    };
    use crate::av1::frame::InterpolationFilter::*;
    use crate::av1::tile_decode::reconstruction::predict_inter_sample;
    for filter in [Regular, Sharp, Smooth] {
        let expected = if filter == Smooth { 76 } else { 52 };
        assert_eq!(
            predict_inter_sample(&plane, 4, 4, 8, 0, (filter, Regular), 8, (4, 8)),
            expected,
            "horizontal {filter:?}"
        );
        // Small vertical dimension must not switch the horizontal kernel.
        let wide = if filter == Sharp {
            40
        } else if filter == Smooth {
            78
        } else {
            50
        };
        assert_eq!(
            predict_inter_sample(&plane, 4, 4, 8, 0, (filter, Regular), 8, (8, 4)),
            wide,
            "wide horizontal {filter:?}"
        );
    }
    let mut transposed = plane.clone();
    for y in 0..16 {
        for x in 0..16 {
            transposed.samples[y * 16 + x] = plane.samples[x * 16 + y];
        }
    }
    for filter in [Regular, Sharp, Smooth] {
        let expected = if filter == Smooth { 76 } else { 52 };
        assert_eq!(
            predict_inter_sample(&transposed, 4, 4, 0, 8, (Regular, filter), 8, (8, 4)),
            expected,
            "vertical {filter:?}"
        );
    }
}

#[test]
fn obmc_filter_choice_is_independent_of_transform_subdivision() {
    use crate::av1::frame::InterpolationFilter;
    use crate::av1::tile_decode::diagnostic::{ObmcNeighbor, ObmcNeighbors};
    use crate::av1::tile_decode::reconstruction::apply_obmc_edge_blend;
    let reference = PlaneBuffer {
        layout: PlaneLayout {
            plane: 0,
            width: 32,
            height: 32,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 1024,
        },
        samples: (0..1024)
            .map(|i| ((i * 37 + i / 32 * 53) % 256) as u16)
            .collect(),
    };
    for above in [true, false] {
        for filter in [
            InterpolationFilter::Regular,
            InterpolationFilter::Sharp,
            InterpolationFilter::Smooth,
        ] {
            let neighbor = ObmcNeighbor {
                origin_x: if above { 8 } else { 0 },
                origin_y: if above { 0 } else { 8 },
                width: 16,
                height: 16,
                motion_vector: (4, 4),
                interpolation_filters: (filter, filter),
            };
            let mut neighbors = ObmcNeighbors::default();
            if above {
                neighbors.above[0] = Some(neighbor);
            } else {
                neighbors.left[0] = Some(neighbor);
            }
            let mut whole = vec![64; 16 * 16];
            apply_obmc_edge_blend(
                &mut whole,
                &reference,
                8,
                8,
                8,
                8,
                16,
                16,
                16,
                16,
                0,
                0,
                &neighbors,
                &mut [0; 256],
                8,
            )
            .unwrap();
            assert!(whole.iter().any(|&sample| sample != 64));
            let mut divided = vec![64; 16 * 16];
            for y in (0..16).step_by(4) {
                for x in (0..16).step_by(4) {
                    let mut tile = [64; 16];
                    apply_obmc_edge_blend(
                        &mut tile,
                        &reference,
                        8,
                        8,
                        8 + x,
                        8 + y,
                        4,
                        4,
                        16,
                        16,
                        0,
                        0,
                        &neighbors,
                        &mut [0; 16],
                        8,
                    )
                    .unwrap();
                    for row in 0..4 {
                        divided[(y + row) * 16 + x..(y + row) * 16 + x + 4]
                            .copy_from_slice(&tile[row * 4..row * 4 + 4]);
                    }
                }
            }
            assert_eq!(whole, divided, "above={above} filter={filter:?}");
        }
    }
}

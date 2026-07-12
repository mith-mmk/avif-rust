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
    apply_cfl_prediction(&mut positive, &luma, 0, 0, 4, 4, 8, 8).unwrap();
    assert_eq!(
        positive,
        vec![
            120, 121, 122, 123, 124, 125, 126, 127, 129, 130, 131, 132, 133, 134, 135, 136
        ]
    );

    let mut negative = vec![128; 16];
    apply_cfl_prediction(&mut negative, &luma, 0, 0, 4, 4, -8, 8).unwrap();
    assert_eq!(
        negative,
        vec![
            136, 135, 134, 133, 132, 131, 130, 129, 127, 126, 125, 124, 123, 122, 121, 120
        ]
    );

    let mut zero = vec![128; 16];
    apply_cfl_prediction(&mut zero, &luma, 0, 0, 4, 4, 0, 8).unwrap();
    assert_eq!(zero, vec![128; 16]);
}

#[test]
fn decodes_sample_first_luma_transform_into_frame_buffer() {
    let data = read_sample_avif();
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
    let data = read_sample_avif();
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
    let data = read_sample_avif();
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
fn decodes_sample_prefix_through_palette_blocks() {
    let data = read_sample_avif();
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

    // Rectangular Tx4x16 traversal now reaches the complete luma tree.
    assert_eq!(prefix.blocks.len(), 2110);
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

fn read_sample_avif() -> Vec<u8> {
    std::fs::read(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("samples")
            .join("WML2Viewer.avif"),
    )
    .expect("sample AVIF should exist")
}

mod support;

use avif_rust::av1::{
    ColorDescription, ColorRange, FrameType, TxMode, alloc_frame_buffers, build_still_decode_plan,
    parse_av1_config, parse_frame_header, parse_sequence_header, parse_tile_group,
};
use avif_rust::container::parse_avif;
use avif_rust::obu::{ObuType, find_obu_payload};
use support::sample_path;

fn sample_avif() -> Vec<u8> {
    std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist")
}

fn sample_payloads() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let data = sample_avif();
    let info = parse_avif(&data).unwrap();
    let config = info.av1_config.expect("sample should contain av1C");
    let sequence = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist")
        .to_vec();
    let frame = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .or_else(|| find_obu_payload(&info.primary_item_payload, ObuType::FrameHeader).unwrap())
        .expect("frame OBU should exist")
        .to_vec();

    (config, sequence, frame)
}

#[test]
fn sample_av1_config_is_exposed_through_public_api() {
    let (config_payload, _, _) = sample_payloads();
    let config = parse_av1_config(&config_payload).unwrap();

    assert_eq!(config.version, 1);
    assert_eq!(config.seq_profile, 1);
    assert_eq!(config.seq_level_idx_0, 5);
    assert_eq!(config.bit_depth(), 8);
    assert!(!config.monochrome);
    assert!(!config.chroma_subsampling_x);
    assert!(!config.chroma_subsampling_y);
    assert!(config.initial_presentation_delay.is_none());
}

#[test]
fn sample_sequence_header_is_exposed_through_public_api() {
    let (_, sequence_payload, _) = sample_payloads();
    let header = parse_sequence_header(&sequence_payload).unwrap();

    assert_eq!(header.seq_profile, 1);
    assert!(!header.still_picture);
    assert!(!header.reduced_still_picture_header);
    assert_eq!(header.seq_level_idx_0, 5);
    assert_eq!(header.max_frame_width, 900);
    assert_eq!(header.max_frame_height, 900);
    assert!(header.enable_cdef);
    assert_eq!(header.color_config.bit_depth, 8);
    assert!(!header.color_config.monochrome);
    assert_eq!(header.color_config.color_range, ColorRange::Full);
    assert!(!header.color_config.subsampling_x);
    assert!(!header.color_config.subsampling_y);
    assert_eq!(
        header.color_config.color_description,
        Some(ColorDescription {
            color_primaries: 1,
            transfer_characteristics: 13,
            matrix_coefficients: 0,
        })
    );
}

#[test]
fn sample_frame_header_and_tile_group_are_exposed_through_public_api() {
    let (_, sequence_payload, frame_payload) = sample_payloads();
    let sequence = parse_sequence_header(&sequence_payload).unwrap();
    let header = parse_frame_header(&frame_payload, &sequence).unwrap();

    assert_eq!(header.frame_type, FrameType::Key);
    assert!(!header.show_existing_frame);
    assert!(header.show_frame);
    assert!(header.error_resilient_mode);
    assert_eq!(header.refresh_frame_flags, 0xff);
    assert_eq!(header.frame_width, 900);
    assert_eq!(header.frame_height, 900);
    assert_eq!(header.upscaled_width, 900);
    assert_eq!(header.render_width, 900);
    assert_eq!(header.render_height, 900);
    assert!(header.tile_info.uniform_tile_spacing);
    assert_eq!(header.tile_info.tile_cols, 1);
    assert_eq!(header.tile_info.tile_rows, 1);
    assert_eq!(header.tile_info.tile_size_bytes, 0);
    assert_eq!(header.tile_info.context_update_tile_id, 0);
    assert_eq!(header.tile_info.mi_col_starts, &[0, 226]);
    assert_eq!(header.tile_info.mi_row_starts, &[0, 226]);
    assert_eq!(header.tx_mode, TxMode::Select);
    assert!(!header.reduced_tx_set);
    assert!(header.uncompressed_header_bits > 0);
    assert!(header.payload_after_header_offset < frame_payload.len());

    let tile_group = parse_tile_group(
        &frame_payload,
        header.uncompressed_header_bits,
        &header.tile_info,
    )
    .unwrap();
    assert_eq!(tile_group.start_tile, 0);
    assert_eq!(tile_group.end_tile, 0);
    assert_eq!(tile_group.tiles.len(), 1);
    assert_eq!(tile_group.tiles[0].tile_id, 0);
    assert_eq!(
        tile_group.tiles.iter().map(|tile| tile.len).sum::<usize>() + tile_group.data_start_offset,
        frame_payload.len()
    );
    assert!(tile_group.tiles.iter().all(|tile| tile.len > 0));
}

#[test]
fn sample_still_decode_plan_and_buffers_are_exposed_through_public_api() {
    let (_, sequence_payload, frame_payload) = sample_payloads();
    let sequence = parse_sequence_header(&sequence_payload).unwrap();
    let frame = parse_frame_header(&frame_payload, &sequence).unwrap();
    let tile_group = parse_tile_group(
        &frame_payload,
        frame.uncompressed_header_bits,
        &frame.tile_info,
    )
    .unwrap();

    let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();

    assert_eq!(plan.width, 900);
    assert_eq!(plan.height, 900);
    assert_eq!(plan.bit_depth, 8);
    assert_eq!(plan.superblock_size, 128);
    assert_eq!(plan.superblock_cols, 8);
    assert_eq!(plan.superblock_rows, 8);
    assert_eq!(plan.planes.len(), 3);
    assert!(plan.planes.iter().all(|plane| plane.width == 900));
    assert!(plan.planes.iter().all(|plane| plane.height == 900));
    assert_eq!(plan.tiles.len(), 1);
    assert_eq!(plan.tiles[0].pixel_x, 0);
    assert_eq!(plan.tiles[0].pixel_y, 0);
    assert_eq!(plan.tiles[0].pixel_width, 900);
    assert_eq!(plan.tiles[0].pixel_height, 900);
    assert_eq!(plan.tiles[0].sb_col_start, 0);
    assert_eq!(plan.tiles[0].sb_col_end, 8);
    assert_eq!(plan.tiles[0].sb_row_start, 0);
    assert_eq!(plan.tiles[0].sb_row_end, 8);
    assert!(plan.tiles[0].payload_len > 0);

    let buffers = alloc_frame_buffers(&plan).unwrap();
    assert_eq!(buffers.planes.len(), 3);
    assert_eq!(buffers.planes[0].samples.len(), 900 * 900);
    assert_eq!(buffers.planes[1].samples.len(), 900 * 900);
    assert_eq!(buffers.planes[2].samples.len(), 900 * 900);
}

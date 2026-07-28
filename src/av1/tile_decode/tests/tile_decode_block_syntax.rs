use super::*;
use crate::av1::frame::SegmentationParams;
use crate::av1::tile_decode::block_syntax::{cfl_is_allowed, cfl_signs, use_angle_delta};
use crate::av1::transform::plan_transform_blocks_with_tx_size;
use crate::av1::{
    BlockSize, Partition, PredictionMode, TxSize, UvPredictionMode, build_still_decode_plan,
    parse_frame_header, parse_sequence_header, parse_tile_group,
};
use crate::container::parse_avif;
use crate::obu::{ObuType, find_obu_payload};

#[test]
fn cfl_availability_matches_lossless_and_block_size_rules() {
    assert!(cfl_is_allowed(true, BlockSize::Block4x4, true, true));
    assert!(cfl_is_allowed(true, BlockSize::Block4x8, true, true));
    assert!(!cfl_is_allowed(true, BlockSize::Block16x4, true, true));
    assert!(!cfl_is_allowed(true, BlockSize::Block8x8, false, false));
    assert!(cfl_is_allowed(false, BlockSize::Block32x16, true, true));
    assert!(!cfl_is_allowed(false, BlockSize::Block64x32, true, true));
}

#[test]
fn cfl_joint_signs_match_av1_symbol_order() {
    let signs = (0..8)
        .map(|joint_sign| cfl_signs(joint_sign).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        signs,
        vec![
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2)
        ]
    );
    assert!(cfl_signs(8).is_err());
}

#[test]
fn angle_delta_availability_matches_av1_block_size_order() {
    assert!(!use_angle_delta(BlockSize::Block4x4));
    assert!(!use_angle_delta(BlockSize::Block4x8));
    assert!(!use_angle_delta(BlockSize::Block8x4));
    assert!(use_angle_delta(BlockSize::Block8x8));
    assert!(use_angle_delta(BlockSize::Block4x16));
    assert!(use_angle_delta(BlockSize::Block16x4));
}

#[test]
fn reads_sample_root_partition_symbol() {
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
    let tile_payload = &tile_group.tiles[0];
    let payload = &frame_payload[tile_payload.offset..tile_payload.offset + tile_payload.len];
    let mut decoder = TileDecoder::new(payload, &frame).unwrap();

    let probe = decoder
        .read_root_partition(&plan.tiles[0], &sequence)
        .unwrap();

    assert_eq!(probe.tile_id, 0);
    assert_eq!(probe.block_size, BlockSize::Block128x128);
    assert_eq!(probe.symbol, 3);
    assert_eq!(probe.partition, Partition::Split);
    assert!(probe.bit_position_after >= 15);
}

#[test]
fn segmentation_alt_q_adjusts_initial_tile_qindex() {
    let data = read_sample_avif();
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    let mut frame = parse_frame_header(frame_payload, &sequence).unwrap();
    frame.segmentation = SegmentationParams {
        enabled: true,
        update_map: true,
        temporal_update: false,
        preskip: false,
        delta_q: 5,
        segment_delta_q: [5, 0, 0, 0, 0, 0, 0, 0],
        segment_delta_lf: [[0; 4]; 8],
        segment_reference_frame: [None; 8],
        segment_global_mv: [false; 8],
        segment_skip: [false; 8],
        last_active_segment: 0,
    };
    let decoder = TileDecoder::new(&[0, 0], &frame).unwrap();
    assert_eq!(
        decoder.current_qindex,
        frame.segmentation.effective_qindex(frame.base_q_idx)
    );
}

#[test]
fn segmentation_id_prediction_uses_normative_negative_deinterleave() {
    assert_eq!(neg_deinterleave(0, 0, 3), 0);
    assert_eq!(neg_deinterleave(1, 0, 3), 1);
    assert_eq!(neg_deinterleave(0, 1, 3), 1);
    assert_eq!(neg_deinterleave(1, 1, 3), 2);
    assert_eq!(neg_deinterleave(2, 1, 3), 0);
}

#[test]
fn segmentation_map_records_every_mi_in_a_decoded_block() {
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
    let mut decoder = TileDecoder::new(&[0, 0], &frame).unwrap();
    decoder.set_segmentation_id(BlockSize::Block8x8, 0, 0, 1);
    assert_eq!(&decoder.segmentation_map[..2], &[1, 1]);
    assert_eq!(
        &decoder.segmentation_map[decoder.mi_cols..decoder.mi_cols + 2],
        &[1, 1]
    );
}

#[test]
fn segmentation_id_entropy_updates_qindex_for_selected_segment() {
    let data = read_sample_avif();
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    let mut frame = parse_frame_header(frame_payload, &sequence).unwrap();
    frame.segmentation = SegmentationParams {
        enabled: true,
        update_map: true,
        temporal_update: false,
        preskip: false,
        delta_q: 0,
        segment_delta_q: [0, 5, 0, 0, 0, 0, 0, 0],
        segment_delta_lf: [[0; 4]; 8],
        segment_reference_frame: [None; 8],
        segment_global_mv: [false; 8],
        segment_skip: [false; 8],
        last_active_segment: 1,
    };
    let mut decoder = TileDecoder::new(&[0; 128], &frame).unwrap();
    let segment_id = decoder
        .read_segmentation_id(&frame, BlockSize::Block8x8, 0, 0, false)
        .unwrap();
    assert!(segment_id <= 1);
    assert_eq!(
        decoder.current_qindex,
        frame
            .segmentation
            .effective_qindex_for_segment(frame.base_q_idx, segment_id)
    );
}

#[test]
fn segmentation_id_segment_zero_still_consumes_entropy_symbol() {
    let data = read_sample_avif();
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    let mut frame = parse_frame_header(frame_payload, &sequence).unwrap();
    frame.segmentation = SegmentationParams {
        enabled: true,
        update_map: true,
        temporal_update: false,
        preskip: false,
        delta_q: 0,
        segment_delta_q: [0; 8],
        segment_delta_lf: [[0; 4]; 8],
        segment_reference_frame: [None; 8],
        segment_global_mv: [false; 8],
        segment_skip: [false; 8],
        last_active_segment: 0,
    };
    let mut decoder = TileDecoder::new(&[0; 128], &frame).unwrap();
    let initial_position = decoder.reader.bit_position();
    let segment_id = decoder
        .read_segmentation_id(&frame, BlockSize::Block8x8, 0, 0, false)
        .unwrap();

    assert_eq!(segment_id, 0);
    assert!(decoder.reader.bit_position() > initial_position);
}

#[test]
fn segmentation_skip_forces_skip_before_the_skip_symbol() {
    let data = read_sample_avif();
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    let mut frame = parse_frame_header(frame_payload, &sequence).unwrap();
    let tile_group = parse_tile_group(
        frame_payload,
        frame.uncompressed_header_bits,
        &frame.tile_info,
    )
    .unwrap();
    let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
    frame.segmentation = SegmentationParams {
        enabled: true,
        update_map: true,
        temporal_update: false,
        preskip: true,
        delta_q: 0,
        segment_delta_q: [0; 8],
        segment_delta_lf: [[0; 4]; 8],
        segment_reference_frame: [None; 8],
        segment_global_mv: [false; 8],
        segment_skip: [true, false, false, false, false, false, false, false],
        last_active_segment: 0,
    };
    let mut decoder = TileDecoder::new(&[0; 128], &frame).unwrap();
    let probe = decoder
        .read_intra_frame_block_mode_with_chroma_reference(
            &sequence,
            &frame,
            &plan.tiles[0],
            BlockSize::Block4x4,
            0,
            0,
            true,
        )
        .unwrap();
    assert!(probe.skip);
    assert_eq!(probe.segment_id, 0);
}

#[test]
fn reads_sample_first_block_mode_symbols() {
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

    let probes =
        probe_tile_block_modes(frame_payload, &tile_group, &sequence, &frame, &plan).unwrap();

    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].tile_id, 0);
    assert_eq!(probes[0].block_size, BlockSize::Block64x64);
    assert_eq!(probes[0].skip_symbol, 0);
    assert_eq!(probes[0].cdef_idx, Some(0));
    assert_eq!(probes[0].y_mode_symbol, 0);
    assert_eq!(probes[0].y_mode, PredictionMode::Dc);
    assert_eq!(probes[0].uv_mode_symbol, Some(0));
    assert_eq!(
        probes[0].uv_mode,
        Some(UvPredictionMode::Intra(PredictionMode::Dc))
    );
    assert_eq!(probes[0].tx_size_symbol, Some(0));
    assert_eq!(probes[0].tx_size, TxSize::Tx64x64);
    assert!(probes[0].bit_position_after > 15);
}

#[test]
fn plans_sample_first_block_transforms() {
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
    let probes =
        probe_tile_block_modes(frame_payload, &tile_group, &sequence, &frame, &plan).unwrap();

    let transforms = plan_transform_blocks_with_tx_size(
        0,
        0,
        0,
        probes[0].block_size,
        probes[0].tx_size,
        plan.width,
        plan.height,
    );

    assert_eq!(transforms.len(), 1);
    assert!(transforms.iter().all(|tx| tx.plane == 0));
    assert!(transforms.iter().all(|tx| tx.tx_size == probes[0].tx_size));
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

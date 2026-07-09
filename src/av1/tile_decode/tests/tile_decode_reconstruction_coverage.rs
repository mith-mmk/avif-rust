use super::TileDecoder;
use super::reconstruction::predict_block;
use crate::av1::decode::{PlaneBuffer, PlaneLayout};
use crate::av1::frame::FrameHeader;
use crate::av1::syntax::{PredictionMode, TxSize};
use crate::av1::transform::TransformBlock;
use crate::av1::{parse_frame_header, parse_sequence_header};
use crate::container::parse_avif;
use crate::obu::{ObuType, find_obu_payload};

#[test]
fn reconstructed_coverage_tracks_top_right_bottom_left_and_frame_edges() {
    let frame = sample_frame();
    let mut decoder = TileDecoder::new(&[0, 0], &frame).unwrap();
    let plane = synthetic_plane(0);
    let current = transform(0, 4, 4);

    assert_eq!(
        decoder
            .reconstructed_extension_availability(&plane, current)
            .unwrap(),
        (false, false)
    );

    decoder
        .mark_reconstructed_transform(transform(0, 8, 0))
        .unwrap();
    assert_eq!(
        decoder
            .reconstructed_extension_availability(&plane, current)
            .unwrap(),
        (true, false)
    );

    decoder
        .mark_reconstructed_transform(transform(0, 0, 8))
        .unwrap();
    assert_eq!(
        decoder
            .reconstructed_extension_availability(&plane, current)
            .unwrap(),
        (true, true)
    );

    let right_edge = transform(0, 12, 4);
    assert_eq!(
        decoder
            .reconstructed_extension_availability(&plane, right_edge)
            .unwrap(),
        (true, false)
    );
    decoder
        .mark_reconstructed_transform(transform(0, 8, 8))
        .unwrap();
    assert_eq!(
        decoder
            .reconstructed_extension_availability(&plane, right_edge)
            .unwrap(),
        (true, true)
    );
}

#[test]
fn reconstruction_coverage_is_plane_and_tile_local() {
    let frame = sample_frame();
    let mut first_tile = TileDecoder::new(&[0, 0], &frame).unwrap();
    let luma = synthetic_plane(0);
    let chroma = synthetic_plane(1);
    let luma_current = transform(0, 4, 4);
    let chroma_current = transform(1, 4, 4);

    first_tile
        .mark_reconstructed_transform(transform(0, 8, 0))
        .unwrap();
    first_tile
        .mark_reconstructed_transform(transform(0, 0, 8))
        .unwrap();
    assert_eq!(
        first_tile
            .reconstructed_extension_availability(&luma, luma_current)
            .unwrap(),
        (true, true)
    );
    assert_eq!(
        first_tile
            .reconstructed_extension_availability(&chroma, chroma_current)
            .unwrap(),
        (false, false)
    );

    let second_tile = TileDecoder::new(&[0, 0], &frame).unwrap();
    assert_eq!(
        second_tile
            .reconstructed_extension_availability(&luma, luma_current)
            .unwrap(),
        (false, false)
    );
}

#[test]
fn reconstructed_coverage_changes_d45_extension_prediction() {
    let frame = sample_frame();
    let mut decoder = TileDecoder::new(&[0, 0], &frame).unwrap();
    let mut plane = synthetic_plane(0);
    for x in 0..plane.layout.width {
        plane.samples[3 * plane.layout.width + x] = (x as u16 + 1) * 10;
    }
    let current = transform(0, 4, 4);

    let (masked_top_right, masked_bottom_left) = decoder
        .reconstructed_extension_availability(&plane, current)
        .unwrap();
    let masked = predict_block(
        &plane,
        PredictionMode::D45,
        current.x,
        current.y,
        current.tx_size.width(),
        current.tx_size.height(),
        Some(0),
        None,
        8,
        false,
        false,
        masked_top_right,
        masked_bottom_left,
    )
    .unwrap();

    decoder
        .mark_reconstructed_transform(transform(0, 8, 0))
        .unwrap();
    let (top_right_available, bottom_left_available) = decoder
        .reconstructed_extension_availability(&plane, current)
        .unwrap();
    let unmasked = predict_block(
        &plane,
        PredictionMode::D45,
        current.x,
        current.y,
        current.tx_size.width(),
        current.tx_size.height(),
        Some(0),
        None,
        8,
        false,
        false,
        top_right_available,
        bottom_left_available,
    )
    .unwrap();

    assert!(top_right_available);
    assert!(!bottom_left_available);
    assert_ne!(masked, unmasked);
}

fn transform(plane: usize, x: usize, y: usize) -> TransformBlock {
    TransformBlock {
        plane,
        x,
        y,
        tx_size: TxSize::Tx4x4,
    }
}

fn synthetic_plane(plane: u8) -> PlaneBuffer {
    let width = 16;
    let height = 16;
    PlaneBuffer {
        layout: PlaneLayout {
            plane,
            width,
            height,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: width * height,
        },
        samples: vec![0; width * height],
    }
}

fn sample_frame() -> FrameHeader {
    let data = std::fs::read(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("samples")
            .join("WML2Viewer.avif"),
    )
    .expect("sample AVIF should exist");
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU should exist");
    parse_frame_header(frame_payload, &sequence).unwrap()
}

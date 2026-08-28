mod support;

use avif_rust::av1::{parse_frame_header, parse_sequence_header};
use avif_rust::container::parse_avif;
use avif_rust::obu::{ObuType, find_obu_payload};
use avif_rust::{NativeDecodeLimits, decode_frame_bytes, decode_frame_bytes_strict_with_limits};
use support::read_sample;

const REDUCED_SEQUENCE: &[u8] = &[0x38, 0x0c, 0xff, 0xd8, 0x40, 0x43, 0x40, 0x08];
const REDUCED_FRAME: &[u8] = &[0xc4, 0x00, 0x00, 0xc2];

#[test]
fn reduced_prefix_keeps_the_legacy_fixed_header_result() {
    let sequence = parse_sequence_header(REDUCED_SEQUENCE).unwrap();
    let frame = parse_frame_header(REDUCED_FRAME, &sequence).unwrap();
    assert!(sequence.reduced_still_picture_header);
    assert!(frame.show_frame && !frame.show_existing_frame);
    assert_eq!(frame.uncompressed_header_bits, 20);
    assert_eq!(frame.payload_after_header_offset, 3);
    assert_eq!(frame.frame_width, sequence.max_frame_width);
    assert_eq!(frame.frame_height, sequence.max_frame_height);
}

#[test]
fn normal_prefix_resumes_with_the_same_public_frame_geometry() {
    let Some(data) = read_sample("WML2Viewer.avif") else {
        return;
    };
    let info = parse_avif(&data).unwrap();
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .unwrap();
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .or_else(|| find_obu_payload(&info.primary_item_payload, ObuType::FrameHeader).unwrap())
        .unwrap();
    let sequence = parse_sequence_header(sequence_payload).unwrap();
    let header = parse_frame_header(frame_payload, &sequence).unwrap();
    let old = decode_frame_bytes(&data).unwrap();
    let native = decode_frame_bytes_strict_with_limits(
        &data,
        &NativeDecodeLimits::new(
            data.len() * 2,
            4096,
            4096,
            1 << 24,
            1 << 28,
            1 << 24,
            1 << 24,
            64,
            64,
            64,
            8,
            1,
        ),
    )
    .unwrap();
    assert_eq!(old.width, native.frame().width);
    assert_eq!(old.height, native.frame().height);
    assert_eq!(native.frame().width, header.upscaled_width as usize);
    assert_eq!(native.frame().height, header.frame_height as usize);
}

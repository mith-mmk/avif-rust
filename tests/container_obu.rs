mod support;

use avif_rust::DecoderError;
use avif_rust::container::{is_avif_file, parse_avif};
use avif_rust::obu::{ObuType, count_obus, find_obu_payload, find_obu_payloads, parse_obu_stream};
use support::sample_path;

fn sample_avif() -> Vec<u8> {
    std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist")
}

#[test]
fn sample_container_metadata_is_exposed_through_public_api() {
    let data = sample_avif();
    let info = parse_avif(&data).unwrap();

    assert!(is_avif_file(&data));
    assert!(info.is_avif_brand());
    assert_eq!(info.width, Some(900));
    assert_eq!(info.height, Some(900));
    assert_eq!(
        info.pixel_information
            .as_ref()
            .map(|pixi| pixi.bits_per_channel.as_slice()),
        Some(&[8, 8, 8][..])
    );
    assert!(!info.primary_item_payload.is_empty());
}

#[test]
fn public_obu_helpers_find_sample_frame_payloads() {
    let data = sample_avif();
    let info = parse_avif(&data).unwrap();
    let obus = parse_obu_stream(&info.primary_item_payload).unwrap();

    assert!(
        obus.iter()
            .any(|obu| matches!(obu.obu_type, ObuType::SequenceHeader))
    );
    assert!(
        obus.iter()
            .any(|obu| matches!(obu.obu_type, ObuType::Frame | ObuType::FrameHeader))
    );
    assert!(
        obus.iter().all(
            |obu| !obu.payload.is_empty() || matches!(obu.obu_type, ObuType::TemporalDelimiter)
        )
    );

    let payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence header OBU should exist");
    assert!(!payload.is_empty());

    let [sequence, frame] = find_obu_payloads(
        &info.primary_item_payload,
        [ObuType::SequenceHeader, ObuType::Frame],
    )
    .unwrap();
    assert!(sequence.is_some());
    assert!(frame.is_some());
}

#[test]
fn public_obu_helpers_count_repeated_types() {
    let data = [
        0x22, 0x01, 0xaa, // tile group, one-byte payload
        0x0a, 0x01, 0xbb, // sequence header, one-byte payload
        0x22, 0x02, 0xcc, 0xdd, // tile group, two-byte payload
    ];

    assert_eq!(count_obus(&data, ObuType::TileGroup).unwrap(), 2);
    assert_eq!(count_obus(&data, ObuType::SequenceHeader).unwrap(), 1);
    assert_eq!(count_obus(&data, ObuType::Frame).unwrap(), 0);
}

#[test]
fn public_parsers_reject_truncated_and_malformed_headers() {
    let err = parse_avif(&[0, 0, 0]).unwrap_err();
    assert!(matches!(err, DecoderError::NotEnoughData(message) if message.contains("box header")));

    let err = parse_obu_stream(&[0x0e]).unwrap_err();
    assert!(
        matches!(err, DecoderError::NotEnoughData(message) if message.contains("extension header"))
    );

    let err = parse_obu_stream(&[0x0a, 0x80]).unwrap_err();
    assert!(matches!(err, DecoderError::NotEnoughData(message) if message.contains("leb128 size")));

    let err = parse_obu_stream(&[0x0a, 0x02, 0xff]).unwrap_err();
    assert!(
        matches!(err, DecoderError::NotEnoughData(message) if message.contains("payload extends"))
    );

    let err = parse_obu_stream(&[0x80]).unwrap_err();
    assert!(err.to_string().contains("forbidden bit"));
}

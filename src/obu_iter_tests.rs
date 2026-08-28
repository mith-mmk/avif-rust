use super::{
    DecoderError, ObuIter, ObuType, find_obu_payload, find_obu_payloads_in_parts, parse_obu_stream,
};

#[test]
fn iterator_preserves_borrowed_payload_and_extension() {
    let data = [0x1e, 0x07, 0x01, 0xbb];
    let mut iter = ObuIter::new(&data);

    let obu = iter.next().expect("one OBU").expect("valid OBU");
    assert_eq!(obu.obu_type, ObuType::FrameHeader);
    assert_eq!(obu.extension_header, Some(0x07));
    assert_eq!(obu.payload, &[0xbb]);
    assert_eq!(obu.payload.as_ptr(), data.as_ptr().wrapping_add(3));
    assert!(iter.next().is_none());
    assert!(iter.next().is_none());
}

#[test]
fn iterator_yields_one_error_then_is_fused() {
    let data = [0x0a, 0x80];
    let mut iter = ObuIter::new(&data);

    let first = iter.next().expect("malformed OBU error");
    assert!(matches!(
        first,
        Err(DecoderError::NotEnoughData(message)) if message.contains("leb128 size")
    ));
    assert!(iter.next().is_none());
    assert!(iter.next().is_none());
}

#[test]
fn legacy_collect_and_search_helpers_keep_framing_behavior() {
    let sequence = [0x0a, 0x01, 0xaa];
    let frame = [0x1a, 0x01, 0xbb];
    let joined = [0x0a, 0x01, 0xaa, 0x1a, 0x01, 0xbb];

    let collected = parse_obu_stream(&joined).expect("collecting parser");
    assert_eq!(collected.len(), 2);
    assert_eq!(
        find_obu_payload(&joined, ObuType::FrameHeader),
        Ok(Some(&[0xbb][..]))
    );
    assert_eq!(
        find_obu_payloads_in_parts(
            &[&sequence, &frame],
            [ObuType::SequenceHeader, ObuType::FrameHeader],
        ),
        Ok([Some(&[0xaa][..]), Some(&[0xbb][..])])
    );
}

#[test]
fn first_match_search_stops_before_a_malformed_suffix() {
    let data = [0x1a, 0x01, 0xbb, 0x0a, 0x80];

    assert_eq!(
        find_obu_payload(&data, ObuType::FrameHeader),
        Ok(Some(&[0xbb][..]))
    );
}

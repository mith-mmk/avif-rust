use super::*;

#[test]
fn child_box_iter_walks_without_materializing_sibling_headers() {
    let payload = [
        0, 0, 0, 12, b'f', b't', b'y', b'p', 1, 2, 3, 4, 0, 0, 0, 8, b'm', b'd', b'a', b't',
    ];
    let headers = child_box_iter(&payload)
        .map(|header| header.expect("valid child box"))
        .collect::<Vec<_>>();
    assert_eq!(headers.len(), 2);
    assert_eq!(headers[0].box_type, *b"ftyp");
    assert_eq!(headers[1].box_type, *b"mdat");
    assert_eq!(headers[0].offset, 0);
    assert_eq!(headers[1].offset, 12);
}

#[test]
fn child_box_and_iterator_keep_malformed_child_diagnostics() {
    let payload = [
        0, 0, 0, 8, b'f', b't', b'y', b'p', 0, 0, 0, 4, b'b', b'a', b'd', b'!',
    ];
    let error = child_box(&payload, b"ftyp").expect_err("malformed trailing child must be checked");
    assert!(
        matches!(error, DecoderError::Bitstream(message) if message.contains("smaller than its header"))
    );
}

fn boxed(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(payload.len() + 8).expect("test box fits");
    let mut output = Vec::with_capacity(payload.len() + 8);
    output.extend_from_slice(&size.to_be_bytes());
    output.extend_from_slice(box_type);
    output.extend_from_slice(payload);
    output
}

fn avis_fixture_with_invalid_and_empty_tracks() -> (Vec<u8>, Vec<u8>) {
    // The first track reaches an stsd payload-too-short error during the
    // second pass.  The second is a valid Box with no mdia and is ignored by
    // the semantic sequence parser.  The final sibling is malformed; it must
    // be reported before the first track's more specific second-pass error.
    let mut hdlr_payload = vec![0; 8];
    hdlr_payload.extend_from_slice(b"vide");
    let stsd = boxed(b"stsd", &[0; 4]);
    let stbl = boxed(b"stbl", &stsd);
    let minf = boxed(b"minf", &stbl);
    let mdia_payload = [boxed(b"hdlr", &hdlr_payload), minf].concat();
    let invalid_trak = boxed(b"trak", &boxed(b"mdia", &mdia_payload));
    let empty_trak = boxed(b"trak", &[]);
    let mut moov_payload = invalid_trak;
    moov_payload.extend_from_slice(&empty_trak);
    moov_payload.extend_from_slice(&[0, 0, 0, 4, b'b', b'a', b'd', b'!']);

    let mut ftyp_payload = Vec::from(*b"avis");
    ftyp_payload.extend_from_slice(&[0, 0, 0, 0]);
    let mut data = boxed(b"ftyp", &ftyp_payload);
    data.extend_from_slice(&boxed(b"moov", &moov_payload));
    (data, moov_payload)
}

#[test]
fn native_sequence_preflight_reports_trailing_malformed_sibling_before_track_processing() {
    let (data, moov_payload) = avis_fixture_with_invalid_and_empty_tracks();
    let limits = NativeDecodeLimits::new(
        data.len(),
        4096,
        4096,
        4096 * 4096,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        64,
        64,
        64,
        8,
        64,
    );
    let mut context = ParseContext::native_sequence(&limits);
    let error = parse_sequence_tracks(&data, &moov_payload, &mut context)
        .expect_err("trailing malformed sibling must be rejected");
    assert!(
        matches!(error, DecoderError::Bitstream(message) if message.contains("smaller than its header"))
    );
}

#[test]
fn public_legacy_animation_keeps_trailing_malformed_sibling_diagnostic() {
    let (data, _) = avis_fixture_with_invalid_and_empty_tracks();
    let error = parse_avif_animation(&data).expect_err("malformed sibling must be rejected");
    assert!(
        matches!(error, DecoderError::Bitstream(message) if message.contains("smaller than its header"))
    );
}

use avif_rust::AvifSequenceDecoder;
use avif_rust::container::{AvifSequenceSampleKind, classify_av1_sequence_sample, parse_avif};

// Required, generated fixtures: these tests cannot skip for missing FFmpeg or
// a sibling checkout. See FIXTURES.md for provenance and regeneration commands.
const AVIS: &[u8] = include_bytes!("avis-subsampled-palette.avif");

#[test]
fn reference_avis_decodes_key_inter_hidden_and_show_existing_frames() {
    let info = parse_avif(AVIS).unwrap();
    assert_eq!(&info.major_brand, b"avis");
    assert_eq!(info.sequence_sample_payloads.len(), 60);
    let kinds: Vec<_> = info
        .sequence_sample_payloads
        .iter()
        .map(|sample| classify_av1_sequence_sample(sample).unwrap().unwrap())
        .collect();
    assert_eq!(kinds[0], AvifSequenceSampleKind::Key);
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, AvifSequenceSampleKind::Inter))
    );
    assert!(info.sequence_sample_payloads.iter().any(|sample| {
        avif_rust::obu::parse_obu_stream(sample)
            .unwrap()
            .iter()
            .filter(|obu| obu.obu_type == avif_rust::obu::ObuType::Frame)
            .count()
            > 1
    }));
    let show_existing = kinds
        .iter()
        .position(|kind| matches!(kind, AvifSequenceSampleKind::ShowExisting { .. }))
        .unwrap();
    let frames = avif_rust::decode_sequence_frames_bytes(AVIS).unwrap();
    assert_eq!(frames.len(), 60);
    assert!(
        frames
            .iter()
            .all(|frame| (frame.width, frame.height, frame.bit_depth) == (64, 64, 8))
    );
    assert!(
        frames
            .windows(2)
            .any(|pair| pair[0].buffers != pair[1].buffers)
    );
    assert_eq!(
        avif_rust::decode_sequence_frame_bytes(AVIS, show_existing).unwrap(),
        frames[show_existing]
    );
    assert_eq!(avif_rust::decode_frame_bytes(AVIS).unwrap(), frames[0]);
    let mut decoder = AvifSequenceDecoder::new(AVIS).unwrap();
    for expected in &frames {
        assert_eq!(&decoder.next_frame().unwrap().unwrap().frame, expected);
    }
    assert!(decoder.next_frame().unwrap().is_none());
}

#[test]
fn sub8x8_chroma_palette_key_matches_independent_yuv_exactly() {
    let data = include_bytes!("lossless-sub8x8-palette.avif");
    let expected = include_bytes!("lossless-sub8x8-palette.yuv");
    let frame = avif_rust::decode_frame_bytes(data).unwrap();
    assert_yuv420_reference(&frame, expected);
}

#[test]
fn subsampled_avis_key_cdef_matches_independent_yuv_exactly() {
    let expected = include_bytes!("avis-subsampled-key-reference.yuv");
    let frame = avif_rust::decode_frame_bytes(AVIS).unwrap();
    assert_yuv420_reference(&frame, expected);
}

fn assert_yuv420_reference(frame: &avif_rust::DecodedFrame, expected: &[u8]) {
    assert_eq!((frame.width, frame.height, frame.bit_depth), (64, 64, 8));
    assert_eq!(frame.buffers.planes.len(), 3);
    let mut offset = 0;
    for (plane_index, plane) in frame.buffers.planes.iter().enumerate() {
        let dimension = if plane_index == 0 { 64 } else { 32 };
        assert_eq!(
            (
                plane.layout.width,
                plane.layout.height,
                plane.layout.stride()
            ),
            (dimension, dimension, dimension)
        );
        let count = dimension * dimension;
        assert_eq!(plane.samples.len(), count);
        for (sample_index, (&actual, &expected)) in plane
            .samples
            .iter()
            .zip(&expected[offset..offset + count])
            .enumerate()
        {
            assert_eq!(
                actual,
                u16::from(expected),
                "plane {plane_index}, sample {sample_index}"
            );
        }
        offset += count;
    }
    assert_eq!(offset, expected.len());
}

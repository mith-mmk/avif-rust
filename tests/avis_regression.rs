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

#[test]
fn first_inter_deblock_uses_logical_last_reference_strength() {
    // Same-bitstream dav1d/libaom/scalar AOM all produce 40 at this point.
    // The pre-deblock input is also 40. Treating physical slot 0 as INTRA
    // instead of logical LAST raises the edge level from 2 to 3 and gives 41.
    let frame = avif_rust::decode_sequence_frame_bytes(AVIS, 1).unwrap();
    let y = &frame.buffers.planes[0];
    assert_eq!(y.samples[5 * y.layout.stride() + 44], 40);
}

#[test]
fn interintra_wedge_prediction_matches_independent_sequence_pixels() {
    // Mandatory same-bitstream dav1d/libaom oracle points. The first hidden
    // inter frame (order16) predicts Y(8,16) as31 from inter58/intra31,
    // adds residual+2, then CDEF produces32. Reversing the inter-intra mask
    // gave60 before filtering and propagated large errors to later frames.
    // No external executable, optional fixture, or sibling checkout is used.
    let frames = avif_rust::decode_sequence_frames_bytes(AVIS).unwrap();
    for (frame_index, x, y, expected) in [(16, 8, 16, 32), (29, 6, 39, 203), (31, 6, 41, 203)] {
        let plane = &frames[frame_index].buffers.planes[0];
        assert_eq!(
            plane.samples[y * plane.layout.stride() + x],
            expected,
            "inter-intra oracle display frame {frame_index} Y({x},{y})"
        );
    }
}

#[test]
fn obmc_neighbor_uses_its_own_primary_reference() {
    // Same-bitstream scalar AOM/dav1d: order8 block(24,40) combines current
    // LAST(order0)=210 with left ALTREF(order16,MV32,-12)=126 using39:25.
    // Prediction177 plus residual-51 gives126, preserved by all loop filters.
    // Rejecting the other-reference neighbor left prediction210 and gave159.
    let frame = avif_rust::decode_sequence_frame_bytes(AVIS, 8).unwrap();
    let y = &frame.buffers.planes[0];
    assert_eq!(y.samples[40 * y.layout.stride() + 24], 126);
}

#[test]
fn subsampled_avis_all_native_frames_match_independent_reference_exactly() {
    let reference = include_bytes!("avis-subsampled-reference.yuv");
    let frames = avif_rust::decode_sequence_frames_bytes(AVIS).unwrap();
    const SAMPLES_PER_FRAME: usize = 64 * 64 + 2 * 32 * 32;
    assert_eq!(frames.len(), 60);
    assert_eq!(reference.len(), frames.len() * SAMPLES_PER_FRAME);
    for (frame_index, (frame, expected)) in frames
        .iter()
        .zip(reference.as_chunks::<SAMPLES_PER_FRAME>().0)
        .enumerate()
    {
        let sample_count: usize = frame.buffers.planes.iter().map(|p| p.samples.len()).sum();
        assert_eq!(sample_count, SAMPLES_PER_FRAME);
        for (sample_index, (&actual, &expected)) in frame
            .buffers
            .planes
            .iter()
            .flat_map(|plane| plane.samples.iter())
            .zip(expected)
            .enumerate()
        {
            assert_eq!(
                actual,
                u16::from(expected),
                "displayed frame {frame_index}, native sample {sample_index}"
            );
        }
    }
}

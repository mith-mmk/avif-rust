use super::*;

fn sequence_track(is_color: bool, is_alpha: bool, sample_len: usize) -> SequenceTrack {
    SequenceTrack {
        samples: vec![vec![0; sample_len]],
        timescale: 1,
        pts_in_timescales: vec![0],
        durations_in_timescales: vec![1],
        durations_ms: vec![1],
        timing_owners: SequenceTimingOwners::default(),
        repetition_count: AvifRepetitionCount::Unknown,
        is_alpha,
        is_color,
    }
}

#[test]
fn native_sequence_selection_keeps_first_color_and_alpha_only() {
    let mut color = None;
    let mut alpha = None;
    assert!(select_native_sequence_track(
        &mut color,
        &mut alpha,
        sequence_track(true, false, 3)
    ));
    assert!(select_native_sequence_track(
        &mut color,
        &mut alpha,
        sequence_track(false, true, 5)
    ));
    assert!(!select_native_sequence_track(
        &mut color,
        &mut alpha,
        sequence_track(true, false, 7)
    ));
    assert!(!select_native_sequence_track(
        &mut color,
        &mut alpha,
        sequence_track(false, true, 11)
    ));
    assert_eq!(color.as_ref().unwrap().samples[0].len(), 3);
    assert_eq!(alpha.as_ref().unwrap().samples[0].len(), 5);
}

#[test]
fn native_sequence_selection_rejects_unclassified_track_without_owner() {
    let mut color = None;
    let mut alpha = None;
    assert!(!select_native_sequence_track(
        &mut color,
        &mut alpha,
        sequence_track(false, false, 13)
    ));
    assert!(color.is_none());
    assert!(alpha.is_none());
}

#[derive(Clone, Copy)]
enum SyntheticSequenceCorruption {
    None,
    DuplicateStts,
    DuplicateDescription,
    DuplicatePayloadRange,
}

fn synthetic_box(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(payload.len() + 8).expect("synthetic box fits");
    let mut output = Vec::with_capacity(payload.len() + 8);
    output.extend_from_slice(&size.to_be_bytes());
    output.extend_from_slice(box_type);
    output.extend_from_slice(payload);
    output
}

fn synthetic_av01_description(alpha: bool, changed: bool) -> Vec<u8> {
    let mut payload = Vec::new();
    if alpha {
        payload.extend_from_slice(b"auxi");
        payload.extend_from_slice(&[0; 4]);
        payload.extend_from_slice(ALPHA_AUX_TYPE.as_bytes());
        payload.push(0);
    }
    if changed {
        payload.push(0x7f);
    }
    synthetic_box(b"av01", &payload)
}

fn synthetic_full_box_payload(version_flags: [u8; 4], body: &[u8]) -> Vec<u8> {
    let mut payload = version_flags.to_vec();
    payload.extend_from_slice(body);
    payload
}

fn synthetic_sequence_track(
    alpha: bool,
    duplicate: bool,
    corruption: SyntheticSequenceCorruption,
) -> Vec<u8> {
    let changed_description = duplicate
        && matches!(
            corruption,
            SyntheticSequenceCorruption::DuplicateDescription
        );
    let first_description = synthetic_av01_description(alpha, false);
    let second_description = synthetic_av01_description(alpha, changed_description);
    let mut stsd_body = 1_u32.to_be_bytes().to_vec();
    if changed_description {
        stsd_body = 2_u32.to_be_bytes().to_vec();
        stsd_body.extend_from_slice(&first_description);
        stsd_body.extend_from_slice(&second_description);
    } else {
        stsd_body.extend_from_slice(&first_description);
    }
    let stsd = synthetic_box(b"stsd", &synthetic_full_box_payload([0; 4], &stsd_body));

    let description_index: u32 = if changed_description { 2 } else { 1 };
    let mut stsc_body = Vec::new();
    stsc_body.extend_from_slice(&1_u32.to_be_bytes());
    stsc_body.extend_from_slice(&1_u32.to_be_bytes());
    stsc_body.extend_from_slice(&2_u32.to_be_bytes());
    stsc_body.extend_from_slice(&description_index.to_be_bytes());
    let stsc = synthetic_box(b"stsc", &synthetic_full_box_payload([0; 4], &stsc_body));

    let mut stsz_body = Vec::new();
    stsz_body.extend_from_slice(&0_u32.to_be_bytes());
    stsz_body.extend_from_slice(&2_u32.to_be_bytes());
    stsz_body.extend_from_slice(&1_u32.to_be_bytes());
    stsz_body.extend_from_slice(&1_u32.to_be_bytes());
    let stsz = synthetic_box(b"stsz", &synthetic_full_box_payload([0; 4], &stsz_body));

    let sample_offset = if duplicate
        && matches!(
            corruption,
            SyntheticSequenceCorruption::DuplicatePayloadRange
        ) {
        u32::MAX
    } else {
        0
    };
    let mut stco_body = 1_u32.to_be_bytes().to_vec();
    stco_body.extend_from_slice(&sample_offset.to_be_bytes());
    let stco = synthetic_box(b"stco", &synthetic_full_box_payload([0; 4], &stco_body));

    let timing_count: u32 =
        if duplicate && matches!(corruption, SyntheticSequenceCorruption::DuplicateStts) {
            1
        } else {
            2
        };
    let mut stts_body = 1_u32.to_be_bytes().to_vec();
    stts_body.extend_from_slice(&timing_count.to_be_bytes());
    stts_body.extend_from_slice(&1_u32.to_be_bytes());
    let stts = synthetic_box(b"stts", &synthetic_full_box_payload([0; 4], &stts_body));

    let mut stbl_payload = Vec::new();
    stbl_payload.extend_from_slice(&stsd);
    stbl_payload.extend_from_slice(&stsc);
    stbl_payload.extend_from_slice(&stsz);
    stbl_payload.extend_from_slice(&stco);
    stbl_payload.extend_from_slice(&stts);
    let stbl = synthetic_box(b"stbl", &stbl_payload);
    let minf = synthetic_box(b"minf", &stbl);

    let mut hdlr_payload = vec![0; 8];
    hdlr_payload.extend_from_slice(b"vide");
    let hdlr = synthetic_box(b"hdlr", &hdlr_payload);
    let mdia = synthetic_box(b"mdia", &[hdlr, minf].concat());
    if alpha {
        let auxl = synthetic_box(b"auxl", &1_u32.to_be_bytes());
        let tref = synthetic_box(b"tref", &auxl);
        synthetic_box(b"trak", &[tref, mdia].concat())
    } else {
        synthetic_box(b"trak", &mdia)
    }
}

fn synthetic_avis(corruption: SyntheticSequenceCorruption) -> Vec<u8> {
    let mut moov_payload = Vec::new();
    for (alpha, duplicate) in [(false, false), (false, true), (true, false), (true, true)] {
        moov_payload.extend_from_slice(&synthetic_sequence_track(alpha, duplicate, corruption));
    }
    let moov = synthetic_box(b"moov", &moov_payload);
    // The sample tables point at the beginning of the file. The movie box is
    // already larger than the selected tracks' two one-byte samples, so no
    // trailing mdat is needed for this container-only parser test.
    moov
}

fn generous_sequence_limits() -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        1 << 20,
        64,
        64,
        4096,
        1 << 20,
        1 << 20,
        1 << 20,
        32,
        32,
        32,
        8,
        8,
    )
}

#[test]
fn native_sequence_synthetic_duplicates_select_first_tracks_and_handoff_timing() {
    let data = synthetic_avis(SyntheticSequenceCorruption::None);
    let limits = generous_sequence_limits();
    let mut context = container_budget::ParseContext::native_sequence(&limits);
    let parsed = parse_avif_animation_with_context(&data, BRAND_AVIS, &[], &mut context)
        .expect("selected and duplicate color/alpha tracks are valid");

    assert_eq!(parsed.sequence.color_samples.len(), 2);
    assert_eq!(parsed.sequence.alpha_samples.len(), 2);
    assert_eq!(parsed.color_timing.len(), 2);
    assert_eq!(parsed.alpha_timing.len(), 2);
    assert_eq!(parsed.sequence.color_durations_ms, vec![1000, 1000]);
    assert_eq!(parsed.sequence.alpha_durations_ms, vec![1000, 1000]);

    let handoff = context
        .take_native_owners()
        .expect("selected payload and final timing owners are handed off");
    let timing_bytes = parsed_animation_timing_metadata_bytes(&ParsedAnimationTiming {
        color_timing: parsed.color_timing.clone(),
        alpha_timing: parsed.alpha_timing.clone(),
        color_durations_ms: parsed.sequence.color_durations_ms.clone(),
        alpha_durations_ms: parsed.sequence.alpha_durations_ms.clone(),
        color_timescale: parsed.color_timescale,
        duration_in_timescales: parsed.duration_in_timescales,
        repetition_count: parsed.repetition_count,
        alpha_samples: parsed.sequence.alpha_samples.clone(),
    })
    .unwrap();
    let sequence_payload_bytes = handoff.ticket_bytes()[4];
    let accounting = context.accounting();
    assert!(accounting.metadata_live >= timing_bytes);
    assert_eq!(
        sequence_payload_bytes,
        2 * std::mem::size_of::<Vec<u8>>() + 4,
        "duplicate tracks must not retain sample payload or outer storage"
    );
    assert_eq!(accounting.payload_live, sequence_payload_bytes);
    let metadata = accounting.metadata_live;
    let payload = accounting.payload_live;
    let icc = accounting.icc_live;
    let budget = context
        .into_budget_with_handoff(metadata, payload, icc, handoff)
        .expect("handoff totals match parser accounting");
    assert_eq!(budget.accounting().aggregate_live, metadata + payload);
}

#[test]
fn native_sequence_synthetic_malformed_duplicate_rolls_back_and_retries() {
    for corruption in [
        SyntheticSequenceCorruption::DuplicateStts,
        SyntheticSequenceCorruption::DuplicateDescription,
        SyntheticSequenceCorruption::DuplicatePayloadRange,
    ] {
        let data = synthetic_avis(corruption);
        let limits = generous_sequence_limits();
        let mut context = container_budget::ParseContext::native_sequence(&limits);
        let checkpoint = context.checkpoint();
        assert!(parse_avif_animation_with_context(&data, BRAND_AVIS, &[], &mut context).is_err());
        context.rollback(checkpoint);
        assert_eq!(context.accounting().aggregate_live, 0);

        let retry_data = synthetic_avis(SyntheticSequenceCorruption::None);
        let parsed = parse_avif_animation_with_context(&retry_data, BRAND_AVIS, &[], &mut context)
            .expect("rollback must leave a retryable context");
        assert_eq!(parsed.sequence.color_samples.len(), 2);
        assert_eq!(parsed.sequence.alpha_samples.len(), 2);
    }
}

#[test]
fn native_sequence_timing_exact_and_one_under_are_transactional_before_reserve() {
    let mut timing_payload = vec![0; 16];
    timing_payload[4..8].copy_from_slice(&1_u32.to_be_bytes());
    timing_payload[8..12].copy_from_slice(&2_u32.to_be_bytes());
    timing_payload[12..16].copy_from_slice(&1_u32.to_be_bytes());
    let requested_bytes = 2 * 2 * std::mem::size_of::<u64>();

    let under_limits = generous_sequence_limits()
        .with_max_live_allocation_bytes(requested_bytes - 1)
        .unwrap();
    let mut under = container_budget::ParseContext::native_sequence(&under_limits);
    let checkpoint = under.checkpoint();
    let observation =
        crate::test_allocation_observer::Observation::begin(2 * std::mem::size_of::<u64>(), false);
    let result = parse_sample_timing_native(&timing_payload, 2, &mut under);
    assert!(result.is_err(), "one-under timing must be rejected");
    assert_eq!(
        observation.requests(),
        0,
        "grouped timing preflight must precede reserve"
    );
    assert_eq!(under.accounting().aggregate_live, 0);
    drop(observation);
    under.rollback(checkpoint);

    let exact_limits = generous_sequence_limits()
        .with_max_live_allocation_bytes(requested_bytes)
        .unwrap();
    let mut exact = container_budget::ParseContext::native_sequence(&exact_limits);
    let (pts, durations, mut owners) =
        parse_sample_timing_native(&timing_payload, 2, &mut exact).expect("exact timing limit");
    assert_eq!(pts, vec![0, 1]);
    assert_eq!(durations, vec![1, 1]);
    exact.release_token(&mut owners.pts).unwrap();
    exact.release_token(&mut owners.durations).unwrap();
    drop(pts);
    drop(durations);
    assert_eq!(exact.accounting().aggregate_live, 0);
}

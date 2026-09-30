use super::{
    SequenceTracksDecoder, SequenceTracksStorage, avif_info_storage_bytes, checked_capacity_bytes,
    sequence_tracks_clone_peak_bytes, validate_exact_timing_sync,
};
use crate::DecoderError;
use crate::container::{
    AvifAnimation, AvifFrameTiming, AvifInfo, AvifSequence, ParsedAnimationTiming,
    parsed_animation_timing_metadata_bytes, validate_sample_timing,
};

const COLOR_16X16_AV1: &[u8] = &[
    0x0a, 0x06, 0x18, 0x0c, 0xff, 0xdb, 0x00, 0x80, 0x32, 0x13, 0x18, 0x00, 0x00, 0x00, 0x50, 0x00,
    0x00, 0x00, 0x00, 0xa9, 0x8e, 0x6c, 0xb6, 0xaa, 0xb1, 0xc6, 0xb6, 0x85, 0x40,
];
const ALPHA_16X16_AV1: &[u8] = &[
    0x0a, 0x05, 0x18, 0x0c, 0xff, 0xdb, 0x54, 0x32, 0x0b, 0x18, 0x00, 0x00, 0x01, 0x40, 0x00, 0x01,
    0xfd, 0x65, 0xe3, 0xd0,
];

fn synthetic_info(primary: &[u8]) -> AvifInfo {
    AvifInfo {
        major_brand: *b"avis",
        // Non-empty metadata makes the strict track clone a real allocation
        // positive control instead of allowing an empty Vec::clone to pass.
        compatible_brands: vec![*b"avis", *b"mif1"],
        primary_item_id: Some(1),
        width: Some(16),
        height: Some(16),
        pixel_information: Some(crate::container::PixelInformation {
            bits_per_channel: vec![8, 8, 8],
            extended_channels: None,
        }),
        color_information: None,
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: None,
        primary_item_payload: primary.to_vec(),
        sequence_sample_payloads: Vec::new(),
    }
}

fn synthetic_static_alpha_info(primary: &[u8]) -> AvifInfo {
    let mut info = synthetic_info(primary);
    info.alpha_auxiliary_items
        .push(crate::container::AuxiliaryImage {
            item_id: 2,
            aux_type: "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha".to_string(),
            payload: ALPHA_16X16_AV1.to_vec(),
        });
    info
}

fn synthetic_animation(sample: &[u8], alpha: Option<&[u8]>) -> AvifAnimation {
    let timing = timing(1, 0, 1);
    AvifAnimation {
        sequence: AvifSequence {
            color_samples: vec![sample.to_vec()],
            color_durations_ms: vec![1000],
            alpha_samples: alpha.into_iter().map(|value| value.to_vec()).collect(),
            alpha_durations_ms: alpha.map(|_| 1000).into_iter().collect(),
        },
        color_timing: vec![timing],
        alpha_timing: alpha.map(|_| timing).into_iter().collect(),
        color_timescale: 1,
        duration_in_timescales: 1,
        repetition_count: Default::default(),
    }
}

fn synthetic_two_frame_animation(sample: &[u8]) -> AvifAnimation {
    let mut animation = synthetic_animation(sample, None);
    animation.sequence.color_samples.push(sample.to_vec());
    animation.sequence.color_durations_ms.push(1000);
    animation.color_timing.push(timing(1, 1, 1));
    animation.duration_in_timescales = 2;
    animation
}

fn synthetic_limits(max_live: Option<usize>) -> crate::limits::NativeDecodeLimits {
    synthetic_limits_with_frame_count(max_live, 1)
}

fn synthetic_limits_with_frame_count(
    max_live: Option<usize>,
    max_frames: usize,
) -> crate::limits::NativeDecodeLimits {
    let limits = crate::limits::NativeDecodeLimits::new(
        usize::MAX,
        16,
        16,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        8,
        8,
        8,
        1,
        max_frames,
    );
    max_live
        .map(|value| limits.with_max_live_allocation_bytes(value).unwrap())
        .unwrap_or(limits)
}

fn generated_strict_avis_fixture() -> Option<Vec<u8>> {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-strict-avis-unit-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).ok()?;
    let output = root.join("sequence.avifs");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "color=c=red:size=64x64:rate=1"])
        .args(["-frames:v", "4", "-c:v", "libaom-av1"])
        .args([
            "-still-picture",
            "0",
            "-g",
            "1",
            "-lag-in-frames",
            "0",
            "-auto-alt-ref",
            "0",
            "-f",
            "avif",
        ])
        .arg(&output)
        .status()
        .ok()?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&root);
        return None;
    }
    let data = std::fs::read(&output).ok()?;
    let _ = std::fs::remove_dir_all(&root);
    Some(data)
}

fn timing(timescale: u64, pts_in_timescales: u64, duration_in_timescales: u64) -> AvifFrameTiming {
    AvifFrameTiming {
        timescale,
        pts_in_timescales,
        duration_in_timescales,
        pts_ms: 0,
        duration_ms: 0,
    }
}

macro_rules! assert_live_accounting_eq {
    ($actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        assert_eq!(
            actual.metadata_live, expected.metadata_live,
            "metadata live"
        );
        assert_eq!(actual.icc_live, expected.icc_live, "ICC live");
        assert_eq!(actual.payload_live, expected.payload_live, "payload live");
        assert_eq!(actual.frame_live, expected.frame_live, "frame live");
        assert_eq!(
            actual.aggregate_live, expected.aggregate_live,
            "aggregate live"
        );
    }};
}

fn assert_committed_track_accounting(decoder: &super::StrictAvifSequenceDecoder) {
    let storage = decoder.tracks.storage_bytes().unwrap();
    assert_eq!(
        decoder.tracks_allocation.metadata.charged_capacity_bytes,
        storage.metadata
    );
    assert_eq!(
        decoder.tracks_allocation.frame.charged_capacity_bytes,
        storage.frame
    );
    assert_eq!(
        decoder.tracks_allocation.state.charged_capacity_bytes,
        storage.state
    );

    let accounting = decoder.budget.accounting();
    assert_eq!(accounting.metadata_live, storage.metadata);
    assert_eq!(accounting.icc_live, 0);
    assert_eq!(accounting.payload_live, 0);
    assert_eq!(accounting.frame_live, storage.frame + storage.state);
    assert_eq!(
        accounting.aggregate_live,
        storage.metadata + storage.frame + storage.state
    );
}

#[test]
fn strict_sequence_timing_accepts_exact_rational_equivalence() {
    validate_exact_timing_sync(&timing(24, 12, 6), &timing(48, 24, 12)).unwrap();
}

#[test]
fn strict_sequence_timing_rejects_pts_or_duration_mismatch() {
    let error = validate_exact_timing_sync(&timing(24, 12, 6), &timing(48, 25, 12))
        .expect_err("mismatched PTS must be rejected");
    assert!(matches!(error, DecoderError::Bitstream(message) if message.contains("synchronized")));

    let error = validate_exact_timing_sync(&timing(24, 12, 6), &timing(48, 24, 13))
        .expect_err("mismatched duration must be rejected");
    assert!(matches!(error, DecoderError::Bitstream(message) if message.contains("synchronized")));
}

#[test]
fn strict_sequence_timing_rejects_zero_timescale() {
    let error = validate_exact_timing_sync(&timing(0, 0, 1), &timing(1, 0, 1))
        .expect_err("zero timescale must be rejected");
    assert!(matches!(error, DecoderError::Bitstream(message) if message.contains("timescale")));
}

#[test]
fn strict_sequence_timing_handles_maximum_u64_without_panic() {
    validate_exact_timing_sync(
        &timing(u64::MAX, u64::MAX - 1, u64::MAX),
        &timing(u64::MAX, u64::MAX - 1, u64::MAX),
    )
    .unwrap();
}

#[test]
fn strict_normal_key_unit_uses_budgeted_frame_and_entropy_owners() {
    let info = synthetic_info(COLOR_16X16_AV1);
    let limits = synthetic_limits(None);
    let run = |max_live: Option<usize>| -> Result<usize, DecoderError> {
        let mut budget = crate::container::DecodeBudget::new(max_live);
        let headers = super::super::parse_av1_sequence_sample_headers_with_budget(
            &info,
            COLOR_16X16_AV1,
            COLOR_16X16_AV1,
            true,
            None,
            &[None; 8],
            &limits,
            &mut budget,
        )?;
        assert!(headers.tile_group.from_frame_obu);
        assert_eq!(headers.frame.frame_type, crate::av1::FrameType::Key);
        let unit = match super::super::decode_sequence_unit_with_native_budget(
            &headers,
            Some(&info),
            &limits,
            &mut budget,
        ) {
            Ok(unit) => unit,
            Err(error) => {
                super::super::drop_native_headers_and_release(headers, &mut budget)?;
                return Err(error);
            }
        };
        assert_eq!((unit.frame.width, unit.frame.height), (16, 16));
        assert_eq!(unit.cdf_states.len(), headers.tile_group.group.tiles.len());
        assert_eq!(unit.motion_field.mi_cols, 4);
        assert_eq!(unit.motion_field.mi_rows, 4);
        unit.release(&mut budget)?;
        super::super::drop_native_headers_and_release(headers, &mut budget)?;
        assert_eq!(budget.accounting().aggregate_live, 0);
        Ok(budget.accounting().aggregate_peak)
    };

    let exact_peak = run(None).expect("the normal key unit must decode");
    assert!(exact_peak > 0);
    run(Some(exact_peak)).expect("the measured exact budget must decode");
    let error = run(Some(exact_peak - 1)).expect_err("one byte under must not enter decode");
    assert!(matches!(error, DecoderError::InvalidParam(_)));
}

#[test]
fn unselected_track_timing_validation_uses_stsz_count_without_materializing_arrays() {
    let mut payload = vec![0; 16];
    payload[4..8].copy_from_slice(&1_u32.to_be_bytes());
    payload[8..12].copy_from_slice(&2_u32.to_be_bytes());
    payload[12..16].copy_from_slice(&3_u32.to_be_bytes());
    validate_sample_timing(&payload, 2).expect("stts must validate against stsz count");
    assert!(validate_sample_timing(&payload, 1).is_err());

    payload.truncate(15);
    assert!(validate_sample_timing(&payload, 2).is_err());
}

#[test]
fn native_timing_handoff_counts_final_vec_capacities() {
    let mut color_timing = Vec::with_capacity(5);
    color_timing.push(timing(1, 0, 1));
    let mut alpha_timing = Vec::with_capacity(3);
    alpha_timing.push(timing(1, 0, 1));
    let mut color_durations_ms = Vec::with_capacity(7);
    color_durations_ms.push(1);
    let mut alpha_durations_ms = Vec::with_capacity(2);
    alpha_durations_ms.push(1);
    let timing = ParsedAnimationTiming {
        color_timing,
        alpha_timing,
        color_durations_ms,
        alpha_durations_ms,
        color_timescale: 1,
        duration_in_timescales: 1,
        repetition_count: Default::default(),
        alpha_samples: Vec::new(),
    };
    let expected =
        (5 + 3) * std::mem::size_of::<AvifFrameTiming>() + (7 + 2) * std::mem::size_of::<u64>();
    assert_eq!(
        parsed_animation_timing_metadata_bytes(&timing).unwrap(),
        expected
    );
}

fn empty_info_with_capacity(capacity: usize) -> crate::container::AvifInfo {
    crate::container::AvifInfo {
        major_brand: *b"avis",
        compatible_brands: Vec::with_capacity(capacity),
        primary_item_id: Some(1),
        width: Some(1),
        height: Some(1),
        pixel_information: Some(crate::container::PixelInformation {
            bits_per_channel: Vec::with_capacity(capacity),
            extended_channels: None,
        }),
        color_information: None,
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: None,
        primary_item_payload: Vec::new(),
        sequence_sample_payloads: Vec::new(),
    }
}

#[test]
fn strict_sequence_clone_storage_uses_checked_actual_capacities() {
    let info = empty_info_with_capacity(7);
    let bytes = avif_info_storage_bytes(&info).expect("capacity walk should succeed");
    assert_eq!(
        bytes,
        7 * std::mem::size_of::<[u8; 4]>() + 7 * std::mem::size_of::<u8>()
    );
    assert!(checked_capacity_bytes::<u16>(usize::MAX, "overflow").is_err());
    assert!(
        SequenceTracksStorage {
            metadata: usize::MAX,
            frame: 1,
            state: 0,
        }
        .total()
        .is_err()
    );
}

#[test]
fn strict_sequence_constructor_clone_peak_preflights_alpha_temporary() {
    let info = empty_info_with_capacity(7);
    let info_clone_bytes = avif_info_storage_bytes(&info).unwrap();
    let no_alpha = sequence_tracks_clone_peak_bytes(&info, &[]).unwrap();
    assert_eq!(no_alpha, info_clone_bytes);

    let alpha_samples = vec![Vec::with_capacity(11)];
    let alpha_peak = sequence_tracks_clone_peak_bytes(&info, &alpha_samples).unwrap();
    assert_eq!(alpha_peak, info_clone_bytes * 2 + 11);

    let under = crate::container::DecodeBudget::new(Some(alpha_peak - 1));
    assert!(under.check_additional_frame(alpha_peak).is_err());
    assert_eq!(under.accounting().aggregate_live, 0);
    assert!(under.check_additional_frame(alpha_peak - 1).is_ok());
}

#[test]
fn strict_sequence_clone_storage_exact_under_and_retry_are_transactional() {
    let storage = SequenceTracksStorage {
        metadata: 5,
        frame: 7,
        state: 0,
    };
    let mut under = crate::container::DecodeBudget::new(Some(storage.total().unwrap() - 1));
    assert!(SequenceTracksDecoder::admit_storage(storage, &mut under).is_err());
    assert_eq!(under.accounting().aggregate_live, 0);

    let mut exact = crate::container::DecodeBudget::new(Some(storage.total().unwrap()));
    let mut allocation =
        SequenceTracksDecoder::admit_storage(storage, &mut exact).expect("exact budget");
    assert_eq!(exact.accounting().aggregate_live, storage.total().unwrap());
    allocation.release(&mut exact).unwrap();
    assert_eq!(exact.accounting().aggregate_live, 0);

    exact.set_max_live_bytes_for_test(storage.total().unwrap() - 1);
    assert!(SequenceTracksDecoder::admit_storage(storage, &mut exact).is_err());
    assert_eq!(exact.accounting().aggregate_live, 0);
    exact.set_max_live_bytes_for_test(storage.total().unwrap());
    let mut retry =
        SequenceTracksDecoder::admit_storage(storage, &mut exact).expect("retry budget");
    retry.release(&mut exact).unwrap();
    assert_eq!(exact.accounting().aggregate_live, 0);
}

#[test]
fn strict_sequence_candidate_guard_releases_all_tickets_on_drop() {
    let storage = SequenceTracksStorage {
        metadata: 5,
        frame: 7,
        state: 0,
    };
    let total = storage.total().unwrap();
    let mut budget = crate::container::DecodeBudget::new(Some(total));
    let allocation = SequenceTracksDecoder::admit_storage(storage, &mut budget)
        .expect("candidate allocation should fit");
    {
        let _guard = super::SequenceTracksAllocationGuard::new(&mut budget, allocation);
    }
    assert_eq!(budget.accounting().aggregate_live, 0);

    let retry = SequenceTracksDecoder::admit_storage(storage, &mut budget)
        .expect("dropped candidate must leave the budget retryable");
    assert_eq!(budget.accounting().aggregate_live, total);
    let mut retry = retry;
    retry.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().aggregate_live, 0);
}

#[test]
fn strict_sequence_state_ticket_is_exact_and_transactional() {
    let storage = SequenceTracksStorage {
        metadata: 0,
        frame: 0,
        state: 17,
    };
    let mut under = crate::container::DecodeBudget::new(Some(16));
    assert!(SequenceTracksDecoder::admit_storage(storage, &mut under).is_err());
    assert_eq!(under.accounting().aggregate_live, 0);

    let mut exact = crate::container::DecodeBudget::new(Some(17));
    let mut allocation =
        SequenceTracksDecoder::admit_storage(storage, &mut exact).expect("state ticket fits");
    assert_eq!(allocation.state.charged_capacity_bytes, 17);
    allocation.release(&mut exact).unwrap();
    assert_eq!(exact.accounting().aggregate_live, 0);
}

#[test]
fn strict_sequence_refresh_underestimate_rejects_without_late_reserve() {
    let storage = SequenceTracksStorage {
        metadata: 0,
        frame: 0,
        state: 4,
    };
    let mut budget = crate::container::DecodeBudget::new(Some(4));
    let allocation = SequenceTracksDecoder::admit_storage(storage, &mut budget)
        .expect("admitted refresh ticket should fit");
    {
        let mut guard = super::SequenceTracksAllocationGuard::new(&mut budget, allocation);
        let error = guard
            .reconcile_state_delta(5)
            .expect_err("a refresh plan underestimate must fail closed");
        assert!(
            matches!(error, DecoderError::InvalidParam(message) if message.contains("underestimated"))
        );
        // No late reserve is possible: the budget remains at the admitted
        // ticket while the guard still owns the candidate allocation.
        assert_eq!(guard.budget.accounting().aggregate_live, 4);
    }
    assert_eq!(budget.accounting().aggregate_live, 0);
    let retry = SequenceTracksDecoder::admit_storage(storage, &mut budget)
        .expect("rollback must leave the budget retryable");
    assert_eq!(budget.accounting().aggregate_live, 4);
    let mut retry = retry;
    retry.release(&mut budget).unwrap();
    assert_eq!(budget.accounting().aggregate_live, 0);
}

#[test]
fn strict_sequence_existing_owner_overflow_rolls_back_the_charge() {
    let mut budget = crate::container::DecodeBudget::new(None);
    let mut metadata = budget
        .reserve_existing_bytes(super::AllocationClass::Metadata, 1, "metadata owner")
        .unwrap();
    let error = budget
        .reserve_existing_bytes(super::AllocationClass::Payload, usize::MAX, "payload owner")
        .expect_err("aggregate overflow must be rejected");
    assert!(matches!(error, DecoderError::InvalidParam(message) if message.contains("overflows")));
    assert_eq!(budget.accounting().aggregate_live, 1);
    budget.release_token(&mut metadata).unwrap();
    assert_eq!(budget.accounting().aggregate_live, 0);
}

#[test]
fn strict_prepare_rgb_preflights_before_allocator_and_commits_once() {
    let animation = synthetic_animation(COLOR_16X16_AV1, None);
    let info = synthetic_info(COLOR_16X16_AV1);
    let mut decoder =
        super::StrictAvifSequenceDecoder::from_test_parts(info, animation, synthetic_limits(None))
            .expect("synthetic RGB strict decoder should construct");
    let baseline = decoder.budget.accounting();
    let required = decoder
        .next_prepare_additional_bytes(COLOR_16X16_AV1, None)
        .expect("strict prepare plan should be finite");
    decoder
        .budget
        .set_max_live_bytes_for_test(baseline.aggregate_live + required - 1);
    let (requests, phases) = {
        let (result, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
            decoder.prepare_next_frame()
        });
        assert!(result.is_err(), "one-under strict prepare must fail");
        (
            requests,
            crate::test_allocation_observer::phase_allocation_requests(),
        )
    };
    assert_eq!(requests, 1, "only the typed error message may allocate");
    assert_eq!(phases[1], 0, "one-under rejection must precede clone");
    assert_eq!(
        phases[2], 0,
        "one-under rejection must precede split/decode"
    );
    assert_eq!(phases[3], 0, "one-under rejection must precede decode");
    assert_eq!(decoder.next_index, 0);
    assert_eq!(decoder.tracks.decoded_sample_counts(), (0, None));
    assert_live_accounting_eq!(decoder.budget.accounting(), baseline);

    decoder
        .budget
        .set_max_live_bytes_for_test(baseline.aggregate_live + required);
    let (dropped, phases) = {
        let (result, _) = crate::test_allocation_observer::count_allocation_requests(|| {
            decoder.prepare_next_frame()
        });
        assert!(result.is_ok(), "exact strict prepare should succeed");
        (
            result
                .expect("exact strict prepare result")
                .expect("first frame should be present"),
            crate::test_allocation_observer::phase_allocation_requests(),
        )
    };
    assert!(phases[1] > 0, "exact prepare must deep-clone metadata");
    assert!(phases[2] > 0, "exact prepare must enter split phase");
    assert!(phases[3] > 0, "exact prepare must enter decode phase");
    assert_eq!(dropped.frame().unwrap().buffers.planes.len(), 3);
    let dropped_frame = dropped.frame().unwrap().clone();
    drop(dropped);
    assert_live_accounting_eq!(decoder.budget.accounting(), baseline);
    assert_eq!(decoder.next_index, 0);
    assert_eq!(decoder.tracks.decoded_sample_counts(), (0, None));
    let prepared = decoder
        .prepare_next_frame()
        .expect("second retry prepare should succeed")
        .expect("first frame should be present");
    assert_eq!(prepared.frame(), Some(&dropped_frame));
    prepared.commit().expect("commit should advance one frame");
    assert_eq!(decoder.next_index, 1);
    assert_eq!(decoder.tracks.decoded_sample_counts(), (1, None));
    assert_committed_track_accounting(&decoder);
}

#[test]
fn synthetic_frame_limit_rejects_before_retained_candidate_allocation() {
    let info = synthetic_info(COLOR_16X16_AV1);
    let animation = synthetic_two_frame_animation(COLOR_16X16_AV1);
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    let result =
        super::StrictAvifSequenceDecoder::from_test_parts(info, animation, synthetic_limits(None));
    assert!(matches!(result, Err(DecoderError::InvalidParam(_))));
    assert_eq!(
        observation.registered_candidate_count(),
        0,
        "frame-count rejection must precede native retained-owner candidates"
    );
    drop(all_candidates);
    drop(observation);
}

#[test]
fn parsed_frame_limit_rejects_before_retained_candidate_allocation() {
    let Some(data) = generated_strict_avis_fixture() else {
        eprintln!("ffmpeg/libaom unavailable; skipping parsed AVIS limit test");
        return;
    };
    let limits = crate::limits::NativeDecodeLimits::new(
        data.len(),
        8192,
        8192,
        8192 * 8192,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        1024,
        1024,
        1 << 20,
        128,
        1,
    );
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let all_candidates = crate::test_allocation_observer::track_all_candidates();
    let result = super::StrictAvifSequenceDecoder::new(&data, limits);
    assert!(matches!(result, Err(crate::DecoderError::InvalidParam(_))));
    assert_eq!(
        observation.registered_candidate_count(),
        0,
        "parsed frame-count rejection must precede retained track candidates"
    );
    drop(all_candidates);
    drop(observation);
}

#[test]
fn strict_prepared_frame_exposes_clone_free_views_and_retained_bytes() {
    let animation = synthetic_animation(COLOR_16X16_AV1, None);
    let info = synthetic_info(COLOR_16X16_AV1);
    let mut decoder =
        super::StrictAvifSequenceDecoder::from_test_parts(info, animation, synthetic_limits(None))
            .expect("synthetic strict decoder should construct");
    let information = decoder.information() as *const _;
    let prepared = decoder
        .prepare_next_frame()
        .expect("strict preparation should succeed")
        .expect("first frame should be present");
    assert_eq!(prepared.frame_index(), 0);
    assert_eq!(prepared.information() as *const _, information);
    let rich_information = prepared.information().rich() as *const _;
    let expected_frame_storage =
        super::decoded_frame_storage_bytes(prepared.frame().expect("prepared frame"))
            .expect("native frame storage should be measurable");
    assert_eq!(
        prepared.retained_live_bytes(),
        prepared.decoder.budget.aggregate_live_bytes()
    );
    assert_eq!(
        prepared.additional_live_bytes(),
        prepared
            .retained_live_bytes()
            .checked_add(expected_frame_storage)
            .expect("prepared live bytes should not overflow")
    );
    assert!(prepared.retained_live_bytes() < prepared.additional_live_bytes());
    let ready = prepared
        .prepare_commit()
        .expect("commit preflight should succeed");
    let observed = ready
        .try_map_and_commit(|frame, rich, _, index| {
            Ok::<_, DecoderError>((
                frame.width,
                frame.height,
                index == 0 && std::ptr::eq(rich, rich_information),
            ))
        })
        .expect("information handoff should succeed");
    assert_eq!(observed, (16, 16, true));
    assert_eq!(decoder.next_index, 1);
}

#[test]
fn strict_prepared_information_callback_panic_rolls_back_and_retries() {
    let animation = synthetic_animation(COLOR_16X16_AV1, None);
    let info = synthetic_info(COLOR_16X16_AV1);
    let mut decoder =
        super::StrictAvifSequenceDecoder::from_test_parts(info, animation, synthetic_limits(None))
            .expect("synthetic strict decoder should construct");
    let baseline = decoder.budget.accounting();
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let prepared = decoder
            .prepare_next_frame()
            .expect("strict preparation should succeed")
            .expect("first frame should be present");
        let ready = prepared
            .prepare_commit()
            .expect("commit token preflight should succeed");
        let _: Result<(), DecoderError> =
            ready.try_map_and_commit(|_, _, _, _| panic!("synthetic mapping panic"));
    }));
    assert!(panic_result.is_err(), "consumer panic must propagate");
    assert_live_accounting_eq!(decoder.budget.accounting(), baseline);
    assert_eq!(decoder.next_index, 0);
    decoder
        .prepare_next_frame()
        .expect("rollback must leave retry available")
        .expect("retry frame should be present")
        .commit()
        .expect("retry commit should succeed");
    assert_eq!(decoder.next_index, 1);
}

#[test]
fn strict_prepared_information_callback_error_rolls_back_and_retries() {
    let animation = synthetic_animation(COLOR_16X16_AV1, None);
    let info = synthetic_info(COLOR_16X16_AV1);
    let mut decoder =
        super::StrictAvifSequenceDecoder::from_test_parts(info, animation, synthetic_limits(None))
            .expect("synthetic strict decoder should construct");
    let baseline = decoder.budget.accounting();
    let prepared = decoder
        .prepare_next_frame()
        .expect("strict preparation should succeed")
        .expect("first frame should be present");
    let retained = prepared.retained_live_bytes();
    let ready = prepared
        .prepare_commit()
        .expect("commit preflight should succeed");
    let result = ready.try_map_and_commit(|frame, _, _, _| {
        assert_eq!(frame.buffers.planes.len(), 3);
        Err::<(), _>(DecoderError::InvalidParam("consumer rejected frame".into()))
    });
    assert!(result.is_err());
    assert!(retained > baseline.aggregate_live);
    assert_live_accounting_eq!(decoder.budget.accounting(), baseline);
    assert_eq!(decoder.next_index, 0);
    decoder
        .prepare_next_frame()
        .expect("rollback must leave retry available")
        .expect("retry frame should be present")
        .commit()
        .expect("retry commit should succeed");
    assert_eq!(decoder.next_index, 1);
}

#[test]
fn strict_prepared_commit_token_drop_rolls_back_without_advancing() {
    let animation = synthetic_animation(COLOR_16X16_AV1, None);
    let info = synthetic_info(COLOR_16X16_AV1);
    let mut decoder =
        super::StrictAvifSequenceDecoder::from_test_parts(info, animation, synthetic_limits(None))
            .expect("synthetic strict decoder should construct");
    let baseline = decoder.budget.accounting();
    let prepared = decoder
        .prepare_next_frame()
        .expect("strict preparation should succeed")
        .expect("first frame should be present");
    let ready = prepared
        .prepare_commit()
        .expect("commit token preflight should succeed");
    assert!(ready.frame().is_some());
    drop(ready);
    assert_live_accounting_eq!(decoder.budget.accounting(), baseline);
    assert_eq!(decoder.next_index, 0);
}

#[test]
fn strict_prepare_rgb_alpha_preflights_and_commits_both_tracks_once() {
    let animation = synthetic_animation(COLOR_16X16_AV1, Some(ALPHA_16X16_AV1));
    let info = synthetic_info(COLOR_16X16_AV1);
    let mut decoder =
        super::StrictAvifSequenceDecoder::from_test_parts(info, animation, synthetic_limits(None))
            .expect("synthetic RGB+alpha strict decoder should construct");
    let baseline = decoder.budget.accounting();
    let required = decoder
        .next_prepare_additional_bytes(COLOR_16X16_AV1, Some(ALPHA_16X16_AV1))
        .expect("strict alpha prepare plan should be finite");
    decoder
        .budget
        .set_max_live_bytes_for_test(baseline.aggregate_live + required - 1);
    let phases = {
        let (result, _) = crate::test_allocation_observer::count_allocation_requests(|| {
            decoder.prepare_next_frame()
        });
        assert!(result.is_err(), "one-under alpha prepare must fail");

        crate::test_allocation_observer::phase_allocation_requests()
    };
    assert_eq!(phases[1], 0, "one-under alpha rejection must precede clone");
    assert_eq!(
        phases[2], 0,
        "one-under alpha rejection must precede decode"
    );
    assert_eq!(
        phases[3], 0,
        "one-under alpha rejection must precede decode"
    );
    assert_eq!(decoder.next_index, 0);
    assert_eq!(decoder.tracks.decoded_sample_counts(), (0, Some(0)));
    assert_live_accounting_eq!(decoder.budget.accounting(), baseline);

    decoder
        .budget
        .set_max_live_bytes_for_test(baseline.aggregate_live + required);
    let (dropped, phases) = {
        let (result, _) = crate::test_allocation_observer::count_allocation_requests(|| {
            decoder.prepare_next_frame()
        });
        assert!(result.is_ok(), "exact alpha strict prepare should succeed");
        (
            result
                .expect("exact alpha strict prepare result")
                .expect("alpha first frame should be present"),
            crate::test_allocation_observer::phase_allocation_requests(),
        )
    };
    assert!(
        phases[1] > 0,
        "exact alpha prepare must deep-clone metadata"
    );
    assert!(phases[2] > 0, "exact alpha prepare must enter split phase");
    assert!(phases[3] > 0, "exact alpha prepare must enter decode phase");
    assert_eq!(dropped.frame().unwrap().buffers.planes.len(), 4);
    let dropped_frame = dropped.frame().unwrap().clone();
    drop(dropped);
    assert_live_accounting_eq!(decoder.budget.accounting(), baseline);
    assert_eq!(decoder.next_index, 0);
    assert_eq!(decoder.tracks.decoded_sample_counts(), (0, Some(0)));
    let prepared = decoder
        .prepare_next_frame()
        .expect("alpha second retry prepare should succeed")
        .expect("alpha first frame should be present");
    assert_eq!(prepared.frame(), Some(&dropped_frame));
    prepared
        .commit()
        .expect("alpha commit should advance one frame");
    assert_eq!(decoder.next_index, 1);
    assert_eq!(decoder.tracks.decoded_sample_counts(), (1, Some(1)));
    assert_committed_track_accounting(&decoder);
}

#[test]
fn strict_static_alpha_frame_ticket_is_not_double_counted_on_continuation() {
    let animation = synthetic_two_frame_animation(COLOR_16X16_AV1);
    let info = synthetic_static_alpha_info(COLOR_16X16_AV1);
    let mut decoder = super::StrictAvifSequenceDecoder::from_test_parts(
        info,
        animation,
        synthetic_limits_with_frame_count(None, 2),
    )
    .expect("synthetic static-alpha decoder should construct");
    let baseline = decoder.budget.accounting();

    let first = decoder
        .prepare_next_frame()
        .expect("first static-alpha preparation should succeed")
        .expect("first frame should be present");
    let first_storage = first
        .candidate_tracks
        .as_ref()
        .expect("candidate tracks")
        .storage_bytes()
        .expect("candidate storage should be measurable");
    let first_frame_storage = first_storage.frame;
    assert!(
        first_frame_storage > 0,
        "first static-alpha frame must create retained frame storage"
    );
    assert_eq!(
        first
            .candidate_allocation
            .as_ref()
            .expect("candidate allocation")
            .frame
            .charged_capacity_bytes,
        first_frame_storage,
        "static-alpha candidate frame ticket must match storage"
    );
    first
        .commit()
        .expect("first static-alpha commit should succeed");
    assert_eq!(decoder.next_index, 1);
    let committed_storage = decoder
        .tracks
        .storage_bytes()
        .expect("committed storage should be measurable");
    assert_eq!(committed_storage.frame, first_frame_storage);
    let after_first = decoder.budget.accounting();
    assert_eq!(
        after_first.frame_live,
        committed_storage.frame + committed_storage.state
    );
    assert_eq!(
        decoder.tracks_allocation.frame.charged_capacity_bytes,
        committed_storage.frame
    );

    let second = decoder
        .prepare_next_frame()
        .expect("second static-alpha preparation should succeed")
        .expect("second frame should be present");
    let second_storage = second
        .candidate_tracks
        .as_ref()
        .expect("candidate tracks")
        .storage_bytes()
        .expect("second candidate storage should be measurable");
    assert_eq!(second_storage.frame, first_frame_storage);
    assert_eq!(
        second
            .candidate_allocation
            .as_ref()
            .expect("second candidate allocation")
            .frame
            .charged_capacity_bytes,
        first_frame_storage,
        "continuation must not charge a second static-alpha frame"
    );
    second
        .commit()
        .expect("second static-alpha commit should succeed");
    assert_eq!(decoder.next_index, 2);
    let final_storage = decoder
        .tracks
        .storage_bytes()
        .expect("final storage should be measurable");
    assert_eq!(final_storage.frame, first_frame_storage);
    let final_accounting = decoder.budget.accounting();
    assert_eq!(
        final_accounting.frame_live,
        final_storage.frame + final_storage.state
    );
    assert_eq!(
        final_accounting.aggregate_live,
        final_accounting.metadata_live
            + final_accounting.payload_live
            + final_accounting.frame_live
    );
    assert!(final_accounting.aggregate_live > baseline.aggregate_live);
}

#[test]
fn strict_prepare_split_overcapacity_rolls_back_guard_and_retries() {
    let animation = synthetic_animation(COLOR_16X16_AV1, None);
    let info = synthetic_info(COLOR_16X16_AV1);
    let mut decoder =
        super::StrictAvifSequenceDecoder::from_test_parts(info, animation, synthetic_limits(None))
            .expect("synthetic strict decoder should construct");
    let baseline = decoder.budget.accounting();
    let baseline_storage = decoder.tracks.storage_bytes().unwrap();
    let required = decoder
        .next_prepare_additional_bytes(COLOR_16X16_AV1, None)
        .expect("strict prepare plan should be finite");
    decoder
        .budget
        .set_max_live_bytes_for_test(baseline.aggregate_live + required);

    let extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        "AVIS strict sequence split",
        1,
    );
    let (result, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
        decoder.prepare_next_frame().map(drop)
    });
    drop(extra);
    let phases = crate::test_allocation_observer::phase_allocation_requests();
    let error = result.expect_err("overcapacity must be rejected by strict prepare");
    assert!(matches!(error, DecoderError::InvalidParam(message) if message.contains("exceeded")));
    assert!(
        requests > 0,
        "overcapacity path must make real candidate requests"
    );
    assert!(
        phases[1] > 0,
        "clone must run before split overcapacity is discovered"
    );
    assert!(
        phases[2] > 0,
        "strict split must allocate the overcapacity candidate"
    );
    assert_eq!(phases[3], 0, "decode must not start after split failure");
    assert_eq!(decoder.next_index, 0);
    assert_eq!(decoder.tracks.decoded_sample_counts(), (0, None));
    assert_live_accounting_eq!(decoder.budget.accounting(), baseline);
    assert_eq!(
        decoder.tracks_allocation.metadata.charged_capacity_bytes,
        baseline_storage.metadata
    );
    assert_eq!(
        decoder.tracks_allocation.frame.charged_capacity_bytes,
        baseline_storage.frame
    );
    assert_eq!(
        decoder.tracks_allocation.state.charged_capacity_bytes,
        baseline_storage.state
    );

    let prepared = decoder
        .prepare_next_frame()
        .expect("rollback must leave strict prepare retryable")
        .expect("first frame should be present after retry");
    prepared.commit().expect("retry commit should succeed");
    assert_eq!(decoder.next_index, 1);
    assert_committed_track_accounting(&decoder);
}

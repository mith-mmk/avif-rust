use super::frame::{
    append_native_alpha_plane, validate_native_frame_prefix_limits, validate_native_sequence_limits,
};
use crate::container::{DecodeBudget, parse_avif};
use crate::limits::NativeDecodeLimits;
use crate::obu::{ObuType, find_obu_payload};
use crate::test_support::fixture_path as sample_path;

fn limits(input_len: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        input_len.saturating_mul(2),
        8192,
        8192,
        1 << 28,
        1 << 30,
        1 << 24,
        1 << 24,
        128,
        256,
        256,
        8,
        8,
    )
}

fn hide_first_frame(payload: &mut [u8]) {
    let frame = find_obu_payload(payload, ObuType::Frame)
        .expect("frame OBU lookup must succeed")
        .or_else(|| find_obu_payload(payload, ObuType::FrameHeader).unwrap())
        .expect("frame OBU must be present");
    let offset = frame.as_ptr() as usize - payload.as_ptr() as usize;
    assert!(
        payload[offset] & 0x80 == 0,
        "fixture must not be show-existing"
    );
    payload[offset] &= !0x10;
}

#[test]
fn native_primary_prefix_rejects_hidden_frame_before_finish() {
    let data = std::fs::read(sample_path("WML2Viewer.avif"))
        .expect("native hookup fixture must be present");
    let mut info = parse_avif(&data).expect("native hookup fixture must parse");
    hide_first_frame(&mut info.primary_item_payload);
    let resource_limits = limits(data.len());
    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let error = match validate_native_frame_prefix_limits(&info, &sequence, &resource_limits) {
        Ok(_) => panic!("hidden native primary must be rejected before finish"),
        Err(error) => error,
    };
    assert!(
        matches!(error, crate::DecoderError::Unsupported(message) if message.contains("displayed"))
    );
}

#[test]
fn native_alpha_prefix_is_checked_before_primary_decode() {
    let data = std::fs::read(sample_path(
        "plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
    ))
    .expect("native alpha hookup fixture must be present");
    let info = parse_avif(&data).expect("native alpha hookup fixture must parse");
    let resource_limits = limits(data.len());
    let prefix = super::still::validate_alpha_auxiliary_header_limits(
        &info,
        &resource_limits,
        info.width.expect("master width must be present") as usize,
        info.height.expect("master height must be present") as usize,
        8,
    )
    .expect("native alpha prefix must pass before primary decode");
    assert!(prefix.prefix.show_frame());
}

#[test]
fn native_selected_alpha_split_obus_use_the_bounded_merge_route() {
    let data = std::fs::read(sample_path(
        "plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
    ))
    .expect("native alpha hookup fixture must be present");
    let info = parse_avif(&data).expect("native alpha hookup fixture must parse");
    let alpha = info
        .alpha_auxiliary_items
        .first()
        .expect("fixture must contain selected alpha");

    let (route, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
        super::select_strict_native_frame_route(&alpha.payload)
            .expect("split alpha route must classify")
    });
    assert_eq!(requests, 0, "strict split classifier must stay borrowed");
    assert!(matches!(route, super::StrictNativeFrameRoute::Split { .. }));

    let resource_limits = limits(data.len());
    let header = super::still::validate_alpha_auxiliary_header_limits(
        &info,
        &resource_limits,
        info.width.expect("master width must be present") as usize,
        info.height.expect("master height must be present") as usize,
        8,
    )
    .expect("split alpha prefix must validate");
    let source = super::ItemHeaderSource::from_auxiliary(
        info.alpha_auxiliary_items
            .first()
            .expect("selected alpha remains present"),
    );
    let mut budget = native_header_budget();
    let headers = super::parse_av1_headers_for_item_with_prefix_and_budget(
        &source,
        header.parts,
        header.prefix,
        &mut budget,
    )
    .expect("split alpha must use strict bounded header assembly");
    assert!(!headers.tile_group.from_frame_obu);
    assert!(headers.tile_group.entropy_states.is_empty());
    super::drop_native_headers_and_release(headers, &mut budget).unwrap();
    assert_eq!(budget.accounting().aggregate_live, 0);
}

#[test]
fn native_alpha_attachment_moves_the_decoded_plane_owner() {
    let data = std::fs::read(sample_path(
        "plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
    ))
    .expect("native alpha hookup fixture must be present");
    let info = parse_avif(&data).expect("native alpha hookup fixture must parse");
    let resource_limits = limits(data.len());

    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let master_prefix = validate_native_frame_prefix_limits(&info, &sequence, &resource_limits)
        .expect("native master prefix must pass");
    let alpha_header = super::still::validate_alpha_auxiliary_header_limits(
        &info,
        &resource_limits,
        info.width.expect("master width must be present") as usize,
        info.height.expect("master height must be present") as usize,
        sequence.color_config.bit_depth,
    )
    .expect("native alpha prefix must pass");

    let mut master = {
        let headers = super::parse_av1_headers_with_frame_prefix(&info, master_prefix)
            .expect("native master headers must parse");
        super::decode_still_frame(&headers, Some(&info)).expect("native master must decode")
    };
    let alpha = super::still::decode_alpha_auxiliary_frame_with_prefix(
        &info,
        alpha_header.parts,
        alpha_header.prefix,
    )
    .expect("native alpha must decode");
    let source_plane = alpha
        .buffers
        .planes
        .first()
        .expect("native alpha must contain one plane");
    let source_ptr = source_plane.samples.as_ptr();
    let source_len = source_plane.samples.len();
    let source_capacity = source_plane.samples.capacity();
    let source_samples = source_plane.samples.clone();

    append_native_alpha_plane(&mut master, alpha).expect("native alpha attachment must pass");

    let attached = master
        .buffers
        .planes
        .iter()
        .find(|plane| plane.layout.plane == 3)
        .expect("native alpha plane must be attached as plane 3");
    assert_eq!(attached.samples.as_ptr(), source_ptr);
    assert_eq!(attached.samples.len(), source_len);
    assert_eq!(attached.samples.capacity(), source_capacity);
    assert_eq!(attached.samples.as_slice(), source_samples.as_slice());
}

fn native_header_budget() -> DecodeBudget {
    DecodeBudget::new(Some(usize::MAX / 2))
}

fn normal_frame_components() -> (
    Vec<u8>,
    crate::av1::SequenceHeader,
    crate::av1::FrameHeader,
    crate::av1::TileGroup,
    Vec<u8>,
) {
    let data = std::fs::read(sample_path("WML2Viewer.avif"))
        .expect("native hookup fixture must be present");
    let info = parse_avif(&data).expect("native header fixture must parse");
    let source = info.primary_item_payload.clone();
    let resource_limits = limits(data.len());
    let bounded_sequence = validate_native_sequence_limits(&info, &resource_limits)
        .expect("normal OBU_FRAME sequence must validate");
    let prefix = validate_native_frame_prefix_limits(&info, &bounded_sequence, &resource_limits)
        .expect("normal OBU_FRAME prefix must validate");
    let mut budget = native_header_budget();
    let headers = super::parse_av1_headers_with_frame_prefix_and_budget(&info, prefix, &mut budget)
        .expect("normal OBU_FRAME must use the strict native header path");
    assert!(headers.tile_group.from_frame_obu);
    let sequence = headers.sequence;
    let frame = headers.frame.clone();
    let group = headers.tile_group.group.clone();
    super::drop_native_headers_and_release(headers, &mut budget)
        .expect("strict header test ownership must release");
    assert_eq!(budget.accounting().aggregate_live, 0);

    let frame_payload = find_obu_payload(&source, ObuType::Frame)
        .expect("frame OBU lookup must succeed")
        .expect("fixture must keep its normal OBU_FRAME")
        .to_vec();
    (source, sequence, frame, group, frame_payload)
}

fn capacity_bytes<T>(values: &Vec<T>) -> usize {
    values
        .capacity()
        .checked_mul(std::mem::size_of::<T>())
        .expect("test owner capacity must fit usize")
}

fn sized_obu(obu_type: ObuType, payload: &[u8]) -> Vec<u8> {
    let obu_type_bits = match obu_type {
        ObuType::SequenceHeader => 1,
        ObuType::FrameHeader => 3,
        ObuType::TileGroup => 4,
        ObuType::Frame => 6,
        _ => panic!("test helper supports only normal-frame selector OBU types"),
    };
    let mut bytes = Vec::with_capacity(payload.len() + 6);
    bytes.push((obu_type_bits << 3) | 0x02);
    let mut length = payload.len();
    loop {
        let mut byte = (length & 0x7f) as u8;
        length >>= 7;
        if length != 0 {
            byte |= 0x80;
        }
        bytes.push(byte);
        if length == 0 {
            break;
        }
    }
    bytes.extend_from_slice(payload);
    bytes
}

fn split_primary_item_payload(payload: &[u8]) -> Vec<u8> {
    let sequence = find_obu_payload(payload, ObuType::SequenceHeader)
        .expect("split fixture sequence OBU lookup must succeed")
        .expect("split fixture must contain one sequence OBU");
    let frame = find_obu_payload(payload, ObuType::Frame)
        .expect("split fixture frame OBU lookup must succeed")
        .expect("split fixture must contain one OBU_FRAME");
    let parsed_sequence = crate::av1::parse_sequence_header(sequence)
        .expect("split fixture sequence header must parse");
    let parsed_frame = crate::av1::parse_frame_header(frame, &parsed_sequence)
        .expect("split fixture frame header must parse");
    let tile_start = parsed_frame.payload_after_header_offset;
    assert!(tile_start < frame.len(), "fixture must contain tile bytes");

    [
        sized_obu(ObuType::SequenceHeader, sequence),
        sized_obu(ObuType::FrameHeader, &frame[..tile_start]),
        sized_obu(ObuType::TileGroup, &frame[tile_start..]),
    ]
    .concat()
}

fn split_master_fixture() -> crate::container::AvifInfo {
    let data = std::fs::read(sample_path("WML2Viewer.avif"))
        .expect("native split master fixture must be present");
    let mut info = parse_avif(&data).expect("native split master fixture must parse");
    info.primary_item_payload = split_primary_item_payload(&info.primary_item_payload);
    info
}

fn two_tile_info() -> crate::av1::TileInfo {
    crate::av1::TileInfo {
        uniform_tile_spacing: true,
        dependent_tiles: false,
        loop_filter_across_tiles: false,
        tile_cols: 2,
        tile_rows: 1,
        tile_cols_log2: 1,
        tile_rows_log2: 0,
        tile_size_bytes: 1,
        context_update_tile_id: 0,
        mi_col_starts: vec![0, 16, 32],
        mi_row_starts: vec![0, 16],
    }
}

fn split_stream(frame_header: &[u8], groups: &[&[u8]]) -> Vec<u8> {
    let mut stream = sized_obu(ObuType::FrameHeader, frame_header);
    for group in groups {
        stream.extend(sized_obu(ObuType::TileGroup, group));
    }
    stream
}

fn parse_split_master_headers_with_budget(
    info: &crate::container::AvifInfo,
    resource_limits: &NativeDecodeLimits,
    budget: &mut DecodeBudget,
) -> Result<super::Av1Headers, crate::DecoderError> {
    let sequence = validate_native_sequence_limits(info, resource_limits)?;
    let prefix = validate_native_frame_prefix_limits(info, &sequence, resource_limits)?;
    super::parse_av1_headers_with_frame_prefix_and_budget(info, prefix, budget)
}

fn parse_split_alpha_headers_with_budget(
    info: &crate::container::AvifInfo,
    resource_limits: &NativeDecodeLimits,
    budget: &mut DecodeBudget,
) -> Result<super::Av1Headers, crate::DecoderError> {
    let header = super::still::validate_alpha_auxiliary_header_limits(
        info,
        resource_limits,
        info.width
            .expect("split alpha master width must be present") as usize,
        info.height
            .expect("split alpha master height must be present") as usize,
        8,
    )?;
    let auxiliary = info
        .alpha_auxiliary_items
        .first()
        .expect("split alpha fixture must contain the selected alpha");
    let source = super::ItemHeaderSource::from_auxiliary(auxiliary);
    super::parse_av1_headers_for_item_with_prefix_and_budget(
        &source,
        header.parts,
        header.prefix,
        budget,
    )
}

fn assert_split_actual_capacity_retry<F>(label: &'static str, mut parse: F)
where
    F: FnMut(&mut DecodeBudget) -> Result<super::Av1Headers, crate::DecoderError>,
{
    let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(label, 1 << 20);

    let mut reference = DecodeBudget::new(None);
    let headers = parse(&mut reference).expect("split reference materialization must succeed");
    let actual_peak = reference.accounting().aggregate_live;
    assert!(actual_peak > 1, "split reference peak must be non-zero");
    super::drop_native_headers_and_release(headers, &mut reference)
        .expect("split reference ownership must release");
    assert_eq!(reference.accounting().aggregate_live, 0);

    let mut under = DecodeBudget::new(Some(actual_peak - 1));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let error = match parse(&mut under) {
        Ok(headers) => {
            super::drop_native_headers_and_release(headers, &mut under)
                .expect("unexpected split M-1 headers must release");
            panic!("split M-1 budget must reject actual capacity");
        }
        Err(error) => error,
    };
    assert!(
        matches!(error, crate::DecoderError::InvalidParam(ref message) if message.contains("live allocation")),
        "split M-1 rejection must be a typed live-allocation error: {error:?}"
    );
    assert_eq!(under.accounting().aggregate_live, 0);
    assert_eq!(
        observation.drops(),
        1,
        "the forced split final-owner candidate must be dropped on M-1"
    );
    assert!(
        observation.restore_count() >= 1,
        "the rejected split transaction must restore its accounting checkpoint"
    );
    drop(observation);

    let mut retry = DecodeBudget::new(Some(actual_peak));
    let headers = parse(&mut retry).expect("split exact-M retry must succeed");
    super::drop_native_headers_and_release(headers, &mut retry)
        .expect("split retry ownership must release");
    assert_eq!(retry.accounting().aggregate_live, 0);
}

fn assert_split_final_owners_drop_before_ticket_release<F>(kind: &str, mut parse: F)
where
    F: FnMut(&mut DecodeBudget) -> Result<super::Av1Headers, crate::DecoderError>,
{
    let observation = crate::test_allocation_observer::Observation::begin(usize::MAX, false);
    let _all_candidates = crate::test_allocation_observer::track_all_candidates();
    let mut budget = native_header_budget();
    let headers = parse(&mut budget).expect("split strict header materialization must succeed");
    assert!(
        !headers.tile_group.from_frame_obu,
        "{kind} must keep split routing"
    );
    assert!(headers.tile_group.entropy_states.is_empty());
    assert_eq!(
        headers.tile_group.group.tiles.len(),
        headers.decode_plan.tiles.len(),
        "{kind} must retain only the final descriptor table"
    );
    super::drop_native_headers_and_release(headers, &mut budget)
        .expect("split strict header ownership must release");
    assert_eq!(budget.accounting().aggregate_live, 0);
    assert!(
        observation
            .registered_drops()
            .into_iter()
            .filter(|drops| *drops != 0)
            .count()
            >= 2,
        "{kind} must physically drop the descriptor and merged tile-data owners"
    );
    assert!(
        observation.release_drop_snapshots()[0] >= 2,
        "{kind} must drop both final split owners before its first ticket release"
    );
}

#[test]
fn native_header_materialization_charges_actual_owners_and_releases_before_alpha() {
    let data = std::fs::read(sample_path("WML2Viewer.avif"))
        .expect("native header fixture must be present");
    let info = parse_avif(&data).expect("native header fixture must parse");
    let resource_limits = limits(data.len());
    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let prefix = validate_native_frame_prefix_limits(&info, &sequence, &resource_limits).unwrap();
    let mut budget = native_header_budget();
    let before = budget.accounting();

    let headers = super::parse_av1_headers_with_frame_prefix_and_budget(&info, prefix, &mut budget)
        .expect("strict header materialization must succeed");
    let actual = headers
        .frame
        .tile_info
        .mi_col_starts
        .capacity()
        .checked_mul(std::mem::size_of::<u32>())
        .unwrap()
        + headers
            .frame
            .tile_info
            .mi_row_starts
            .capacity()
            .checked_mul(std::mem::size_of::<u32>())
            .unwrap()
        + headers
            .tile_group
            .group
            .tiles
            .capacity()
            .checked_mul(std::mem::size_of::<crate::av1::TilePayload>())
            .unwrap()
        + headers
            .decode_plan
            .planes
            .capacity()
            .checked_mul(std::mem::size_of::<crate::av1::PlaneLayout>())
            .unwrap()
        + headers
            .decode_plan
            .tiles
            .capacity()
            .checked_mul(std::mem::size_of::<crate::av1::TileDecodePlan>())
            .unwrap()
        + headers.tile_group.tile_data.capacity();
    assert_eq!(budget.accounting().frame_live, actual);
    assert!(actual > 0);
    assert!(
        headers.tile_group.entropy_states.is_empty(),
        "normal strict OBU_FRAME path must not allocate diagnostic entropy state"
    );

    super::drop_native_headers_and_release(headers, &mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, before.frame_live);
}

#[test]
fn native_header_materialization_uses_the_same_owners_for_split_frame_header_and_tile_group() {
    fn leb128(mut value: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            bytes.push(byte);
            if value == 0 {
                return bytes;
            }
        }
    }
    fn obu(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut result = vec![(kind << 3) | 0x02];
        result.extend(leb128(payload.len()));
        result.extend_from_slice(payload);
        result
    }

    let data = std::fs::read(sample_path("WML2Viewer.avif"))
        .expect("native header fixture must be present");
    let mut info = parse_avif(&data).expect("native header fixture must parse");
    let sequence = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .unwrap()
        .expect("sequence OBU must be present");
    let frame = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .unwrap()
        .expect("frame OBU must be present");
    let parsed_sequence = crate::av1::parse_sequence_header(sequence).unwrap();
    let parsed_frame = crate::av1::parse_frame_header(frame, &parsed_sequence).unwrap();
    let tile_start = parsed_frame.payload_after_header_offset;
    assert!(tile_start < frame.len(), "fixture must include tile data");
    info.primary_item_payload = [
        obu(1, sequence),
        obu(3, &frame[..tile_start]),
        obu(4, &frame[tile_start..]),
    ]
    .concat();

    let resource_limits = limits(data.len());
    let bounded_sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let prefix =
        validate_native_frame_prefix_limits(&info, &bounded_sequence, &resource_limits).unwrap();
    let mut budget = native_header_budget();
    let headers = super::parse_av1_headers_with_frame_prefix_and_budget(&info, prefix, &mut budget)
        .expect("split strict header materialization must succeed");
    assert!(!headers.tile_group.from_frame_obu);
    assert_eq!(
        headers.tile_group.group.tiles.len(),
        headers.decode_plan.tiles.len()
    );
    super::drop_native_headers_and_release(headers, &mut budget).unwrap();
    assert_eq!(budget.accounting().frame_live, 0);
}

#[test]
fn native_header_drops_tile_vectors_before_releasing_their_tickets() {
    let data = std::fs::read(sample_path("WML2Viewer.avif"))
        .expect("native header fixture must be present");
    let info = parse_avif(&data).expect("native header fixture must parse");
    let resource_limits = limits(data.len());
    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let prefix = validate_native_frame_prefix_limits(&info, &sequence, &resource_limits).unwrap();
    let mut budget = native_header_budget();
    let observation = crate::test_allocation_observer::Observation::begin(usize::MAX, false);
    let _all_candidates = crate::test_allocation_observer::track_all_candidates();
    let headers = super::parse_av1_headers_with_frame_prefix_and_budget(&info, prefix, &mut budget)
        .expect("strict header materialization must succeed");

    super::drop_native_headers_and_release(headers, &mut budget).unwrap();

    assert!(
        observation
            .registered_drops()
            .into_iter()
            .any(|drops| drops != 0),
        "strict header must physically drop at least one ticketed tile vector"
    );
    assert!(observation.release_count() >= 1);
    assert!(
        observation.release_drop_snapshots()[0] > 0,
        "the first ticket release must observe a preceding vector drop"
    );
}

#[test]
fn native_normal_frame_tile_copy_rejects_actual_capacity_then_retries() {
    let data = std::fs::read(sample_path("WML2Viewer.avif"))
        .expect("native hookup fixture must be present");
    let info = parse_avif(&data).expect("native header fixture must parse");
    let source_snapshot = info.primary_item_payload.clone();
    let resource_limits = limits(data.len());
    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let prefix = validate_native_frame_prefix_limits(&info, &sequence, &resource_limits).unwrap();
    let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        "native AV1 frame tile data",
        1 << 20,
    );

    let mut reference = native_header_budget();
    let headers =
        super::parse_av1_headers_with_frame_prefix_and_budget(&info, prefix, &mut reference)
            .expect("reference normal OBU_FRAME assembly must succeed");
    assert!(headers.tile_group.from_frame_obu);
    let actual_peak = reference.accounting().aggregate_live;
    super::drop_native_headers_and_release(headers, &mut reference).unwrap();
    assert_eq!(reference.accounting().aggregate_live, 0);

    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let prefix = validate_native_frame_prefix_limits(&info, &sequence, &resource_limits).unwrap();
    let mut under = DecodeBudget::new(Some(actual_peak - 1));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let error =
        match super::parse_av1_headers_with_frame_prefix_and_budget(&info, prefix, &mut under) {
            Ok(_) => {
                panic!("actual final tile-data capacity must be admitted, not only requested bytes")
            }
            Err(error) => error,
        };
    assert!(
        matches!(error, crate::DecoderError::InvalidParam(message) if message.contains("live allocation"))
    );
    assert_eq!(under.accounting().aggregate_live, 0);
    assert_eq!(observation.drops(), 1);
    assert!(observation.restore_count() >= 1);
    assert_eq!(
        info.primary_item_payload, source_snapshot,
        "rejected tile-data candidate must not mutate the normal OBU_FRAME source"
    );
    drop(observation);

    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let prefix = validate_native_frame_prefix_limits(&info, &sequence, &resource_limits).unwrap();
    let mut retry = DecodeBudget::new(Some(actual_peak));
    let headers = super::parse_av1_headers_with_frame_prefix_and_budget(&info, prefix, &mut retry)
        .expect("same normal OBU_FRAME source must retry after a rejected actual capacity");
    super::drop_native_headers_and_release(headers, &mut retry).unwrap();
    assert_eq!(retry.accounting().aggregate_live, 0);
    assert_eq!(
        info.primary_item_payload, source_snapshot,
        "retry must reuse the unchanged normal OBU_FRAME source"
    );
}

#[test]
fn native_normal_frame_selector_is_borrowed_and_uses_the_frame_route() {
    let data = std::fs::read(sample_path("WML2Viewer.avif"))
        .expect("native hookup fixture must be present");
    let info = parse_avif(&data).expect("native header fixture must parse");

    let (frame, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
        super::select_native_normal_frame_obu_for_test(&info.primary_item_payload)
            .expect("normal OBU_FRAME selector must parse the stream")
            .map(|payload| (payload.as_ptr(), payload.len()))
    });
    assert_eq!(requests, 0, "normal OBU selector must stay borrowed");
    let (frame_ptr, frame_len) = frame.expect("fixture must select a normal OBU_FRAME");
    let source_start = info.primary_item_payload.as_ptr() as usize;
    let source_end = source_start + info.primary_item_payload.len();
    let frame_start = frame_ptr as usize;
    let frame_end = frame_start + frame_len;
    assert!(frame_start >= source_start && frame_end <= source_end);

    let resource_limits = limits(data.len());
    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let prefix = validate_native_frame_prefix_limits(&info, &sequence, &resource_limits).unwrap();
    let mut budget = native_header_budget();
    let headers = super::parse_av1_headers_with_frame_prefix_and_budget(&info, prefix, &mut budget)
        .expect("normal OBU_FRAME must use the strict native frame route");
    assert!(headers.tile_group.from_frame_obu);
    assert!(
        headers.tile_group.entropy_states.is_empty(),
        "normal header assembly must not route through legacy diagnostic state"
    );
    super::drop_native_headers_and_release(headers, &mut budget).unwrap();
    assert_eq!(budget.accounting().aggregate_live, 0);
}

#[test]
fn native_normal_frame_selector_keeps_split_frame_header_tile_group_on_legacy_route() {
    let split = [
        sized_obu(ObuType::FrameHeader, &[0x00]),
        sized_obu(ObuType::TileGroup, &[0x00]),
    ]
    .concat();
    let (selected, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
        super::select_native_normal_frame_obu_for_test(&split)
            .expect("split OBU stream must frame cleanly")
            .map(|payload| (payload.as_ptr(), payload.len()))
    });

    assert_eq!(requests, 0, "split OBU selection must remain borrowed");
    assert!(
        selected.is_none(),
        "split FRAME_HEADER plus TILE_GROUP must not enter normal OBU_FRAME routing"
    );
}

#[test]
fn native_normal_frame_selector_rejects_ambiguous_duplicate_frame_obus() {
    let ambiguous = [
        sized_obu(ObuType::Frame, &[0x11]),
        sized_obu(ObuType::Frame, &[0x22]),
    ]
    .concat();
    let (selected, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
        super::select_native_normal_frame_obu_for_test(&ambiguous)
            .expect("duplicate OBU stream must frame cleanly")
            .map(|payload| (payload.as_ptr(), payload.len()))
    });

    assert_eq!(requests, 0, "ambiguous OBU selection must remain borrowed");
    assert!(
        selected.is_none(),
        "ambiguous duplicate OBU_FRAME payloads must not enter normal routing"
    );
}

#[test]
fn strict_split_selector_is_borrowed_and_fail_closed() {
    let split = [
        sized_obu(ObuType::FrameHeader, &[0x00]),
        sized_obu(ObuType::TileGroup, &[0x00]),
    ]
    .concat();
    let (result, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
        super::select_strict_native_frame_route(&split)
    });
    assert_eq!(requests, 0, "strict split classifier must not allocate");
    assert!(matches!(
        result,
        Ok(super::StrictNativeFrameRoute::Split { .. })
    ));

    let interstitial = [
        sized_obu(ObuType::FrameHeader, &[0x00]),
        sized_obu(ObuType::SequenceHeader, &[0x00]),
        sized_obu(ObuType::TileGroup, &[0x00]),
    ]
    .concat();
    assert!(matches!(
        super::select_strict_native_frame_route(&interstitial),
        Err(crate::DecoderError::Unsupported(message)) if message.contains("unsupported OBU")
    ));

    let duplicate = [
        sized_obu(ObuType::Frame, &[0x00]),
        sized_obu(ObuType::Frame, &[0x00]),
    ]
    .concat();
    assert!(matches!(
        super::select_strict_native_frame_route(&duplicate),
        Err(crate::DecoderError::Unsupported(message)) if message.contains("exactly one")
    ));
}

#[test]
fn legacy_frame_prefix_wrapper_remains_outside_native_selector_budget_route() {
    let data = std::fs::read(sample_path("WML2Viewer.avif"))
        .expect("native hookup fixture must be present");
    let info = parse_avif(&data).expect("native header fixture must parse");
    let resource_limits = limits(data.len());
    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let prefix = validate_native_frame_prefix_limits(&info, &sequence, &resource_limits).unwrap();

    let headers = super::parse_av1_headers_with_frame_prefix(&info, prefix)
        .expect("legacy prefix wrapper must retain its established parser route");
    assert!(headers.tile_group.from_frame_obu);
    assert!(
        !headers.tile_group.entropy_states.is_empty(),
        "legacy wrapper must not adopt strict native selector/budget materialization"
    );
}

#[test]
fn native_normal_frame_tile_group_entries_reject_actual_capacity_then_retry() {
    let (source, _sequence, frame, _group, frame_payload) = normal_frame_components();
    let source_snapshot = source.clone();
    let start_bit_offset = frame.payload_after_header_offset * 8;
    let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        "native AV1 tile group entries",
        1 << 20,
    );

    let mut reference_budget = DecodeBudget::new(None);
    let (reference, mut reference_allocation) = crate::av1::parse_tile_group_with_budget(
        &frame_payload,
        start_bit_offset,
        &frame.tile_info,
        &mut reference_budget,
    )
    .expect("normal OBU_FRAME tile descriptors must materialize");
    let actual = capacity_bytes(&reference.tiles);
    assert!(
        actual > reference.tiles.len() * std::mem::size_of::<crate::av1::TilePayload>(),
        "test must exercise actual rather than requested tile-descriptor capacity"
    );
    drop(reference);
    reference_allocation.release(&mut reference_budget).unwrap();
    assert_eq!(reference_budget.accounting().aggregate_live, 0);

    let mut under = DecodeBudget::new(Some(actual - 1));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let error = match crate::av1::parse_tile_group_with_budget(
        &frame_payload,
        start_bit_offset,
        &frame.tile_info,
        &mut under,
    ) {
        Ok(_) => panic!("actual tile-descriptor capacity must be admitted before publication"),
        Err(error) => error,
    };
    assert!(
        matches!(error, crate::DecoderError::InvalidParam(message) if message.contains("live allocation"))
    );
    assert_eq!(under.accounting().aggregate_live, 0);
    assert_eq!(observation.drops(), 1);
    assert_eq!(observation.restore_count(), 1);
    assert_eq!(
        source, source_snapshot,
        "failure must leave OBU_FRAME source intact"
    );
    drop(observation);

    let mut retry = DecodeBudget::new(Some(actual));
    let (group, mut allocation) = crate::av1::parse_tile_group_with_budget(
        &frame_payload,
        start_bit_offset,
        &frame.tile_info,
        &mut retry,
    )
    .expect("same normal OBU_FRAME tile descriptors must retry at actual capacity");
    drop(group);
    allocation.release(&mut retry).unwrap();
    assert_eq!(retry.accounting().aggregate_live, 0);
    assert_eq!(
        source, source_snapshot,
        "retry must reuse normal OBU_FRAME source"
    );
}

#[test]
fn native_normal_frame_decode_plan_planes_reject_actual_capacity_then_retry() {
    let (source, sequence, frame, group, _frame_payload) = normal_frame_components();
    let source_snapshot = source.clone();
    let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        "native AV1 decode plan planes",
        1 << 20,
    );

    let mut reference_budget = DecodeBudget::new(None);
    let (reference, mut reference_allocation) = crate::av1::build_still_decode_plan_with_budget(
        &sequence,
        &frame,
        &group,
        &mut reference_budget,
    )
    .expect("normal OBU_FRAME plane plan must materialize");
    let actual = capacity_bytes(&reference.planes);
    let retry_peak = actual
        .checked_add(capacity_bytes(&reference.tiles))
        .expect("plan retry peak must fit usize");
    assert!(
        actual > reference.planes.len() * std::mem::size_of::<crate::av1::PlaneLayout>(),
        "test must exercise actual rather than requested plane-plan capacity"
    );
    drop(reference);
    reference_allocation.release(&mut reference_budget).unwrap();
    assert_eq!(reference_budget.accounting().aggregate_live, 0);

    let mut under = DecodeBudget::new(Some(actual - 1));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let error = match crate::av1::build_still_decode_plan_with_budget(
        &sequence, &frame, &group, &mut under,
    ) {
        Ok(_) => panic!("actual plane-plan capacity must be admitted before publication"),
        Err(error) => error,
    };
    assert!(
        matches!(error, crate::DecoderError::InvalidParam(message) if message.contains("live allocation"))
    );
    assert_eq!(under.accounting().aggregate_live, 0);
    assert_eq!(observation.drops(), 1);
    assert_eq!(observation.restore_count(), 1);
    assert_eq!(
        source, source_snapshot,
        "failure must leave OBU_FRAME source intact"
    );
    drop(observation);

    let mut retry = DecodeBudget::new(Some(retry_peak));
    let (plan, mut allocation) =
        crate::av1::build_still_decode_plan_with_budget(&sequence, &frame, &group, &mut retry)
            .expect("same normal OBU_FRAME plane plan must retry at actual capacity");
    drop(plan);
    allocation.release(&mut retry).unwrap();
    assert_eq!(retry.accounting().aggregate_live, 0);
    assert_eq!(
        source, source_snapshot,
        "retry must reuse normal OBU_FRAME source"
    );
}

#[test]
fn native_normal_frame_decode_plan_tiles_reject_actual_capacity_then_retry() {
    let (source, sequence, frame, group, _frame_payload) = normal_frame_components();
    let source_snapshot = source.clone();
    let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(
        "native AV1 decode plan tiles",
        1 << 20,
    );

    let mut reference_budget = DecodeBudget::new(None);
    let (reference, mut reference_allocation) = crate::av1::build_still_decode_plan_with_budget(
        &sequence,
        &frame,
        &group,
        &mut reference_budget,
    )
    .expect("normal OBU_FRAME tile plan must materialize");
    let plane_bytes = capacity_bytes(&reference.planes);
    let actual = capacity_bytes(&reference.tiles);
    let retry_peak = plane_bytes
        .checked_add(actual)
        .expect("plan retry peak must fit usize");
    assert!(
        actual > reference.tiles.len() * std::mem::size_of::<crate::av1::TileDecodePlan>(),
        "test must exercise actual rather than requested tile-plan capacity"
    );
    drop(reference);
    reference_allocation.release(&mut reference_budget).unwrap();
    assert_eq!(reference_budget.accounting().aggregate_live, 0);

    let mut under = DecodeBudget::new(Some(retry_peak - 1));
    let observation = crate::test_allocation_observer::Observation::begin(1, false);
    let error = match crate::av1::build_still_decode_plan_with_budget(
        &sequence, &frame, &group, &mut under,
    ) {
        Ok(_) => panic!("actual tile-plan capacity must be admitted before publication"),
        Err(error) => error,
    };
    assert!(
        matches!(error, crate::DecoderError::InvalidParam(message) if message.contains("live allocation"))
    );
    assert_eq!(under.accounting().aggregate_live, 0);
    assert_eq!(observation.drops(), 1);
    assert_eq!(observation.restore_count(), 1);
    assert_eq!(
        source, source_snapshot,
        "failure must leave OBU_FRAME source intact"
    );
    drop(observation);

    let mut retry = DecodeBudget::new(Some(retry_peak));
    let (plan, mut allocation) =
        crate::av1::build_still_decode_plan_with_budget(&sequence, &frame, &group, &mut retry)
            .expect("same normal OBU_FRAME tile plan must retry at actual capacity");
    drop(plan);
    allocation.release(&mut retry).unwrap();
    assert_eq!(retry.accounting().aggregate_live, 0);
    assert_eq!(
        source, source_snapshot,
        "retry must reuse normal OBU_FRAME source"
    );
}

#[test]
fn native_split_master_tile_owners_use_actual_capacity_and_retry() {
    let info = split_master_fixture();
    let source_snapshot = info.primary_item_payload.clone();
    let resource_limits = limits(source_snapshot.len().saturating_mul(2));

    for label in [
        "native AV1 split tile group entries",
        "native AV1 split frame tile data",
    ] {
        assert_split_actual_capacity_retry(label, |budget| {
            parse_split_master_headers_with_budget(&info, &resource_limits, budget)
        });
        assert_eq!(
            info.primary_item_payload, source_snapshot,
            "split master source must remain byte-identical after {label} M/M-1/retry"
        );
    }
}

#[test]
fn native_selected_alpha_split_tile_owners_use_actual_capacity_and_retry() {
    let data = std::fs::read(sample_path(
        "plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
    ))
    .expect("native split alpha fixture must be present");
    let info = parse_avif(&data).expect("native split alpha fixture must parse");
    let source_snapshot = info
        .alpha_auxiliary_items
        .first()
        .expect("native split alpha fixture must include alpha")
        .payload
        .clone();
    let resource_limits = limits(data.len());

    for label in [
        "native AV1 split tile group entries",
        "native AV1 split frame tile data",
    ] {
        assert_split_actual_capacity_retry(label, |budget| {
            parse_split_alpha_headers_with_budget(&info, &resource_limits, budget)
        });
        assert_eq!(
            info.alpha_auxiliary_items
                .first()
                .expect("selected alpha must remain present")
                .payload,
            source_snapshot,
            "selected alpha source must remain byte-identical after {label} M/M-1/retry"
        );
    }
}

#[test]
fn native_split_master_and_alpha_drop_final_owners_before_ticket_release() {
    let master = split_master_fixture();
    let master_limits = limits(master.primary_item_payload.len().saturating_mul(2));
    let alpha_data = std::fs::read(sample_path(
        "plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
    ))
    .expect("native split alpha fixture must be present");
    let alpha = parse_avif(&alpha_data).expect("native split alpha fixture must parse");
    let alpha_limits = limits(alpha_data.len());

    assert_split_final_owners_drop_before_ticket_release("master", |budget| {
        parse_split_master_headers_with_budget(&master, &master_limits, budget)
    });
    assert_split_final_owners_drop_before_ticket_release("selected alpha", |budget| {
        parse_split_alpha_headers_with_budget(&alpha, &alpha_limits, budget)
    });
}

#[test]
fn native_split_obu_error_contract_is_borrowed_fail_closed_and_restores_budget() {
    let header = [0u8];
    let tile = [0u8];
    let cases: [(&str, Vec<u8>); 5] = [
        ("tile-before-header", sized_obu(ObuType::TileGroup, &tile)),
        (
            "duplicate-header",
            [
                sized_obu(ObuType::FrameHeader, &header),
                sized_obu(ObuType::FrameHeader, &header),
                sized_obu(ObuType::TileGroup, &tile),
            ]
            .concat(),
        ),
        (
            "mixed-frame-and-tile",
            [
                sized_obu(ObuType::Frame, &header),
                sized_obu(ObuType::TileGroup, &tile),
            ]
            .concat(),
        ),
        (
            "non-tile-between-header-and-group",
            [
                sized_obu(ObuType::FrameHeader, &header),
                sized_obu(ObuType::SequenceHeader, &header),
                sized_obu(ObuType::TileGroup, &tile),
            ]
            .concat(),
        ),
        (
            "missing-tile-group",
            sized_obu(ObuType::FrameHeader, &header),
        ),
    ];

    for (name, payload) in cases {
        let result = super::select_strict_native_frame_route(&payload);
        assert!(result.is_err(), "{name} must be rejected");
    }

    let tile_info = two_tile_info();
    let malformed = [
        ("reverse-range", split_stream(&header, &[&[0b1100_0000]])),
        ("hole", split_stream(&header, &[&[0b1110_0000, 0x55]])),
        ("bad-size", split_stream(&header, &[&[0]])),
        (
            "duplicate-tile",
            split_stream(&header, &[&[0, 0x55], &[0, 0x66]]),
        ),
    ];
    for (name, payload) in malformed {
        let budget = DecodeBudget::new(Some(1024));
        let before = budget.accounting();
        let result = super::walk_strict_split_tile_groups(&payload, &tile_info, |_, _| Ok(()));
        assert!(result.is_err(), "{name} must be rejected");
        assert_eq!(
            budget.accounting(),
            before,
            "{name} must leave budget untouched"
        );
    }

    let complete = split_stream(&header, &[&[0b1000_0000, 0x55], &[0b1110_0000, 0x66]]);
    let (result, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
        super::walk_strict_split_tile_groups(&complete, &tile_info, |_, _| Ok(()))
    });
    assert_eq!(
        requests, 0,
        "the complete split validation pass must not create per-tile payload owners"
    );
    assert_eq!(result.unwrap().tile_count, 2);
}
